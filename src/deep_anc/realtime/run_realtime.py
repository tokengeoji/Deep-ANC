"""실시간 Deep ANC 런타임 — 3-스레드 구조 (콜백 / 추론 / 제어).

  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml
  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --set controller=fxlms
  python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate

구조 (docs/06):
  [콜백]   입력 변환/DC차단 → in_ring, 소음(ch0) 생성, out_ring→리미터/게이트→ch1 출력
  [추론]   in_ring 에서 hop 단위 소비 → engine.step → out_ring  (콜백은 절대 대기 안 함)
  [제어]   키보드, 1초 통계, 워치독 메시지
파이프라인 핸드오프 지연 = 1 hop — 학습 플랜트의 handoff_extra_samples 와 정합 [C1].
이 1 hop 은 이제 주석이 아니라 **강제**다: 입출력 링버퍼의 백로그 허용치를
safety.PipelineHandoffBudget 이 단 한 곳에서 유도하고, 비대칭이면 생성 자체가
거부된다 (예전에는 입력만 8 hop = 42.7 ms 여서 추론이 뒤처지면 상쇄가 증폭이 됐다).
시작은 항상 ANC OFF. 시스템(전원모드/RT우선순위 등)은 건드리지 않는다 — 프로젝트 정책.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..audio_io import (
    assert_measurement_preconditions,
    capture_input_probe,
    float32_to_pcm_int16,
    format_sounddevice_devices,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
)
from ..config import REPO_ROOT, load_runtime_config
from ..dsp.filters import DCBlocker
from .engines import (
    build_engine,
    checkpoint_digital_reference_lead_samples,
    engine_digital_reference_lead_samples_from_config,
    secondary_path_npz,
)
from .clock_telemetry import (
    ClockTelemetryRecorder,
    RuntimeCounterSnapshot,
    bind_recording_to_clock_receipt,
    payload_sha256,
    sha256_file,
    write_clock_receipt_exclusive,
)
from .noise_gen import DigitalReferenceBuffer, NoiseProgram
from .plant_contract import validate_runtime_plant_contract
from .ring_buffer import SPSCRing
from .safety import (
    BlockObservation,
    FadeGate,
    PipelineHandoffBudget,
    PowerEMA,
    SafetySupervisor,
)
from .ui import KeyboardController, RuntimeState


def power_to_db(power: float, floor_db: float = -200.0) -> float:
    if not np.isfinite(power) or power <= 0.0:
        return floor_db
    return max(floor_db, 10.0 * float(np.log10(power)))


def fxlms_adaptation_allowed(
    *,
    requested: bool,
    full_anc_gain: bool,
    full_noise_gain: bool,
    hold_samples: int,
    output_clip_fraction: float,
    input_clip_fraction: float,
    reference_power: float,
    stream_ok: bool,
) -> bool:
    """FxLMS가 현재 ERR 블록으로 갱신해도 되는 안전 조건."""
    return bool(
        requested
        and full_anc_gain
        and full_noise_gain
        and int(hold_samples) == 0
        and float(output_clip_fraction) == 0.0
        and float(input_clip_fraction) == 0.0
        and float(reference_power) > 1.0e-12
        and stream_ok
    )


def reduction_measurement_eligibility(
    *,
    anc_enabled: bool,
    anc_output_active: bool,
    baseline_valid: bool,
    error_power: float,
    had_output_data: bool,
    callback_status: bool,
    xruns: int,
    fallback_silence_blocks: int,
    deadline_miss_blocks: int,
    engine_error_blocks: int,
    full_anc_gain: bool,
    full_noise_gain: bool,
) -> tuple[bool, str]:
    """실시간 ``저감``을 유효한 OFF/ON 비교로 표시할 수 있는지 판정한다.

    ``anc_output_active``만 보면 안 된다. 게이트가 열려 있어도 output ring이 비어
    fallback 무음을 냈거나, 페이드 중이거나, callback/inference timing 오류가 있으면
    현재 ERR은 상쇄음에 대한 측정값이 아니다. 이 함수는 UI 숫자를 성능 증거로
    오인하지 않도록 fail-closed 한다.
    """

    reasons: list[str] = []
    if not anc_enabled:
        reasons.append("anc_off")
    if not anc_output_active:
        reasons.append("output_gate")
    if not baseline_valid:
        reasons.append("baseline")
    if not np.isfinite(error_power) or error_power <= 0.0:
        reasons.append("error_power")
    if not had_output_data:
        reasons.append("fallback")
    if callback_status:
        reasons.append("callback_status")
    if int(xruns) > 0:
        reasons.append("xrun")
    if int(fallback_silence_blocks) > 0:
        reasons.append("fallback")
    if int(deadline_miss_blocks) > 0:
        reasons.append("deadline")
    if int(engine_error_blocks) > 0:
        reasons.append("engine_error")
    if not full_anc_gain or not full_noise_gain:
        reasons.append("fade")
    return (not reasons, ",".join(dict.fromkeys(reasons)) or "ok")


def input_preflight(cfg: dict, seconds: float = 2.0) -> bool:
    """스피커를 열기 전에 필수 I2S 입력 채널이 살아 있는지 확인한다."""
    report = capture_input_probe(cfg["hardware"]["audio"], seconds=seconds)
    names = ("ERR", "REF")
    for item in report["channels"][:2]:
        index = int(item["channel"])
        verdict = "PASS" if item["valid"] else "FAIL"
        print(
            f"[{verdict}] {names[index]} ch{index}: RMS {item['rms_dbfs']:.2f}dBFS, "
            f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}, "
            f"unique {item['unique_codes']}, raw [{item['raw_min']}, {item['raw_max']}]"
        )

    required = (0, 1) if cfg.get("reference") == "mic" else (0,)
    failed = [index for index in required if not report["channels"][index]["valid"]]
    if failed:
        labels = ", ".join(f"{names[index]} ch{index}" for index in failed)
        print(
            f"[중단] 필수 입력({labels})이 무효입니다. 오디오 출력을 시작하지 않습니다.",
            file=sys.stderr,
        )
        return False
    if not report["channels"][1]["valid"]:
        print(
            "[경고] REF ch1 무효 — digital-reference만 허용하며 mic-reference는 금지합니다.",
            file=sys.stderr,
        )
    return True


_ENGINE_ARTIFACT_KEYS = {
    "torch": ("ckpt",),
    "ort": ("onnx",),
    "trt": ("plan",),
}
"""엔진 종류별로 **실제 로드에 쓰이는** 키. 나머지 키는 읽히지 않는다."""

_ENGINE_ARTIFACT_SUFFIX = {"ckpt": (".pt", ".pth"), "onnx": (".onnx",), "plan": (".plan", ".engine")}
"""키별로 허용되는 확장자. 파일 시스템을 보지 않고도 판정할 수 있는 부패 검사다."""


class EngineArtifactIssue(BaseModel):
    """엔진 아티팩트 preflight 가 찾은 문제 하나.

    왜 문자열이 아니라 타입인가
    --------------------------
    2026-08-06 실측 반증: ``main()`` 이 이 목록을 통째로 fail-closed 로 처리해서
    ``engine.type=ort`` 인 배포가 **읽히지도 않는** ``ckpt``/``plan`` 이 없다는
    이유로 시작을 거부했다. 모델은 GitHub Release 로 배포되므로 "필요한 것만 받은
    트리" 가 정상 배포인데 그것이 하드 중단된 것이다.

    호출부가 "무엇이 치명적인가" 를 문자열 매칭으로 다시 판정하면 그것이 두 번째
    유도가 된다(발생기 A). 그래서 치명 여부를 **여기서 한 번** 정하고 타입에 실어
    보낸다: :attr:`fatal` 은 "이것 때문에 오디오를 열면 안 되는가" 이고,
    나머지는 설정 부패 경고다 — 저장소 위생 검사(pytest)는 경고까지 전부 막는다.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    active: bool
    """이 키를 현재 engine.type 이 실제로 로드하는가."""

    missing_file: bool
    """파일 부재가 원인인가 (배포에서 아직 안 받았을 수 있는 종류)."""

    fatal: bool
    """시작 자체를 막아야 하는가."""

    detail: str

    def __str__(self) -> str:  # 로그/에러 메시지에서 그대로 쓰인다
        return self.detail


def engine_artifact_preflight(
    cfg: dict, *, require_all: bool = True
) -> list[EngineArtifactIssue]:
    """``engine`` 블록이 가리키는 파일이 실제로 존재하는지 오디오를 열기 전에 본다.

    왜 활성 엔진만 보면 안 되는가
    ----------------------------
    2026-08-05 실측: ``configs/runtime_tiny.yaml`` 의 ``plan:
    runs/export/tiny_fp16.plan`` 은 **존재하지 않는 파일**이었다. 실제 파일 이름은
    ``tiny_corrected_fp16.plan`` 이다. 그런데 ``engine.type`` 이 ``ort`` 라 이 키는
    한 번도 읽히지 않았고, 따라서 **조용히 썩어 있었다** — ``trt`` 로 바꾸는 순간
    터졌을 것이고, 그 시점은 대개 실기 앞이다.

    그래서 기본값 ``require_all=True`` 는 **선언된 모든 아티팩트**를 검사한다.

    왜 그것이 시작을 막으면 안 되는가
    --------------------------------
    ``runs/`` 는 ``.gitignore`` 대상이고 모델은 GitHub Release 로 배포된다
    (``git ls-files runs/`` = 0). ``engine.type=ort`` 배포가 onnx 하나만 받는 것은
    **정상**이다. 그러므로 "선언됐는데 파일이 없다"는 활성 키에서만 치명적이고,
    미사용 키에서는 경고다. 확장자 오류처럼 **파일 시스템과 무관한 부패**는
    어디서나 치명적이다 — 그것은 받아 놓지 않은 것이 아니라 잘못 적은 것이다.
    """

    engine = (cfg or {}).get("engine", {}) or {}
    kind = str(engine.get("type", "torch"))
    if kind not in _ENGINE_ARTIFACT_KEYS:
        return [
            EngineArtifactIssue(
                key="type",
                active=True,
                missing_file=False,
                fatal=True,
                detail=f"알 수 없는 engine.type={kind!r}; 허용={sorted(_ENGINE_ARTIFACT_KEYS)}",
            )
        ]

    active_keys = _ENGINE_ARTIFACT_KEYS[kind]
    checked = (
        tuple(key for key in ("ckpt", "onnx", "plan") if engine.get(key))
        if require_all
        else active_keys
    )
    problems: list[EngineArtifactIssue] = []
    for key in active_keys:
        if not engine.get(key):
            problems.append(
                EngineArtifactIssue(
                    key=key,
                    active=True,
                    missing_file=False,
                    fatal=True,
                    detail=(
                        f"engine.type={kind} 인데 engine.{key} 가 비었습니다 — "
                        "로드할 것이 없습니다"
                    ),
                )
            )
    for key in checked:
        value = str(engine.get(key, ""))
        if not value:
            continue
        active = key in active_keys
        role = "활성" if active else "미사용(지금은 읽히지 않음)"
        suffixes = _ENGINE_ARTIFACT_SUFFIX[key]
        if not value.endswith(suffixes):
            problems.append(
                EngineArtifactIssue(
                    key=key,
                    active=active,
                    missing_file=False,
                    fatal=True,
                    detail=(
                        f"engine.{key} 확장자가 규약과 다릅니다 [{role}]: {value} — "
                        f"허용 {list(suffixes)}. 이것은 아티팩트를 안 받은 것이 아니라 "
                        "설정을 잘못 적은 것이므로 어떤 환경에서도 거부합니다"
                    ),
                )
            )
            continue
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            problems.append(
                EngineArtifactIssue(
                    key=key,
                    active=active,
                    missing_file=True,
                    fatal=active,
                    detail=(
                        f"engine.{key} 아티팩트가 없습니다 [{role}]: {value} — "
                        "설정에 적혀 있으면 존재해야 합니다. 쓰지 않는다면 그 줄을 "
                        "지우고, 배포라면 GitHub Release 에서 받으세요"
                    ),
                )
            )
    return problems


def require_engine_artifacts(cfg: dict, *, require_all: bool = True) -> None:
    """:func:`engine_artifact_preflight` 를 실패 폐쇄로 감싼다 (저장소 위생 검사용)."""

    problems = engine_artifact_preflight(cfg, require_all=require_all)
    if problems:
        raise FileNotFoundError(
            "런타임 엔진 아티팩트 preflight 실패:\n- "
            + "\n- ".join(item.detail for item in problems)
        )


def require_engine_artifacts_to_start(cfg: dict) -> list[str]:
    """오디오를 열기 전 preflight — **치명적인 것만** 시작을 막는다.

    반환값은 사람에게 보여 줄 경고 목록(치명적이지 않은 문제)이다.
    """

    problems = engine_artifact_preflight(cfg, require_all=True)
    fatal = [item for item in problems if item.fatal]
    if fatal:
        raise FileNotFoundError(
            "런타임 엔진 아티팩트 preflight 실패:\n- "
            + "\n- ".join(item.detail for item in fatal)
        )
    return [item.detail for item in problems]


def validate_digital_reference_lead(
    reference: str,
    configured_lead: int,
    checkpoint_lead: int | None = None,
) -> int:
    """reference 모드와 digital lead 정합을 검증하고 정규화된 값을 반환한다.

    ``digital_reference_lead_samples``는 digital-reference에서만 의미가 있다.
    acoustic-reference에서는 체크포인트의 digital lead metadata를 비교하면 안 된다.
    두 모드는 같은 ``[reference, error]`` 입력 모양을 쓰더라도 reference의 시간축
    계약이 서로 다르기 때문이다. acoustic 모드는 runtime lead=0만 허용하고, 실제
    inference 입력 선택은 ``ref_mic`` 경로가 담당한다.
    """
    lead = int(configured_lead)
    if lead < 0:
        raise ValueError("digital_reference_lead_samples는 0 이상이어야 합니다")
    if reference != "digital":
        if lead:
            raise ValueError(
                "digital_reference_lead_samples는 reference=digital에서만 사용할 수 있습니다"
            )
        # acoustic-reference에서는 checkpoint/ONNX의 digital lead를 검증 대상에서
        # 제외한다. checkpoint가 digital로 학습됐는지는 별도의 실험 경고/모델 계약
        # 문제이며, acoustic 입력을 선택하는 런타임 자체를 막는 값은 아니다.
        return 0
    if checkpoint_lead is not None and lead != int(checkpoint_lead):
        raise ValueError(
            "digital-reference lead 불일치: "
            f"runtime={lead}, checkpoint={int(checkpoint_lead)}. "
            "학습과 배포의 digital_reference_lead_samples를 동일하게 맞추세요."
        )
    return lead


_LEGACY_DIAGNOSTIC_LEAD_SAMPLES = 109
_LEGACY_DIAGNOSTIC_PHYSICS_STATUS = "secondary_surrogate_representation_pretrain"


def validate_legacy_diagnostic_descriptor(
    cfg: dict,
    *,
    checkpoint_cfg: dict,
    checkpoint_lead: int,
    artifact_lead: int,
) -> None:
    """기존 surrogate pretrain을 다시 확인할 때만 허용하는 명시적 계약.

    이 경로는 strict P/S를 느슨하게 만드는 일반 우회로가 아니다. 현재 저장소에
    남아 있는 ``secondary_surrogate_representation_pretrain`` + lead=109 artifact만
    지정된 ``--legacy-diagnostic`` CLI 경로에서 허용한다. checkpoint/ONNX가 서로
    맞는지는 계속 검사하고, 실제 plant가 맞는지는 별도의 current strict 계약으로
    판정한다.
    """

    reference = str(cfg.get("reference", "digital"))
    controller = str(cfg.get("controller", "dl"))
    engine_type = str((cfg.get("engine") or {}).get("type", "torch"))
    if reference != "digital" or controller != "dl" or engine_type != "ort":
        raise ValueError(
            "--legacy-diagnostic은 reference=digital, controller=dl, engine.type=ort인 "
            "기존 Tiny surrogate runtime에서만 사용할 수 있습니다"
        )

    configured_lead = validate_digital_reference_lead(
        reference, cfg.get("digital_reference_lead_samples", 0)
    )
    if configured_lead != _LEGACY_DIAGNOSTIC_LEAD_SAMPLES:
        raise ValueError(
            "--legacy-diagnostic은 legacy lead=109 설정에서만 사용할 수 있습니다: "
            f"runtime={configured_lead}"
        )
    if str(checkpoint_cfg.get("physics_status", "")) != _LEGACY_DIAGNOSTIC_PHYSICS_STATUS:
        raise ValueError(
            "--legacy-diagnostic 대상 checkpoint가 기존 surrogate pretrain이 아닙니다: "
            f"physics_status={checkpoint_cfg.get('physics_status')!r}"
        )
    checkpoint_data = checkpoint_cfg.get("data") or {}
    if str(checkpoint_data.get("reference_mode", "digital")) != "digital":
        raise ValueError("legacy diagnostic checkpoint의 reference_mode가 digital이 아닙니다")
    if str(checkpoint_data.get("digital_primary_path_mode", "")) != "secondary_surrogate":
        raise ValueError(
            "legacy diagnostic checkpoint의 digital_primary_path_mode가 "
            "secondary_surrogate가 아닙니다"
        )

    if int(checkpoint_lead) != _LEGACY_DIAGNOSTIC_LEAD_SAMPLES:
        raise ValueError(
            "legacy diagnostic checkpoint lead가 109가 아닙니다: "
            f"checkpoint={int(checkpoint_lead)}"
        )
    if int(artifact_lead) != _LEGACY_DIAGNOSTIC_LEAD_SAMPLES:
        raise ValueError(
            "legacy diagnostic ONNX metadata lead가 109가 아닙니다: "
            f"onnx={int(artifact_lead)}"
        )
    validate_digital_reference_lead(reference, configured_lead, int(checkpoint_lead))
    validate_digital_reference_lead(reference, configured_lead, int(artifact_lead))


def validate_legacy_diagnostic_config(cfg: dict) -> None:
    """legacy diagnostic 실행 전 checkpoint/ONNX의 정체성과 lead를 검증한다."""

    engine = cfg.get("engine") or {}
    checkpoint_value = engine.get("ckpt")
    if not checkpoint_value:
        raise ValueError("--legacy-diagnostic에는 engine.ckpt가 필요합니다")
    checkpoint_path = Path(str(checkpoint_value)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"legacy diagnostic checkpoint가 없습니다: {checkpoint_path}"
        )

    import torch

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"legacy diagnostic checkpoint를 읽을 수 없습니다: {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise ValueError(
            f"legacy diagnostic checkpoint에 resolved cfg가 없습니다: {checkpoint_path}"
        )
    checkpoint_cfg = state["cfg"]
    checkpoint_lead = checkpoint_digital_reference_lead_samples(state)
    artifact_lead = engine_digital_reference_lead_samples_from_config(cfg)
    validate_legacy_diagnostic_descriptor(
        cfg,
        checkpoint_cfg=checkpoint_cfg,
        checkpoint_lead=checkpoint_lead,
        artifact_lead=int(artifact_lead if artifact_lead is not None else -1),
    )


class RealtimeANC:
    """프로그래밍 API — evaluate_session 등에서 재사용. CLI 는 main() 참조."""

    def __init__(
        self,
        cfg: dict,
        record_seconds: float = 0.0,
        *,
        validate_plant_contract: bool = True,
    ) -> None:
        if bool(cfg.get("start_on", False)):
            raise ValueError(
                "안전 규약상 start_on=true는 허용되지 않습니다. "
                "ANC는 OFF로 시작한 뒤 현장에서 명시적으로 켜야 합니다."
            )

        # 실제 P/S와의 lead 검증은 sounddevice import·engine 생성보다도 앞선다.
        # legacy 109-sample artifact가 strict plant(115)에 맞는 것처럼 시작하는 경로를
        # 없앤다. --calibrate의 ChirpEngine은 모델 배포가 아니라 별도 경로 지연 측정이라
        # 해당 함수가 명시적으로 이 검사를 끈다.
        self.plant_contract = (
            validate_runtime_plant_contract(cfg) if validate_plant_contract else None
        )
        self._runtime_deployment_snapshot_start = None
        self._runtime_physical_fingerprint_start = None
        if (
            float(record_seconds) > 0.0
            and str(cfg.get("controller", "dl")) == "dl"
            and self.plant_contract is not None
        ):
            # 실시간 증거를 녹음할 때는 실제로 로드할 checkpoint/export/
            # sidecar/P/S bytes를 stream open 전에 고정한다. 종료 후 동일
            # snapshot을 다시 계산해 캡처 중 변경을 거부한다.
            from ..eval.broadband_runtime import snapshot_runtime_deployment_files

            self._runtime_deployment_snapshot_start = (
                snapshot_runtime_deployment_files(
                    runtime_cfg=cfg,
                    plant=self.plant_contract,
                    repo_root=REPO_ROOT,
                )
            )
            from ..dsp.measurement_level import collect_alsa_physical_fingerprint

            self._runtime_physical_fingerprint_start = (
                collect_alsa_physical_fingerprint(cfg["hardware"])
            )

        reference = str(cfg.get("reference", "digital"))
        digital_reference_lead = validate_digital_reference_lead(
            reference, cfg.get("digital_reference_lead_samples", 0)
        )

        # API 호출도 CLI preflight를 우회할 수 없다. artifact의 lead를 먼저 metadata만
        # 읽어 비교하므로 legacy config를 115로 덮어도 PortAudio import 이전에 막힌다.
        preflight_engine_lead = engine_digital_reference_lead_samples_from_config(cfg)
        validate_digital_reference_lead(
            reference, digital_reference_lead, preflight_engine_lead
        )

        import sounddevice as sd

        self.sd = sd
        self.cfg = cfg
        hw = cfg["hardware"]["audio"]
        self.fs = int(hw["sample_rate"])
        self.block = int(hw["block_size"])
        self.latency = str(hw.get("latency", "low"))
        if self.latency not in {"low", "high"}:
            raise ValueError(f"hardware.audio.latency는 low/high여야 합니다: {self.latency!r}")
        self.hop = int(cfg.get("hop", self.block))
        if self.hop != self.block:
            raise ValueError("현재 구현은 hop == block_size 를 요구합니다")
        ch = cfg["hardware"]["channels"]
        self.ch_err, self.ch_ref = int(ch["error_mic"]), int(ch["reference_mic"])
        self.ch_noise, self.ch_cancel = int(ch["noise_out"]), int(ch["cancel_out"])
        self.reference = reference
        self.digital_reference_lead = digital_reference_lead

        self.in_dev = resolve_alsa_portaudio_device(hw["input"]["card"], hw["input"]["pcm"], "input", 2)
        self.out_dev = resolve_alsa_portaudio_device(hw["output"]["card"], hw["output"]["pcm"], "output", 2)

        self.engine = build_engine(cfg)
        checkpoint_lead = getattr(self.engine, "digital_reference_lead_samples", None)
        validate_digital_reference_lead(
            self.reference, self.digital_reference_lead, checkpoint_lead
        )
        self.program = NoiseProgram(cfg.get("noise", {}), self.fs)
        self.digital_reference_buffer = DigitalReferenceBuffer(self.digital_reference_lead)

        dc_r = float(cfg["hardware"].get("dc_blocker_r", 0.995))
        self.err_dc, self.ref_dc = DCBlocker(dc_r), DCBlocker(dc_r)

        self.safety = SafetySupervisor(cfg.get("safety", {}), self.fs, self.block)
        # 페이드 길이도 검증된 설정에서만 읽는다 (cfg.get 재유도 금지).
        fade = int(self.safety.limits.fade_ms * self.fs / 1000.0)
        self._fade_samples = fade
        # 링버퍼 백로그 예산의 단일 출처. 입력 8 hop / 출력 1 hop 의 비대칭은
        # 여기서 구조적으로 불가능하다 (safety.PipelineHandoffBudget 참조).
        self.handoff_budget = PipelineHandoffBudget.derive(
            duct_cfg=cfg.get("duct", {}), hop=self.hop
        )
        self.state = RuntimeState(start_on=False)
        self.anc_gate = FadeGate(fade, initial=0.0)
        self.noise_gate = FadeGate(max(fade, int(0.1 * self.fs)), initial=0.0)
        self.noise_gate.set_target(1.0)

        # 네 번째 채널은 callback이 판정한 FxLMS 적응 허용 플래그다. 가중치
        # 소유자인 추론 스레드만 이 플래그를 읽어 adapt 상태를 바꾼다.
        self.in_ring = SPSCRing(4, self.hop * 64)      # err, ref_mic, ref_digital, adapt
        self.out_ring = SPSCRing(1, self.hop * 64)
        self.clock_telemetry = ClockTelemetryRecorder(
            sample_rate=self.fs,
            block_size=self.block,
            input_device=(
                f"{hw['input']['card']}:{int(hw['input']['pcm'])} "
                f"(PortAudio index {self.in_dev})"
            ),
            output_device=(
                f"{hw['output']['card']}:{int(hw['output']['pcm'])} "
                f"(PortAudio index {self.out_dev})"
            ),
            allowed_input_backlog_samples=(
                self.handoff_budget.input_keep_backlog_samples
            ),
            allowed_output_backlog_samples=(
                self.handoff_budget.output_keep_backlog_samples
            ),
        )

        self.err_meter = PowerEMA(self.fs, 0.4)
        self.ctrl_meter = PowerEMA(self.fs, 0.4)
        # 베이스라인(=ANC 없는 에러 파워)의 수집·유효성·강제 갱신은 전부
        # self.safety.baseline 이 소유한다. 예전에는 EMA 가 여기, 유효성 규칙이
        # safety 에 있어서 같은 물리량의 부기가 두 파일로 갈라져 있었다(발생기 A).
        self._last_input_drops = 0
        self.step_times_ms: list[float] = []
        self.xruns = 0
        # inference ``engine.step`` 자체가 1 block wall-time을 넘긴 횟수다.
        # output ring이 비어 실제 무음을 낸 fallback과 같은 사건으로 중복 계상하지 않는다.
        self._deadline_miss_blocks = 0
        # engine.step 예외를 무음으로 바꾸는 것은 오디오 안전을 위해
        # 필요하지만, 그 블록을 정상 추론으로 숨기면 latency/연속성 증거가
        # 위조된다. deadline/fallback과 독립적으로 exact 0을 강제한다.
        self._engine_error_blocks = 0
        self._fallback_silence_blocks = 0
        # 현재 runtime은 hard sample insertion으로 clock을 맞추지 않는다. 향후
        # rate matcher가 추가되면 이 counter를 실제 삽입 sample 수에 연결해야 한다.
        self._ring_add_samples = 0
        # 첫 callback이 추론 결과 없이 시작하는 구조적 fallback을 없애기 위한
        # 의도된 1-hop 무음 프라임. start()가 오디오 stream을 열기 전에 단 한
        # 번만 발행하며, runtime raw에 그 수를 보존한다.
        self._intentional_startup_prime_blocks = 0
        self._inference_step_count = 0
        self._last_anc = False
        self._adaptation_hold_samples = 0

        self.record_len = int(record_seconds * self.fs)
        self._record_runtime_evidence = self.record_len > 0
        self.rec_pos = 0
        if self.record_len > 0:
            self.rec = {
                "err": np.zeros(self.record_len, dtype=np.float32),
                "ref": np.zeros(self.record_len, dtype=np.float32),
                "source": np.zeros(self.record_len, dtype=np.float32),
                "control": np.zeros(self.record_len, dtype=np.float32),
                "anc_gain": np.zeros(self.record_len, dtype=np.float32),
            }
        else:
            self.rec = None

        self._infer_thread: threading.Thread | None = None
        self._stream = None

    # ---------- 콜백 (PortAudio 스레드) ----------

    def _ensure_clock_telemetry(self) -> ClockTelemetryRecorder:
        """정상 생성 경로와 기존 ``__new__`` 기반 callback harness를 함께 지원한다."""

        recorder = getattr(self, "clock_telemetry", None)
        if recorder is not None:
            return recorder
        budget = self.handoff_budget
        recorder = ClockTelemetryRecorder(
            sample_rate=int(self.fs),
            block_size=int(self.block),
            input_device=f"unbound (PortAudio index {getattr(self, 'in_dev', 'unknown')})",
            output_device=f"unbound (PortAudio index {getattr(self, 'out_dev', 'unknown')})",
            allowed_input_backlog_samples=int(
                budget.input_keep_backlog_samples
            ),
            allowed_output_backlog_samples=int(
                budget.output_keep_backlog_samples
            ),
        )
        self.clock_telemetry = recorder
        return recorder

    def _runtime_counter_snapshot(self) -> RuntimeCounterSnapshot:
        return RuntimeCounterSnapshot(
            xrun_count=int(getattr(self, "xruns", 0)),
            deadline_miss_count=int(getattr(self, "_deadline_miss_blocks", 0)),
            engine_error_blocks=int(getattr(self, "_engine_error_blocks", 0)),
            input_ring_drop_samples=int(self.in_ring.drops),
            output_ring_drop_samples=int(self.out_ring.drops),
            input_ring_overrun_blocks=int(self.in_ring.overruns),
            output_ring_overrun_blocks=int(self.out_ring.overruns),
            input_ring_underrun_blocks=int(self.in_ring.underruns),
            output_ring_underrun_blocks=int(self.out_ring.underruns),
            ring_add_samples=int(getattr(self, "_ring_add_samples", 0)),
            input_backlog_samples=int(self.in_ring.available()),
            output_backlog_samples=int(self.out_ring.available()),
            fallback_silence_blocks=int(
                getattr(self, "_fallback_silence_blocks", 0)
            ),
            watchdog_trip_counts={
                item.value: int(count)
                for item, count in self.safety.trip_counts.items()
            },
        )

    def clock_telemetry_receipt(self) -> dict:
        """현재 session의 fail-closed clock/queue receipt payload를 반환한다."""

        return self._ensure_clock_telemetry().build_receipt(
            final_snapshot=self._runtime_counter_snapshot()
        )

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        clock_token = None
        clock_entry_snapshot = None
        clock_telemetry = self._ensure_clock_telemetry()
        try:
            # PortAudio time_info를 버리지 않는다. 이 frame counter는 callback에서
            # 관측한 exact 정수일 뿐, callback 전에 silent-drop된 ADC period가 없다는
            # 물리 증거는 아니다(clock receipt는 그래서 최대 INCONCLUSIVE다).
            clock_token = clock_telemetry.begin_callback(
                frames=frames, time_info=time_info, status=status
            )
            if status:
                self.xruns += 1
            # output ring을 pop하기 전 backlog를 보존해야 callback 종료 snapshot에서
            # 사라지는 순간 최대치를 놓치지 않는다.
            clock_entry_snapshot = self._runtime_counter_snapshot()

            mics = pcm_int32_to_float32(indata[:, :2])
            err = self.err_dc.process(mics[:, self.ch_err])
            ref_mic = self.ref_dc.process(mics[:, self.ch_ref])

            noise_gain = self.noise_gate.process(frames)
            self.noise_gate.set_target(1.0 if self.state.noise_enabled else 0.0)
            # 자기생성 소스는 지금 만든 신호를 ref로 즉시 공급하고, 실제 ch0 재생만
            # lead만큼 늦춘다. 게이트도 같은 FIFO를 통과시켜 ON/OFF 전환 중에도
            # ref[t] == source[t+lead] 정렬을 보존한다 (digital-ref 전용).
            future_source = self.program.generate(frames) * noise_gain
            played, future = self.digital_reference_buffer.process(
                np.stack([future_source, noise_gain])
            )
            source, played_noise_gain = played
            ref_digital = future[0]

            # 백로그가 예산을 넘으면 최신으로 재동기 — 언더런에 의한 지연 누적 방지 (#9).
            # 허용치는 handoff_budget 이 유일하게 유도한다 (입출력 대칭 강제).
            y_blk, had_data = self.out_ring.pop_latest(
                frames, keep_backlog=self.handoff_budget.output_keep_backlog_samples
            )
            if not had_data:
                self._fallback_silence_blocks = int(
                    getattr(self, "_fallback_silence_blocks", 0)
                ) + 1
            out_report = self.safety.limit_output(y_blk[0])
            y_lim, clip_frac = out_report.signal, out_report.clipped_fraction

            # 상쇄 출력 게이트의 목표는 매 블록 유도한다: 운용자 스위치 AND
            # "발산 검증 프로브 중이 아님". 프로브 구간은 음향적으로 mute 와 완전히
            # 같고(출력 0), 다른 것은 측정 결과에 따라 재개할 수 있다는 것뿐이다.
            anc_output_wanted = bool(self.state.anc_enabled) and not self.safety.probe_active
            if anc_output_wanted != self._last_anc:
                self.anc_gate.set_target(1.0 if anc_output_wanted else 0.0)
                if anc_output_wanted:
                    secondary_total = int(
                        getattr(self.engine, "secondary_total_length", 0)
                    )
                    self._adaptation_hold_samples = secondary_total + self._fade_samples
                else:
                    self._adaptation_hold_samples = 0
                self._last_anc = anc_output_wanted
            gain = self.anc_gate.process(frames)
            control = y_lim * gain

            out = np.zeros((frames, 2), dtype=np.float32)
            out[:, self.ch_noise] = source
            out[:, self.ch_cancel] = control
            outdata[:] = float32_to_pcm_int16(out)

            err_power = self.err_meter.update(err)
            ctrl_power = self.ctrl_meter.update(control)

            # 베이스라인 수집 조건은 **"상쇄 출력이 나가지 않는다"** 하나다.
            # (소음 재생 조건을 넣으면 외부 소음원 운용에서 영원히 안 잡히고, 빼기만
            #  하면 정숙한 방의 플로어가 굳어 소음을 켜는 순간 발산으로 오판한다.
            #  그래서 수집은 넓게 두고 판정을 프로브로 바꿨다 — safety.BaselineTracker.)
            anc_output_active = bool(gain.size and float(np.max(gain)) > 0.001)

            input_drops = int(self.in_ring.drops)
            stale_input = max(0, input_drops - self._last_input_drops)
            self._last_input_drops = input_drops
            verdict = self.safety.check_block(
                BlockObservation(
                    anc_on=bool(self.state.anc_enabled),
                    output=out_report,
                    error_power=float(err_power),
                    anc_output_active=anc_output_active,
                    had_output_data=bool(had_data),
                    stale_input_samples=stale_input,
                )
            )
            mute = verdict.mute
            if mute:
                self.state.anc_enabled = False
            for msg in verdict.messages:
                self.state.messages.put(msg)

            if self._adaptation_hold_samples > 0:
                self._adaptation_hold_samples = max(
                    0, self._adaptation_hold_samples - frames
                )
            full_anc_gain = bool(gain.size and float(np.min(gain)) >= 0.999)
            full_noise_gain = bool(
                played_noise_gain.size and float(np.min(played_noise_gain)) >= 0.999
            )
            input_clip_fraction = float(
                np.mean(np.abs(mics[:, self.ch_err]) >= 0.98)
            )
            selected_reference = ref_digital if self.reference == "digital" else ref_mic
            reference_power = float(
                np.mean(selected_reference.astype(np.float64) ** 2)
            )
            adapt_allowed = fxlms_adaptation_allowed(
                requested=self.state.anc_enabled and not mute,
                full_anc_gain=full_anc_gain,
                full_noise_gain=full_noise_gain,
                hold_samples=self._adaptation_hold_samples,
                output_clip_fraction=clip_frac,
                input_clip_fraction=input_clip_fraction,
                reference_power=reference_power,
                stream_ok=not bool(status) and had_data,
            )
            adapt_gate = np.full(frames, float(adapt_allowed), dtype=np.float32)
            self.in_ring.push(np.stack([err, ref_mic, ref_digital, adapt_gate]))

            if self.rec is not None and self.rec_pos < self.record_len:
                n = min(frames, self.record_len - self.rec_pos)
                sl = slice(self.rec_pos, self.rec_pos + n)
                self.rec["err"][sl] = err[:n]
                self.rec["ref"][sl] = ref_mic[:n]
                self.rec["source"][sl] = source[:n]
                self.rec["control"][sl] = control[:n]
                self.rec["anc_gain"][sl] = gain[:n]
                self.rec_pos += n

            baseline = self.safety.baseline
            # 일부 오디오 없는 callback fixture는 구형 BaselineTracker 대역을
            # ``initialized``만 가진 SimpleNamespace로 주입한다. 실제 Tracker는
            # ``valid``를 제공하지만, 판정 의미는 둘 다 "유효한 baseline인가"로
            # 동일하므로 callback 테스트/호환 경로에서도 예외를 내지 않는다.
            baseline_valid = bool(
                getattr(
                    baseline,
                    "valid",
                    getattr(baseline, "initialized", False),
                )
            )
            reduction_valid, reduction_reason = reduction_measurement_eligibility(
                anc_enabled=bool(self.state.anc_enabled),
                anc_output_active=anc_output_active,
                baseline_valid=baseline_valid,
                error_power=float(err_power),
                had_output_data=bool(had_data),
                callback_status=bool(status),
                xruns=int(self.xruns),
                fallback_silence_blocks=int(
                    getattr(self, "_fallback_silence_blocks", 0)
                ),
                deadline_miss_blocks=int(
                    getattr(self, "_deadline_miss_blocks", 0)
                ),
                engine_error_blocks=int(
                    getattr(self, "_engine_error_blocks", 0)
                ),
                full_anc_gain=full_anc_gain,
                full_noise_gain=full_noise_gain,
            )
            reduction = float("nan")
            # 기준은 상쇄 출력이 닫힌 동안 수집한 ERR power다. 다만 fallback/xrun/
            # deadline이 한 번이라도 있으면 현재 session의 OFF/ON 비교를 성능값으로
            # 승격하지 않는다. 이전 구현은 게이트가 열려 있다는 사실만 보고 fallback
            # 무음도 정상 상쇄음처럼 계산했다.
            if reduction_valid:
                reduction = 10.0 * np.log10((baseline.power + 1e-30) / (err_power + 1e-30))
            self.state.latest_stats = {
                "anc": self.state.anc_enabled,
                "anc_output": anc_output_active,
                "divergence_probe": self.safety.probe_active,
                "err_dbfs": power_to_db(err_power),
                # acoustic-reference 현장에서 실제 REF 입력이 들어오는지를 확인하기
                # 위한 raw block meter. model input은 아래 inference loop에서 별도로
                # ref_mic를 선택하며, ref_digital은 이 값 계산에 사용하지 않는다.
                "ref_mic_dbfs": power_to_db(
                    float(np.mean(ref_mic.astype(np.float64) ** 2))
                ),
                "ctrl_dbfs": power_to_db(ctrl_power),
                "reduction_db": reduction,
                "baseline_dbfs": (
                    power_to_db(baseline.power) if baseline_valid else float("nan")
                ),
                "reduction_valid": reduction_valid,
                "reduction_reason": reduction_reason,
                "fxlms_adapt_allowed": adapt_allowed,
                "fxlms_adapt_hold_samples": self._adaptation_hold_samples,
                "underruns": self.out_ring.underruns,
                "fallback_silence_blocks": int(
                    getattr(self, "_fallback_silence_blocks", 0)
                ),
                "deadline_miss_blocks": int(
                    getattr(self, "_deadline_miss_blocks", 0)
                ),
                "engine_error_blocks": int(
                    getattr(self, "_engine_error_blocks", 0)
                ),
                "drops": self.out_ring.drops,
                "stale_input_drops": input_drops,
                "watchdog_trips": {
                    item.value: count
                    for item, count in self.safety.trip_counts.items()
                    if count
                },
                "xruns": self.xruns,
                "step_ms": float(np.mean(self.step_times_ms[-50:])) if self.step_times_ms else 0.0,
            }
            clock_telemetry.finish_callback(
                clock_token,
                entry_snapshot=clock_entry_snapshot,
                snapshot=self._runtime_counter_snapshot(),
            )
            clock_token = None
            self.state.latest_stats["clock_telemetry_status"] = (
                clock_telemetry.live_status()
            )
        except BaseException as exc:      # 콜백 예외 → 안전 정지
            if clock_token is not None:
                clock_telemetry.abort_callback(clock_token, error=exc)
            outdata.fill(0)
            self.state.fatal_error = exc
            self.state.quit_event.set()
            raise self.sd.CallbackAbort from exc

    # ---------- 추론 스레드 ----------

    def _inference_loop(self) -> None:
        affinity = self.cfg.get("engine", {}).get("cpu_affinity")
        if affinity:
            try:
                os.sched_setaffinity(0, set(int(c) for c in affinity))
            except OSError:
                pass
        while not self.state.quit_event.is_set():
            if self.state.reset_event.is_set():
                self.engine.reset()
                # SPSC: 이 스레드는 in_ring 의 소비자만이다 — out_ring 의 read_pos 는
                # 콜백 소유이므로 건드리지 않는다 (콜백의 pop_latest 가 자연 배출).
                self.in_ring.consumer_reset()
                self.state.reset_event.clear()
            if not self.in_ring.wait_for(self.hop, timeout=0.1):
                continue
            # 입력 백로그 허용치는 출력과 **같아야 한다**. 예전에는 여기만 8 hop
            # (42.7 ms) 이어서 추론이 뒤처지면 실효 핸드오프가 학습 가정의 8배로
            # 조용히 늘었다 — 그 지연에서는 상쇄가 증폭이 된다.
            blk, ok = self.in_ring.pop_latest(
                self.hop, keep_backlog=self.handoff_budget.input_keep_backlog_samples
            )
            if not ok:
                continue
            err, ref_mic, ref_digital, adapt_gate = blk
            ref = ref_digital if self.reference == "digital" else ref_mic
            t0 = time.perf_counter()
            try:
                set_adapt = getattr(self.engine, "set_adapt_enabled", None)
                if set_adapt is not None:
                    set_adapt(bool(np.all(adapt_gate >= 0.5)))
                y = self.engine.step(ref.copy(), err.copy())
            except Exception as exc:
                self.state.messages.put(f"엔진 오류: {exc!r} — 무음 출력")
                self._engine_error_blocks = int(
                    getattr(self, "_engine_error_blocks", 0)
                ) + 1
                y = np.zeros(self.hop, dtype=np.float32)
            dt = (time.perf_counter() - t0) * 1000.0
            self.step_times_ms.append(dt)
            self._inference_step_count = int(
                getattr(self, "_inference_step_count", 0)
            ) + 1
            if dt >= 1000.0 * self.hop / self.fs:
                self._deadline_miss_blocks += 1
            # 녹음 session은 초반 spike를 삭제하면 max/deadline 증거를 잃는다.
            # 미녹음 장시간 interactive 세션만 메모리 상한을 둔다.
            if (
                not bool(getattr(self, "_record_runtime_evidence", False))
                and len(self.step_times_ms) > 10000
            ):
                del self.step_times_ms[:5000]
            self.out_ring.push(y.reshape(1, -1))

    # ---------- 실행 ----------

    def _prime_output_handoff(self) -> None:
        """stream open 전 exact 1-hop 무음을 발행해 첫 callback fallback을 없앤다.

        이 push는 inference producer thread가 시작하기 전에만 수행한다. 그 뒤에는
        inference thread만 ``out_ring.write_pos``의 producer다. 두 블록을 넣어
        실효 handoff를 늘리는 경로는 즉시 거부한다.
        """

        if int(getattr(self, "_intentional_startup_prime_blocks", 0)) != 0:
            raise RuntimeError("runtime output handoff prime은 단 한 번만 허용됩니다")
        if (
            self.out_ring.available() != 0
            or self.out_ring.write_pos != 0
            or self.out_ring.read_pos != 0
        ):
            raise RuntimeError("runtime output ring이 비어 있지 않아 startup prime을 거부합니다")
        self.out_ring.push(np.zeros((1, self.hop), dtype=np.float32))
        if self.out_ring.available() != self.hop or self.out_ring.overruns != 0:
            raise RuntimeError("runtime output handoff prime 발행에 실패했습니다")
        self._intentional_startup_prime_blocks = 1

    def start(self) -> None:
        # PortAudio가 prime callback을 즉시 호출해도 output ring은 이미 exact
        # 1-hop 무음을 가진다. inference thread보다도 먼저여야 SPSC producer
        # 소유권 인계가 명확하다.
        self._prime_output_handoff()
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="anc-inference"
        )
        self._infer_thread.start()
        try:
            self._stream = self.sd.Stream(
                samplerate=self.fs,
                blocksize=self.block,
                device=(self.in_dev, self.out_dev),
                channels=(2, 2),
                dtype=("int32", "int16"),
                latency=(self.latency, self.latency),
                callback=self._callback,
                prime_output_buffers_using_stream_callback=True,
            )
            self._stream.start()
        except BaseException:
            # Stream 생성/start 실패가 inference producer를 뒤에 남기면 다음
            # PCM 실행과 겹쳐 timing을 오염시킬 수 있다. 원래 예외를 보존하되
            # 이 start가 만든 자원은 반환 전에 모두 닫는다.
            self.state.quit_event.set()
            if self._stream is not None:
                try:
                    self._stream.abort()
                except BaseException:
                    pass
                try:
                    self._stream.close()
                except BaseException:
                    pass
            self._infer_thread.join(timeout=1.0)
            raise

    def stop(self) -> None:
        # 종료 페이드 시퀀스 (안전장치 8)
        self.state.anc_enabled = False
        self.state.noise_enabled = False
        time.sleep(0.2)
        self.state.quit_event.set()
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=1.0)

    def session_data(self) -> dict[str, np.ndarray]:
        if self.rec is None:
            return {}
        n = self.rec_pos
        return {k: v[:n].copy() for k, v in self.rec.items()}

    def runtime_timing_data(self) -> dict[str, np.ndarray]:
        """session NPZ에 결속할 전수 inference wall-time raw를 반환한다."""

        times = np.asarray(self.step_times_ms, dtype=np.float64)
        count = int(getattr(self, "_inference_step_count", len(times)))
        if bool(getattr(self, "_record_runtime_evidence", False)) and count != len(times):
            raise RuntimeError(
                "녹음 runtime inference step count와 보존 latency raw 수가 다릅니다"
            )
        if times.ndim != 1 or np.any(~np.isfinite(times)) or np.any(times < 0.0):
            raise RuntimeError("runtime inference latency raw가 유한한 1-D 양수가 아닙니다")
        payload = {
            "inference_step_times_ms": times,
            "inference_step_count": np.asarray(count, dtype=np.int64),
            "intentional_startup_prime_blocks": np.asarray(
                int(getattr(self, "_intentional_startup_prime_blocks", 0)),
                dtype=np.int64,
            ),
        }
        start_snapshot = getattr(self, "_runtime_deployment_snapshot_start", None)
        if start_snapshot is not None:
            from ..eval.broadband_runtime import snapshot_runtime_deployment_files

            end_snapshot = snapshot_runtime_deployment_files(
                runtime_cfg=self.cfg,
                plant=self.plant_contract,
                repo_root=REPO_ROOT,
            )
            if end_snapshot != start_snapshot:
                raise RuntimeError(
                    "runtime 녹음 중 checkpoint/export/metadata/plant bytes가 변경됐습니다"
                )
            snapshot_json = json.dumps(
                start_snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            payload.update(
                {
                    "runtime_deployment_snapshot_json": np.asarray(snapshot_json),
                    "runtime_deployment_snapshot_sha256": np.asarray(
                        start_snapshot["snapshot_sha256"]
                    ),
                }
            )
            from ..dsp.measurement_level import collect_alsa_physical_fingerprint

            end_fingerprint = collect_alsa_physical_fingerprint(
                self.cfg["hardware"]
            )
            start_fingerprint = getattr(
                self, "_runtime_physical_fingerprint_start", None
            )
            if end_fingerprint != start_fingerprint:
                raise RuntimeError(
                    "runtime 녹음 중 ALSA 물리 hardware fingerprint가 변경됐습니다"
                )
            fingerprint_json = json.dumps(
                start_fingerprint,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            payload.update(
                {
                    "runtime_physical_fingerprint_json": np.asarray(
                        fingerprint_json
                    ),
                    "runtime_physical_fingerprint_sha256": np.asarray(
                        start_fingerprint["sha256"]
                    ),
                }
            )
        elif (
            bool(getattr(self, "_record_runtime_evidence", False))
            and isinstance(getattr(self, "cfg", None), dict)
            and str(self.cfg.get("controller", "dl")) == "dl"
        ):
            raise RuntimeError("DL runtime 녹음에 deployment start snapshot이 없습니다")
        return payload


def _prepare_runtime_record_targets(record_path: str | Path) -> tuple[Path, Path]:
    """오디오 시작 전에 no-replace session/receipt 목적지를 fail-closed로 검사한다."""

    base = Path(record_path)
    npz_path = base.with_suffix(".npz")
    receipt_path = base.with_suffix(".runtime_clock.json")
    if npz_path == receipt_path:
        raise ValueError("runtime NPZ와 clock receipt 경로가 같을 수 없습니다")
    parent = npz_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise ValueError(f"runtime record parent가 디렉터리가 아닙니다: {parent}")
    # no-replace 증거 경로가 symlink를 통해 실행 중 다른 곳으로 바뀌는 것을 막는다.
    cursor = parent.absolute()
    while True:
        if cursor.is_symlink():
            raise ValueError(f"runtime record parent에 symlink가 포함됩니다: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    for target in (npz_path, receipt_path):
        if os.path.lexists(target):
            raise FileExistsError(
                f"runtime no-replace target이 이미 존재합니다: {target}"
            )
    return npz_path, receipt_path


def run_cli(
    cfg: dict,
    run_seconds: float,
    record_path: str | None,
    *,
    validate_plant_contract: bool = True,
    start_noise: bool = False,
) -> int:
    # target 충돌을 소리를 낸 뒤 발견하면 실험과 speaker 연결 시간을 낭비한다.
    # 끝의 xb/O_EXCL 검사도 유지해 이 preflight 뒤 TOCTOU를 다시 막는다.
    record_targets = (
        _prepare_runtime_record_targets(record_path) if record_path else None
    )
    anc = RealtimeANC(
        cfg,
        record_seconds=run_seconds if record_path else 0.0,
        validate_plant_contract=validate_plant_contract,
    )
    # ANC는 항상 OFF로 시작하되, 대화형 band/tone 재현에서는 사용자가 N을
    # 먼저 누르지 않아도 소음 source만 켜 둘 수 있다.
    anc.state.noise_enabled = bool(start_noise)
    keyboard = KeyboardController(anc.state)

    engine_desc = cfg.get("controller", "dl")
    if engine_desc == "dl":
        engine_desc = f"dl/{cfg.get('engine', {}).get('type', 'torch')}"
    print("=" * 72)
    print(f"Deep ANC 실시간 런타임 | 컨트롤러: {engine_desc} | reference: {cfg.get('reference')}")
    print(f"블록 {anc.block} ({1000*anc.block/anc.fs:.2f}ms) @ {anc.fs}Hz | 시작: ANC OFF")
    print(f"소음 시작: {'ON' if anc.state.noise_enabled else 'OFF'} | A=ANC 토글 | N=소음 토글")
    if str(cfg.get("reference", "digital")) == "mic":
        print("음향 레퍼런스: 실제 reference_mic ch1 → 모델 입력 | ref_digital 미사용")
        if str(cfg.get("controller", "dl")) == "dl":
            print(
                "[경고] 현재 DL artifact는 digital-reference 학습본입니다. "
                "mic 입력 실행은 acoustic 경로 일회성 실험입니다."
            )
    budget = anc.handoff_budget
    print(
        f"실효 핸드오프 {budget.effective_handoff_samples}샘플 "
        f"({1000*budget.effective_handoff_samples/anc.fs:.2f}ms) "
        f"| 백로그 허용 입력={budget.input_keep_backlog_samples} "
        f"출력={budget.output_keep_backlog_samples}"
    )
    for note in anc.safety.limits.legacy_notes:
        print(f"[설정 경고] {note}")
    print(KeyboardController.help_text())
    print("주의: TPA3116D2 볼륨을 낮춘 상태에서 시작하세요.")
    print("=" * 72)

    anc.start()
    keyboard.start()
    started = time.monotonic()
    next_report = started
    run_error: Exception | None = None
    try:
        while not anc.state.quit_event.is_set():
            now = time.monotonic()
            if run_seconds > 0 and now - started >= run_seconds:
                break
            while True:
                try:
                    print(f"\n[명령] {anc.state.messages.get_nowait()}")
                except Exception:
                    break
            if now >= next_report:
                s = anc.state.latest_stats
                if s:
                    red = s.get("reduction_db", float("nan"))
                    red_txt = "  n/a" if not np.isfinite(red) else f"{red:6.2f}"
                    if s.get("reduction_valid"):
                        reduction_status = "VALID"
                    elif s.get("anc"):
                        reduction_status = f"INVALID:{s.get('reduction_reason', 'unknown')}"
                    else:
                        reduction_status = "OFF"
                    print(
                        f"[{'ON ' if s['anc'] else 'OFF'}] e={s['err_dbfs']:7.2f} dBFS | "
                        f"ref_mic={s.get('ref_mic_dbfs', float('nan')):7.2f} dBFS | "
                        f"base={s.get('baseline_dbfs', float('nan')):7.2f} dBFS | "
                        f"ctrl={s['ctrl_dbfs']:7.2f} | 저감={red_txt} dB | "
                        f"판정={reduction_status} | "
                        f"step={s['step_ms']:5.2f}ms | "
                        f"deadline={s.get('deadline_miss_blocks', 0)} | "
                        f"fallback={s.get('fallback_silence_blocks', s['underruns'])} | "
                        f"engine_err={s.get('engine_error_blocks', 0)} | "
                        f"xrun={s['xruns']}"
                    )
                next_report = now + 1.0
            if anc.state.fatal_error is not None:
                raise RuntimeError("오디오 콜백 실패") from anc.state.fatal_error
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # callback 실패 receipt도 먼저 보존한 뒤 기존처럼 예외를 다시 올린다.
        run_error = exc
    finally:
        keyboard.stop()
        anc.stop()

    clock_payload = anc.clock_telemetry_receipt()
    if record_path:
        assert record_targets is not None
        data = anc.session_data()
        timing_data = anc.runtime_timing_data()
        npz_path, clock_receipt_path = record_targets
        telemetry_digest = payload_sha256(clock_payload)
        # raw와 telemetry binding을 덮어쓰지 못하게 session NPZ도 exclusive-create한다.
        with npz_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                fs=anc.fs,
                runtime_clock_telemetry_sha256=np.asarray(telemetry_digest),
                runtime_clock_authority_status=np.asarray(
                    clock_payload["authority_status"]
                ),
                **data,
                **timing_data,
            )
            handle.flush()
            os.fsync(handle.fileno())
        receipt_bundle = bind_recording_to_clock_receipt(
            clock_payload,
            recording_path=npz_path,
            recording_sha256=sha256_file(npz_path),
        )
        receipt_path, receipt_sha = write_clock_receipt_exclusive(
            clock_receipt_path, receipt_bundle
        )
        print(f"세션 저장: {npz_path}")
        print(
            f"clock receipt: {receipt_path} | {clock_payload['authority_status']} "
            f"| sha256={receipt_sha}"
        )
    else:
        print(f"clock telemetry: {clock_payload['authority_status']} (미저장)")
    print("종료 — 양 채널 무음.")
    if run_error is not None:
        raise run_error
    return 0


def run_calibrate(cfg: dict) -> int:
    """--calibrate: 3-스레드 경로 그대로의 실효 상쇄경로 지연 실측 [C1].

    추론 엔진 자리에 '처프 재생 엔진'을 넣어 out_ring→콜백→스피커→에러마이크
    왕복 지연을 상호상관으로 측정하고, 학습에 쓰는 지연(캘리브레이션+핸드오프)과
    비교해 어긋남을 리포트한다.
    """
    from scipy import signal as sp_signal

    from ..dsp.secondary_path import load_secondary_path
    from ..dsp.timing import handoff_samples_from_config  # handoff 의 단일 출처

    fs = int(cfg["hardware"]["audio"]["sample_rate"])
    seconds = 6.0
    t = np.arange(int(seconds * fs)) / fs
    chirp = (
        0.05 * sp_signal.chirp(t, 100.0, seconds, 2000.0, method="logarithmic")
    ).astype(np.float32)
    fade = int(0.05 * fs)
    chirp[:fade] *= np.linspace(0, 1, fade)
    chirp[-fade:] *= np.linspace(1, 0, fade)

    class ChirpEngine:
        def __init__(self, hop: int) -> None:
            self.hop = hop
            self.pos = 0

        def reset(self) -> None:
            self.pos = 0

        def step(self, ref, err):
            out = np.zeros(self.hop, dtype=np.float32)
            n = min(self.hop, chirp.size - self.pos)
            if n > 0:
                out[:n] = chirp[self.pos : self.pos + n]
                self.pos += n
            return out

    cfg = dict(cfg)
    cfg["noise"] = {"type": "silence"}
    # 측정 모드: 음향 성능 워치독(포화/RMS/데드라인/발산/백로그)은 자문으로 내린다.
    # 임계값을 1e12 로 무력화하던 예전 방식과 달리 **하드웨어 보호(DC·NaN)는 그대로
    # mute 한다** — 캘리브레이션 중이라고 보이스코일에 DC 를 흘려도 되는 것은 아니다.
    cfg["safety"] = dict(cfg.get("safety", {}))
    cfg["safety"]["measurement_mode"] = True
    anc = RealtimeANC(
        cfg,
        record_seconds=seconds + 2.0,
        validate_plant_contract=False,
    )
    anc.engine = ChirpEngine(anc.hop)
    anc.state.anc_enabled = True          # 게이트를 열어 처프를 내보낸다
    anc.anc_gate.set_target(1.0)
    print(f"실효 지연 측정: 처프 {seconds:.0f}s 재생 (상쇄 스피커 ch1) ...")
    anc.start()
    time.sleep(seconds + 1.5)
    anc.stop()

    data = anc.session_data()
    err = data["err"].astype(np.float64)
    ctrl = data["control"].astype(np.float64)
    if np.max(np.abs(ctrl)) < 1e-6:
        print("[실패] 출력이 재생되지 않았습니다", file=sys.stderr)
        return 1
    corr = sp_signal.fftconvolve(err, ctrl[::-1], mode="full")
    lag = int(np.argmax(np.abs(corr))) - (ctrl.size - 1)
    sp = load_secondary_path(secondary_path_npz(cfg))
    handoff = handoff_samples_from_config(cfg["duct"])
    # rec['control'] 은 블록이 실제 "출력"되는 콜백 시점(핸드오프 이후)에 기록되므로,
    # 측정 lag 의 기대값은 캘리브레이션 지연(1342)뿐이다 — 핸드오프는 별도 합산 (#11)
    expected = sp.delay_samples
    total_training = sp.delay_samples + handoff
    print(f"측정 지연(출력→에러마이크): {lag}샘플 ({1000*lag/fs:.2f}ms)")
    print(f"캘리브레이션 기대값        : {expected}샘플 | 차이 {lag - expected:+d}샘플 "
          f"({1000*(lag-expected)/fs:+.2f}ms)")
    print(f"학습 플랜트 총지연         : 측정 {lag} + 핸드오프 {handoff} = {lag + handoff} "
          f"(설정값 {total_training})")
    if abs(lag - expected) > 512:
        print(
            "→ 차이가 지터 증강 범위(+512)를 벗어납니다. duct.yaml 의 "
            "handoff_extra_samples 조정 또는 재캘리브레이션 후 파인튜닝을 권장합니다."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-seconds", type=float, default=None)
    parser.add_argument("--record", default=None, help="세션 npz 저장 경로")
    parser.add_argument("--calibrate", action="store_true", help="실효 지연 측정 모드")
    parser.add_argument(
        "--start-noise",
        action="store_true",
        help="ANC는 OFF로 유지하고 소음 source만 시작부터 ON",
    )
    parser.add_argument(
        "--legacy-diagnostic",
        action="store_true",
        help=(
            "기존 Tiny surrogate(lead=109) ANC 재현용. strict P/S plant 계약을 "
            "건너뛰며 물리 성능/녹음 증거로 사용할 수 없음"
        ),
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument(
        "--input-probe-seconds",
        type=float,
        default=2.0,
        help="스피커 출력 전 무출력 마이크 사전점검 길이",
    )
    args = parser.parse_args()

    if args.calibrate and args.legacy_diagnostic:
        parser.error("--calibrate와 --legacy-diagnostic은 함께 사용할 수 없습니다")

    if args.list_devices:
        print(format_sounddevice_devices())
        return 0

    try:
        cfg = load_runtime_config(args.config, args.overrides)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] runtime 설정 해석 실패: {exc}", file=sys.stderr)
        return 2

    run_seconds = (
        args.run_seconds
        if args.run_seconds is not None
        else float(cfg.get("run_seconds", 0.0))
    )
    record = args.record or cfg.get("record")
    if args.legacy_diagnostic:
        if record:
            print(
                "[중단] --legacy-diagnostic은 녹음 경로를 허용하지 않습니다. "
                "기존 모델 재현은 진단용 무저장 실행만 지원합니다.",
                file=sys.stderr,
            )
            return 2
        if not (0.0 < run_seconds <= 60.0):
            print(
                "[중단] --legacy-diagnostic은 --run-seconds를 0 초과 60초 이하로 "
                "명시해야 합니다.",
                file=sys.stderr,
            )
            return 2

    # 이 검사는 파일/metadata만 읽으며 스피커·마이크를 열지 않는다. 사용자 확인보다
    # 앞서 수행해 legacy artifact를 명시적으로 요청했는지와 현재 strict plant 정합을
    # 구분한다. 기본 경로의 strict 검사는 계속 fail-closed다.
    if not args.calibrate:
        try:
            if args.legacy_diagnostic:
                validate_legacy_diagnostic_config(cfg)
                print(
                    "[경고] legacy diagnostic: 기존 surrogate checkpoint/ONNX(lead=109)를 "
                    "재현합니다. strict P/S plant 계약은 검사하지 않으며 ANC 감쇠·녹음 "
                    "증거로 승격할 수 없습니다.",
                    file=sys.stderr,
                )
            else:
                validate_runtime_plant_contract(cfg)
        except (OSError, RuntimeError, ValueError) as exc:
            label = (
                "legacy diagnostic preflight"
                if args.legacy_diagnostic
                else "strict runtime plant 계약"
            )
            print(f"[중단] {label} 실패: {exc}", file=sys.stderr)
            return 2

    # 엔진 artifact와 lead metadata도 hardware를 열기 전에 검사한다. 그 뒤에야
    # confirmation·PortAudio input probe 순으로 진행한다.
    try:
        for warning in require_engine_artifacts_to_start(cfg):
            print(f"[경고] {warning}", file=sys.stderr)
        preflight_engine_lead = engine_digital_reference_lead_samples_from_config(cfg)
        validate_digital_reference_lead(
            str(cfg.get("reference", "digital")),
            int(cfg.get("digital_reference_lead_samples", 0)),
            preflight_engine_lead,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] runtime engine preflight 실패: {exc}", file=sys.stderr)
        return 2

    if not (args.confirm_speaker and args.confirm_user_present and args.confirm_volume_minimum):
        print(
            "[중단] 런타임 출력에는 --confirm-speaker, --confirm-user-present, "
            "--confirm-volume-minimum이 모두 필요합니다.", file=sys.stderr
        )
        return 2
    try:
        import sounddevice as sd
        from ..dsp.measurement_level import assert_live_pcm_clock_preconditions
        assert_live_pcm_clock_preconditions(cfg["hardware"]["audio"])
        assert_measurement_preconditions(sd, cfg["hardware"]["audio"])
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 오디오 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    try:
        if not input_preflight(cfg, seconds=args.input_probe_seconds):
            return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 입력 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    if args.calibrate:
        return run_calibrate(cfg)
    if record and run_seconds <= 0:
        parser.error("--record 는 녹음 버퍼 크기 산정을 위해 --run-seconds 가 필요합니다")
    return run_cli(
        cfg,
        run_seconds,
        record,
        validate_plant_contract=not args.legacy_diagnostic,
        start_noise=args.start_noise,
    )


if __name__ == "__main__":
    raise SystemExit(main())
