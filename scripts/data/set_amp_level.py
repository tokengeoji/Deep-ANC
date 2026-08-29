#!/usr/bin/env python3
"""앰프 볼륨을 **교정된 레벨**에 맞춘다 — 실시간 미터.

    .venv/bin/python scripts/data/set_amp_level.py --self-test        # 소리 없음
    # 최초 canonical evidence를 만드는 1회 meter raw(소리 20초)
    .venv/bin/python scripts/data/set_amp_level.py --bootstrap-level-evidence \
        --confirm-speaker --confirm-user-present --confirm-volume-minimum

왜 이 도구가 있는가
------------------
2026-08-06: 앰프 볼륨을 눈감고 맞추다가 하루를 날렸다. 이 시스템은 **구동 레벨에
창(window)이 있고 그 창이 좁다.** 실측(분석 구간 ERR 대역 RMS 기준):

    -68.9 dBFS  →  신호가 잡음 바닥(-69)에 묻혀 P/S 측정이 전량 기각
    -48.3 dBFS  →  **정상. P−S = 140 (기하 예측 147과 일치)**   ← 목표
    -37   dBFS  →  P−S = 1 로 붕괴. 두 채널이 결합된다

위쪽이 왜 깨지는가: TPA3116D2 가 Jetson USB-C 에서 어댑터로 전원을 받는다. 이 앰프는
12~24V 용인데 5V 로는 부족해서, 세게 구동하면 전원이 주저앉고 **공유 전원·접지를 통해
두 채널이 결합**한다. 그 상태에서는 P(소음경로)와 S(취소경로)를 구분할 수 없고,
ANC 는 원리적으로 성립하지 않는다 (안티노이즈가 소음 스피커에서도 나온다).

즉 이 레벨은 취향이 아니라 **측정이 성립하는 조건**이다. 그래서 숫자로 맞춘다.

무엇을 재는가
------------
``measure_paths_interleaved.py`` 와 **똑같은 인터리브 멀티톤**의 소음 채널 성분을
ch0 으로만 흘리고, ERR 마이크의 대역 RMS 를 0.25초마다 출력한다.

프로브를 같게 두는 것이 핵심이다 — 자체 신호를 만들면 눈금이 그 측정과 어긋난다.
실제로 밴드 노이즈(크레스트 12dB)로 만들었다가 멀티톤(5.6dB) 대비 6dB 넘게 어긋났다.

⚠ 취소 스피커(ch1)는 무음으로 둔다. 두 채널을 동시에 울리면 결합이 있을 때 레벨이
   부풀어 보여서 맞출 수가 없다.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import shlex
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_yaml  # noqa: E402
from deep_anc.data.repository_fd import repository_execution_identity  # noqa: E402
from deep_anc.dsp.fullband_v5_meter import (  # noqa: E402
    CONFIRMATION_KEYS as _V5_CONFIRMATION_KEYS,
    DEFAULT_HARDWARE_PATH as V5_HARDWARE_PATH,
    DEFAULT_LEVEL_EVIDENCE_PATH as V5_LEVEL_EVIDENCE_PATH,
    DEFAULT_LIVE_AUTHORITY_PATH as V5_LIVE_AUTHORITY_PATH,
    DEFAULT_PLAN_ENVELOPE_PATH as V5_PLAN_ENVELOPE_PATH,
    DEFAULT_RAW_TARGET_PATH as V5_RAW_TARGET_PATH,
    FULLBAND_V5_FOLLOWUP_SCHEMA,
    build_fullband_v5_followup as _build_fullband_v5_followup,
    resolve_fullband_v5_devices as _package_resolve_fullband_v5_devices,
    validate_fullband_v5_meter_raw,
    validate_fullband_v5_static_contract,
    write_fullband_v5_meter_raw_atomic,
)
from deep_anc.dsp.fullband_v6_meter import (  # noqa: E402
    CONFIRMATION_KEYS as _V6_CONFIRMATION_KEYS,
    DEFAULT_HARDWARE_PATH as V6_HARDWARE_PATH,
    DEFAULT_LEVEL_EVIDENCE_PATH as V6_LEVEL_EVIDENCE_PATH,
    DEFAULT_LIVE_AUTHORITY_PATH as V6_LIVE_AUTHORITY_PATH,
    DEFAULT_PLAN_ENVELOPE_PATH as V6_PLAN_ENVELOPE_PATH,
    DEFAULT_RAW_TARGET_PATH as V6_RAW_TARGET_PATH,
    FOLLOWUP_SCHEMA as FULLBAND_V6_FOLLOWUP_SCHEMA,
    build_fullband_v6_followup as _build_fullband_v6_followup,
    resolve_fullband_v6_devices as _package_resolve_fullband_v6_devices,
    validate_fullband_v6_meter_raw,
    validate_fullband_v6_static_contract,
    write_fullband_v6_meter_raw_atomic,
)
from deep_anc.dsp.measurement_level import (  # noqa: E402
    BOOTSTRAP_METER_RAW_SCHEMA,
    LIVE_WATCHDOG_GRACE_SECONDS,
    LiveAudioTermination,
    OFFICIAL_MEASUREMENT_LEVEL,
    assert_live_pcm_clock_preconditions,
    band_rms_dbfs as contract_band_rms_dbfs,
    collect_alsa_physical_fingerprint,
    load_measurement_level_evidence,
    measurement_hardware_identity,
    repository_audio_lock,
    scoped_live_audio_signal_handlers,
    validate_measurement_hardware_contract,
    write_bootstrap_meter_raw_atomic,
)

METER_MODE_STRICT = "strict"
METER_MODE_FULLBAND_V5 = "fullband-v5"
METER_MODE_FULLBAND_V6 = "fullband-v6"
FULLBAND_V5_DEFAULT_DIAGNOSTICS_ROOT = "results/fullband_causal_v5/level_meter"
FULLBAND_V6_DEFAULT_DIAGNOSTICS_ROOT = "results/fullband_causal_v6/level_meter"
FULLBAND_V5_ADAPTER_SCRIPT = "scripts/data/measure_paths_fullband_causal_v5.py"
FULLBAND_V6_ADAPTER_SCRIPT = "scripts/data/measure_paths_fullband_causal_v6.py"
SET_AMP_REPOSITORY_PATH = "scripts/data/set_amp_level.py"

SPEAKER_DISCONNECT_NOTICE = (
    "[스피커 출력 종료] 오디오 스트림을 닫았습니다. "
    "지금 스피커/앰프 연결을 즉시 해제하세요."
)
SPEAKER_STOP_UNCONFIRMED_NOTICE = (
    "[스피커 정지 확인 불가] 오디오 스트림 close를 확인하지 못했습니다. "
    "지금 스피커/앰프를 즉시 물리 분리하세요."
)
SPEAKER_PREFLIGHT_ABORT_NOTICE = (
    "[실기 측정 중단] 출력 스트림을 열기 전에 실패했습니다. "
    "스피커/앰프가 연결되어 있으면 지금 물리적으로 분리하세요."
)


def _load_fullband_v5_static_contract(
    args, *, require_sealed_raw_fresh: bool  # noqa: ANN001
) -> dict[str, Any]:
    """CLI namespace를 package 공용 static validator에 전달한다."""

    return validate_fullband_v5_static_contract(
        repository_root=REPO_ROOT,
        plan_envelope_path=args.plan_envelope,
        live_authority_path=args.live_authority,
        level_evidence_path=args.level_evidence,
        hardware_path=args.hardware,
        raw_target_path=args.raw_target,
        require_sealed_raw_fresh=require_sealed_raw_fresh,
    )


def _v5_repository_execution_identity() -> dict[str, Any]:
    return repository_execution_identity(REPO_ROOT, SET_AMP_REPOSITORY_PATH)


def _resolve_fullband_v5_devices(sd, contract):  # noqa: ANN001
    return _package_resolve_fullband_v5_devices(contract, sd_module=sd)


def _v5_confirmations(args) -> dict[str, bool]:  # noqa: ANN001
    return {
        "speaker_output": bool(args.confirm_speaker),
        "user_present": bool(args.confirm_user_present),
        "volume_minimum": bool(args.confirm_volume_minimum),
        "routing_and_geometry": bool(args.confirm_routing_and_geometry),
        "same_amplifier_setting": bool(args.confirm_same_amplifier_setting),
    }


def _load_fullband_v6_static_contract(
    args, *, require_sealed_raw_fresh: bool  # noqa: ANN001
) -> dict[str, Any]:
    """CLI namespace를 v6 package static validator에 전달한다."""

    return validate_fullband_v6_static_contract(
        repository_root=REPO_ROOT,
        plan_envelope_path=args.plan_envelope,
        live_authority_path=args.live_authority,
        level_evidence_path=args.level_evidence,
        hardware_path=args.hardware,
        raw_target_path=args.raw_target,
        require_sealed_raw_fresh=require_sealed_raw_fresh,
    )


def _v6_repository_execution_identity() -> dict[str, Any]:
    return repository_execution_identity(REPO_ROOT, SET_AMP_REPOSITORY_PATH)


def _resolve_fullband_v6_devices(sd, contract):  # noqa: ANN001
    return _package_resolve_fullband_v6_devices(contract, sd_module=sd)


def _v6_confirmations(args) -> dict[str, bool]:  # noqa: ANN001
    value = {
        "speaker_output": bool(args.confirm_speaker),
        "user_present": bool(args.confirm_user_present),
        "volume_minimum": bool(args.confirm_volume_minimum),
        "routing_and_geometry": bool(args.confirm_routing_and_geometry),
        "same_amplifier_setting": bool(args.confirm_same_amplifier_setting),
    }
    if set(value) != _V6_CONFIRMATION_KEYS:
        raise AssertionError("v6 confirmation key 집합이 package 계약과 다릅니다")
    return value


# 목표 ERR 대역 RMS (dBFS) — 과거 실측 메모에서 유도된 후보값.
#
# 이 미터는 ch0 만 울리고, 기준이 되는 인터리브 측정은 두 채널을 함께 울린다.
# 그래서 눈금이 1:1 이 아니다. 같은 노브 위치에서 둘을 나란히 재서 오프셋을 잡았다
# (2026-08-06):
#
#   같은 노브 위치      미터(ch0 단독)           -43.70 dBFS
#                       인터리브 분석구간(양채널) -39.91 dBFS
#   기준 (8/4, P-S=140) 인터리브 분석구간         -46.33 dBFS
#   → 필요한 감쇠 6.42 dB → 미터 목표 -43.70 - 6.42 = -50.1
#
# ⚠ 이 paired 기록의 raw는 보존되지 않았고 당시 probe peak도 현행 0.003이 아니었다.
#   따라서 아래 수치는 JSON에 적혀 있다는 이유만으로 실기 근거가 되지 않는다.
#   현행 peak로 같은 노브 위치에서 얻은 ch0 단독 raw + interleaved raw의 SHA evidence가
#   정상 live는 그 evidence가 있어야만 스트림을 연다. 최초 한 번은 명시적
#   ``--bootstrap-level-evidence``가 PASS raw+SHA receipt를 남기고, 바로 이어지는 strict
#   raw와 원자적으로 evidence를 만들므로 순환을 안전하게 끊는다.
#
# 프로브 peak·주기·대역과 미터 목표/허용오차는
# ``OFFICIAL_MEASUREMENT_LEVEL``이 유일한 출처다. 이 파일에 숫자를 다시 적지 마라.


def band_rms_dbfs(x: np.ndarray) -> float:
    """대역 안 RMS(dBFS). 대역 밖 험·DC 가 눈금을 흔들지 않게 대역제한한다."""
    return contract_band_rms_dbfs(x)


def verdict(level: float) -> str:
    if level < OFFICIAL_MEASUREMENT_LEVEL.meter_min_dbfs:
        return (
            f"↑ 올리세요 "
            f"({OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs - level:+.1f} dB 부족)"
        )
    if level > OFFICIAL_MEASUREMENT_LEVEL.meter_max_dbfs:
        return (
            f"↓ 내리세요 "
            f"({level - OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs:+.1f} dB 초과)"
        )
    return "✅ 맞았습니다 — 여기서 멈추세요"


def strict_followup_command(
    meter_raw: str | Path, *, capture_id: str, bootstrap: bool
) -> str:
    """PASS meter 뒤 그대로 복사할 strict 명령을 한 곳에서 만든다."""

    output_suffix = str(capture_id)[:8]
    bootstrap_arg = "--bootstrap-level-evidence " if bootstrap else ""
    return (
        "  .venv/bin/python scripts/data/measure_paths_interleaved.py "
        f"{bootstrap_arg}--meter-raw {meter_raw} "
        "--confirm-same-amplifier-setting --confirm-user-present "
        "--confirm-volume-minimum --confirm-routing-and-geometry "
        f"--primary-out assets/measured/primary_path_il_strict_{output_suffix}.npz "
        f"--secondary-out assets/measured/secondary_path_il_strict_{output_suffix}.npz"
    )


def _live_adapter_marker_available(script: str, marker: str) -> bool:
    """오디오/스크립트 import 없이 reviewed live adapter marker만 확인한다."""

    path = REPO_ROOT / script
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, ast.Constant) or value.value is not True:
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == marker
            for target in targets
        ):
            return True
    return False


def fullband_v5_live_adapter_available() -> bool:
    return _live_adapter_marker_available(
        FULLBAND_V5_ADAPTER_SCRIPT,
        "FULLBAND_V5_LIVE_ADAPTER_IMPLEMENTED",
    )


def fullband_v6_live_adapter_available() -> bool:
    return _live_adapter_marker_available(
        FULLBAND_V6_ADAPTER_SCRIPT,
        "FULLBAND_V6_LIVE_ADAPTER_IMPLEMENTED",
    )


def fullband_v5_followup_command(meter_raw: str | Path) -> str:
    """PASS v5 meter에 exact 결속된 live capture 명령을 생성한다."""

    values = [
        str(Path(sys.executable).absolute()),
        str((REPO_ROOT / FULLBAND_V5_ADAPTER_SCRIPT).resolve(strict=True)),
        "--execute-live",
        "--plan-envelope",
        V5_PLAN_ENVELOPE_PATH,
        "--live-authority",
        V5_LIVE_AUTHORITY_PATH,
        "--meter-raw",
        str(meter_raw),
        "--level-evidence",
        V5_LEVEL_EVIDENCE_PATH,
        "--hardware",
        V5_HARDWARE_PATH,
        "--raw-target",
        V5_RAW_TARGET_PATH,
        "--confirm-speaker",
        "--confirm-user-present",
        "--confirm-volume-minimum",
        "--confirm-routing-and-geometry",
        "--confirm-same-amplifier-setting",
    ]
    return "  " + " ".join(shlex.quote(value) for value in values)


def fullband_v6_followup_command(meter_raw: str | Path) -> str:
    """PASS v6 meter에 exact 결속된 live capture 명령을 생성한다.

    adapter 파일의 존재/marker 판정은 ``fullband_v6_live_adapter_available``가
    담당한다. 아직 adapter가 없더라도 이 함수는 검토 가능한 exact 명령 문자열만
    만들며 실행하거나 module을 import하지 않는다.
    """

    values = [
        str(Path(sys.executable).absolute()),
        str((REPO_ROOT / FULLBAND_V6_ADAPTER_SCRIPT).absolute()),
        "--execute-live",
        "--plan-envelope",
        V6_PLAN_ENVELOPE_PATH,
        "--live-authority",
        V6_LIVE_AUTHORITY_PATH,
        "--meter-raw",
        str(meter_raw),
        "--level-evidence",
        V6_LEVEL_EVIDENCE_PATH,
        "--hardware",
        V6_HARDWARE_PATH,
        "--raw-target",
        V6_RAW_TARGET_PATH,
        "--confirm-speaker",
        "--confirm-user-present",
        "--confirm-volume-minimum",
        "--confirm-routing-and-geometry",
        "--confirm-same-amplifier-setting",
    ]
    return "  " + " ".join(shlex.quote(value) for value in values)


def _validated_meter_identity_sha256(precommand: dict[str, Any]) -> str:
    """meter validator가 발행한 generation-specific identity만 허용한다."""

    value = precommand.get("identity_sha256")
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("meter validator identity_sha256이 exact lowercase SHA-256이 아닙니다")
    return value


def probe_signal(seconds: float) -> np.ndarray:
    """``measure_paths_interleaved.py`` 와 **동일한** 자극을 반복해 만든다.

    여기서 자체 신호를 만들면 눈금이 그 측정과 어긋난다 — 크레스트가 다르면 같은 peak
    에서 RMS 가 6dB 넘게 갈린다. 이 저장소를 무너뜨린 발생기 A(같은 물리량을 두 곳에서
    따로 유도)를 계측 도구에서 반복하지 않는다.

    소음 채널(ch0) 성분만 쓴다. 취소 채널까지 함께 울리면 결합이 있을 때 레벨이 부풀어
    보여서 노브를 맞출 수가 없다.
    """

    from deep_anc.dsp.interleaved_probe import build_interleaved_probe

    probe = build_interleaved_probe(
        sample_rate=OFFICIAL_MEASUREMENT_LEVEL.sample_rate,
        period_seconds=OFFICIAL_MEASUREMENT_LEVEL.period_seconds,
        band_hz=OFFICIAL_MEASUREMENT_LEVEL.design_band_hz,
        amplitude=OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
        tone_spacing_hz=None,
    )
    period = np.asarray(probe.noise_signal, dtype=np.float32).reshape(-1)
    fs = OFFICIAL_MEASUREMENT_LEVEL.sample_rate
    repeats = int(np.ceil(seconds * fs / period.size))
    return np.tile(period, repeats)[: int(seconds * fs)].astype(np.float32)


def self_test() -> int:
    """미터와 판정을 소리 없이 검증한다."""

    print("[self-test] 알려진 레벨을 주입해 미터를 검증한다")
    ok = True
    for target in (-60.0, -44.0, -33.0):
        n = OFFICIAL_MEASUREMENT_LEVEL.sample_rate
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1.0 / OFFICIAL_MEASUREMENT_LEVEL.sample_rate)
        lo, hi = OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz
        X[(f < lo) | (f > hi)] = 0.0
        x = np.fft.irfft(X, n)
        x *= 10 ** (target / 20.0) / np.sqrt(np.mean(x**2))
        got = band_rms_dbfs(x)
        good = abs(got - target) < 0.5
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] 주입 {target:+7.1f} → 측정 {got:+7.1f} dBFS | {verdict(got)}")
    # 대역 밖 험이 눈금을 흔들지 않는지
    n = OFFICIAL_MEASUREMENT_LEVEL.sample_rate
    t = np.arange(n) / OFFICIAL_MEASUREMENT_LEVEL.sample_rate
    hum = 0.05 * np.sin(2 * np.pi * 60.0 * t)
    got = band_rms_dbfs(hum)
    good = got < -40.0
    ok &= good
    print(f"  [{'OK' if good else 'FAIL'}] 60Hz 험 진폭 0.05 → 대역내 {got:+7.1f} dBFS (대역 밖이라 낮아야 한다)")
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def meter_capture_invalid_reasons(
    telemetry: dict,
    levels: list[float],
    *,
    expected_frames: int,
) -> list[str]:
    """레벨 값이 좋아 보여도 캡처/종료 결함이면 무조건 실패시킨다."""

    reasons: list[str] = []
    if telemetry.get("interrupted"):
        reasons.append("operator_interrupt")
    if int(telemetry.get("xrun_count", 0)):
        reasons.append(f"xrun_{telemetry['xrun_count']}")
    if int(telemetry.get("unexpected_status_count", 0)):
        reasons.append(
            f"unexpected_callback_status_{telemetry['unexpected_status_count']}"
        )
    if telemetry.get("callback_error"):
        reasons.append(f"callback_error_{telemetry['callback_error']}")
    if telemetry.get("stream_abort_error"):
        reasons.append(f"stream_abort_error_{telemetry['stream_abort_error']}")
    if telemetry.get("stream_close_error"):
        reasons.append(f"stream_close_error_{telemetry['stream_close_error']}")
    if telemetry.get("output_stop_confirmed") is not True:
        reasons.append("output_stop_unconfirmed")
    if not telemetry.get("completed") or int(telemetry.get("output_frames", 0)) != int(
        expected_frames
    ):
        reasons.append(
            f"capture_incomplete_{telemetry.get('output_frames', 0)}_of_{expected_frames}"
        )
    if int(telemetry.get("meter_drop_count", 0)):
        reasons.append(f"meter_queue_drop_{telemetry['meter_drop_count']}")
    if not levels:
        reasons.append("no_meter_levels")
    elif not np.all(np.isfinite(np.asarray(levels, dtype=np.float64))):
        reasons.append("meter_nonfinite")
    return reasons


def _status_snapshot(status) -> tuple[bool, str]:  # noqa: ANN001
    names = (
        "input_overflow",
        "input_underflow",
        "output_overflow",
        "output_underflow",
    )
    is_xrun = any(bool(getattr(status, name, False)) for name in names)
    return is_xrun, str(status)


def capture_meter_stream(
    sd,
    *,
    noise: np.ndarray,
    fs: int,
    in_dev: int,
    out_dev: int,
    err_ch: int,
    noise_out_ch: int,
    block_size: int = 256,
    latency: str = "low",
    include_raw: bool = False,
    pre_open_check: Callable[[], None] | None = None,
) -> tuple[list[float], dict] | tuple[list[float], dict, np.ndarray, np.ndarray]:
    """20초 full-duplex meter capture와 stream 종료 telemetry를 반환한다."""

    import queue as _queue
    import time

    from deep_anc.audio_io import (
        MEASUREMENT_DTYPE,
        float32_to_pcm_int16,
        pcm_int32_to_float32,
    )

    hop = int(0.25 * fs)
    cursor = {"out": 0}
    meter: "_queue.Queue[float]" = _queue.Queue(maxsize=128)
    pending = {"buf": np.zeros(0, dtype=np.float64)}
    submitted_output_pcm = np.zeros((noise.size, 2), dtype=np.int16)
    input_raw = np.zeros((noise.size, 2), dtype=np.int32)
    telemetry = {
        "completed": False,
        "interrupted": False,
        "output_frames": 0,
        "callback_count": 0,
        "xrun_count": 0,
        "unexpected_status_count": 0,
        "statuses": [],
        "callback_error": None,
        "meter_drop_count": 0,
        "stream_abort_error": None,
        "stream_close_error": None,
        "output_stop_confirmed": False,
        "stream_started_at_utc": None,
        "termination_signal": None,
        "nominal_output_seconds": float(noise.size) / float(fs),
        "hard_max_output_seconds": (
            float(noise.size) / float(fs) + LIVE_WATCHDOG_GRACE_SECONDS
        ),
    }

    def callback(indata, outdata, frames, _time, status):  # noqa: ANN001
        outdata[:] = 0
        try:
            telemetry["callback_count"] += 1
            if status:
                is_xrun, text = _status_snapshot(status)
                telemetry["xrun_count"] += int(is_xrun)
                telemetry["unexpected_status_count"] += 1
                telemetry["statuses"].append(text)
            start = int(cursor["out"])
            end = min(start + int(frames), int(noise.size))
            take = end - start
            if take > 0:
                submitted = float32_to_pcm_int16(noise[start:end])
                outdata[:take, noise_out_ch] = submitted
                if include_raw:
                    submitted_output_pcm[start:end, noise_out_ch] = submitted
                    input_raw[start:end] = np.asarray(indata[:take, :2], dtype=np.int32)
                buf = np.concatenate(
                    [
                        pending["buf"],
                        pcm_int32_to_float32(indata[:take, err_ch]).astype(
                            np.float64
                        ),
                    ]
                )
                while buf.size >= hop:
                    level = band_rms_dbfs(buf[:hop])
                    try:
                        meter.put_nowait(level)
                    except _queue.Full:
                        telemetry["meter_drop_count"] += 1
                    buf = buf[hop:]
                pending["buf"] = buf
            cursor["out"] = end
            telemetry["output_frames"] = end
            if end >= noise.size:
                telemetry["completed"] = True
                raise sd.CallbackStop
        except sd.CallbackStop:
            raise
        except BaseException as exc:
            telemetry["callback_error"] = f"{type(exc).__name__}: {exc}"
            outdata[:] = 0
            raise sd.CallbackAbort

    levels: list[float] = []
    stream = None
    failure: BaseException | None = None
    call_started = time.monotonic()
    stream_started: float | None = None
    deadline: float | None = None
    signal_scope = scoped_live_audio_signal_handlers()
    signal_scope.__enter__()
    try:
        try:
            if pre_open_check is not None:
                pre_open_check()
            stream = sd.Stream(
                samplerate=fs,
                blocksize=int(block_size),
                dtype=MEASUREMENT_DTYPE,
                channels=(2, 2),
                device=(in_dev, out_dev),
                latency=(str(latency), str(latency)),
                callback=callback,
            )
            stream.start()
            stream_started = time.monotonic()
            telemetry["stream_started_at_utc"] = dt.datetime.now(
                dt.timezone.utc
            ).isoformat()
            deadline = (
                stream_started
                + float(noise.size) / float(fs)
                + LIVE_WATCHDOG_GRACE_SECONDS
            )
            while not telemetry["completed"]:
                if telemetry["callback_error"]:
                    raise RuntimeError(telemetry["callback_error"])
                if not stream.active:
                    raise RuntimeError("오디오 스트림이 20초 전에 종료됐습니다")
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError("레벨 미터 hard-max watchdog을 넘었습니다")
                try:
                    level = meter.get(timeout=min(0.25, max(0.01, deadline - now)))
                except _queue.Empty:
                    continue
                levels.append(float(level))
                filled = max(0, min(40, int(level + 70.0))) if np.isfinite(level) else 0
                bar = "█" * filled
                print(f"  {level:+7.1f} dBFS  {bar:<40} {verdict(level)}", flush=True)
        except LiveAudioTermination as exc:
            telemetry["interrupted"] = True
            telemetry["termination_signal"] = int(exc.signum)
            failure = exc
        except KeyboardInterrupt as exc:
            telemetry["interrupted"] = True
            failure = exc
        except BaseException as exc:
            failure = exc

        if stream is not None:
            try:
                stream.abort()
            except BaseException as exc:
                telemetry["stream_abort_error"] = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, LiveAudioTermination):
                    telemetry["interrupted"] = True
                    telemetry["termination_signal"] = int(exc.signum)
                    failure = failure or exc
            try:
                stream.close()
            except BaseException as exc:
                telemetry["stream_close_error"] = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, LiveAudioTermination):
                    telemetry["interrupted"] = True
                    telemetry["termination_signal"] = int(exc.signum)
                    failure = failure or exc
        telemetry["output_stop_confirmed"] = bool(
            stream is None or telemetry["stream_close_error"] is None
        )
        print(
            SPEAKER_DISCONNECT_NOTICE
            if telemetry["output_stop_confirmed"]
            else SPEAKER_STOP_UNCONFIRMED_NOTICE,
            flush=True,
        )
    finally:
        signal_scope.__exit__(None, None, None)

    while True:
        try:
            levels.append(float(meter.get_nowait()))
        except _queue.Empty:
            break
    telemetry["elapsed_seconds"] = float(time.monotonic() - call_started)
    telemetry["output_elapsed_seconds"] = (
        None
        if stream_started is None
        else float(time.monotonic() - stream_started)
    )
    if failure is not None and telemetry["callback_error"] is None:
        telemetry["callback_error"] = f"{type(failure).__name__}: {failure}"
    if include_raw:
        return levels, telemetry, submitted_output_pcm, input_raw
    return levels, telemetry


def _bootstrap_meter_session(args) -> tuple[Path, str]:  # noqa: ANN001
    root = (REPO_ROOT / args.diagnostics_root).resolve()
    results_root = (REPO_ROOT / "results").resolve()
    try:
        root.relative_to(results_root)
    except ValueError as exc:
        raise ValueError("bootstrap meter diagnostics는 results/ 아래여야 합니다") from exc
    root.mkdir(parents=True, exist_ok=True)
    capture_id = uuid.uuid4().hex
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session = root / f"{stamp}_{capture_id[:8]}"
    session.mkdir(exist_ok=False)
    return session, capture_id


def measure(args) -> int:
    from deep_anc.audio_io import (
        assert_measurement_preconditions,
        resolve_alsa_portaudio_device,
    )

    if not np.isclose(
        float(args.seconds),
        OFFICIAL_MEASUREMENT_LEVEL.meter_seconds,
        rtol=0.0,
        atol=1e-12,
    ):
        print(
            f"[중단] official meter는 정확히 {OFFICIAL_MEASUREMENT_LEVEL.meter_seconds:.0f}초입니다: "
            f"{args.seconds!r}",
            file=sys.stderr,
        )
        return 2
    mode = str(getattr(args, "mode", METER_MODE_STRICT))
    fullband_modes = {METER_MODE_FULLBAND_V5, METER_MODE_FULLBAND_V6}
    if mode not in {METER_MODE_STRICT, *fullband_modes}:
        print(f"[중단] 알 수 없는 meter mode: {mode!r}", file=sys.stderr)
        return 2
    bootstrap = bool(getattr(args, "bootstrap_level_evidence", False))
    if mode in fullband_modes and bootstrap:
        print(
            f"[중단] {mode}는 기존 strict paired level evidence만 사용하며 "
            "bootstrap 우회를 허용하지 않습니다",
            file=sys.stderr,
        )
        return 2
    calibration_evidence: dict | None = None
    fullband_contract: dict[str, Any] | None = None
    fullband_execution_identity: dict[str, Any] | None = None
    fullband_resolve_devices: Callable[..., dict[str, int]] | None = None
    fullband_build_followup: Callable[..., dict[str, Any]] | None = None
    fullband_confirmations: dict[str, bool] | None = None
    try:
        if mode == METER_MODE_FULLBAND_V5:
            # authority/plan/hardware/evidence/sealed raw freshness를 PortAudio import보다
            # 먼저 검증한다. old broadband/v4 파일은 loader schema/SHA에서 거부된다.
            fullband_execution_identity = _v5_repository_execution_identity()
            fullband_contract = _load_fullband_v5_static_contract(
                args,
                require_sealed_raw_fresh=True,
            )
            calibration_evidence = fullband_contract["evidence"]
            fullband_resolve_devices = _resolve_fullband_v5_devices
            fullband_build_followup = _build_fullband_v5_followup
            fullband_confirmations = _v5_confirmations(args)
        elif mode == METER_MODE_FULLBAND_V6:
            # v6도 clean exact checkout과 v6 sealed plan/authority/raw path를
            # PortAudio import보다 먼저 검증한다. v5 path splice는 validator가 거부한다.
            fullband_execution_identity = _v6_repository_execution_identity()
            fullband_contract = _load_fullband_v6_static_contract(
                args,
                require_sealed_raw_fresh=True,
            )
            calibration_evidence = {
                "hardware_identity": fullband_contract["hardware_identity"]
            }
            fullband_resolve_devices = _resolve_fullband_v6_devices
            fullband_build_followup = _build_fullband_v6_followup
            fullband_confirmations = _v6_confirmations(args)
        if fullband_contract is not None:
            hardware_config = fullband_contract["hardware_config"]
            hardware = fullband_contract["hardware_audio"]
            channel_map = fullband_contract["channel_map"]
            physical_fingerprint = fullband_contract["physical_fingerprint"]
            hardware_identity = fullband_contract["hardware_identity"]
        elif bootstrap:
            evidence_path = (REPO_ROOT / args.level_evidence).resolve()
            evidence_path.relative_to(REPO_ROOT.resolve())
            if evidence_path.exists():
                raise FileExistsError(
                    "canonical level evidence가 이미 있어 bootstrap을 다시 실행할 수 없습니다: "
                    f"{evidence_path}"
                )
            hardware_config = load_yaml(REPO_ROOT / args.hardware)
            hardware, channel_map = validate_measurement_hardware_contract(
                hardware_config
            )
            physical_fingerprint = collect_alsa_physical_fingerprint(hardware_config)
            hardware_identity = measurement_hardware_identity(
                hardware_config,
                physical_fingerprint=physical_fingerprint,
            )
        else:
            calibration_evidence = load_measurement_level_evidence(
                args.level_evidence,
                repository_root=REPO_ROOT,
            )
            hardware_config = load_yaml(REPO_ROOT / args.hardware)
            hardware, channel_map = validate_measurement_hardware_contract(
                hardware_config
            )
            physical_fingerprint = collect_alsa_physical_fingerprint(hardware_config)
            hardware_identity = measurement_hardware_identity(
                hardware_config,
                physical_fingerprint=physical_fingerprint,
            )
        if calibration_evidence is not None and (
            calibration_evidence.get("hardware_identity") != hardware_identity
        ):
            raise ValueError(
                "영구 calibration evidence의 hardware identity가 현재 meter "
                "hardware/channel 계약과 다릅니다"
            )

        # 위 read-only file/identity gate가 모두 끝난 뒤에만 PortAudio를 import/query한다.
        import sounddevice as sd

        if fullband_contract is not None:
            assert fullband_resolve_devices is not None
            devices = fullband_resolve_devices(sd, fullband_contract)
            in_dev = devices["input"]
            out_dev = devices["output"]
        else:
            in_dev = resolve_alsa_portaudio_device(
                hardware["input"]["card"], hardware["input"]["pcm"], "input", 2
            )
            out_dev = resolve_alsa_portaudio_device(
                hardware["output"]["card"], hardware["output"]["pcm"], "output", 2
            )
        bootstrap_session, capture_id = _bootstrap_meter_session(args)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 실기 preflight 실패: {exc}", file=sys.stderr)
        print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        return 2

    fs = OFFICIAL_MEASUREMENT_LEVEL.sample_rate
    err_ch = channel_map["error_mic"]
    noise_out_ch = channel_map["noise_out"]
    noise = probe_signal(args.seconds)
    print(
        f"목표 {OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs:+.1f} dBFS "
        f"(허용 ±{OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db:.0f} dB) · "
        f"프로브 peak {OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude:.3f} · "
        f"ch{noise_out_ch}(소음 스피커)만 재생"
    )
    print(
        f"nominal 출력 {args.seconds:.1f}초 / hard-max "
        f"{args.seconds + LIVE_WATCHDOG_GRACE_SECONDS:.1f}초 동안 표시합니다. "
        "그 전에 무출력 입력 preflight 1.5초를 한 번 수행합니다. "
        "숫자를 보며 노브를 돌리세요.\n"
        "Ctrl-C는 안전 중단이지만 결과는 무조건 FAIL입니다.\n",
        flush=True,
    )
    post_revalidation_error: str | None = None
    try:
        with repository_audio_lock(REPO_ROOT, purpose="measurement_level_meter") as audio_lock:
            print(
                f"저장소/UID audio lock 획득: {audio_lock['path']} (pid={audio_lock['pid']}). "
                "다른 저장소나 lock 미준수 프로세스는 별도 장치 점유 gate로만 방어합니다."
            )
            # 두 입력/rail은 무출력 preflight로 정확히 한 번 판정한다.
            clip_ratio = assert_measurement_preconditions(sd, hardware)
            print(
                f"마이크/배선 점검 PASS: ERR=ch{err_ch}, "
                f"REF=ch{channel_map['reference_mic']}, "
                f"noise_out=ch{noise_out_ch}, cancel_out=ch{channel_map['cancel_out']} · "
                f"레일 비율 {clip_ratio[0]:.4f}/{clip_ratio[1]:.4f}"
            )
            def _pre_open_check() -> None:
                # capture 함수가 큰 raw buffer를 준비한 **뒤**, sd.Stream open 직전에
                # 실제 codec/DAC와 PCM/clock을 read-only로 확인한다.
                if fullband_contract is not None:
                    assert (
                        fullband_resolve_devices is not None
                        and fullband_build_followup is not None
                        and fullband_confirmations is not None
                    )
                    static_loader = (
                        _load_fullband_v5_static_contract
                        if mode == METER_MODE_FULLBAND_V5
                        else _load_fullband_v6_static_contract
                    )
                    refreshed_contract = static_loader(
                        args,
                        require_sealed_raw_fresh=True,
                    )
                    refreshed_devices = fullband_resolve_devices(
                        sd, refreshed_contract
                    )
                    expected_followup = fullband_build_followup(
                        fullband_contract,
                        resolved_devices={"input": in_dev, "output": out_dev},
                        confirmations=fullband_confirmations,
                    )
                    refreshed_followup = fullband_build_followup(
                        refreshed_contract,
                        resolved_devices=refreshed_devices,
                        confirmations=fullband_confirmations,
                    )
                    if refreshed_followup != expected_followup:
                        raise RuntimeError(
                            f"{mode} output 직전 plan/authority/hardware/evidence/device "
                            "결속이 변경됐습니다"
                        )
                else:
                    refreshed = collect_alsa_physical_fingerprint(hardware_config)
                    if refreshed != physical_fingerprint:
                        raise RuntimeError(
                            "output 직전 ALSA physical fingerprint가 preflight 이후 변경됐습니다"
                        )
                assert_live_pcm_clock_preconditions(hardware)

            started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
            captured = capture_meter_stream(
                sd,
                noise=noise,
                fs=fs,
                in_dev=in_dev,
                out_dev=out_dev,
                err_ch=err_ch,
                noise_out_ch=noise_out_ch,
                block_size=int(hardware["block_size"]),
                latency=str(hardware["latency"]),
                include_raw=True,
                pre_open_check=_pre_open_check,
            )
            if fullband_contract is not None:
                try:
                    assert (
                        fullband_resolve_devices is not None
                        and fullband_build_followup is not None
                        and fullband_confirmations is not None
                    )
                    static_loader = (
                        _load_fullband_v5_static_contract
                        if mode == METER_MODE_FULLBAND_V5
                        else _load_fullband_v6_static_contract
                    )
                    refreshed_contract = static_loader(
                        args,
                        require_sealed_raw_fresh=True,
                    )
                    refreshed_devices = fullband_resolve_devices(
                        sd, refreshed_contract
                    )
                    if fullband_build_followup(
                        refreshed_contract,
                        resolved_devices=refreshed_devices,
                        confirmations=fullband_confirmations,
                    ) != fullband_build_followup(
                        fullband_contract,
                        resolved_devices={"input": in_dev, "output": out_dev},
                        confirmations=fullband_confirmations,
                    ):
                        raise RuntimeError(
                            f"{mode} capture 뒤 plan/authority/hardware/evidence/device "
                            "결속이 변경됐습니다"
                        )
                except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                    # 출력은 이미 close됐다. 유일한 raw는 보존하되 PASS 승격만 막는다.
                    post_revalidation_error = f"{type(exc).__name__}: {exc}"
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] output 직전 precondition/audio lock 실패: {exc}", file=sys.stderr)
        print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        return 2
    completed_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    levels, telemetry, submitted_output_pcm, input_raw = captured
    reasons = meter_capture_invalid_reasons(
        telemetry,
        levels,
        expected_frames=int(noise.size),
    )
    if post_revalidation_error is not None:
        reasons.append(f"{mode.replace('-', '_')}_post_capture_binding_invalid")
    final = (
        float(np.median(levels[-8:] if len(levels) >= 8 else levels))
        if levels and np.all(np.isfinite(np.asarray(levels, dtype=np.float64)))
        else None
    )
    target_pass = bool(
        final is not None
        and abs(final - OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs)
        <= OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db
    )
    bootstrap_paths = None
    status = "PASS" if not reasons and target_pass else "FAIL"
    metadata = {
            "schema": BOOTSTRAP_METER_RAW_SCHEMA,
            "capture_id": capture_id,
            "status": status,
            "passed": status == "PASS",
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "hardware_identity": hardware_identity,
            "resolved_devices": {"input": int(in_dev), "output": int(out_dev)},
            "audio_lock": dict(audio_lock),
            "calibration_evidence": (
                {
                    "mode": (
                        "fullband_v5_tracked_attestation"
                        if mode == METER_MODE_FULLBAND_V5
                        else "fullband_v6_tracked_attestation"
                    ),
                    "level_evidence": {
                        "path": fullband_contract["level_evidence"]["path"],
                        "file_sha256": fullband_contract["level_evidence"][
                            "file_sha256"
                        ],
                        "scope": fullband_contract["level_evidence"]["scope"],
                        "preserved_raw_revalidated": False,
                    },
                }
                if fullband_contract is not None
                else {
                    "mode": "bootstrap_pending" if bootstrap else "verified_existing",
                    "path": (
                        None
                        if calibration_evidence is None
                        else calibration_evidence.get("_evidence_path")
                    ),
                    "sha256": (
                        None
                        if calibration_evidence is None
                        else calibration_evidence.get("_evidence_sha256")
                    ),
                }
            ),
            "operator_confirmations": {
                "speaker_output": bool(args.confirm_speaker),
                "user_present": bool(args.confirm_user_present),
                "volume_minimum_before_start": bool(args.confirm_volume_minimum),
            },
            "recipe": {
                "sample_rate": OFFICIAL_MEASUREMENT_LEVEL.sample_rate,
                "block_size": int(hardware["block_size"]),
                "latency": str(hardware["latency"]),
                "seconds": OFFICIAL_MEASUREMENT_LEVEL.meter_seconds,
                "probe_amplitude": OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
                "period_seconds": OFFICIAL_MEASUREMENT_LEVEL.period_seconds,
                "design_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.design_band_hz),
                "meter_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
                "meter_target_dbfs": OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs,
                "meter_tolerance_db": OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db,
                "noise_output_channel": noise_out_ch,
                "cancel_output_silent": True,
            },
            "meter_ch0_dbfs": final,
            "telemetry": telemetry,
            "invalid_reasons": list(reasons),
    }
    if fullband_contract is not None:
        assert fullband_build_followup is not None and fullband_confirmations is not None
        metadata["repository_execution"] = fullband_execution_identity
        generation_key = "fullband_v5" if mode == METER_MODE_FULLBAND_V5 else "fullband_v6"
        metadata[f"{generation_key}_followup"] = fullband_build_followup(
            fullband_contract,
            resolved_devices={"input": in_dev, "output": out_dev},
            confirmations=fullband_confirmations,
        )
        metadata[f"{generation_key}_post_capture_revalidation"] = {
            "passed": post_revalidation_error is None,
            "error": post_revalidation_error,
        }
    try:
        if mode == METER_MODE_FULLBAND_V5:
            raw_writer = write_fullband_v5_meter_raw_atomic
        elif mode == METER_MODE_FULLBAND_V6:
            raw_writer = write_fullband_v6_meter_raw_atomic
        else:
            raw_writer = write_bootstrap_meter_raw_atomic
        bootstrap_paths = raw_writer(
            bootstrap_session / "meter_raw.npz",
            repository_root=REPO_ROOT,
            metadata=metadata,
            submitted_output_pcm_int16=submitted_output_pcm,
            input_raw_int32=input_raw,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raw_candidate = bootstrap_session / "meter_raw.npz"
        recovery = (
            f" immutable raw는 보존됐습니다: {raw_candidate}"
            if raw_candidate.is_file()
            else " final raw는 노출되지 않았습니다."
        )
        print(
            f"[실패] fresh meter raw 저장 실패: {exc}.{recovery}",
            file=sys.stderr,
        )
        return 1
    print(
        "fresh meter immutable raw 저장: "
        f"{bootstrap_paths['raw'].relative_to(REPO_ROOT)}\n"
        f"  SHA256 {bootstrap_paths['sha256']}"
    )
    if "recovery_relative_path" in bootstrap_paths:
        print(
            "  durable same-inode recovery hardlink (중복 캡처 아님): "
            f"{bootstrap_paths['recovery_relative_path']}\n"
            f"  recovery SHA256 {bootstrap_paths['recovery_sha256']}"
        )
    if reasons:
        print(
            "[실패] 레벨 미터 캡처 계약 위반: " + ", ".join(reasons),
            file=sys.stderr,
        )
        return 1
    assert final is not None
    print(f"\n마지막 구간 중앙값 {final:+.1f} dBFS — {verdict(final)}")
    if target_pass:
        assert bootstrap_paths is not None
        raw_rel = bootstrap_paths["raw"].relative_to(REPO_ROOT)
        if fullband_contract is not None:
            try:
                if mode == METER_MODE_FULLBAND_V5:
                    precommand = validate_fullband_v5_meter_raw(
                        raw_rel,
                        repository_root=REPO_ROOT,
                        require_fresh=True,
                        require_sealed_raw_fresh=True,
                    )
                    identity = _validated_meter_identity_sha256(precommand)
                    command = fullband_v5_followup_command(raw_rel)
                    adapter_available = fullband_v5_live_adapter_available()
                else:
                    precommand = validate_fullband_v6_meter_raw(
                        raw_rel,
                        repository_root=REPO_ROOT,
                        require_fresh=True,
                        require_sealed_raw_fresh=True,
                    )
                    identity = _validated_meter_identity_sha256(precommand)
                    command = fullband_v6_followup_command(raw_rel)
                    adapter_available = fullband_v6_live_adapter_available()
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                print(
                    f"[실패] {mode} capture 명령 직전 meter/authority 재검증 실패: {exc}",
                    file=sys.stderr,
                )
                return 1
            print(
                f"{mode} fresh meter/plan/authority/hardware/evidence 결속 PASS:\n"
                f"  meter identity {identity}\n"
                f"  followup contract {precommand['followup_contract_sha256']}\n"
                f"같은 앰프 노브를 유지하고 10분 안에 아래 exact {mode} capture를 실행하세요:\n"
                + command
            )
            if adapter_available:
                print(f"[READY] reviewed {mode} --execute-live adapter marker를 확인했습니다")
            else:
                print(
                    f"[차단] {mode} --execute-live adapter가 아직 구현/표시되지 않았습니다. "
                    "위 명령은 기록용이며 실행하면 안 됩니다.",
                    file=sys.stderr,
                )
            return 0
        command = strict_followup_command(
            raw_rel, capture_id=capture_id, bootstrap=bootstrap
        )
        if bootstrap:
            print(
                "같은 앰프 노브 설정을 유지하고 아래 strict bootstrap만 실행하세요. "
                "추가 probe 없이 기존 12.5초 official capture가 evidence의 두 번째 raw가 됩니다:\n"
                + command
            )
        else:
            print(
                "영구 evidence는 calibration 근거일 뿐 현재 노브 증거가 아닙니다. "
                "같은 노브를 유지하고 10분 안에 strict를 실행하세요:\n"
                + command
            )
        return 0
    print("목표 범위 밖입니다. 노브를 조정하고 다시 실행하세요.", file=sys.stderr)
    return 1


def _mode_path_defaults(mode: str) -> dict[str, str]:
    if mode == METER_MODE_FULLBAND_V6:
        return {
            "hardware": V6_HARDWARE_PATH,
            "plan_envelope": V6_PLAN_ENVELOPE_PATH,
            "live_authority": V6_LIVE_AUTHORITY_PATH,
            "raw_target": V6_RAW_TARGET_PATH,
            "level_evidence": V6_LEVEL_EVIDENCE_PATH,
        }
    return {
        "hardware": V5_HARDWARE_PATH,
        "plan_envelope": V5_PLAN_ENVELOPE_PATH,
        "live_authority": V5_LIVE_AUTHORITY_PATH,
        "raw_target": V5_RAW_TARGET_PATH,
        "level_evidence": V5_LEVEL_EVIDENCE_PATH,
    }


def _lexical_repository_path(value: str | Path) -> Path:
    supplied = Path(os.fspath(value))
    return Path(os.path.abspath(supplied if supplied.is_absolute() else REPO_ROOT / supplied))


def _apply_mode_defaults(args) -> None:  # noqa: ANN001
    """mode별 sealed path를 채우고 v5/v6 cross-generation splice를 차단한다."""

    defaults = _mode_path_defaults(str(args.mode))
    fullband = args.mode in {METER_MODE_FULLBAND_V5, METER_MODE_FULLBAND_V6}
    for name, expected in defaults.items():
        supplied = getattr(args, name)
        if supplied is None:
            setattr(args, name, expected)
            continue
        if fullband and _lexical_repository_path(supplied) != _lexical_repository_path(expected):
            raise ValueError(
                f"{args.mode} --{name.replace('_', '-')}가 exact generation path와 다릅니다"
            )

    if args.diagnostics_root is None:
        if args.mode == METER_MODE_FULLBAND_V5:
            args.diagnostics_root = FULLBAND_V5_DEFAULT_DIAGNOSTICS_ROOT
        elif args.mode == METER_MODE_FULLBAND_V6:
            args.diagnostics_root = FULLBAND_V6_DEFAULT_DIAGNOSTICS_ROOT
        else:
            args.diagnostics_root = "results/calibration_interleaved/level_bootstrap"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(METER_MODE_STRICT, METER_MODE_FULLBAND_V5, METER_MODE_FULLBAND_V6),
        default=METER_MODE_STRICT,
        help="strict interleaved 또는 committed causal fullband-v5/v6 후속 측정",
    )
    parser.add_argument("--hardware", default=None)
    parser.add_argument(
        "--plan-envelope",
        default=None,
        help="fullband-v5/v6에서만 사용; mode별 sealed exact path만 허용",
    )
    parser.add_argument(
        "--live-authority",
        default=None,
        help="fullband-v5/v6에서만 사용; mode별 capture-only authority exact path",
    )
    parser.add_argument(
        "--raw-target",
        default=None,
        help="fullband-v5/v6에서만 사용; mode별 아직 존재하지 않는 raw exact path",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=OFFICIAL_MEASUREMENT_LEVEL.meter_seconds,
        help="official에서는 정확히 20초로 고정",
    )
    parser.add_argument(
        "--level-evidence",
        default=None,
        help="peak 0.003 paired raw evidence JSON",
    )
    parser.add_argument(
        "--bootstrap-level-evidence",
        action="store_true",
        help=(
            "canonical paired evidence가 아직 없을 때만 쓰는 1회 경로. 기존 evidence "
            "gate를 우회하는 대신 PASS meter raw+SHA receipt를 새 경로에 보존한다"
        ),
    )
    parser.add_argument(
        "--diagnostics-root",
        default=None,
        help="bootstrap meter immutable raw의 새 session 상위 경로(results/ 아래만 허용)",
    )
    parser.add_argument("--self-test", action="store_true", help="소리 없이 미터만 검증")
    parser.add_argument("--confirm-speaker", action="store_true")
    parser.add_argument(
        "--confirm-user-present",
        action="store_true",
        help="실제 출력 동안 사용자가 입회함을 명시적으로 확인",
    )
    parser.add_argument(
        "--confirm-volume-minimum",
        action="store_true",
        help="출력 시작 전 앰프 볼륨이 최저임을 명시적으로 확인",
    )
    parser.add_argument(
        "--confirm-routing-and-geometry",
        action="store_true",
        help="fullband-v5/v6 ERR/REF 및 NS/CS 배선·덕트 기하가 authority와 같음을 확인",
    )
    parser.add_argument(
        "--confirm-same-amplifier-setting",
        action="store_true",
        help="fullband-v5/v6에서 paired evidence와 같은 앰프/노브 설정을 유지함을 확인",
    )
    args = parser.parse_args(argv)

    try:
        _apply_mode_defaults(args)
    except ValueError as error:
        print(f"[중단] meter mode/path 결속 실패: {error}", file=sys.stderr)
        return 2

    if args.self_test:
        return self_test()
    if not np.isclose(
        float(args.seconds),
        OFFICIAL_MEASUREMENT_LEVEL.meter_seconds,
        rtol=0.0,
        atol=1e-12,
    ):
        print(
            f"[중단] official meter는 정확히 "
            f"{OFFICIAL_MEASUREMENT_LEVEL.meter_seconds:.0f}초입니다: "
            f"{args.seconds!r}",
            file=sys.stderr,
        )
        return 2
    if not args.confirm_speaker:
        print(
            f"스피커에서 {args.seconds:.0f}초 동안 소리가 납니다. "
            "--confirm-speaker 를 붙여 실행하세요.",
            file=sys.stderr,
        )
        return 2
    if not (args.confirm_user_present and args.confirm_volume_minimum):
        print(
            "[중단] 모든 live meter는 사용자 입회와 시작 전 볼륨 최저 확인이 "
            "모두 필요합니다: --confirm-user-present --confirm-volume-minimum",
            file=sys.stderr,
        )
        return 2
    if args.mode in {METER_MODE_FULLBAND_V5, METER_MODE_FULLBAND_V6} and not (
        args.confirm_routing_and_geometry and args.confirm_same_amplifier_setting
    ):
        print(
            f"[중단] {args.mode} meter는 다섯 확인이 모두 필요합니다: "
            "--confirm-speaker --confirm-user-present --confirm-volume-minimum "
            "--confirm-routing-and-geometry --confirm-same-amplifier-setting",
            file=sys.stderr,
        )
        return 2
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
