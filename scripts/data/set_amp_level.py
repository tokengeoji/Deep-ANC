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
import datetime as dt
import hashlib
import shlex
import sys
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_yaml  # noqa: E402
from deep_anc.dsp.measurement_level import (  # noqa: E402
    BOOTSTRAP_METER_RAW_SCHEMA,
    DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH,
    LIVE_WATCHDOG_GRACE_SECONDS,
    LiveAudioTermination,
    MEASUREMENT_LEVEL_EVIDENCE_SCHEMA,
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
        below_min = OFFICIAL_MEASUREMENT_LEVEL.meter_min_dbfs - level
        return (
            f"↑ 올리세요 "
            f"(하한 {OFFICIAL_MEASUREMENT_LEVEL.meter_min_dbfs:+.4f} 대비 "
            f"{below_min:.4f} dB 부족; 목표 대비 "
            f"{OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs - level:+.1f} dB)"
        )
    if level > OFFICIAL_MEASUREMENT_LEVEL.meter_max_dbfs:
        above_max = level - OFFICIAL_MEASUREMENT_LEVEL.meter_max_dbfs
        return (
            f"↓ 내리세요 "
            f"(상한 {OFFICIAL_MEASUREMENT_LEVEL.meter_max_dbfs:+.4f} 대비 "
            f"{above_max:.4f} dB 초과; 목표 대비 "
            f"{level - OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs:+.1f} dB)"
        )
    margin = min(
        level - OFFICIAL_MEASUREMENT_LEVEL.meter_min_dbfs,
        OFFICIAL_MEASUREMENT_LEVEL.meter_max_dbfs - level,
    )
    return (
        "✅ 맞았습니다 — 여기서 멈추세요 "
        f"(가장 가까운 경계 여유 {margin:.4f} dB)"
    )


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


def broadband_followup_command(
    meter_raw: str | Path,
    *,
    plan: str | Path,
    raw_session_dir: str | Path,
    level_evidence: str | Path,
    hardware: str | Path,
) -> str:
    """PASS meter 뒤 그대로 복사할 광대역 P/S live 명령을 만든다."""

    argv = [
        ".venv/bin/python",
        "scripts/data/measure_paths_broadband_interleaved.py",
        "--execute-live",
        "--hardware",
        str(hardware),
        "--plan",
        str(plan),
        "--meter-raw",
        str(meter_raw),
        "--level-evidence",
        str(level_evidence),
        "--raw-session-dir",
        str(raw_session_dir),
        "--confirm-speaker",
        "--confirm-user-present",
        "--confirm-volume-minimum",
        "--confirm-routing-and-geometry",
        "--confirm-same-amplifier-setting",
    ]
    return "  " + " ".join(shlex.quote(value) for value in argv)


def fullband_v5_followup_command(
    meter_raw: str | Path, *, contract: dict
) -> str:
    """현재는 fail-closed adapter를 가리키는 exact v5 capture-only 명령."""

    argv = [
        ".venv/bin/python",
        "scripts/data/measure_paths_fullband_causal_v5.py",
        "--execute-live",
        "--plan-envelope",
        contract["plan_envelope"]["path"],
        "--live-authority",
        contract["live_capture_authority"]["path"],
        "--meter-raw",
        str(meter_raw),
        "--level-evidence",
        contract["level_evidence"]["path"],
        "--hardware",
        contract["hardware"]["path"],
        "--raw-target",
        contract["sealed_raw"]["path"],
        "--confirm-speaker",
        "--confirm-user-present",
        "--confirm-volume-minimum",
        "--confirm-routing-and-geometry",
        "--confirm-same-amplifier-setting",
    ]
    return "  " + " ".join(shlex.quote(value) for value in argv)


def _repository_cli_path(path: str | Path) -> str:
    """저장소 안 경로는 cwd에서 바로 실행 가능한 상대경로로 표시한다."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    return resolved.relative_to(REPO_ROOT.resolve()).as_posix()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_sha256(value: dict) -> str:
    import json

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_fullband_v5_followup_contract(
    *, hardware_arg: str | Path, level_evidence_arg: str | Path
) -> dict:
    from deep_anc.dsp import fullband_live_authority_v5 as authority

    hardware_path = (REPO_ROOT / Path(hardware_arg)).resolve()
    if _repository_cli_path(hardware_path) != authority.SEALED_HARDWARE_RELATIVE_PATH:
        raise ValueError("fullband-v5 hardware는 sealed exact path여야 합니다")
    evidence = load_measurement_level_evidence(
        level_evidence_arg, repository_root=REPO_ROOT
    )
    evidence_path = Path(evidence["_evidence_path"])
    evidence_relative = _repository_cli_path(evidence_path)
    if evidence_relative != Path(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH).as_posix():
        raise ValueError("fullband-v5 level evidence는 canonical exact path여야 합니다")
    if evidence.get("schema") != MEASUREMENT_LEVEL_EVIDENCE_SCHEMA:
        raise ValueError("fullband-v5 level evidence schema가 canonical exact 값과 다릅니다")
    loaded_authority = authority.load_exact_saved_live_capture_authority_v5(
        REPO_ROOT / authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
        repository_root=REPO_ROOT,
        expected_file_sha256=authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_FILE_SHA256,
        expected_payload_sha256=authority.EXPECTED_LIVE_CAPTURE_AUTHORITY_PAYLOAD_SHA256,
    )
    validation = loaded_authority["validation"]
    if validation["sealed_raw"] != {
        "path": authority.SEALED_RAW_RELATIVE_PATH,
        "fresh": True,
    }:
        raise ValueError("fullband-v5 sealed raw freshness receipt가 exact하지 않습니다")
    core = {
        "schema": "fullband_v5_meter_followup_v1",
        "mode": "fullband-v5",
        "status": "blocked_until_v5_live_adapter_implementation",
        "capture_only": True,
        "plan_live_capture_enabled": False,
        "plan_envelope": {
            "path": authority.SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
            "file_sha256": authority.EXPECTED_PLAN_ENVELOPE_FILE_SHA256,
            "payload_sha256": authority.EXPECTED_PLAN_PAYLOAD_SHA256,
            "pcm_sha256": authority.EXPECTED_PCM_SHA256,
        },
        "live_capture_authority": {
            "path": authority.SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
            "file_sha256": loaded_authority["file_sha256"],
            "payload_sha256": loaded_authority["payload_sha256"],
        },
        "hardware": {
            "path": authority.SEALED_HARDWARE_RELATIVE_PATH,
            "sha256": authority.EXPECTED_HARDWARE_FILE_SHA256,
        },
        "level_evidence": {
            "path": evidence_relative,
            "sha256": str(evidence["_evidence_sha256"]),
            "schema": str(evidence["schema"]),
        },
        "sealed_raw": {
            "path": authority.SEALED_RAW_RELATIVE_PATH,
            "fresh": True,
        },
    }
    return {**core, "canonical_payload_sha256": _canonical_payload_sha256(core)}


def _validate_broadband_followup_contract(
    *,
    plan_arg: str | Path,
    raw_target_arg: str | Path,
    hardware_arg: str | Path,
    level_evidence_arg: str | Path,
) -> dict:
    from scripts.data import measure_paths_broadband_interleaved as broadband

    raw_session = broadband.raw_session_path_for_plan({}, raw_target_arg)
    broadband.validate_fresh_raw_session_target(raw_session)

    plan_candidate = Path(plan_arg).expanduser()
    if not plan_candidate.is_absolute():
        plan_candidate = REPO_ROOT / plan_candidate
    plan_candidate = plan_candidate.resolve()
    try:
        plan_candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"광대역 saved plan은 저장소 안에 있어야 합니다: {plan_candidate}"
        ) from exc
    if not plan_candidate.is_file():
        raise FileNotFoundError(f"저장된 광대역 plan이 없습니다: {plan_candidate}")

    hardware_candidate = Path(hardware_arg).expanduser()
    if not hardware_candidate.is_absolute():
        hardware_candidate = REPO_ROOT / hardware_candidate
    hardware_candidate = hardware_candidate.resolve()
    hardware_relative = _repository_cli_path(hardware_candidate)
    if not hardware_candidate.is_file():
        raise FileNotFoundError(f"광대역 hardware YAML이 없습니다: {hardware_candidate}")
    hardware_sha = _sha256_file(hardware_candidate)

    evidence_candidate = Path(level_evidence_arg).expanduser()
    if not evidence_candidate.is_absolute():
        evidence_candidate = REPO_ROOT / evidence_candidate
    evidence_candidate = evidence_candidate.resolve()
    evidence_relative = _repository_cli_path(evidence_candidate)
    if not evidence_candidate.is_file():
        raise FileNotFoundError(
            f"광대역 paired level evidence가 없습니다: {evidence_candidate}"
        )
    evidence_sha = _sha256_file(evidence_candidate)

    expected_plan, expected_pcm = broadband.build_signal_plan(
        hardware_path=hardware_candidate,
    )
    saved_plan = broadband.load_exact_saved_plan(
        plan_candidate,
        expected_plan=expected_plan,
        expected_pcm=expected_pcm,
    )
    authority = broadband.validate_live_authority_binding(saved_plan, expected_pcm)
    return {
        "schema": "broadband_meter_followup_v1",
        "mode": "broadband",
        "plan": {
            "path": str(authority["path"]),
            "file_sha256": str(authority["file_sha256"]),
            "payload_sha256": str(authority["payload_sha256"]),
            "pcm_sha256": str(authority["pcm_sha256"]),
        },
        "raw_session_dir": _repository_cli_path(raw_session),
        "hardware": {"path": hardware_relative, "sha256": hardware_sha},
        "level_evidence": {
            "path": evidence_relative,
            "sha256": evidence_sha,
        },
    }


def revalidate_followup_contract(contract: dict) -> dict:
    """meter open/종료 직전에 followup 입력 bytes와 raw freshness를 다시 읽는다."""

    if contract.get("mode") == "fullband-v5":
        refreshed = _validate_fullband_v5_followup_contract(
            hardware_arg=contract["hardware"]["path"],
            level_evidence_arg=contract["level_evidence"]["path"],
        )
        if refreshed != contract:
            raise ValueError(
                "fullband-v5 meter followup contract가 최초 preflight 이후 변경됐습니다"
            )
        return refreshed
    if contract.get("mode") != "broadband":
        return contract
    refreshed = _validate_broadband_followup_contract(
        plan_arg=contract["plan"]["path"],
        raw_target_arg=contract["raw_session_dir"],
        hardware_arg=contract["hardware"]["path"],
        level_evidence_arg=contract["level_evidence"]["path"],
    )
    if refreshed != contract:
        raise ValueError("광대역 meter followup contract가 최초 preflight 이후 변경됐습니다")
    return refreshed


def validate_followup_contract(args) -> dict:  # noqa: ANN001
    """미터가 안내할 후속 측정을 장치 import/open 전에 fail-closed 검증한다."""

    mode = str(getattr(args, "followup_mode", "strict"))
    plan_arg = getattr(args, "broadband_plan", None)
    raw_target_arg = getattr(args, "broadband_raw_session_dir", None)
    if mode == "strict":
        if plan_arg is not None or raw_target_arg is not None:
            raise ValueError(
                "--broadband-plan/--broadband-raw-session-dir는 "
                "--followup-mode broadband에서만 사용할 수 있습니다"
            )
        return {"mode": "strict"}
    if mode == "fullband-v5":
        if plan_arg is not None or raw_target_arg is not None:
            raise ValueError(
                "fullband-v5에서 old --broadband-plan/--broadband-raw-session-dir를 거부합니다"
            )
        if bool(getattr(args, "bootstrap_level_evidence", False)):
            raise ValueError("fullband-v5는 verified existing level evidence가 필요합니다")
        return _validate_fullband_v5_followup_contract(
            hardware_arg=args.hardware,
            level_evidence_arg=args.level_evidence,
        )
    if mode != "broadband":
        raise ValueError(f"알 수 없는 meter followup mode입니다: {mode!r}")
    if bool(getattr(args, "bootstrap_level_evidence", False)):
        raise ValueError(
            "광대역 followup은 이미 존재하는 paired level evidence가 필요하므로 "
            "--bootstrap-level-evidence와 함께 사용할 수 없습니다"
        )
    if plan_arg is None or raw_target_arg is None:
        raise ValueError(
            "광대역 followup은 --broadband-plan과 "
            "--broadband-raw-session-dir를 모두 명시해야 합니다"
        )

    # sounddevice를 import하지 않는 signal-only 모듈이다. 이 검증은 main/measure의
    # live audio import보다 앞서 실행되므로 잘못된 plan/target으로 20초 meter조차
    # 시작하지 않는다.
    return _validate_broadband_followup_contract(
        plan_arg=plan_arg,
        raw_target_arg=raw_target_arg,
        hardware_arg=args.hardware,
        level_evidence_arg=args.level_evidence,
    )


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
    try:
        with scoped_live_audio_signal_handlers():
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
        try:
            stream.close()
        except BaseException as exc:
            telemetry["stream_close_error"] = f"{type(exc).__name__}: {exc}"
    telemetry["output_stop_confirmed"] = bool(
        stream is None or telemetry["stream_close_error"] is None
    )
    print(
        SPEAKER_DISCONNECT_NOTICE
        if telemetry["output_stop_confirmed"]
        else SPEAKER_STOP_UNCONFIRMED_NOTICE,
        flush=True,
    )

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
    try:
        followup = getattr(args, "_validated_followup_contract", None)
        if followup is None:
            followup = validate_followup_contract(args)
        followup = revalidate_followup_contract(followup)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 후속 측정 preflight 실패: {exc}", file=sys.stderr)
        print(SPEAKER_PREFLIGHT_ABORT_NOTICE, file=sys.stderr, flush=True)
        return 2

    import sounddevice as sd

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
    bootstrap = bool(getattr(args, "bootstrap_level_evidence", False))
    calibration_evidence: dict | None = None
    try:
        if bootstrap:
            evidence_path = (REPO_ROOT / args.level_evidence).resolve()
            evidence_path.relative_to(REPO_ROOT.resolve())
            if evidence_path.exists():
                raise FileExistsError(
                    "canonical level evidence가 이미 있어 bootstrap을 다시 실행할 수 없습니다: "
                    f"{evidence_path}"
                )
        else:
            calibration_evidence = load_measurement_level_evidence(
                args.level_evidence,
                repository_root=REPO_ROOT,
            )
        hardware_path = (REPO_ROOT / args.hardware).resolve()
        hardware_config = load_yaml(hardware_path)
        hardware, channel_map = validate_measurement_hardware_contract(
            hardware_config
        )
        if followup["mode"] == "fullband-v5" and (
            channel_map.get("noise_out") != 0
            or channel_map.get("cancel_out") != 1
        ):
            raise ValueError(
                "fullband-v5 meter는 noise speaker ch0/cancel speaker ch1 exact mapping이 필요합니다"
            )
        physical_fingerprint = collect_alsa_physical_fingerprint(hardware_config)
        hardware_identity = measurement_hardware_identity(
            hardware_config,
            physical_fingerprint=physical_fingerprint,
        )
        if (
            calibration_evidence is not None
            and calibration_evidence.get("hardware_identity") != hardware_identity
        ):
            raise ValueError(
                "영구 calibration evidence의 hardware identity가 현재 meter "
                "hardware/channel 계약과 다릅니다"
            )
        if followup["mode"] in {"broadband", "fullband-v5"}:
            observed_hardware = {
                "path": _repository_cli_path(hardware_path),
                "sha256": _sha256_file(hardware_path),
            }
            if observed_hardware != followup["hardware"]:
                raise ValueError(
                    "광대역 followup hardware가 meter preflight와 다릅니다"
                )
            if calibration_evidence is None:
                raise ValueError("live followup은 verified_existing evidence가 필요합니다")
            observed_evidence = {
                "path": _repository_cli_path(calibration_evidence["_evidence_path"]),
                "sha256": str(calibration_evidence["_evidence_sha256"]),
            }
            if followup["mode"] == "fullband-v5":
                observed_evidence["schema"] = str(calibration_evidence["schema"])
            if observed_evidence != followup["level_evidence"]:
                raise ValueError(
                    "광대역 followup level evidence가 meter preflight와 다릅니다"
                )
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
                revalidate_followup_contract(followup)
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
    if followup["mode"] in {"broadband", "fullband-v5"}:
        try:
            revalidate_followup_contract(followup)
        except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
            reasons.append(
                "followup_contract_post_capture_revalidation_failed:"
                f"{type(exc).__name__}:{exc}"
            )
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
    calibration_metadata = {
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
    if followup["mode"] in {"broadband", "fullband-v5"}:
        calibration_metadata = {
            "mode": "verified_existing",
            "path": followup["level_evidence"]["path"],
            "sha256": followup["level_evidence"]["sha256"],
        }
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
            "calibration_evidence": calibration_metadata,
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
    if followup["mode"] in {"broadband", "fullband-v5"}:
        metadata["hardware"] = dict(followup["hardware"])
        metadata["followup_contract"] = dict(followup)
    if followup["mode"] == "fullband-v5":
        metadata["fullband_v5_operator_confirmations"] = {
            "speaker_output": args.confirm_speaker is True,
            "user_present": args.confirm_user_present is True,
            "volume_minimum": args.confirm_volume_minimum is True,
            "routing_and_geometry": args.confirm_routing_and_geometry is True,
            "same_amplifier_setting": args.confirm_same_amplifier_setting is True,
        }
    try:
        bootstrap_paths = write_bootstrap_meter_raw_atomic(
            bootstrap_session / "meter_raw.npz",
            repository_root=REPO_ROOT,
            metadata=metadata,
            submitted_output_pcm_int16=submitted_output_pcm,
            input_raw_int32=input_raw,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
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
    if reasons:
        print(
            "[실패] 레벨 미터 캡처 계약 위반: " + ", ".join(reasons),
            file=sys.stderr,
        )
        return 1
    assert final is not None
    print(f"\n마지막 구간 중앙값 {final:+.4f} dBFS — {verdict(final)}")
    if target_pass:
        assert bootstrap_paths is not None
        raw_rel = bootstrap_paths["raw"].relative_to(REPO_ROOT)
        if followup["mode"] in {"broadband", "fullband-v5"}:
            try:
                revalidate_followup_contract(followup)
            except (
                FileNotFoundError,
                FileExistsError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                print(
                    "[실패] meter 종료 뒤 광대역 followup 재검증 실패. "
                    f"명령을 출력하지 않습니다: {exc}",
                    file=sys.stderr,
                )
                return 1
            if followup["mode"] == "fullband-v5":
                command = fullband_v5_followup_command(raw_rel, contract=followup)
                print(
                    "[BLOCKED] blocked_until_v5_live_adapter_implementation. "
                    "아래 명령은 exact capture-only handoff이며 현재 adapter는 exit 2로 닫힙니다. "
                    "실행 권위나 준비 완료를 주장하지 않습니다:\n" + command
                )
            else:
                command = broadband_followup_command(
                    raw_rel,
                    plan=followup["plan"]["path"],
                    raw_session_dir=followup["raw_session_dir"],
                    level_evidence=followup["level_evidence"]["path"],
                    hardware=followup["hardware"]["path"],
                )
                print(
                    "영구 evidence는 calibration 근거일 뿐 현재 노브 증거가 아닙니다. "
                    "같은 노브를 유지하고 10분 안에 아래 exact 광대역 P/S만 실행하세요:\n"
                    + command
                )
        else:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument(
        "--seconds",
        type=float,
        default=OFFICIAL_MEASUREMENT_LEVEL.meter_seconds,
        help="official에서는 정확히 20초로 고정",
    )
    parser.add_argument(
        "--level-evidence",
        default=str(DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH),
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
        default="results/calibration_interleaved/level_bootstrap",
        help="bootstrap meter immutable raw의 새 session 상위 경로(results/ 아래만 허용)",
    )
    parser.add_argument(
        "--followup-mode",
        choices=("strict", "broadband", "fullband-v5"),
        default="strict",
        help="PASS 뒤 출력할 P/S 명령. 기본 strict 동작은 기존과 동일",
    )
    parser.add_argument(
        "--broadband-plan",
        type=Path,
        help="broadband followup에 사용할 exact saved signal plan JSON",
    )
    parser.add_argument(
        "--broadband-raw-session-dir",
        type=Path,
        help="broadband followup의 새 no-replace results/ raw session 경로",
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
        help="fullband-v5의 routing/geometry가 검토된 상태임을 확인",
    )
    parser.add_argument(
        "--confirm-same-amplifier-setting",
        action="store_true",
        help="fullband-v5 handoff까지 같은 amplifier setting을 유지함을 확인",
    )
    args = parser.parse_args(argv)

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
    # live preflight/raw/PortAudio보다 operator safety confirmation을 먼저 닫는다.
    # 특히 fullband-v5에서 누락된 confirmation이 있으면 tracked raw나 host 장치
    # 상태를 읽지도 않고 즉시 중단해야 한다.
    if not (args.confirm_user_present and args.confirm_volume_minimum):
        print(
            "[중단] 모든 live meter는 사용자 입회와 시작 전 볼륨 최저 확인이 "
            "모두 필요합니다: --confirm-user-present --confirm-volume-minimum",
            file=sys.stderr,
        )
        return 2
    if args.followup_mode == "fullband-v5" and not (
        args.confirm_routing_and_geometry and args.confirm_same_amplifier_setting
    ):
        print(
            "[중단] fullband-v5 meter는 5개 confirmation이 모두 필요합니다: "
            "--confirm-routing-and-geometry --confirm-same-amplifier-setting",
            file=sys.stderr,
        )
        return 2
    try:
        args._validated_followup_contract = validate_followup_contract(args)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 후속 측정 preflight 실패: {exc}", file=sys.stderr)
        return 2
    return measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
