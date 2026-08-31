#!/usr/bin/env python3
"""덕트 실측 수집 — 소음 재생(ch0) + 레퍼런스/에러 마이크 동시 녹음 (ANC OFF).

  .venv/bin/python scripts/data/record_duct.py --program tone --frequency 300 --seconds 15 --dry-run
  # dry-run PASS 뒤 실제 출력은 세 확인을 모두 명시한다.
  .venv/bin/python scripts/data/record_duct.py --program tone --frequency 300 --seconds 15 \
    --confirm-user-present --confirm-volume-minimum --confirm-routing-and-geometry

저장: data/recorded/<타임스탬프_프로그램>/
      {mics.wav(2ch PCM_32), source.wav(원본 provenance), source_aligned.wav(학습용), session.json}
시작 시 레퍼런스 마이크(ch1) 자가진단 — 과거 무신호 이력 대응 (docs/02).
상쇄 스피커(ch1 출력)는 전 구간 무음을 유지한다.

시간축 규약 (2026-08-05 결함 2 수정)
-----------------------------------
이 스크립트는 예전에 ``cursor["in"] == cursor["out"]`` 이라는 이유만으로
``source[t]`` 와 ``mics[t]`` 가 **같은 물리 시각**이라고 단언했다. 그것은 DAC→ADC
왕복지연이 0 이라는 가정이고, 실제로는 0 도 아니고 상수도 아니다 — AB13X USB DAC 이
UAC1 ADAPTIVE(full speed, 피드백 엔드포인트 없음)라 장치 PLL 이 주기 4~5 초, 진폭
259~407 샘플로 헌팅한다. 그 결과 80 세션 전부가 ``coh²(source→ERR)=0.02~0.13`` 인
채로 QA 를 통과했다.

지금은 **단언하지 않고 측정한다**:

* ``source.wav`` 는 재생한 그대로 남긴다 (원본 provenance — 절대 덮어쓰지 않는다).
* REF 마이크(ch1)는 ERR 과 **같은 ADC** 를 타므로 재생 신호의 시간축 증인이다.
  이 증인으로 시변 지연 L(t) 를 추정해 ``source_aligned.wav`` 를 만든다.
* 검증은 추정에 쓰지 않은 ERR 채널로 한다(홀드아웃). 기준 미달 raw는 canonical 세션으로
  발행하지 않고 ``--failed-root`` 아래 no-replace 진단 증거로 보존한다.
* PortAudio 콜백 타임스탬프는 provenance 로만 남긴다 — 실측상 ``dac−adc`` 가
  0.010/0.020 s 두 값 사이를 16 샘플 단위로만 튀어 실제 ±130 샘플 변조를 전혀
  보여주지 않는다. **진단용이지 수정 수단이 아니다.**
"""

import argparse
import contextlib
import csv
import ctypes
import datetime
import errno
import hashlib
import io
import json
import math
import os
import secrets
import signal
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import (                              # noqa: E402
    MAX_RECORDING_OUTPUT_PEAK,
    MIN_PROBE_DBFS,
    pcm_int32_to_float32,
    resolve_alsa_portaudio_device,
    rms_dbfs,
)
from deep_anc.config import REPO_ROOT, load_yaml             # noqa: E402
from deep_anc.data.manifest import (                         # noqa: E402
    VALID_SPLITS,
    validate_group_id,
    validate_session_id,
    validate_source_family,
)
from deep_anc.data.recorded_qa import (                      # noqa: E402
    CAPTURE_MIN_LOW_BAND_COHERENCE,
    CAPTURE_MIN_RAW_VALID_WINDOW_RATIO,
    DEFAULT_RECORDED_CAPTURE_GATE,
    RecordedCaptureGateContract,
    RecordedCaptureGateResult,
    evaluate_recorded_capture_gate,
)
from deep_anc.data.recording_level_campaign import (        # noqa: E402
    RECORDING_LEVEL_PLAYBACK_AMPLITUDE,
    RECORDING_LEVEL_SAMPLE_RATE,
    RECORDING_LEVEL_SESSION_FRAMES,
    RecordingLevelCampaignError,
    build_recording_level_session_binding,
    build_recording_level_source_gain_session_binding,
    rendered_source_level_evidence,
    rendered_source_level_evidence_v2,
    validate_recording_level_campaign,
    validate_recording_level_session_binding,
    validate_recording_level_source_gain_session_binding,
)
from deep_anc.data.recording_source_gain import (            # noqa: E402
    RecordingSourceGainError,
    build_recording_source_gain_session_binding,
    validate_recording_source_gain_plan,
    validate_recording_source_gain_session_binding,
)
from deep_anc.data.recorded_generation import (               # noqa: E402
    ADDITIONS_ROOT,
    SOURCE_PLAN_ROOT,
)
from deep_anc.data.holdout_contract import (                 # noqa: E402
    read_regular_file_snapshot,
)
from deep_anc.data.repository_fd import (                     # noqa: E402
    RepositoryFileGuard,
    repository_execution_identity,
)
from deep_anc.data.timeline import (                         # noqa: E402
    TimelineSettings,
    align_source_to_adc,
)
from deep_anc.audio_io import (                             # noqa: E402
    MAX_PROBE_CLIP_RATIO,
    input_rail_gate,
)
from deep_anc.dsp.measurement_level import (                 # noqa: E402
    assert_live_pcm_clock_preconditions,
    collect_alsa_physical_fingerprint,
)
from deep_anc.realtime.noise_gen import (                    # noqa: E402
    NoiseProgram,
    render_recording_file_window,
)

# 게이트 하한. CLI 는 이 값 **이상**만 받는다(강화 전용).
# 0.90 의 근거: 선형 Wiener 하한 10·log10(1−coh²) 로 −10 dB. 재정렬이 성공한 실측
# 세션은 0.87~0.96 이고 붕괴 세션은 0.02~0.13 이라 그 사이 골짜기가 넓다.
DEFAULT_MIN_TIMELINE_COHERENCE = CAPTURE_MIN_LOW_BAND_COHERENCE
DEFAULT_MIN_VALID_WINDOW_RATIO = CAPTURE_MIN_RAW_VALID_WINDOW_RATIO
PROGRAM_PEAK_FACTORS = {
    "white": 4.0,
    "band": 4.0,
    "multitone": 1.5,
}

# 부모 오케스트레이터가 사람이 읽는 마지막 stdout 줄을 실패 원인으로 오인하지 않도록,
# durable ``failure.json``이 발행된 **뒤에만** 이 한 줄짜리 기계 판독 포인터를 stderr로
# 내보낸다. 기존 한국어 안내 줄은 현장 운용/로그 하위 호환성을 위해 그대로 유지한다.
FAILURE_RECEIPT_MARKER = "DEEP_ANC_RECORD_DUCT_FAILURE_JSON="
SCRIPT_RELATIVE_PATH = "scripts/data/record_duct.py"


def _prepare_file_source_timeline(
    program: NoiseProgram,
    *,
    settle_frames: int,
    keep_frames: int,
    sample_rate: int,
) -> np.ndarray:
    """settle zero prefix 뒤에 planned file window를 exact 배치한다."""

    settle_frames = int(settle_frames)
    keep_frames = int(keep_frames)
    if settle_frames < 0 or keep_frames <= 0:
        raise ValueError("settle_frames는 0 이상, keep_frames는 양수여야 합니다")
    timeline = np.zeros(settle_frames + keep_frames, dtype=np.float32)
    timeline[settle_frames:] = render_recording_file_window(
        program,
        keep_frames,
        sample_rate=sample_rate,
    )
    return timeline


# 자가진단 상한. QA 의 max_clip_ratio(0.005)와 같은 자리에서 판정하되, 재생 전에 본다.
# 레일 게이트와 임계는 src/deep_anc/audio_io.py 가 단일 출처다.
# 여기 두면 다른 도구가 sys.path 를 조작해 스크립트에서 import 해야 하고,
# 실제로 그 불편함이 "새 도구는 그냥 안 쓴다" 로 이어졌다(2026-08-06).


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_recording_level_authority(
    args: argparse.Namespace,
    *,
    sample_rate: int,
    now_utc: datetime.datetime,
    source_gain_binding: dict | None = None,
) -> dict | None:
    """Canonical additions의 fresh meter campaign을 audio open 전에 검증한다."""

    receipt = args.recording_level_campaign
    expected_sha = args.recording_level_campaign_sha256
    supplied = (receipt is not None, expected_sha is not None)
    # 무음 dry-run은 meter/campaign 발행보다 먼저 수행할 수 있어야 한다. hidden
    # canonical flag가 있어도 live 전환 시에만 fresh campaign을 필수화한다.
    required = bool(args.require_recording_level_campaign and not args.dry_run)
    if not any(supplied):
        if required:
            raise RecordingLevelCampaignError(
                "canonical 수집에는 --recording-level-campaign과 외부 "
                "--recording-level-campaign-sha256이 필요합니다"
            )
        if args.confirm_same_amplifier_setting:
            raise RecordingLevelCampaignError(
                "campaign receipt 없이 --confirm-same-amplifier-setting만 지정할 수 없습니다"
            )
        return None
    if not all(supplied):
        raise RecordingLevelCampaignError(
            "--recording-level-campaign과 --recording-level-campaign-sha256은 함께 필요합니다"
        )
    if args.confirm_same_amplifier_setting is not True:
        raise RecordingLevelCampaignError(
            "recording level campaign 사용에는 --confirm-same-amplifier-setting이 필요합니다"
        )
    if args.program != "file":
        raise RecordingLevelCampaignError(
            "recording level campaign은 planned file session에만 사용할 수 있습니다"
        )
    frames = float(args.seconds) * float(sample_rate)
    if (
        sample_rate != RECORDING_LEVEL_SAMPLE_RATE
        or not np.isclose(
            frames,
            float(RECORDING_LEVEL_SESSION_FRAMES),
            rtol=0.0,
            atol=1e-9,
        )
        or (
            source_gain_binding is None
            and not np.isclose(
                float(args.amplitude),
                RECORDING_LEVEL_PLAYBACK_AMPLITUDE,
                rtol=0.0,
                atol=1e-12,
            )
        )
    ):
        raise RecordingLevelCampaignError(
            "recording level campaign session은 exact 48kHz/15초이며, legacy는 amplitude "
            "0.06, 현행 canonical은 verified receipt-bound source-gain row amplitude여야 합니다"
        )
    summary = validate_recording_level_campaign(
        repo_root=REPO_ROOT,
        campaign_receipt=str(receipt),
        expected_sha256=str(expected_sha).lower(),
        now_utc=now_utc,
        require_fresh=True,
    )
    payload = summary.get("payload")
    hardware = payload.get("hardware") if isinstance(payload, dict) else None
    config_ref = hardware.get("config") if isinstance(hardware, dict) else None
    config_path = config_ref.get("path") if isinstance(config_ref, dict) else None
    if (
        not isinstance(config_path, str)
        or Path(os.path.abspath(_repo_path(args.hardware)))
        != Path(os.path.abspath(_repo_path(config_path)))
    ):
        raise RecordingLevelCampaignError(
            "record_duct --hardware가 fresh campaign에 결속된 hardware config와 다릅니다"
        )
    if source_gain_binding is not None:
        gain_hardware = source_gain_binding.get("gain_linearity_hardware")
        campaign_hardware = payload.get("hardware") if isinstance(payload, dict) else None
        if (
            not isinstance(gain_hardware, dict)
            or not isinstance(campaign_hardware, dict)
            or config_ref.get("sha256") != gain_hardware.get("sha256")
            or config_ref.get("path") != gain_hardware.get("path")
            or campaign_hardware.get("physical_fingerprint_sha256")
            != gain_hardware.get("physical_fingerprint_sha256")
        ):
            raise RecordingLevelCampaignError(
                "fresh meter campaign hardware가 gain-linearity receipt hardware와 다릅니다"
            )
    return summary


def _validate_source_gain_authority(
    args: argparse.Namespace,
    *,
    collection_plan: dict,
    require_clean_execution: bool,
    authority_required: bool = False,
) -> dict | None:
    """현행 receipt-bound source-gain plan/row/amplitude/commit을 open 전에 결속한다."""

    plan_path = args.source_gain_plan
    expected_sha = args.source_gain_plan_sha256
    required = bool(
        (args.require_source_gain_plan or authority_required) and not args.dry_run
    )
    if plan_path is None and expected_sha is None:
        if required:
            raise RecordingSourceGainError(
                "canonical live에는 현행 --source-gain-plan과 외부 SHA가 필요합니다"
            )
        return None
    if plan_path is None or expected_sha is None:
        raise RecordingSourceGainError(
            "--source-gain-plan과 --source-gain-plan-sha256은 함께 필요합니다"
        )
    if collection_plan.get("status") != "exact":
        raise RecordingSourceGainError("source-gain plan은 exact collection row에만 쓸 수 있습니다")
    source_list_path = _repo_path(str(collection_plan.get("source_list", "")))
    source_file_path = _repo_path(str(args.file or ""))
    if (
        _sha256(source_list_path) != collection_plan.get("source_list_sha256")
        or _sha256(source_file_path) != collection_plan.get("source_file_sha256")
    ):
        raise RecordingSourceGainError(
            "current source-list/source file bytes가 initial collection snapshot과 다릅니다"
        )
    summary = validate_recording_source_gain_plan(
        repo_root=REPO_ROOT,
        plan_path=str(plan_path),
        expected_sha256=str(expected_sha).lower(),
    )
    payload = summary.get("payload")
    source_ref = payload.get("source_plan") if isinstance(payload, dict) else None
    if (
        summary.get("canonical_live_eligible") is not True
        or not isinstance(source_ref, dict)
        or source_ref.get("sha256") != collection_plan.get("source_list_sha256")
    ):
        raise RecordingSourceGainError(
            "현행 source-gain live eligibility/source-list SHA 결속 실패"
        )
    if require_clean_execution:
        execution = repository_execution_identity(REPO_ROOT, SCRIPT_RELATIVE_PATH)
        expected_commit = str(execution["repository_commit"])
    else:
        contract = payload.get("contract") if isinstance(payload, dict) else None
        expected_commit = (
            str(contract.get("gain_linearity_source_commit", ""))
            if isinstance(contract, dict)
            else ""
        )
        execution = None
    binding = build_recording_source_gain_session_binding(
        summary,
        source_row_number=int(collection_plan["source_row_number"]),
        expected_source_commit=expected_commit,
    )
    gain_hardware = binding.get("gain_linearity_hardware")
    if not isinstance(gain_hardware, dict):
        raise RecordingSourceGainError("source-gain binding에 gain-linearity hardware가 없습니다")
    current_hardware_path = _repo_path(args.hardware)
    expected_hardware_path = _repo_path(str(gain_hardware.get("path", "")))
    current_config = load_yaml(current_hardware_path)
    current_fingerprint = collect_alsa_physical_fingerprint(current_config)
    if (
        Path(os.path.abspath(current_hardware_path))
        != Path(os.path.abspath(expected_hardware_path))
        or _sha256(current_hardware_path) != gain_hardware.get("sha256")
        or current_fingerprint != gain_hardware.get("physical_fingerprint")
    ):
        raise RecordingSourceGainError(
            "current hardware YAML/physical fingerprint가 gain-linearity receipt와 다릅니다"
        )
    if binding["source_file"].get("sha256") != collection_plan.get(
        "source_file_sha256"
    ):
        raise RecordingSourceGainError("source-gain row source bytes가 collection과 다릅니다")
    millionths_float = float(args.amplitude) * 1_000_000.0
    millionths = int(round(millionths_float))
    if (
        not math.isclose(millionths_float, float(millionths), rel_tol=0.0, abs_tol=1e-9)
        or millionths != binding["amplitude_millionths"]
        or not math.isclose(
            float(args.amplitude), float(binding["amplitude"]), rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise RecordingSourceGainError(
            "CLI amplitude가 source-gain integer-millionths selected row와 다릅니다"
        )
    return {"summary": summary, "binding": binding, "execution": execution}


def _is_within_repo_subtree(value: str | Path, relative_root: str) -> bool:
    candidate = Path(os.path.abspath(_repo_path(value)))
    root = Path(os.path.abspath(_repo_path(relative_root)))
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _requires_v2_source_gain_authority(
    args: argparse.Namespace, collection_plan: dict
) -> bool:
    """Canonical additions 경로는 hidden flag가 없어도 현행 gain authority를 요구한다."""

    source_list = collection_plan.get("source_list")
    return bool(
        args.require_source_gain_plan
        or args.require_recording_level_campaign
        or (
            collection_plan.get("status") == "exact"
            and isinstance(source_list, str)
            and _is_within_repo_subtree(source_list, SOURCE_PLAN_ROOT)
        )
        or _is_within_repo_subtree(args.out_root, ADDITIONS_ROOT)
    )


def _uses_managed_recorded_additions_namespace(
    args: argparse.Namespace, collection_plan: dict
) -> bool:
    """공식 additions path를 generic diagnostic으로 위장한 live 우회를 막는다."""

    source_list = collection_plan.get("source_list")
    return bool(
        isinstance(source_list, str)
        and _is_within_repo_subtree(source_list, SOURCE_PLAN_ROOT)
        or _is_within_repo_subtree(args.out_root, ADDITIONS_ROOT)
    )


def _guard_relative_path(value: str | Path, *, label: str) -> str:
    absolute = Path(os.path.abspath(_repo_path(value)))
    root = Path(os.path.abspath(REPO_ROOT))
    try:
        relative = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise RecordingSourceGainError(
            f"{label}는 canonical live에서 저장소 내부 regular file이어야 합니다"
        ) from exc
    return relative


def _open_live_authority_guards(
    stack: contextlib.ExitStack,
    *,
    args: argparse.Namespace,
    collection_plan: dict,
    source_gain_binding: dict,
) -> list[RepositoryFileGuard]:
    """Stream 시작 전부터 종료까지 모든 source-gain authority inode/bytes를 고정한다."""

    receipt = source_gain_binding.get("gain_linearity_receipt")
    paths: list[tuple[str | Path, str]] = [
        (args.hardware, "hardware config"),
        (args.file, "rendered source file"),
        (collection_plan["source_list"], "source list"),
        (args.source_gain_plan, "source gain plan"),
        (args.recording_level_campaign, "recording level campaign"),
    ]
    if isinstance(receipt, dict) and isinstance(receipt.get("path"), str):
        paths.append((receipt["path"], "gain-linearity receipt"))
    guards: list[RepositoryFileGuard] = []
    seen: set[str] = set()
    for value, label in paths:
        relative = _guard_relative_path(value, label=label)
        if relative in seen:
            continue
        seen.add(relative)
        guards.append(
            stack.enter_context(RepositoryFileGuard(REPO_ROOT, relative, label=label))
        )
    return guards


def _construct_stream_after_authority_check(
    sd,
    *,
    pre_open_check,
    stream_kwargs: dict,
):
    """Pa_OpenStream 직전과 constructor 직후(아직 start 전)에 authority를 검사한다."""

    pre_open_check()
    stream = sd.Stream(**stream_kwargs)
    try:
        # sounddevice constructor가 PortAudio stream을 열 수 있지만 context enter 전이라
        # callback/start는 아직 아니다. constructor 사이 mutation은 여기서 잡고 닫는다.
        pre_open_check()
    except BaseException:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
        raise
    return stream


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _artifact_evidence(paths: list[Path], *, base: Path) -> list[dict]:
    evidence: list[dict] = []
    for path in paths:
        _fsync_file(path)
        stat = path.stat()
        evidence.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size_bytes": int(stat.st_size),
                "sha256": _sha256(path),
            }
        )
    return evidence


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Linux renameat2(RENAME_NOREPLACE)로 디렉터리를 원자 발행한다."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE)를 지원하지 않아 안전 발행을 거부합니다")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, "발행 대상이 이미 존재합니다", destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _announce_failure_receipt(
    *, failure_dir: Path, stage: str, reason: str
) -> None:
    """durable failure evidence의 기계 판독 포인터를 한 줄로 알린다."""

    receipt = failure_dir / "failure.json"
    marker = {
        "schema_version": 1,
        "failure_stage": str(stage),
        "failure_reason": str(reason),
        "failure_artifact": os.path.abspath(failure_dir),
        "failure_receipt": os.path.abspath(receipt),
    }
    print(
        FAILURE_RECEIPT_MARKER
        + json.dumps(marker, ensure_ascii=False, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


def _seal_staging_failure(
    *, staging_dir: Path, failed_root: Path, stage: str, reason: str, metadata: dict
) -> Path:
    """results staging 전체를 failure evidence로 봉인해 원자 이동한다."""

    artifact_paths = sorted(
        path for path in staging_dir.iterdir() if path.is_file() and path.name != "failure.json"
    )
    artifacts = _artifact_evidence(artifact_paths, base=staging_dir)
    _write_json_exclusive(
        staging_dir / "failure.json",
        {
            **metadata,
            "schema_version": 1,
            "status": "failed_capture",
            "failure_stage": stage,
            "failure_reason": reason,
            "raw_available": any(item["path"].endswith(".wav") for item in artifacts),
            "artifacts": artifacts,
        },
    )
    _fsync_directory(staging_dir)
    failed_root.mkdir(parents=True, exist_ok=True)
    final = failed_root / f"{staging_dir.name.removeprefix('.staging_')}_{stage}"
    _atomic_rename_noreplace(staging_dir, final)
    _fsync_directory(staging_dir.parent)
    _fsync_directory(failed_root)
    print(f"[보존] 실패 staging 전체: {final}", file=sys.stderr)
    _announce_failure_receipt(failure_dir=final, stage=stage, reason=reason)
    return final


def _validate_collection_plan(args, parser: argparse.ArgumentParser) -> dict:
    """CLI provenance가 지정됐다면 source-list의 exact byte/row와 교차 검증한다."""

    names = (
        "source_list",
        "source_list_sha256",
        "source_row_number",
        "lineage_key",
        "preassigned_split",
    )
    supplied = [getattr(args, name) is not None for name in names]
    if not any(supplied):
        return {
            "status": "unbound_diagnostic",
            "reason": "source-list SHA/row/lineage/preassigned split이 지정되지 않음",
        }
    if not all(supplied):
        missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None]
        parser.error("collection plan 일부만 지정됨: " + ", ".join(missing))

    source_list = _repo_path(args.source_list)
    if not source_list.is_file():
        parser.error(f"--source-list 파일 없음: {source_list}")
    expected_sha = str(args.source_list_sha256).lower()
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        parser.error("--source-list-sha256은 64자리 소문자 hex여야 합니다")
    raw = source_list.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        parser.error(
            f"source-list SHA 불일치: expected={expected_sha}, actual={actual_sha}, path={source_list}"
        )
    if args.source_row_number < 2:
        parser.error("--source-row-number는 CSV header 다음인 2 이상이어야 합니다")
    try:
        lineage_key = validate_group_id(args.lineage_key)
    except ValueError as exc:
        parser.error(str(exc))
    if args.preassigned_split not in VALID_SPLITS:
        parser.error(f"--preassigned-split은 {VALID_SPLITS} 중 하나여야 합니다")

    try:
        reader = csv.DictReader(raw.decode("utf-8").splitlines())
    except UnicodeDecodeError as exc:
        parser.error(f"source-list UTF-8 오류: {exc}")
    selected = None
    for logical_row, row in enumerate(reader, start=2):
        if logical_row == args.source_row_number:
            selected = row
            break
    if selected is None:
        parser.error(f"source-list에 {args.source_row_number}행이 없습니다")

    row_lineage = selected.get("lineage_key") or selected.get("group_id")
    comparisons = {
        "path": (selected.get("path"), args.file),
        "source_family": (selected.get("source_family"), args.source_family),
        "group_id": (selected.get("group_id"), args.group_id),
        "lineage_key": (row_lineage, lineage_key),
    }
    for field, (planned, observed) in comparisons.items():
        if planned != observed:
            parser.error(
                f"source-list {args.source_row_number}행 {field} 불일치: "
                f"planned={planned!r}, cli={observed!r}"
            )
    row_split = (selected.get("split") or "").strip()
    if row_split and row_split != args.preassigned_split:
        parser.error(
            f"source-list split 불일치: planned={row_split!r}, cli={args.preassigned_split!r}"
        )
    try:
        row_seconds = float(selected.get("seconds", ""))
    except ValueError:
        parser.error(f"source-list {args.source_row_number}행 seconds가 숫자가 아닙니다")
    if not np.isclose(row_seconds, args.seconds, rtol=0.0, atol=1e-9):
        parser.error(
            f"source-list seconds 불일치: planned={row_seconds}, cli={args.seconds}"
        )
    try:
        row_start = float(selected.get("start_seconds") or 0.0)
    except ValueError:
        parser.error(f"source-list {args.source_row_number}행 start_seconds가 숫자가 아닙니다")
    if not np.isclose(row_start, args.file_start_seconds, rtol=0.0, atol=1e-9):
        parser.error(
            f"source-list start_seconds 불일치: planned={row_start}, "
            f"cli={args.file_start_seconds}"
        )

    source_file = _repo_path(args.file)
    if not source_file.is_file():
        parser.error(f"계획된 source 파일 없음: {source_file}")
    source_digest = _sha256(source_file)
    declared_source_digest = (selected.get("source_file_sha256") or "").strip()
    if declared_source_digest:
        if (
            len(declared_source_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in declared_source_digest)
        ):
            parser.error(
                f"source-list {args.source_row_number}행 source_file_sha256은 "
                "64자리 소문자 hex여야 합니다"
            )
        if source_digest != declared_source_digest:
            parser.error(
                f"source-list 원본 SHA 불일치: planned={declared_source_digest}, "
                f"actual={source_digest}"
            )
    return {
        "status": "exact",
        "source_list": str(source_list),
        "source_list_sha256": actual_sha,
        "source_row_number": int(args.source_row_number),
        "lineage_key": lineage_key,
        "preassigned_split": args.preassigned_split,
        "split_source": "csv" if row_split else "explicit_cli",
        "source_file_sha256": source_digest,
        "start_seconds": float(args.file_start_seconds),
    }


def _preserve_failed_capture(
    *,
    failed_root: Path,
    stage: str,
    reason: str,
    sample_rate: int,
    metadata: dict,
    mics_raw: np.ndarray | None = None,
    source_raw: np.ndarray | None = None,
) -> Path:
    """실패 증거를 새 디렉터리에 no-replace로 보존한다.

    출력 stream이 닫힌 뒤 메모리에 존재하는 raw만 보존할 수 있다. 프로세스 강제 종료,
    전원 손실 또는 PortAudio가 콜백을 돌려주기 전의 실패에는 raw가 없으며 metadata에
    ``raw_available=false``가 남는다.
    """

    failed_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    token = secrets.token_hex(4)
    staging_dir = failed_root / f".staging_{stamp}_{token}"
    staging_dir.mkdir(exist_ok=False)
    raw_paths: list[Path] = []
    if mics_raw is not None and np.asarray(mics_raw).size:
        path = staging_dir / "mics_raw.wav"
        sf.write(path, np.asarray(mics_raw), sample_rate, subtype="PCM_32")
        raw_paths.append(path)
    if source_raw is not None and np.asarray(source_raw).size:
        path = staging_dir / "source_raw.wav"
        sf.write(path, np.asarray(source_raw), sample_rate, subtype="FLOAT")
        raw_paths.append(path)
    artifacts = _artifact_evidence(raw_paths, base=staging_dir)
    payload = {
        **metadata,
        "schema_version": 1,
        "status": "failed_capture",
        "failure_stage": stage,
        "failure_reason": reason,
        "raw_available": bool(raw_paths),
        "sample_rate": int(sample_rate),
        "artifacts": artifacts,
    }
    _write_json_exclusive(staging_dir / "failure.json", payload)
    _fsync_directory(staging_dir)
    failure_dir = failed_root / f"{stamp}_{stage}_{token}"
    _atomic_rename_noreplace(staging_dir, failure_dir)
    _fsync_directory(failed_root)
    print(f"[보존] 실패 raw/metadata: {failure_dir}", file=sys.stderr)
    _announce_failure_receipt(failure_dir=failure_dir, stage=stage, reason=reason)
    return failure_dir


def _publish_session(
    *,
    out_root: Path,
    staging_root: Path,
    failed_root: Path,
    session_name: str,
    sample_rate: int,
    mics: np.ndarray,
    source: np.ndarray,
    aligned: np.ndarray | None,
    metadata: dict,
) -> Path:
    """results staging을 durable하게 만든 뒤 active tree에 원자 발행한다."""

    staging_root.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(4)
    staging_dir = staging_root / f".staging_{session_name}_{token}"
    staging_dir.mkdir(exist_ok=False)
    _fsync_directory(staging_root)
    try:
        mics_path = staging_dir / "mics.wav"
        source_path = staging_dir / "source.wav"
        sf.write(mics_path, mics, sample_rate, subtype="PCM_32")
        sf.write(source_path, source, sample_rate, subtype="FLOAT")
        artifact_paths = [mics_path, source_path]
        if aligned is not None:
            aligned_path = staging_dir / "source_aligned.wav"
            sf.write(aligned_path, aligned, sample_rate, subtype="FLOAT")
            artifact_paths.append(aligned_path)
        artifacts = _artifact_evidence(artifact_paths, base=staging_dir)
        session_meta = {**metadata, "artifacts": artifacts}
        _write_json_exclusive(staging_dir / "session.json", session_meta)
        _fsync_directory(staging_dir)

        out_root.mkdir(parents=True, exist_ok=True)
        destination = out_root / session_name
        _atomic_rename_noreplace(staging_dir, destination)
        _fsync_directory(staging_root)
        _fsync_directory(out_root)
        return destination
    except Exception as exc:
        if staging_dir.exists():
            _seal_staging_failure(
                staging_dir=staging_dir,
                failed_root=failed_root,
                stage="canonical_publish",
                reason=repr(exc),
                metadata={"sample_rate": int(sample_rate), **metadata},
            )
        raise


def timeline_gate_result(
    report, *, min_coherence: float, min_valid_window_ratio: float
) -> RecordedCaptureGateResult:
    """gate: ``recording_timeline_fail_closed`` — active corpus 발행 전 단일 판정.

    저역·고역·REF 대조군·원시 추적률·잔여 지연 안정성을 공용 계약으로 함께 본다.
    호출부가 조건을 다시 조립하지 않도록 판정 결과와 실패 조건을 그대로 반환한다.
    """

    contract = RecordedCaptureGateContract(
        min_low_band_coherence=float(min_coherence),
        min_high_band_coherence=DEFAULT_RECORDED_CAPTURE_GATE.min_high_band_coherence,
        min_ref_err_coherence=DEFAULT_RECORDED_CAPTURE_GATE.min_ref_err_coherence,
        min_raw_valid_window_ratio=float(min_valid_window_ratio),
        max_residual_robust_std_samples=(
            DEFAULT_RECORDED_CAPTURE_GATE.max_residual_robust_std_samples
        ),
        max_residual_p95_p5_samples=(
            DEFAULT_RECORDED_CAPTURE_GATE.max_residual_p95_p5_samples
        ),
    )
    return evaluate_recorded_capture_gate(report, contract)


def timeline_gate(report, *, min_coherence: float, min_valid_window_ratio: float) -> bool:
    """기존 bool 호출자를 위한 얇은 호환 경로. 판정식은 공용 계약에만 있다."""

    return timeline_gate_result(
        report,
        min_coherence=min_coherence,
        min_valid_window_ratio=min_valid_window_ratio,
    ).ok


def _timeline_metadata_with_capture_gate(
    report, *, min_coherence: float, min_valid_window_ratio: float
) -> tuple[dict, RecordedCaptureGateResult]:
    """판정 metadata를 만들고 실패 시 digital-reference 사용 가능성을 닫는다."""

    result = timeline_gate_result(
        report,
        min_coherence=min_coherence,
        min_valid_window_ratio=min_valid_window_ratio,
    )
    metadata = report.as_metadata()
    metadata["capture_gate"] = result.as_metadata()
    metadata["usable_for_digital_reference"] = result.ok
    return metadata, result


def _summarise_io_timestamps(stamps: np.ndarray, fs: int) -> dict:
    """PortAudio 콜백 타임스탬프 요약 — **provenance 전용, 수정 수단 아님.**

    실측(무음 40초 전이중 프로브, record_duct 와 동일 스트림 설정):
    콜백 7500회 / 프레임 1,920,000 / status 이벤트 **0회**, adc rate +5.0 ppm(잔차
    0.018 ms). 그런데 ``dac − adc`` 는 0.010 s 와 0.020 s 두 값 사이를 16 샘플 단위로만
    튄다. 실제로 일어나고 있는 ±130 샘플 변조를 **전혀 보여주지 않는다** — 이 값들은
    호스트 시계에서 나온 예측치이지 장치가 실제로 소리를 낸 시각이 아니기 때문이다.

    그래서 여기에 남기는 것은 "그때 호스트가 뭐라고 믿었는가" 뿐이고, 정렬은 REF 증인
    재정렬이 담당한다. 이 구분을 흐리면 결함 2 가 그대로 재발한다.
    """

    if stamps.size == 0:
        return {"callbacks": 0, "note": "타임스탬프 없음"}
    adc = stamps[:, 0]
    dac = stamps[:, 1]
    frames = stamps[:, 2]
    elapsed = np.cumsum(np.concatenate([[0.0], frames[:-1]])) / float(fs)
    summary: dict = {
        "callbacks": int(stamps.shape[0]),
        "frames_total": int(frames.sum()),
        "unique_frames": sorted({int(value) for value in frames}),
        "dac_minus_adc_s": {
            "min": float(np.min(dac - adc)),
            "median": float(np.median(dac - adc)),
            "max": float(np.max(dac - adc)),
            "unique_values": int(np.unique(np.round(dac - adc, 6)).size),
        },
        "note": (
            "provenance 전용. 실측상 dac−adc 는 16샘플 단위 계단이라 실제 DAC 헌팅"
            "(4~5초 주기, 259~407샘플)을 보여주지 못한다. 정렬 복원에 쓰지 말 것."
        ),
    }
    if stamps.shape[0] >= 8:
        for name, series in (("adc", adc), ("dac", dac)):
            slope, intercept = np.polyfit(elapsed, series - series[0], 1)
            residual = (series - series[0]) - (slope * elapsed + intercept)
            summary[f"{name}_rate_ppm"] = float((slope - 1.0) * 1.0e6)
            summary[f"{name}_residual_rms_ms"] = float(np.sqrt(np.mean(residual**2)) * 1000.0)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware_jetson.yaml")
    parser.add_argument(
        "--program",
        default="tone",
        choices=["tone", "multitone", "white", "band", "nonlinear", "sweep", "file", "silence"],
    )
    parser.add_argument("--frequency", type=float, default=300.0)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--band", type=float, nargs=2, default=[80.0, 1000.0])
    parser.add_argument("--file", default=None, help="program=file 재생 wav")
    parser.add_argument(
        "--file-start-seconds",
        type=float,
        default=0.0,
        help="program=file의 사전 계획된 시작 offset",
    )
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out-root", default="data/recorded")
    parser.add_argument(
        "--failed-root",
        default="results/recording_failures/record_duct",
        help="실패 raw/metadata를 no-replace로 보존할 루트",
    )
    parser.add_argument(
        "--staging-root",
        default="results/recording_staging/record_duct",
        help="active recorded tree 발행 전 durable staging 루트",
    )
    parser.add_argument(
        "--source-family",
        default=None,
        help=(
            "소스 계열 ID(예: speech/music/environment). 생략 시 program 이름을 사용하며, "
            "program=file은 명시를 권장"
        ),
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help=(
            "분할 누수를 막을 상관 그룹 ID(같은 화자/곡/원본/환경의 반복 세션은 같은 값). "
            "생략 시 현재 세션만의 ID 사용"
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help=(
            "스트림을 연 뒤 무음으로 흘려보내고 버릴 길이. I2S 기동 트랜지언트가 "
            "약 0.5초 지속되므로 여유를 둔 1.0초가 기본값"
        ),
    )
    parser.add_argument("--force", action="store_true", help="ref 마이크 무신호여도 진행")
    parser.add_argument("--ref-check-dbfs", type=float, default=-80.0)
    parser.add_argument(
        "--min-timeline-coherence",
        type=float,
        default=0.90,
        help=(
            "재정렬 후 coh²(source_aligned→ERR, 150-600Hz) 하한. 이 값 미만이면 세션을 "
            "저장하지 않는다(실패-폐쇄). 강화(올리기)만 허용된다"
        ),
    )
    parser.add_argument(
        "--min-valid-window-ratio",
        type=float,
        default=0.90,
        help="지연 추정에 성공한 창의 비율 하한. 강화만 허용된다",
    )
    parser.add_argument("--dry-run", action="store_true", help="파일/오디오를 변경하지 않고 계획만 검증")
    parser.add_argument("--confirm-speaker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--confirm-user-present", action="store_true")
    parser.add_argument("--confirm-volume-minimum", action="store_true")
    parser.add_argument("--confirm-routing-and-geometry", action="store_true")
    parser.add_argument("--source-list", default=None)
    parser.add_argument("--source-list-sha256", default=None)
    parser.add_argument("--source-row-number", type=int, default=None)
    parser.add_argument("--lineage-key", default=None)
    parser.add_argument("--preassigned-split", choices=VALID_SPLITS, default=None)
    parser.add_argument(
        "--recording-level-campaign",
        default=None,
        help="fresh recording-level campaign.json 저장소 상대경로",
    )
    parser.add_argument(
        "--recording-level-campaign-sha256",
        default=None,
        help="campaign.json의 외부 SHA-256 anchor",
    )
    parser.add_argument("--confirm-same-amplifier-setting", action="store_true")
    parser.add_argument(
        "--require-recording-level-campaign",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--source-gain-plan",
        default=None,
        help="현행 source별 physical gain plan 저장소 상대경로",
    )
    parser.add_argument(
        "--source-gain-plan-sha256",
        default=None,
        help="현행 source별 gain plan의 외부 SHA-256 anchor",
    )
    parser.add_argument(
        "--require-source-gain-plan",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 게이트 인자는 **강화 방향으로만** 열려 있다. NaN은 하한 비교가
    # False이며 1 초과값은 녹음 뒤 계약 생성에서야 예외가 나므로, 둘 다 오디오
    # primitive에 닿기 전에 finite 구간과 canonical 하한을 함께 검사한다.
    for option, value, floor in (
        (
            "--min-timeline-coherence",
            args.min_timeline_coherence,
            DEFAULT_MIN_TIMELINE_COHERENCE,
        ),
        (
            "--min-valid-window-ratio",
            args.min_valid_window_ratio,
            DEFAULT_MIN_VALID_WINDOW_RATIO,
        ),
    ):
        if not np.isfinite(value) or not (floor <= value <= 1.0):
            parser.error(
                f"{option} 는 finite이며 {floor} 이상 1.0 이하여야 합니다 "
                f"(받은 값 {value}) — 게이트는 강화만 합니다"
            )
    if args.force:
        parser.error("--force는 입력 무신호/rail 안전 게이트를 우회하므로 더 이상 허용하지 않습니다")
    if args.ref_check_dbfs < MIN_PROBE_DBFS:
        parser.error(
            f"--ref-check-dbfs는 공용 하한 {MIN_PROBE_DBFS:.0f} dBFS 이상으로만 강화할 수 있습니다"
        )
    if not np.isfinite(args.seconds) or args.seconds <= 0.0:
        parser.error("--seconds는 양수 finite여야 합니다")
    if not np.isfinite(args.settle_seconds) or args.settle_seconds < 0.0:
        parser.error("--settle-seconds는 0 이상 finite여야 합니다")
    if not np.isfinite(args.amplitude) or args.amplitude < 0.0:
        parser.error("--amplitude는 0 이상 finite여야 합니다")
    peak_factor = PROGRAM_PEAK_FACTORS.get(args.program, 1.0)
    maximum_peak = float(args.amplitude) * peak_factor
    if args.program != "silence" and args.amplitude <= 0.0:
        parser.error("무음 외 program의 --amplitude는 0보다 커야 합니다")
    if maximum_peak > MAX_RECORDING_OUTPUT_PEAK + 1e-12:
        parser.error(
            f"program={args.program}의 최대 출력 peak 상한은 "
            f"{MAX_RECORDING_OUTPUT_PEAK:.3f}입니다: amplitude {args.amplitude} × "
            f"factor {peak_factor} = {maximum_peak:.3f}"
        )
    if not np.isfinite(args.file_start_seconds) or args.file_start_seconds < 0.0:
        parser.error("--file-start-seconds는 0 이상 finite여야 합니다")

    try:
        source_family = validate_source_family(args.source_family or args.program)
        requested_group_id = (
            validate_group_id(args.group_id) if args.group_id is not None else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    collection_plan = _validate_collection_plan(args, parser)
    if (
        not args.dry_run
        and _uses_managed_recorded_additions_namespace(args, collection_plan)
    ):
        parser.error(
            "관리되는 recorded-additions source-plan/out-root를 record_duct로 직접 live 실행할 수 없습니다. "
            "Stage-1 19-session은 retired이고, Stage-2 47-slot은 actual P/S·source-gain·"
            "raw-first authority 전까지 audio open=0입니다."
        )
    v2_authority_required = _requires_v2_source_gain_authority(
        args, collection_plan
    )
    hw = load_yaml(_repo_path(args.hardware))["audio"]
    fs = int(hw["sample_rate"])
    block = int(hw["block_size"])
    if args.program == "file":
        if not args.file:
            parser.error("--program file에는 --file이 필요합니다")
        source_path = _repo_path(args.file)
        if not source_path.is_file():
            parser.error(f"재생 파일 없음: {source_path}")
        try:
            source_info = sf.info(str(source_path))
        except RuntimeError as exc:
            parser.error(f"재생 파일을 열 수 없습니다: {source_path}: {exc}")
        source_duration = float(source_info.frames) / float(source_info.samplerate)
        if args.file_start_seconds + args.seconds > source_duration + 1e-9:
            parser.error(
                f"계획 구간이 source 길이를 넘습니다: start={args.file_start_seconds}, "
                f"seconds={args.seconds}, duration={source_duration}"
            )

    try:
        source_gain_authority = _validate_source_gain_authority(
            args,
            collection_plan=collection_plan,
            require_clean_execution=not args.dry_run,
            authority_required=v2_authority_required,
        )
    except (OSError, RecordingSourceGainError, RuntimeError, ValueError) as exc:
        parser.error(f"recording source gain 검증 실패: {exc}")
    source_gain_binding = (
        source_gain_authority["binding"]
        if source_gain_authority is not None
        else None
    )
    try:
        recording_level_campaign = _validate_recording_level_authority(
            args,
            sample_rate=fs,
            now_utc=datetime.datetime.now(datetime.timezone.utc),
            source_gain_binding=source_gain_binding,
        )
    except (OSError, RecordingLevelCampaignError, ValueError) as exc:
        parser.error(f"recording level campaign 검증 실패: {exc}")

    audible_seconds = 0.0 if args.program == "silence" else float(args.seconds)
    print(
        f"수집 계획: program={args.program}, audible={audible_seconds:.1f}초, "
        f"output-open={args.seconds + args.settle_seconds:.1f}초, "
        "input-only preflight≈3.5초(PCM/CPU/clock gate 2회 포함)\n"
        f"collection provenance: {collection_plan['status']}"
    )
    if args.dry_run:
        if args.require_recording_level_campaign and recording_level_campaign is None:
            print(
                "[DRY-RUN 안내] live canonical 실행에는 fresh recording-level "
                "campaign path/SHA와 same-amplifier 확인이 필수입니다"
            )
        if v2_authority_required and source_gain_authority is None:
            print(
                "[DRY-RUN 안내] live canonical 실행에는 현행 PASS source-gain plan path/SHA가 "
                "필수입니다"
            )
        print("[DRY-RUN PASS] 파일 생성/수정 및 오디오 장치 open 없음")
        return 0

    confirmations = {
        "--confirm-user-present": args.confirm_user_present,
        "--confirm-volume-minimum": args.confirm_volume_minimum,
        "--confirm-routing-and-geometry": args.confirm_routing_and_geometry,
    }
    missing_confirmations = [name for name, confirmed in confirmations.items() if not confirmed]
    if missing_confirmations:
        print("[중단] 실기 확인 누락: " + ", ".join(missing_confirmations), file=sys.stderr)
        return 2

    import sounddevice as sd

    failure_meta = {
        "requested_program": args.program,
        "requested_file": args.file,
        "requested_seconds": float(args.seconds),
        "source_family": source_family,
        "requested_group_id": requested_group_id,
        "collection_plan": collection_plan,
        "recording_level_campaign": (
            {
                "campaign_id": recording_level_campaign["campaign_id"],
                "receipt_path": recording_level_campaign["receipt_path"],
                "receipt_sha256": recording_level_campaign["receipt_sha256"],
            }
            if recording_level_campaign is not None
            else None
        ),
        "source_gain_binding": source_gain_binding,
        "source_gain_execution": (
            source_gain_authority.get("execution")
            if source_gain_authority is not None
            else None
        ),
    }
    try:
        in_dev = resolve_alsa_portaudio_device(
            hw["input"]["card"], hw["input"]["pcm"], "input", 2
        )
        out_dev = resolve_alsa_portaudio_device(
            hw["output"]["card"], hw["output"]["pcm"], "output", 2
        )
        # /dev/snd 점유, CPU idle, capture clock을 입력 probe보다 먼저 확인한다.
        # 출력 open 직전에도 한 번 더 호출해 probe 동안 생긴 점유 race를 닫는다.
        assert_live_pcm_clock_preconditions(hw)
    except (OSError, RuntimeError, ValueError) as exc:
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="device_preflight",
            reason=str(exc),
            sample_rate=fs,
            metadata=failure_meta,
        )
        print(f"[중단] 장치/PCM/CPU/clock preflight 실패: {exc}", file=sys.stderr)
        return 1

    # ----- 1) 레퍼런스 마이크 자가진단 (2초 무음 캡처) -----
    print("레퍼런스 마이크 점검 중 (2초)...")
    # 앞 1초는 기동 트랜지언트라 버린다. 이걸 포함해서 재면 무신호 마이크도 -42dBFS 로
    # 보여 "살아 있다"고 오판한다 — 이 점검의 목적을 정확히 무력화한다.
    probe_settle = int(1.0 * fs)
    try:
        probe = sd.rec(
            probe_settle + int(2 * fs), samplerate=fs, channels=2, dtype="int32", device=in_dev
        )
        sd.wait()
    except (OSError, RuntimeError, ValueError) as exc:
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="input_probe",
            reason=str(exc),
            sample_rate=fs,
            metadata=failure_meta,
        )
        print(f"[중단] input-only probe 실패: {exc}", file=sys.stderr)
        return 1
    probe_f = pcm_int32_to_float32(probe[probe_settle:])
    err_db = rms_dbfs(probe_f[:, 0])
    ref_db = rms_dbfs(probe_f[:, 1])
    rail_ok, clip_ratio = input_rail_gate(probe_f)
    print(
        f"  ch0(err) {err_db:7.2f} dBFS | ch1(ref) {ref_db:7.2f} dBFS | "
        f"레일 비율 {clip_ratio[0]:.4f}/{clip_ratio[1]:.4f}"
    )
    dead_channels = [
        name for name, level in (("ERR ch0", err_db), ("REF ch1", ref_db))
        if level < args.ref_check_dbfs
    ]
    if dead_channels:
        reason = (
            f"마이크 무신호: {', '.join(dead_channels)}; ERR={err_db:.1f}, REF={ref_db:.1f}, "
            f"threshold={args.ref_check_dbfs:.1f} dBFS"
        )
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="input_level_gate",
            reason=reason,
            sample_rate=fs,
            metadata={**failure_meta, "probe_dbfs": {"err": err_db, "ref": ref_db}},
        )
        print(
            f"[중단] {reason}. 배선 점검(docs/02_hardware_setup.md) 후 다시 dry-run부터 "
            "시작하세요.",
            file=sys.stderr,
        )
        return 1
    # 판정 근거는 input_rail_gate() 의 docstring 에 있다. 재생 **전에** 막는다.
    if not rail_ok:
        reason = (
            f"마이크 입력 rail: ch0={clip_ratio[0]:.4f}, ch1={clip_ratio[1]:.4f}, "
            f"limit={MAX_PROBE_CLIP_RATIO}"
        )
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="input_rail_gate",
            reason=reason,
            sample_rate=fs,
            metadata={
                **failure_meta,
                "probe_dbfs": {"err": err_db, "ref": ref_db},
                "probe_rail_ratio": clip_ratio,
            },
        )
        print(
            f"[중단] 마이크 입력이 풀스케일에 붙어 있습니다 (레일 비율 "
            f"ch0 {clip_ratio[0]:.4f} / ch1 {clip_ratio[1]:.4f} > {MAX_PROBE_CLIP_RATIO}). "
            "입력단 전원/배선을 확인하세요 — 이 상태의 녹음은 클리핑으로 전량 폐기됩니다. "
            "스피커를 울리기 전에 멈춥니다.",
            file=sys.stderr,
        )
        return 1

    # ----- 2) 프로그램 준비 -----
    prog_cfg = {
        "type": args.program,
        "frequency": args.frequency,
        "amplitude": args.amplitude,
        "band": args.band,
        "file": args.file,
        "file_start_seconds": args.file_start_seconds,
    }
    try:
        source_bytes = None
        if args.program == "file":
            source_snapshot = read_regular_file_snapshot(
                _repo_path(args.file),
                root=REPO_ROOT,
                label="record_duct planned source",
                capture_bytes=True,
            )
            assert source_snapshot.data is not None
            source_bytes = source_snapshot.data
            planned_sha = collection_plan.get("source_file_sha256")
            if (
                collection_plan.get("status") == "exact"
                and source_snapshot.sha256 != planned_sha
            ):
                raise ValueError(
                    "재생 직전 source snapshot SHA가 collection plan과 다릅니다: "
                    f"planned={planned_sha}, actual={source_snapshot.sha256}"
                )
            snapshot_info = sf.info(io.BytesIO(source_bytes))
            snapshot_duration = float(snapshot_info.frames) / float(
                snapshot_info.samplerate
            )
            if args.file_start_seconds + args.seconds > snapshot_duration + 1e-9:
                raise ValueError(
                    "재생 직전 source snapshot window가 파일 길이를 넘습니다: "
                    f"start={args.file_start_seconds}, seconds={args.seconds}, "
                    f"duration={snapshot_duration}"
                )
        program = NoiseProgram(prog_cfg, fs, file_bytes=source_bytes)
    except (OSError, RuntimeError, ValueError) as exc:
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="program_prepare",
            reason=str(exc),
            sample_rate=fs,
            metadata={**failure_meta, "program": prog_cfg},
        )
        print(f"[중단] 재생 프로그램 준비 실패: {exc}", file=sys.stderr)
        return 1

    # I2S 입력은 스트림을 연 직후 약 0.5초 동안 큰 기동 트랜지언트를 낸다
    # (실측: 0.0-0.5초 -36.3 dBFS peak 0.062 → 0.5초 이후 -67.4 dBFS peak 0.002).
    # 이 구간을 세션에 남기면 (a) 학습 데이터 앞머리가 잡음이 되고 (b) 세션 QA 의
    # peak/RMS 통계가 트랜지언트를 재게 된다. 무음으로 흘려보내고 잘라낸다.
    # 출력과 입력을 같은 길이만큼 버리므로 정렬은 유지된다.
    settle = int(max(0.0, args.settle_seconds) * fs)
    keep = int(args.seconds * fs)
    total = keep + settle
    source = (
        _prepare_file_source_timeline(
            program,
            settle_frames=settle,
            keep_frames=keep,
            sample_rate=fs,
        )
        if args.program == "file"
        else np.zeros(total, dtype=np.float32)
    )
    recorded = np.zeros((total, 2), dtype=np.float32)
    cursor = {"in": 0, "out": 0}
    xrun_state: dict = {"count": 0, "flags": set()}

    fade = np.linspace(0.0, 1.0, int(0.1 * fs), dtype=np.float32)
    # 재생 진폭 포락선을 미리 만들어 둔다. 예전에는 콜백 안에서 `for k in range(frames)`
    # 로 샘플마다 파이썬 분기를 돌았다 — 블록 256 샘플에 파이썬 루프 256회를 5.33 ms
    # 마감 안에서 하는 것은 그 자체가 xrun 위험이다. 여기서 한 번 만들고 콜백은
    # 슬라이스 곱셈만 한다.
    envelope = np.zeros(total + 8 * block, dtype=np.float32)
    envelope[settle : settle + keep] = 1.0
    ramp = min(fade.size, keep // 2)
    if ramp > 0:
        envelope[settle : settle + ramp] = fade[:ramp]
        envelope[settle + keep - ramp : settle + keep] = fade[:ramp][::-1]

    # file plan의 start_seconds는 audible 15초의 첫 샘플이다. 과거에는 settle
    # 무음 중에도 program.generate()를 호출해 cursor가 1초 전진했고, 15초 composite의
    # 끝은 처음으로 wrap됐다. file만은 planned window를 먼저 exact 렌더링하고 settle
    # 앞에는 zero를 둔다. validator도 같은 공용 렌더러를 사용한다.
    # 콜백 타임스탬프는 **provenance 전용**이다. 실측(무음 40초 프로브): status 0회,
    # adc/cur rate +5.0 ppm, 그런데 dac−adc 는 0.010/0.020 s 두 값 사이를 16 샘플
    # 단위로만 튄다 → 실제 ±130 샘플 변조를 전혀 보여주지 않는다. 이 값으로 정렬을
    # 고치려 들면 안 된다. 정렬은 REF 증인으로 사후 추정한다.
    max_callbacks = total // max(1, block) + 64
    stamps = np.zeros((max_callbacks, 3), dtype=np.float64)
    stamp_count = {"n": 0}
    graceful_stop = {"signal": None}

    def callback(indata, outdata, frames, time_info, status):
        if graceful_stop["signal"] is not None:
            # Parent hard-timeout의 첫 단계는 SIGTERM이다. 다음 callback에서 두 DAC
            # 채널을 exact zero로 제출하고 stream을 정상 close시킨 뒤 partial raw를
            # durable failure receipt로 발행한다.
            outdata[:] = 0
            raise sd.CallbackStop
        if status:
            # 콜백 안에서 print 하면 그 자체가 다음 xrun 을 만든다. 세어만 두고 밖에서 판정한다.
            # xrun 은 source 와 mics 사이에 **영구 오프셋**을 남긴다 — 커서는 frames 만큼
            # 계속 전진하므로 드롭된 블록만큼 두 배열이 세션 끝까지 어긋난다.
            # ⚠ 이 가드는 **필요조건일 뿐 충분조건이 아니다.** status==0 이어도 시간축은
            #   깨진다: 무음 40초 프로브에서 status 0회였고, PortAudio 를 완전히 배제한
            #   aplay+arecord 직결 경로에서도 4~5초 주기 5.4~8.5 ms 변조가 같은 파형으로
            #   재현됐다. 그래서 저장 시점의 REF 증인 재정렬이 진짜 판정이다.
            xrun_state["count"] += 1
            xrun_state["flags"].add(str(status))
        idx = stamp_count["n"]
        if idx < max_callbacks:
            stamps[idx, 0] = time_info.inputBufferAdcTime
            stamps[idx, 1] = time_info.outputBufferDacTime
            stamps[idx, 2] = float(frames)
            stamp_count["n"] = idx + 1

        i = cursor["in"]
        n = min(frames, total - i)
        recorded[i : i + n] = pcm_int32_to_float32(indata[:n, :2])
        cursor["in"] = i + n

        o = cursor["out"]
        m = max(0, min(frames, total - o))
        if args.program == "file":
            # settle은 pre-rendered source의 exact zero prefix다. callback block이
            # settle 경계를 가로질러도 planned file cursor는 소비되지 않는다.
            blk = np.zeros(frames, dtype=np.float32)
            if m > 0:
                blk[:m] = source[o : o + m]
        else:
            blk = program.generate(frames)
            # 생성형 program은 기존 위상/settle 동작을 보존한다.
            blk *= envelope[o : o + frames]
            if m > 0:
                source[o : o + m] = blk[:m]
        out = np.zeros((frames, 2), dtype=np.float32)
        out[:, 0] = blk                             # ch0 = 소음 스피커
        # ch1(상쇄 스피커)은 무음 유지
        outdata[:] = np.rint(np.clip(out, -1, 1) * 32767).astype(np.int16)
        cursor["out"] = o + m
        if cursor["in"] >= total:
            raise sd.CallbackStop

    try:
        # input probe 뒤 다른 프로세스가 PCM을 잡았거나 CPU 부하가 생긴 race를 닫는다.
        current_source_gain = _validate_source_gain_authority(
            args,
            collection_plan=collection_plan,
            require_clean_execution=source_gain_authority is not None,
            authority_required=v2_authority_required,
        )
        if (
            (current_source_gain is None) != (source_gain_authority is None)
            or (
                current_source_gain is not None
                and current_source_gain["binding"] != source_gain_binding
            )
        ):
            raise RecordingSourceGainError(
                "input probe 뒤 source-gain plan/row/commit binding이 변경됐습니다"
            )
        current_level_campaign = _validate_recording_level_authority(
            args,
            sample_rate=fs,
            now_utc=datetime.datetime.now(datetime.timezone.utc),
            source_gain_binding=source_gain_binding,
        )
        recording_level_binding = None
        rendered_level = None
        if current_level_campaign is not None:
            if source_gain_binding is None:
                rendered_level = rendered_source_level_evidence(
                    source[settle : settle + keep]
                )
            else:
                rendered_level = rendered_source_level_evidence_v2(
                    source[settle : settle + keep],
                    amplitude_millionths=source_gain_binding[
                        "amplitude_millionths"
                    ],
                )
        assert_live_pcm_clock_preconditions(hw)
        # 이 timestamp는 CLI 시작이나 input probe 시작이 아니라, 마지막 live gate가
        # 끝난 뒤 duplex stream을 여는 바로 직전의 시각이다.
        session_started_at_utc = datetime.datetime.now(datetime.timezone.utc)
        if current_level_campaign is not None:
            assert rendered_level is not None
            if source_gain_binding is None:
                recording_level_binding = build_recording_level_session_binding(
                    current_level_campaign,
                    session_started_at_utc=session_started_at_utc,
                    same_amplifier_setting=args.confirm_same_amplifier_setting,
                    rendered_source=rendered_level,
                )
            else:
                assert current_source_gain is not None
                recording_level_binding = (
                    build_recording_level_source_gain_session_binding(
                        current_level_campaign,
                        current_source_gain["summary"],
                        session_started_at_utc=session_started_at_utc,
                        same_amplifier_setting=args.confirm_same_amplifier_setting,
                        source_gain_binding=source_gain_binding,
                        rendered_source=rendered_level,
                    )
                )
    except (
        OSError,
        RuntimeError,
        RecordingLevelCampaignError,
        RecordingSourceGainError,
        ValueError,
    ) as exc:
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="immediate_live_gate",
            reason=str(exc),
            sample_rate=fs,
            metadata={**failure_meta, "program": prog_cfg},
        )
        print(f"[중단] 출력 직전 PCM/CPU/clock gate 실패: {exc}", file=sys.stderr)
        return 1

    print(f"녹음 시작: {args.program}, {args.seconds:.0f}초 (ANC 없음, ch1 무음)")
    stream_error: Exception | None = None
    previous_handlers: dict[int, object] = {}

    def _request_graceful_stop(signum, _frame):
        graceful_stop["signal"] = signal.Signals(signum).name

    authority_guards: list[RepositoryFileGuard] = []

    def _immediate_pre_open_authority_check() -> None:
        for guard in authority_guards:
            guard.verify()
        immediate_source_gain = _validate_source_gain_authority(
            args,
            collection_plan=collection_plan,
            require_clean_execution=True,
            authority_required=v2_authority_required,
        )
        if (
            immediate_source_gain is None
            or immediate_source_gain["binding"] != source_gain_binding
        ):
            raise RecordingSourceGainError(
                "audio start 직전 source-gain plan/row/commit binding이 변경됐습니다"
            )
        immediate_campaign = _validate_recording_level_authority(
            args,
            sample_rate=fs,
            now_utc=datetime.datetime.now(datetime.timezone.utc),
            source_gain_binding=source_gain_binding,
        )
        if (
            immediate_campaign is None
            or recording_level_campaign is None
            or immediate_campaign.get("campaign_id")
            != recording_level_campaign.get("campaign_id")
            or immediate_campaign.get("receipt_sha256")
            != recording_level_campaign.get("receipt_sha256")
        ):
            raise RecordingLevelCampaignError(
                "audio start 직전 recording-level campaign binding이 변경됐습니다"
            )
        assert_live_pcm_clock_preconditions(hw)
        for guard in authority_guards:
            guard.verify()

    try:
        for stop_signal in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[int(stop_signal)] = signal.getsignal(stop_signal)
            signal.signal(stop_signal, _request_graceful_stop)
        stream_kwargs = {
            "samplerate": fs,
            "blocksize": block,
            "device": (in_dev, out_dev),
            "channels": (2, 2),
            "dtype": ("int32", "int16"),
            "latency": ("low", "low"),
            "callback": callback,
            "prime_output_buffers_using_stream_callback": True,
        }
        with contextlib.ExitStack() as authority_stack:
            if source_gain_binding is not None:
                authority_guards = _open_live_authority_guards(
                    authority_stack,
                    args=args,
                    collection_plan=collection_plan,
                    source_gain_binding=source_gain_binding,
                )
                stream = _construct_stream_after_authority_check(
                    sd,
                    pre_open_check=_immediate_pre_open_authority_check,
                    stream_kwargs=stream_kwargs,
                )
            else:
                # legacy gain은 diagnostic-only이며 canonical authority 경로에서는 위에서
                # 이미 거절된다. 일반 진단 수집의 기존 동작만 보존한다.
                stream = sd.Stream(**stream_kwargs)
            try:
                if source_gain_binding is not None:
                    # sounddevice Stream constructor는 Pa_OpenStream만 수행하고 start하지
                    # 않는다. context manager의 암시적 start 대신 authority를 한 번 더
                    # 확인한 뒤 explicit start한다.
                    _immediate_pre_open_authority_check()
                stream.start()
                while cursor["in"] < total and graceful_stop["signal"] is None:
                    time.sleep(0.1)
            finally:
                try:
                    stream.stop()
                finally:
                    stream.close()
    except Exception as exc:  # PortAudio/callback 오류도 메모리에 남은 partial raw를 보존한다.
        stream_error = exc
    finally:
        for stop_signal in (signal.SIGTERM, signal.SIGINT):
            previous = previous_handlers.get(int(stop_signal))
            if previous is not None:
                signal.signal(stop_signal, previous)
        # 정렬/저장보다 먼저 알려야 스피커가 분석 시간 동안 연결된 채 방치되지 않는다.
        print("출력 종료 — 지금 스피커를 분리하세요.", flush=True)

    if graceful_stop["signal"] is not None and stream_error is None:
        stream_error = RuntimeError(
            f"graceful_parent_stop:{graceful_stop['signal']}"
        )

    captured = min(int(cursor["in"]), int(cursor["out"]), total)
    raw_mics = recorded[:captured]
    raw_source = source[:captured]
    if stream_error is not None:
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="duplex_stream",
            reason=repr(stream_error),
            sample_rate=fs,
            metadata={
                **failure_meta,
                "program": prog_cfg,
                "captured_frames": captured,
                "requested_frames": total,
                "xrun_count": int(xrun_state["count"]),
                "xrun_flags": sorted(xrun_state["flags"]),
            },
            mics_raw=raw_mics,
            source_raw=raw_source,
        )
        print(f"[중단] duplex stream 실패: {stream_error}", file=sys.stderr)
        return 1

    # Campaign/source binding은 live 전뿐 아니라 stream close 뒤에도 같은 source bytes와
    # receipt SHA에서 재유도한다. 출력 중 receipt/raw가 바뀐 경우 active tree에 발행하지 않는다.
    if recording_level_binding is not None:
        try:
            post_source_gain = _validate_source_gain_authority(
                args,
                collection_plan=collection_plan,
                require_clean_execution=source_gain_authority is not None,
                authority_required=v2_authority_required,
            )
            if (
                (post_source_gain is None) != (source_gain_authority is None)
                or (
                    post_source_gain is not None
                    and post_source_gain["binding"] != source_gain_binding
                )
            ):
                raise RecordingSourceGainError(
                    "stream close 뒤 source-gain plan/row/commit binding이 변경됐습니다"
                )
            post_campaign = validate_recording_level_campaign(
                repo_root=REPO_ROOT,
                campaign_receipt=str(args.recording_level_campaign),
                expected_sha256=str(args.recording_level_campaign_sha256).lower(),
                now_utc=session_started_at_utc,
                require_fresh=True,
            )
            if source_gain_binding is None:
                validate_recording_level_session_binding(
                    post_campaign, recording_level_binding
                )
                post_rendered = rendered_source_level_evidence(
                    source[settle : settle + keep]
                )
            else:
                assert post_source_gain is not None
                validate_recording_source_gain_session_binding(
                    post_source_gain["summary"], source_gain_binding
                )
                validate_recording_level_source_gain_session_binding(
                    post_campaign,
                    post_source_gain["summary"],
                    recording_level_binding,
                )
                post_rendered = rendered_source_level_evidence_v2(
                    source[settle : settle + keep],
                    amplitude_millionths=source_gain_binding[
                        "amplitude_millionths"
                    ],
                )
            if post_rendered != recording_level_binding.get("rendered_source"):
                raise RecordingLevelCampaignError(
                    "stream close 뒤 rendered source binding이 pre-live bytes와 다릅니다"
                )
        except (
            OSError,
            RecordingLevelCampaignError,
            RecordingSourceGainError,
            RuntimeError,
            ValueError,
        ) as exc:
            _preserve_failed_capture(
                failed_root=_repo_path(args.failed_root),
                stage="recording_level_binding",
                reason=str(exc),
                sample_rate=fs,
                metadata={
                    **failure_meta,
                    "program": prog_cfg,
                    "captured_frames": captured,
                    "recording_level_binding": recording_level_binding,
                },
                mics_raw=raw_mics,
                source_raw=raw_source,
            )
            print(
                f"[중단] recording level/source binding 재검증 실패: {exc}",
                file=sys.stderr,
            )
            return 1

    # ----- 3) 저장 -----
    # xrun 이 하나라도 있으면 source↔mics 정렬이 깨졌다. 전달맵은 이미 xrun 을 무효화
    # 사유로 쓰는데(measure_duct_transfer_map) 학습데이터 수집기만 기준이 느슨했다.
    if xrun_state["count"] > 0:
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="xrun_gate",
            reason=(
                f"xrun {xrun_state['count']}회: "
                f"{', '.join(sorted(xrun_state['flags']))}"
            ),
            sample_rate=fs,
            metadata={
                **failure_meta,
                "program": prog_cfg,
                "captured_frames": captured,
                "requested_frames": total,
                "io_timestamps": _summarise_io_timestamps(stamps[: stamp_count["n"]], fs),
            },
            mics_raw=raw_mics,
            source_raw=raw_source,
        )
        print(
            f"[중단] 오디오 xrun {xrun_state['count']}회 ({', '.join(sorted(xrun_state['flags']))}) — "
            "source 와 mics 의 정렬이 깨져 학습에 쓸 수 없습니다. 세션을 저장하지 않습니다.",
            file=sys.stderr,
        )
        return 1
    if captured != total:
        _preserve_failed_capture(
            failed_root=_repo_path(args.failed_root),
            stage="incomplete_capture",
            reason=f"captured_frames={captured}, requested_frames={total}",
            sample_rate=fs,
            metadata={**failure_meta, "program": prog_cfg},
            mics_raw=raw_mics,
            source_raw=raw_source,
        )
        print(
            f"[중단] 캡처 길이 부족: {captured}/{total} frames. 실패 raw를 보존했습니다.",
            file=sys.stderr,
        )
        return 1

    # ----- 3a) 시간축 재정렬 (저장 **전에** 판정한다) -----
    # settle 구간을 양쪽에서 동일하게 잘라낸다. 이 자르기가 보존하는 것은 인덱스
    # 동일성일 뿐 물리 시각 동일성이 아니다 — 그래서 바로 아래에서 실제로 측정한다.
    mics_keep = recorded[settle:captured]
    source_keep = source[settle:captured]

    silent_program = args.program == "silence" or source_family == "silence"
    timeline_meta: dict = {}
    aligned = None
    if silent_program:
        # 무음 세션은 재생 신호가 없어 L(t) 를 추정할 수 없다. 추정할 수 없다는 사실을
        # 그대로 기록한다 — "검사하지 않음"을 "통과"로 적으면 그게 결함 2 의 재발이다.
        timeline_meta = {
            "method": "skipped_silent_program",
            "usable_for_digital_reference": False,
            "reason": "무음 프로그램은 재생↔녹음 대응을 측정할 수 없습니다",
        }
        print("[안내] 무음 세션이라 시간축 재정렬을 건너뜁니다 (digital-ref 학습에 쓸 수 없음)")
    else:
        print("시간축 재정렬 중 (REF 마이크를 증인으로 사용)...")
        try:
            aligned, report = align_source_to_adc(
                source_keep,
                mics_keep[:, 1],   # 증인 = REF (추정 전용)
                mics_keep[:, 0],   # 홀드아웃 = ERR (검증 전용)
                fs,
                settings=TimelineSettings(sample_rate=fs),
            )
        except (RuntimeError, ValueError) as exc:
            _preserve_failed_capture(
                failed_root=_repo_path(args.failed_root),
                stage="timeline_alignment",
                reason=str(exc),
                sample_rate=fs,
                metadata={
                    **failure_meta,
                    "program": prog_cfg,
                    "captured_frames": captured,
                    "io_timestamps": _summarise_io_timestamps(
                        stamps[: stamp_count["n"]], fs
                    ),
                },
                mics_raw=raw_mics,
                source_raw=raw_source,
            )
            print(f"[중단] 시간축 재정렬 예외: {exc}", file=sys.stderr)
            return 1
        timeline_meta, capture_gate = _timeline_metadata_with_capture_gate(
            report,
            min_coherence=args.min_timeline_coherence,
            min_valid_window_ratio=args.min_valid_window_ratio,
        )
        print(
            f"  coh²(source→ERR,150-600Hz) {report.coh2_150_600_before:.3f} → "
            f"{report.coh2_150_600_after:.3f} | 600-1600Hz "
            f"{report.coh2_600_1600_before:.3f} → {report.coh2_600_1600_after:.3f}"
        )
        print(
            f"  유효창 {report.valid_window_ratio:.3f} | 원시지연 중앙 "
            f"{report.raw_lag_median_samples:.1f} ptp {report.raw_lag_ptp_samples:.1f} | "
            f"잔여지연 중앙 {report.aligned_lag_median_samples:.2f} "
            f"robust-std {report.aligned_lag_robust_std_samples:.2f} "
            f"p95-p5 {report.aligned_lag_p95_p5_samples:.2f}"
        )
        if not capture_gate.ok:
            _preserve_failed_capture(
                failed_root=_repo_path(args.failed_root),
                stage="timeline_gate",
                reason=capture_gate.failure_text,
                sample_rate=fs,
                metadata={
                    **failure_meta,
                    "program": prog_cfg,
                    "timeline": timeline_meta,
                    "captured_frames": captured,
                    "io_timestamps": _summarise_io_timestamps(
                        stamps[: stamp_count["n"]], fs
                    ),
                },
                mics_raw=raw_mics,
                source_raw=raw_source,
            )
            print(
                "[중단] 시간축 capture gate 실패 — 세션을 저장하지 않습니다: "
                + capture_gate.failure_text,
                file=sys.stderr,
            )
            return 1

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{stamp}_{args.program}_{secrets.token_hex(4)}"
    session_id = validate_session_id(session_name)
    group_id = requested_group_id or validate_group_id(session_id)
    meta = {
        "session_id": session_id,
        "program": prog_cfg,
        "source_family": source_family,
        "group_id": group_id,
        "seconds": args.seconds,
        "sample_rate": fs,
        "block_size": block,
        "channels": {"err_mic": 0, "ref_mic": 1, "noise_out": 0, "cancel_out": 1},
        "ref_check_dbfs": {"err": err_db, "ref": ref_db},
        "timestamp": stamp,
        "collection_plan": collection_plan,
        "preassigned_split": collection_plan.get("preassigned_split"),
        "safety_confirmations": {
            "user_present": True,
            "volume_minimum": True,
            "routing_and_geometry": True,
        },
        "timeline": timeline_meta,
        "recording_level_binding": recording_level_binding,
        "source_gain_binding": source_gain_binding,
        "plant_domain": (
            "current_strict"
            if source_gain_binding is not None and recording_level_binding is not None
            else (
                "diagnostic_legacy_gain"
                if recording_level_binding is not None
                else None
            )
        ),
        "io_timestamps": _summarise_io_timestamps(stamps[: stamp_count["n"]], fs),
    }
    try:
        session_dir = _publish_session(
            out_root=_repo_path(args.out_root),
            staging_root=_repo_path(args.staging_root),
            failed_root=_repo_path(args.failed_root),
            session_name=session_name,
            sample_rate=fs,
            mics=mics_keep,
            source=source_keep,
            aligned=aligned,
            metadata=meta,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"[중단] canonical 세션 원자 발행 실패: {exc}. active recorded tree에는 "
            "partial session을 발행하지 않았습니다.",
            file=sys.stderr,
        )
        return 1
    print(f"저장 완료: {session_dir}")
    print("다음: .venv/bin/python scripts/data/make_recorded_manifest.py 로 manifest 갱신")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
