"""장치·추론 의존성 없는 런타임 녹음 provenance와 덮어쓰기 방지 저장.

health는 시작/종료를 포함한 전체 런타임의 누적치다. 녹음 구간별 건전성이나
입력 clipping/적응 이력은 저장하지 않으므로 이 메타만으로 성능 PASS를 주장하지 않는다.
exclusive 생성 후 쓰기가 실패하면 불완전한 새 파일이 남을 수 있다. 자동으로 지우거나
덮어쓰지 않으며, 다음 저장도 해당 파일이 있으면 거부한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


RECORDING_SCHEMA_VERSION = 1
SIGNAL_KEYS = ("err", "ref", "source", "control", "anc_gain")


def prepare_recording_path(path: str | Path) -> Path:
    """장치 초기화 전에 호출한다. 실제 쓰기 시 exclusive open으로 다시 보호한다."""
    target = Path(path).with_suffix(".npz")
    # exists()가 False인 깨진 symlink도 기존 사용자 경로이므로 거부한다.
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"녹음 파일이 이미 존재합니다 — 덮어쓰지 않습니다: {target}")
    return target


def capture_recording_provenance(runtime, secondary, secondary_path: str | Path) -> dict:
    """녹음 활성 생성자에서 한 번 호출해 실행 설정과 S 파일 해시를 고정한다."""
    digest = hashlib.sha256()
    with Path(secondary_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "reference": str(runtime.reference),
        "controller": str(runtime.cfg.get("controller", "dl")),
        "sample_rate": int(runtime.fs),
        "block_size": int(runtime.block),
        "hop": int(runtime.hop),
        "latency": str(runtime.latency),
        "digital_reference_lead_samples": int(runtime.digital_reference_lead),
        "handoff_extra_samples": int(runtime.hop),
        "secondary_sha256": digest.hexdigest(),
        "secondary_delay_samples": int(secondary.delay_samples),
        "secondary_fir_length": int(secondary.fir.size),
        "control_limit": float(runtime.safety.control_limit),
        "dc_blocker_r": float(runtime.cfg["hardware"].get("dc_blocker_r", 0.995)),
        "channels": {
            "error_mic": int(runtime.ch_err),
            "reference_mic": int(runtime.ch_ref),
            "noise_out": int(runtime.ch_noise),
            "cancel_out": int(runtime.ch_cancel),
        },
        "record_requested_samples": int(runtime.record_len),
    }


def build_recording_payload(runtime, data: dict | None = None) -> dict[str, np.ndarray]:
    """session_data() 배열 API를 유지하며 저장할 때만 scalar 메타를 붙인다."""
    if runtime._recording_provenance is None:
        raise ValueError("녹음 provenance가 없습니다 — 녹음 활성 생성자가 필요합니다")
    if data is None:
        data = runtime.session_data()
    payload = {key: np.asarray(data[key]) for key in SIGNAL_KEYS}
    meta = dict(runtime._recording_provenance)
    meta["recorded_samples"] = int(runtime.rec_pos)
    meta["runtime_health"] = {
        "xrun_count": int(runtime.xruns),
        # ring.drops는 횟수가 아니라 backlog에서 폐기한 샘플 수다.
        "input_ring_drops": int(runtime.in_ring.drops),
        "output_ring_drops": int(runtime.out_ring.drops),
        "output_ring_underruns": int(runtime.out_ring.underruns),
        "fatal_error": runtime.state.fatal_error is not None,
        "scope": "whole_runtime_including_startup_stop",
    }
    payload["fs"] = np.asarray(meta["sample_rate"], dtype=np.int64)
    payload["recording_schema_version"] = np.asarray(RECORDING_SCHEMA_VERSION, dtype=np.int64)
    payload["recording_meta_json"] = np.asarray(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    _validate_recording_payload(payload)
    return payload


def _validate_recording_payload(payload: dict) -> None:
    required = {*SIGNAL_KEYS, "fs", "recording_schema_version", "recording_meta_json"}
    if set(payload) != required:
        raise ValueError("녹음 payload 키가 schema version 1 규약과 다릅니다")
    fs = np.asarray(payload["fs"])
    version = np.asarray(payload["recording_schema_version"])
    text = np.asarray(payload["recording_meta_json"])
    if fs.ndim != 0 or fs.dtype.kind not in "iu" or int(fs) <= 0:
        raise ValueError("녹음 fs는 양의 정수 scalar여야 합니다")
    if version.ndim != 0 or version.dtype.kind not in "iu" or int(version) != RECORDING_SCHEMA_VERSION:
        raise ValueError("녹음 recording_schema_version이 지원되지 않습니다")
    if text.ndim != 0 or text.dtype.kind != "U":
        raise ValueError("녹음 recording_meta_json은 문자열 scalar여야 합니다")
    meta = json.loads(str(text))
    if not isinstance(meta, dict):
        raise ValueError("녹음 recording_meta_json은 JSON object여야 합니다")
    # NaN/Infinity 같은 비표준 JSON 값이 유입된 payload도 파일 생성 전에 거부한다.
    json.dumps(meta, allow_nan=False)
    if meta.get("sample_rate") != int(fs):
        raise ValueError("녹음 metadata sample_rate와 fs가 다릅니다")
    n = None
    for key in SIGNAL_KEYS:
        values = np.asarray(payload[key])
        if values.ndim != 1 or values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError(f"녹음 {key}는 유한한 1차원 실수 배열이어야 합니다")
        if n is None:
            n = values.size
        elif values.size != n:
            raise ValueError("녹음 신호 배열 길이가 다릅니다")
    if meta.get("recorded_samples") != n or meta.get("record_requested_samples", -1) < n:
        raise ValueError("녹음 recorded_samples/record_requested_samples가 배열 길이와 다릅니다")
    gain = np.asarray(payload["anc_gain"])
    if np.any((gain < 0.0) | (gain > 1.0)):
        raise ValueError("녹음 anc_gain은 0..1 범위여야 합니다")


def save_recording(path: str | Path, payload: dict) -> Path:
    """검증 후 .npz를 exclusive 생성한다. 쓰기 실패 시 부분 파일이 남을 수 있다."""
    _validate_recording_payload(payload)
    target = prepare_recording_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 초기 존재 검사와 실제 저장 사이에 다른 파일이 생겨도 절대로 덮어쓰지 않는다.
    with target.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    return target
