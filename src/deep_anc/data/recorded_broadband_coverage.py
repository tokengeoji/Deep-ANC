"""광대역 recorded 데이터의 실제 target-d/coherence coverage 진단.

기존 150--1600 Hz coverage schema를 조용히 확장하지 않는다. 이 모듈은 광대역 v2
계약의 독립 진단 경로이며, source WAV에 고역이 있다는 사실과 ERR target ``d``에
제어할 에너지가 실제 존재한다는 사실을 분리한다.

공식 campaign receipt는 이 진단 결과를 그대로 신뢰하면 안 된다. manifest/session/WAV,
timing, deterministic segment population을 모두 SHA로 결속하는 후속 schema가 필요하다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import coherence, welch

from ..dsp.control_band_contract import ControlBandContract
from ..dsp.invariants import coherence_from_delay_jitter


BROADBAND_COVERAGE_DIAGNOSTIC_SCHEMA = "recorded_broadband_coverage_diagnostic_v1"
MIN_BROADBAND_COHERENCE = 0.60
MIN_TARGET_ENERGY_DENSITY_RATIO = 0.25


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mono_float64(value: np.ndarray, *, channel: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 2:
        if channel is None:
            if raw.shape[1] != 1:
                raise ValueError(f"mono source를 기대했지만 shape={raw.shape}입니다")
            raw = raw[:, 0]
        else:
            if not 0 <= int(channel) < raw.shape[1]:
                raise ValueError(f"channel {channel}이 shape={raw.shape}에 없습니다")
            raw = raw[:, int(channel)]
    elif raw.ndim != 1:
        raise ValueError(f"audio는 1D/2D여야 합니다: shape={raw.shape}")
    result = np.asarray(raw, dtype=np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        scale = float(max(abs(int(info.min)), abs(int(info.max))))
        result = result / scale
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("audio가 비었거나 NaN/Inf를 포함합니다")
    return result


def measure_broadband_session(
    source: np.ndarray,
    target_d: np.ndarray,
    *,
    sample_rate: int,
    subbands_hz: Sequence[Sequence[float]],
    nperseg: int = 8192,
    noverlap: int = 4096,
    min_coherence: float = MIN_BROADBAND_COHERENCE,
    min_target_density: float = MIN_TARGET_ENERGY_DENSITY_RATIO,
) -> dict[str, Any]:
    """한 세션의 source→ERR coherence와 ERR target-d density를 계산한다."""

    src = _mono_float64(source)
    target = _mono_float64(target_d)
    if src.size != target.size:
        raise ValueError(f"source/target 길이가 다릅니다: {src.size} != {target.size}")
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError("sample_rate는 양수여야 합니다")
    nper = int(nperseg)
    overlap = int(noverlap)
    if src.size < nper or nper < 256 or not 0 <= overlap < nper:
        raise ValueError("Welch/coherence window 설정 또는 신호 길이가 잘못됐습니다")
    bands = tuple(tuple(float(value) for value in band) for band in subbands_hz)
    if not bands:
        raise ValueError("subband가 비었습니다")

    frequency, coh = coherence(
        src,
        target,
        fs=rate,
        nperseg=nper,
        noverlap=overlap,
        detrend=False,
    )
    psd_frequency, target_psd = welch(
        target,
        fs=rate,
        nperseg=nper,
        noverlap=overlap,
        detrend=False,
    )
    total_mask = (psd_frequency >= bands[0][0]) & (psd_frequency < bands[-1][1])
    if not np.any(total_mask):
        raise ValueError("전체 point-control 대역에 Welch bin이 없습니다")
    flat_target_density = float(np.mean(target_psd[total_mask]))
    if not math.isfinite(flat_target_density):
        raise ValueError("target-d PSD가 유한하지 않습니다")

    coherence_values: list[float] = []
    density_values: list[float] = []
    for lo, hi in bands:
        coh_mask = (frequency >= lo) & (frequency < hi)
        psd_mask = (psd_frequency >= lo) & (psd_frequency < hi)
        if not np.any(coh_mask) or not np.any(psd_mask):
            raise ValueError(f"subband [{lo:g}, {hi:g})에 FFT bin이 없습니다")
        coh_value = float(np.median(coh[coh_mask]))
        band_density = float(np.mean(target_psd[psd_mask]))
        density_ratio = (
            0.0
            if flat_target_density <= np.finfo(np.float64).tiny
            else float(band_density / flat_target_density)
        )
        if not (math.isfinite(coh_value) and math.isfinite(density_ratio)):
            raise ValueError(f"subband [{lo:g}, {hi:g}) 결과가 유한하지 않습니다")
        coherence_values.append(coh_value)
        density_values.append(density_ratio)

    coherence_pass = tuple(value >= float(min_coherence) for value in coherence_values)
    density_pass = tuple(value >= float(min_target_density) for value in density_values)
    return {
        "coherence": tuple(coherence_values),
        "target_energy_density_ratio": tuple(density_values),
        "coherence_pass": coherence_pass,
        "target_energy_density_pass": density_pass,
        "joint_pass": tuple(
            left and right for left, right in zip(coherence_pass, density_pass, strict=True)
        ),
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return {"median": float("nan"), "p10": float("nan")}
    return {
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
    }


def summarize_broadband_sessions(
    sessions: Sequence[dict[str, Any]],
    *,
    subbands_hz: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """ALL/family/split×family를 session 단위로 집계한다."""

    rows = tuple(sessions)
    n_bands = len(tuple(subbands_hz))
    if not rows:
        raise ValueError("집계할 session이 없습니다")

    def one_group(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        result: list[dict[str, Any]] = []
        for index in range(n_bands):
            coh = [float(row["coherence"][index]) for row in selected]
            density = [float(row["target_energy_density_ratio"][index]) for row in selected]
            coh_pass = [bool(row["coherence_pass"][index]) for row in selected]
            density_pass = [bool(row["target_energy_density_pass"][index]) for row in selected]
            joint = [bool(row["joint_pass"][index]) for row in selected]
            groups = {
                str(row["group_id"])
                for row in selected
                if bool(row["joint_pass"][index])
            }
            result.append(
                {
                    "coherence": _distribution(coh),
                    "coherence_pass_fraction": float(np.mean(coh_pass)),
                    "target_energy_density": _distribution(density),
                    "target_energy_density_pass_fraction": float(np.mean(density_pass)),
                    "joint_pass_fraction": float(np.mean(joint)),
                    "joint_pass_sessions": int(np.sum(joint)),
                    "joint_pass_independent_groups": len(groups),
                }
            )
        return {"sessions": len(selected), "subbands": result}

    families = sorted({str(row["source_family"]) for row in rows})
    splits = sorted({str(row["split"]) for row in rows})
    return {
        "all": one_group(rows),
        "by_family": {
            family: one_group([row for row in rows if row["source_family"] == family])
            for family in families
        },
        "by_split_family": {
            split: {
                family: one_group(
                    [
                        row
                        for row in rows
                        if row["split"] == split and row["source_family"] == family
                    ]
                )
                for family in families
                if any(
                    row["split"] == split and row["source_family"] == family for row in rows
                )
            }
            for split in splits
        },
    }


def scan_recorded_broadband_coverage(
    manifest_path: str | Path,
    *,
    contract: ControlBandContract,
    qa_path: str | Path | None = None,
    start_seconds: float = 5.0,
    stop_seconds: float = 65.0,
    nperseg: int = 8192,
    noverlap: int = 4096,
) -> dict[str, Any]:
    """manifest 전 세션의 고정 5--65초 population을 실제 WAV에서 계산한다."""

    if contract.role != "broadband_point_control":
        raise ValueError("광대역 coverage에는 broadband_point_control 계약이 필요합니다")
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    start = float(start_seconds)
    stop = float(stop_seconds)
    if not (math.isfinite(start) and math.isfinite(stop) and 0.0 <= start < stop):
        raise ValueError("start/stop seconds가 잘못됐습니다")

    qa_by_session: dict[str, dict[str, Any]] = {}
    qa_snapshot: dict[str, Any] | None = None
    if qa_path is not None:
        qa_file = Path(qa_path).expanduser().resolve()
        qa_payload = json.loads(qa_file.read_text(encoding="utf-8"))
        qa_by_session = {str(row["session_id"]): row for row in qa_payload["sessions"]}
        qa_snapshot = {
            "path": str(qa_file),
            "size_bytes": qa_file.stat().st_size,
            "sha256": sha256_file(qa_file),
        }

    rows: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        session_id = str(entry.get("session_id", "")).strip()
        if not session_id or session_id in seen_sessions:
            raise ValueError(f"manifest line {line_number}: session_id가 비었거나 중복입니다")
        seen_sessions.add(session_id)
        base = (manifest.parent / str(entry["path"])).resolve()
        source_path = base / "source_aligned.wav"
        mics_path = base / "mics.wav"
        source_rate, source_raw = wavfile.read(source_path, mmap=True)
        mics_rate, mics_raw = wavfile.read(mics_path, mmap=True)
        if int(source_rate) != contract.sample_rate or int(mics_rate) != contract.sample_rate:
            raise ValueError(f"{session_id}: sample rate가 48kHz가 아닙니다")
        begin = int(round(start * source_rate))
        end = int(round(stop * source_rate))
        if len(source_raw) < end or len(mics_raw) < end:
            raise ValueError(f"{session_id}: 고정 {start:g}--{stop:g}s population보다 짧습니다")
        source = _mono_float64(source_raw[begin:end])
        target = _mono_float64(mics_raw[begin:end], channel=0)
        metrics = measure_broadband_session(
            source,
            target,
            sample_rate=contract.sample_rate,
            subbands_hz=contract.point_control_subbands_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        row: dict[str, Any] = {
            "session_id": session_id,
            "source_family": str(entry["source_family"]),
            "split": str(entry["split"]),
            "group_id": str(entry["group_id"]),
            **{key: list(value) for key, value in metrics.items()},
        }
        qa_row = qa_by_session.get(session_id)
        if qa_row is not None:
            jitter = float(qa_row["alignment"]["source_err_delay_robust_std_samples"])
            row["alignment_robust_std_samples"] = jitter
            row["jitter_coherence_ceiling_at_subband_upper"] = [
                coherence_from_delay_jitter(
                    band[1], jitter, sample_rate=float(contract.sample_rate)
                )
                for band in contract.point_control_subbands_hz
            ]
        rows.append(row)

    summary = summarize_broadband_sessions(
        rows, subbands_hz=contract.point_control_subbands_hz
    )
    payload: dict[str, Any] = {
        "schema": BROADBAND_COVERAGE_DIAGNOSTIC_SCHEMA,
        "role": "diagnostic_only_not_campaign_receipt",
        "control_band_contract": contract.model_dump(mode="json"),
        "control_band_contract_sha256": contract.digest(),
        "manifest": {
            "path": str(manifest),
            "size_bytes": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
        },
        "qa": qa_snapshot,
        "population": {
            "start_seconds": start,
            "stop_seconds": stop,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "min_coherence": MIN_BROADBAND_COHERENCE,
            "min_target_energy_density_ratio": MIN_TARGET_ENERGY_DENSITY_RATIO,
            "target_channel": "mics.wav ch0 ERR",
            "source_file": "source_aligned.wav",
        },
        "sessions": rows,
        "summary": summary,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


__all__ = [
    "BROADBAND_COVERAGE_DIAGNOSTIC_SCHEMA",
    "MIN_BROADBAND_COHERENCE",
    "MIN_TARGET_ENERGY_DENSITY_RATIO",
    "measure_broadband_session",
    "scan_recorded_broadband_coverage",
    "sha256_file",
    "summarize_broadband_sessions",
]
