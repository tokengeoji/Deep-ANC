"""기존 실측 82세션의 1차경로 레벨을 strict P 단위로 보정하는 계약.

2026-08-04/06 녹음의 ``source_aligned.wav -> ERR`` 크기는 현행 strict P보다
약 20--25 dB 작다. 이 차이를 그대로 두고 measured synthetic과 70:30으로 섞으면
같은 디지털 기준 입력에 서로 다른 플랜트 이득을 가르친다. 이 모듈은 원본 WAV를
바꾸지 않고, **train split만으로** 날짜 cohort별 ERR 이득을 적합해 receipt에 봉인한다.

보정은 digital-reference에서만 물리적으로 정의된다. acoustic-reference의 REF/ERR
비율에 이 이득을 적용하는 것은 다른 문제를 만드는 것이므로 fail-closed한다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import csd, welch

from .holdout_contract import read_regular_file_snapshot
from .manifest import read_manifest
from .source_trust import (
    PROTECTED_IGNORED_ROOTS,
    SOURCE_TRUST_SCHEMA,
    SourceTrustError,
    exact_clean_source_evidence,
)


SCHEMA = "recorded_primary_level_calibration_v1"
RECEIPT_ROOT = Path("data/manifests/recorded_level_calibration")
HISTORICAL_DOMAIN = "historical_calibrated"
CURRENT_DOMAIN = "current_strict"
CALIBRATION_BANDS_HZ = (
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
)
WELCH_RECIPE: dict[str, Any] = {
    "sample_rate": 48_000,
    "window": "hann",
    "nperseg": 8192,
    "noverlap": 4096,
    "detrend": "constant",
    "scaling": "density",
    "average": "mean",
    "plant_transfer_estimator": "H1=CSD(source,ERR)/PSD(source)",
    "analysis_start_seconds": 5.0,
    "analysis_stop_seconds": 65.0,
    "fit_split": "train",
    "cohort_estimator": "median_session_power_ratio_db",
    "trusted_band_hz": [150.0, 1600.0],
    "subbands_hz": [list(band) for band in CALIBRATION_BANDS_HZ],
}
EXPECTED_HISTORICAL_COHORTS = {
    "20260804": "historical_20260804",
    "20260806": "historical_20260806",
}
CALIBRATION_QUALITY_CONTRACT = {
    "heldout_split_median_max_abs_db": 1.0,
    "all_session_residual_max_abs_db": 6.0,
    "train_complex_agreement_min": 0.95,
    "train_complex_relative_error_max": 0.25,
    "calibrated_err_abs_peak_max": 0.8,
}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RecordedLevelCalibrationError(ValueError):
    """보정 receipt 또는 그 입력이 계약을 위반했다."""


@dataclass(frozen=True)
class RecordedLevelCalibration:
    path: Path
    sha256: str
    payload: dict[str, Any]
    err_gain_by_session: dict[str, float]
    plant_domain_by_session: dict[str, str]


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def require_clean_exact_commit(repo_root: str | Path) -> str:
    """권위 source 검사를 통과한 실제 40자리 commit만 반환한다.

    단순 ``git status``는 replace/graft, 숨은 index flag, 실제 tracked blob/mode
    변조와 protected-root injection을 놓친다. Receipt 발행 시작점은 repository의
    공용 exact-source 구현만 사용하며, 실행 가능한 stale bytecode도 허용하지 않는다.
    """

    try:
        evidence = exact_clean_source_evidence(
            repo_root,
            reject_runtime_bytecode=True,
        )
    except SourceTrustError as exc:
        raise RecordedLevelCalibrationError(
            f"calibration receipt clean exact source 검증 실패: {exc}"
        ) from exc
    commit = str(evidence.get("commit") or "")
    if not _COMMIT_RE.fullmatch(commit):
        raise RecordedLevelCalibrationError(f"git HEAD가 40자리 commit이 아닙니다: {commit}")
    return commit


def _validate_clean_source_evidence(
    value: object,
    *,
    source_commit: str,
) -> Mapping[str, Any]:
    """Receipt 내부 exact-source evidence의 canonical shape를 검증한다."""

    if not isinstance(value, Mapping):
        raise RecordedLevelCalibrationError(
            "calibration clean_source evidence가 없습니다"
        )
    required = {
        "schema",
        "commit",
        "head_tree_object_id",
        "git_object_format",
        "tracked_file_count",
        "tracked_inventory_sha256",
        "policy",
    }
    if set(value) != required:
        raise RecordedLevelCalibrationError(
            "calibration clean_source evidence 필드 집합이 exact하지 않습니다"
        )
    object_format = value.get("git_object_format")
    object_id = value.get("head_tree_object_id")
    object_id_length = 40 if object_format == "sha1" else 64 if object_format == "sha256" else 0
    tracked_count = value.get("tracked_file_count")
    if (
        value.get("schema") != SOURCE_TRUST_SCHEMA
        or value.get("commit") != source_commit
        or object_id_length == 0
        or not isinstance(object_id, str)
        or len(object_id) != object_id_length
        or any(character not in "0123456789abcdef" for character in object_id)
        or isinstance(tracked_count, bool)
        or not isinstance(tracked_count, int)
        or tracked_count <= 0
        or _SHA256_RE.fullmatch(
            str(value.get("tracked_inventory_sha256") or "")
        )
        is None
    ):
        raise RecordedLevelCalibrationError(
            "calibration clean_source evidence identity가 유효하지 않습니다"
        )
    policy = value.get("policy")
    expected_policy = {
        "tracked_worktree": "exact_HEAD_blob_and_mode",
        "index": "exact_HEAD_tree_no_hidden_flags",
        "nonignored_untracked": "forbidden",
        "protected_ignored_roots": list(PROTECTED_IGNORED_ROOTS),
        "protected_runtime_bytecode": "forbidden",
        "ignored_artifacts_outside_protected_roots": "allowed",
        "replace_refs_and_grafts": "forbidden",
    }
    if policy != expected_policy:
        raise RecordedLevelCalibrationError(
            "calibration clean_source fail-closed policy가 exact하지 않습니다"
        )
    return value


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise RecordedLevelCalibrationError(
            f"보정 입력이 저장소 밖입니다: {path}"
        ) from exc


def canonical_recorded_level_calibration_output(
    repo_root: str | Path, output: str | Path
) -> Path:
    """Canonical 발행 경로를 저장소 내부 전용 namespace로 제한한다."""

    root = Path(repo_root).resolve()
    relative = Path(output)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.parts[: len(RECEIPT_ROOT.parts)] != RECEIPT_ROOT.parts
        or len(relative.parts) <= len(RECEIPT_ROOT.parts)
        or relative.suffix != ".json"
    ):
        raise RecordedLevelCalibrationError(
            "calibration receipt --out은 저장소 상대 "
            f"{RECEIPT_ROOT.as_posix()}/<name>.json 이어야 합니다: {output}"
        )
    candidate = root.joinpath(*relative.parts)
    # parent가 아직 없어도 lexical containment가 exact하다. symlink/non-directory는
    # 실제 발행 시 dirfd+O_NOFOLLOW writer가 다시 차단한다.
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - 위 lexical 검사의 이중 방어
        raise RecordedLevelCalibrationError(
            f"calibration receipt 출력이 저장소 밖입니다: {output}"
        ) from exc
    return candidate


def _file_ref(path: Path, *, repo_root: Path, label: str) -> dict[str, Any]:
    try:
        snapshot = read_regular_file_snapshot(
            path, root=repo_root, label=label, capture_bytes=False
        )
    except ValueError as exc:
        raise RecordedLevelCalibrationError(str(exc)) from exc
    return {
        "path": _repo_relative(path, repo_root),
        "size": int(snapshot.size),
        "sha256": snapshot.sha256,
    }


def _cohort(session_id: str) -> str:
    for prefix, cohort in EXPECTED_HISTORICAL_COHORTS.items():
        if session_id.startswith(prefix + "_"):
            return cohort
    raise RecordedLevelCalibrationError(
        f"지원하지 않는 historical session cohort입니다: {session_id}"
    )


def _load_mono(path: Path, *, sample_rate: int, channel: int = 0) -> np.ndarray:
    values, actual_rate = sf.read(path, dtype="float64", always_2d=True)
    if int(actual_rate) != int(sample_rate):
        raise RecordedLevelCalibrationError(
            f"{path}: sample rate {actual_rate} != {sample_rate}"
        )
    if values.shape[1] <= channel:
        raise RecordedLevelCalibrationError(
            f"{path}: 요청 채널 {channel}이 없습니다 (channels={values.shape[1]})"
        )
    result = np.asarray(values[:, channel], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise RecordedLevelCalibrationError(f"{path}: NaN/Inf가 있습니다")
    return result


def _strict_primary(path: Path) -> tuple[np.ndarray, int, int]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            fir = np.asarray(archive["fir"], dtype=np.float64).reshape(-1)
            sample_rate = int(np.asarray(archive["sample_rate"]).item())
            delay_samples = int(np.asarray(archive["delay_samples"]).item())
            band = np.asarray(archive["consistency_band_hz"], dtype=np.float64)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RecordedLevelCalibrationError(
            f"strict primary NPZ를 읽을 수 없습니다: {path}: {exc}"
        ) from exc
    if (
        sample_rate != WELCH_RECIPE["sample_rate"]
        or fir.size < 2
        or not np.all(np.isfinite(fir))
        or band.tolist() != WELCH_RECIPE["trusted_band_hz"]
    ):
        raise RecordedLevelCalibrationError(
            "strict primary의 sample rate/FIR/trusted band가 보정 계약과 다릅니다"
        )
    return fir, sample_rate, delay_samples


def _session_power_ratio_db(
    source_path: Path,
    mics_path: Path,
    *,
    strict_fir: np.ndarray,
    sample_rate: int,
) -> tuple[float, list[float], np.ndarray, np.ndarray, np.ndarray, float]:
    source = _load_mono(source_path, sample_rate=sample_rate)
    err = _load_mono(mics_path, sample_rate=sample_rate, channel=0)
    start = int(round(float(WELCH_RECIPE["analysis_start_seconds"]) * sample_rate))
    stop = int(round(float(WELCH_RECIPE["analysis_stop_seconds"]) * sample_rate))
    if min(source.size, err.size) < stop:
        raise RecordedLevelCalibrationError(
            f"60초 Welch 구간이 없습니다: {source_path.parent}"
        )
    kwargs = {
        "fs": sample_rate,
        "window": WELCH_RECIPE["window"],
        "nperseg": WELCH_RECIPE["nperseg"],
        "noverlap": WELCH_RECIPE["noverlap"],
        "detrend": WELCH_RECIPE["detrend"],
        "scaling": WELCH_RECIPE["scaling"],
        "average": WELCH_RECIPE["average"],
    }
    frequency, source_psd = welch(source[start:stop], **kwargs)
    cross_frequency, source_err_csd = csd(
        source[start:stop], err[start:stop], **kwargs
    )
    if not np.array_equal(frequency, cross_frequency):  # pragma: no cover - scipy 방어
        raise RecordedLevelCalibrationError("source/ERR Welch frequency grid가 다릅니다")
    # 마이크 자기잡음까지 strict P gain으로 키우면 물리 플랜트가 아니라 noise floor를
    # 맞추게 된다. H1 coherent transfer가 설명하는 ERR power만 apples-to-apples로 쓴다.
    coherent_err_psd = np.square(np.abs(source_err_csd)) / np.maximum(
        source_psd, np.finfo(np.float64).tiny
    )
    strict_transfer = np.fft.rfft(strict_fir, n=int(WELCH_RECIPE["nperseg"]))
    strict_err_psd = source_psd * np.square(np.abs(strict_transfer))

    def ratio_db(low: float, high: float) -> float:
        mask = (frequency >= low) & (frequency <= high)
        observed = float(np.sum(coherent_err_psd[mask]))
        expected = float(np.sum(strict_err_psd[mask]))
        if not observed > 0.0 or not expected > 0.0:
            raise RecordedLevelCalibrationError(
                f"Welch band power가 0입니다: {source_path.parent}, {low}-{high}Hz"
            )
        return float(10.0 * math.log10(observed / expected))

    subbands = [ratio_db(low, high) for low, high in CALIBRATION_BANDS_HZ]
    fullband = ratio_db(*WELCH_RECIPE["trusted_band_hz"])
    return (
        fullband,
        subbands,
        frequency,
        source_psd,
        source_err_csd,
        float(np.max(np.abs(err))),
    )


def _aggregate_shape_diagnostic(
    *,
    frequency: np.ndarray,
    source_psd: np.ndarray,
    source_err_csd: np.ndarray,
    strict_fir: np.ndarray,
    strict_delay_samples: int,
    sample_rate: int,
) -> dict[str, Any]:
    """gain 하나로 해결되지 않는 complex plant shape 잔차를 정량화한다.

    strict full transfer와 corpus aggregate transfer 사이의 상대 integer delay 및
    complex scalar를 최소제곱으로 제거한다. 이 값은 scalar calibration의 한계를
    숨기지 않기 위한 진단이며 threshold를 결과에 맞춰 낮추는 PASS gate가 아니다.
    """

    low, high = WELCH_RECIPE["trusted_band_hz"]
    mask = (frequency >= low) & (frequency <= high) & (source_psd > 0.0)
    old = source_err_csd[mask] / source_psd[mask]
    omega = 2.0 * np.pi * frequency[mask] / float(sample_rate)
    strict = np.fft.rfft(strict_fir, n=int(WELCH_RECIPE["nperseg"]))[mask]
    strict = strict * np.exp(-1j * omega * int(strict_delay_samples))
    weight = np.asarray(source_psd[mask], dtype=np.float64)
    if old.size < 8 or not np.all(np.isfinite(old)) or not np.all(np.isfinite(strict)):
        raise RecordedLevelCalibrationError("complex shape diagnostic 입력이 부족합니다")

    best_delay = 0
    best_agreement = -1.0
    best_aligned = old
    strict_norm = float(np.sum(weight * np.square(np.abs(strict))))
    for delay in range(-2048, 2049):
        aligned = old * np.exp(-1j * omega * delay)
        old_norm = float(np.sum(weight * np.square(np.abs(aligned))))
        numerator = abs(np.sum(weight * np.conj(aligned) * strict))
        agreement = float(numerator / math.sqrt(max(old_norm * strict_norm, 1e-300)))
        if agreement > best_agreement:
            best_delay = delay
            best_agreement = agreement
            best_aligned = aligned
    denominator = np.sum(weight * np.square(np.abs(best_aligned)))
    scale = np.sum(weight * np.conj(best_aligned) * strict) / denominator
    residual = float(
        math.sqrt(
            float(np.sum(weight * np.square(np.abs(scale * best_aligned - strict))))
            / max(strict_norm, 1e-300)
        )
    )
    return {
        "definition": "aggregate_CSD_over_PSD_then_best_integer_delay_and_complex_scalar",
        "band_hz": [low, high],
        "delay_search_samples": [-2048, 2048],
        "best_relative_delay_samples": int(best_delay),
        "complex_agreement": float(best_agreement),
        "relative_error_after_scalar_and_delay": residual,
        "fit_scope": "train_split_only_shape_diagnostic",
        "interpretation": "scalar_level_calibration_does_not_replace_plant_shape_ablation",
        "required_ablation_domains": [HISTORICAL_DOMAIN, CURRENT_DOMAIN],
    }


def build_recorded_level_calibration_payload(
    *,
    repo_root: str | Path,
    recorded_manifest: str | Path,
    strict_primary_npz: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    """old82를 읽어 train-only cohort gain receipt payload를 만든다."""

    root = Path(repo_root).resolve()
    commit = str(source_commit).lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise RecordedLevelCalibrationError(
            "source_commit은 clean exact 40자리 SHA여야 합니다"
        )
    manifest_path = Path(recorded_manifest)
    primary_path = Path(strict_primary_npz)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not primary_path.is_absolute():
        primary_path = root / primary_path
    strict_fir, sample_rate, strict_delay_samples = _strict_primary(primary_path)
    entries = read_manifest(manifest_path)
    if len(entries) != 82:
        raise RecordedLevelCalibrationError(
            f"historical calibration manifest는 82세션이어야 합니다: {len(entries)}"
        )

    sessions: list[dict[str, Any]] = []
    aggregate_source_psd: np.ndarray | None = None
    aggregate_source_err_csd: np.ndarray | None = None
    aggregate_frequency: np.ndarray | None = None
    seen_ids: set[str] = set()
    for entry in entries:
        session_id = str(entry.get("session_id") or "")
        split = str(entry.get("split") or "")
        if not session_id or session_id in seen_ids or split not in {"train", "val", "test"}:
            raise RecordedLevelCalibrationError(
                f"manifest session_id/split이 유효하지 않습니다: {entry}"
            )
        seen_ids.add(session_id)
        session_dir = Path(str(entry["path"]))
        source = session_dir / "source_aligned.wav"
        mics = session_dir / "mics.wav"
        source_ref = _file_ref(source, repo_root=root, label=f"{session_id} aligned source")
        mics_ref = _file_ref(mics, repo_root=root, label=f"{session_id} mics")
        (
            ratio_db,
            band_ratios,
            frequency,
            source_psd,
            source_err_csd,
            raw_err_abs_peak,
        ) = _session_power_ratio_db(
            source,
            mics,
            strict_fir=strict_fir,
            sample_rate=sample_rate,
        )
        if aggregate_frequency is None:
            aggregate_frequency = frequency
            aggregate_source_psd = np.zeros_like(source_psd)
            aggregate_source_err_csd = np.zeros_like(source_err_csd)
        elif not np.array_equal(aggregate_frequency, frequency):  # pragma: no cover
            raise RecordedLevelCalibrationError("session Welch grid가 다릅니다")
        assert aggregate_source_psd is not None
        assert aggregate_source_err_csd is not None
        if split == "train":
            aggregate_source_psd += source_psd
            aggregate_source_err_csd += source_err_csd
        sessions.append(
            {
                "session_id": session_id,
                "split": split,
                "source_family": str(entry.get("source_family") or ""),
                "cohort": _cohort(session_id),
                "plant_domain": HISTORICAL_DOMAIN,
                "source_aligned": source_ref,
                "mics": mics_ref,
                "observed_to_strict_power_ratio_db": ratio_db,
                "subband_observed_to_strict_power_ratio_db": band_ratios,
                "raw_err_abs_peak": raw_err_abs_peak,
            }
        )

    cohort_payload: dict[str, Any] = {}
    for cohort in sorted(set(EXPECTED_HISTORICAL_COHORTS.values())):
        members = [item for item in sessions if item["cohort"] == cohort]
        fit = [item for item in members if item["split"] == "train"]
        if not fit:
            raise RecordedLevelCalibrationError(f"{cohort}: train fit session이 없습니다")
        fitted_db = float(
            np.median([item["observed_to_strict_power_ratio_db"] for item in fit])
        )
        err_gain = float(10.0 ** (-fitted_db / 20.0))
        residuals: dict[str, Any] = {}
        for split in ("train", "val", "test"):
            split_values = [
                float(item["observed_to_strict_power_ratio_db"]) - fitted_db
                for item in members
                if item["split"] == split
            ]
            residuals[split] = {
                "session_count": len(split_values),
                "median_db": None if not split_values else float(np.median(split_values)),
                "max_abs_db": None
                if not split_values
                else float(np.max(np.abs(split_values))),
            }
        cohort_payload[cohort] = {
            "fit_split": "train",
            "train_fit_count": len(fit),
            "fit_session_ids": sorted(item["session_id"] for item in fit),
            "member_session_ids": sorted(item["session_id"] for item in members),
            "fitted_observed_to_strict_power_ratio_db": fitted_db,
            "err_amplitude_gain": err_gain,
            "heldout_residual_diagnostics": residuals,
        }
        for item in members:
            item["calibrated_err_abs_peak"] = float(
                item["raw_err_abs_peak"] * err_gain
            )

    assert aggregate_frequency is not None
    assert aggregate_source_psd is not None
    assert aggregate_source_err_csd is not None
    shape_diagnostic = _aggregate_shape_diagnostic(
        frequency=aggregate_frequency,
        source_psd=aggregate_source_psd,
        source_err_csd=aggregate_source_err_csd,
        strict_fir=strict_fir,
        strict_delay_samples=strict_delay_samples,
        sample_rate=sample_rate,
    )
    quality_observed = {
        "heldout_split_median_max_abs_db": float(
            max(
                abs(float(cohort_payload[cohort]["heldout_residual_diagnostics"][split]["median_db"]))
                for cohort in sorted(cohort_payload)
                for split in ("val", "test")
                if cohort_payload[cohort]["heldout_residual_diagnostics"][split]["median_db"]
                is not None
            )
        ),
        "all_session_residual_max_abs_db": float(
            max(
                abs(
                    float(item["observed_to_strict_power_ratio_db"])
                    - float(
                        cohort_payload[str(item["cohort"])][
                            "fitted_observed_to_strict_power_ratio_db"
                        ]
                    )
                )
                for item in sessions
            )
        ),
        "train_complex_agreement": float(shape_diagnostic["complex_agreement"]),
        "train_complex_relative_error": float(
            shape_diagnostic["relative_error_after_scalar_and_delay"]
        ),
        "calibrated_err_abs_peak": float(
            max(float(item["calibrated_err_abs_peak"]) for item in sessions)
        ),
    }
    quality_pass = (
        quality_observed["heldout_split_median_max_abs_db"]
        <= CALIBRATION_QUALITY_CONTRACT["heldout_split_median_max_abs_db"]
        and quality_observed["all_session_residual_max_abs_db"]
        <= CALIBRATION_QUALITY_CONTRACT["all_session_residual_max_abs_db"]
        and quality_observed["train_complex_agreement"]
        >= CALIBRATION_QUALITY_CONTRACT["train_complex_agreement_min"]
        and quality_observed["train_complex_relative_error"]
        <= CALIBRATION_QUALITY_CONTRACT["train_complex_relative_error_max"]
        and quality_observed["calibrated_err_abs_peak"]
        <= CALIBRATION_QUALITY_CONTRACT["calibrated_err_abs_peak_max"]
    )
    if not quality_pass:
        raise RecordedLevelCalibrationError(
            "historical scalar calibration 사전 고정 품질 gate가 FAIL했습니다: "
            f"observed={quality_observed}, thresholds={CALIBRATION_QUALITY_CONTRACT}"
        )
    implementation_ref = _file_ref(
        Path(__file__).resolve(), repo_root=root, label="calibration implementation"
    )
    analysis_identity = {
        "schema": SCHEMA,
        "welch_recipe": WELCH_RECIPE,
        "power_ratio_definition": "10log10(sum(abs(CSD_source_ERR)^2/PSD_source)/sum(PSD_source*abs(H_strict)^2))",
        "cohort_gain_definition": "10**(-median_train_power_ratio_db/20)",
        "shape_definition": shape_diagnostic["definition"],
        "implementation_sha256": implementation_ref["sha256"],
    }
    # 시작점은 ``require_clean_exact_commit``이 검사하고, 긴 82세션 분석이 끝난
    # 뒤에는 같은 commit을 권위 구현으로 다시 검사한다. 최종 evidence 자체를
    # receipt에 넣어 단순 boolean clean 주장을 provenance로 쓰지 않는다.
    try:
        clean_source = exact_clean_source_evidence(
            root,
            expected_commit=commit,
            reject_runtime_bytecode=True,
        )
    except SourceTrustError as exc:
        raise RecordedLevelCalibrationError(
            f"calibration 분석 종료 clean exact source 재검증 실패: {exc}"
        ) from exc
    return {
        "schema": SCHEMA,
        "source_commit": commit,
        "source_tree_clean_at_issue": True,
        "clean_source": clean_source,
        "analysis_contract": analysis_identity,
        "analysis_contract_sha256": _sha256_bytes(_canonical_json(analysis_identity)),
        "implementation_source": implementation_ref,
        "purpose": "old82_ERR_to_current_strict_primary_level_only",
        "reference_mode": "digital",
        "apply_to": ["ERR", "d"],
        "forbidden_apply_to": ["source_aligned", "REF", "acoustic_reference"],
        "fit_policy": {
            "allowed_split": "train",
            "heldout_splits_are_diagnostics_only": ["val", "test"],
            "wav_mutation": False,
        },
        "welch_recipe": WELCH_RECIPE,
        "recorded_manifest": _file_ref(
            manifest_path, repo_root=root, label="historical recorded manifest"
        ),
        "strict_primary_npz": _file_ref(
            primary_path, repo_root=root, label="strict primary NPZ"
        ),
        "cohorts": cohort_payload,
        "plant_shape_diagnostic": shape_diagnostic,
        "quality_gate": {
            "thresholds": CALIBRATION_QUALITY_CONTRACT,
            "observed": quality_observed,
            "pass": True,
            "threshold_policy": "predeclared_not_result_tuned",
        },
        "sessions": sorted(sessions, key=lambda item: item["session_id"]),
    }


def write_recorded_level_calibration_receipt(
    payload: dict[str, Any], output: str | Path
) -> tuple[Path, str]:
    """receipt를 symlink 추적/덮어쓰기 없이 발행하고 외부 SHA를 반환한다.

    Canonical receipt는 immutable authority이므로 ``replace`` 경로를 제공하지 않는다.
    각 parent를 dirfd+``O_NOFOLLOW``로 열고 최종 파일은 ``O_EXCL``로 생성해
    exists-check와 write 사이의 TOCTOU 및 symlink 교체를 차단한다.
    """

    raw = _canonical_json(payload)
    path = Path(os.path.abspath(os.fspath(output)))
    parts = path.parts
    if not path.is_absolute() or len(parts) < 2 or path.name in {"", ".", ".."}:
        raise RecordedLevelCalibrationError(f"receipt 출력 경로가 유효하지 않습니다: {output}")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(parts[0], directory_flags | nofollow)
    try:
        for component in parts[1:-1]:
            try:
                next_fd = os.open(
                    component,
                    directory_flags | nofollow,
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                except FileExistsError:
                    # 다른 writer가 먼저 만들었을 수 있다. 아래 no-follow open이
                    # 그것이 실제 directory인지 다시 판정한다.
                    pass
                try:
                    next_fd = os.open(
                        component,
                        directory_flags | nofollow,
                        dir_fd=current_fd,
                    )
                except OSError as exc:
                    raise RecordedLevelCalibrationError(
                        "receipt parent가 regular directory가 아니거나 symlink입니다: "
                        f"{path.parent}"
                    ) from exc
            except OSError as exc:
                raise RecordedLevelCalibrationError(
                    "receipt parent가 regular directory가 아니거나 symlink입니다: "
                    f"{path.parent}"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd

        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | nofollow
        try:
            file_fd = os.open(path.name, file_flags, 0o644, dir_fd=current_fd)
        except FileExistsError as exc:
            raise FileExistsError(
                f"기존 calibration receipt를 덮어쓰지 않습니다: {path}"
            ) from exc
        try:
            created_stat = os.fstat(file_fd)
            if not stat.S_ISREG(created_stat.st_mode):
                raise RecordedLevelCalibrationError(
                    f"calibration receipt가 regular file이 아닙니다: {path}"
                )
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                written = os.write(file_fd, view[offset:])
                if written <= 0:
                    raise OSError("calibration receipt short write")
                offset += written
            os.fsync(file_fd)
        except BaseException:
            # 생성 뒤 write/fsync가 실패하면 우리가 만든 동일 inode만 제거한다.
            try:
                named_stat = os.stat(path.name, dir_fd=current_fd, follow_symlinks=False)
                if (
                    named_stat.st_dev == created_stat.st_dev
                    and named_stat.st_ino == created_stat.st_ino
                ):
                    os.unlink(path.name, dir_fd=current_fd)
            except (NameError, OSError):
                pass
            raise
        finally:
            os.close(file_fd)
    finally:
        os.close(current_fd)
    return path, _sha256_bytes(raw)


def validate_recorded_level_calibration_receipt(
    receipt: str | Path,
    *,
    expected_sha256: str,
    repo_root: str | Path,
    verify_bound_audio: bool = False,
    verify_current_commit: bool = False,
) -> RecordedLevelCalibration:
    """외부 SHA와 immutable 입력 결속을 검증해 session gain map을 반환한다."""

    root = Path(repo_root).resolve()
    path = Path(receipt)
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    actual_sha = _sha256_bytes(raw)
    expected = str(expected_sha256 or "").lower()
    if len(expected) != 64 or expected != actual_sha:
        raise RecordedLevelCalibrationError(
            "recorded level calibration receipt 외부 SHA가 없거나 다릅니다: "
            f"configured={expected!r}, actual={actual_sha}"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordedLevelCalibrationError("calibration receipt JSON 오류") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise RecordedLevelCalibrationError("calibration receipt schema가 다릅니다")
    commit = str(payload.get("source_commit") or "").lower()
    if not _COMMIT_RE.fullmatch(commit) or payload.get("source_tree_clean_at_issue") is not True:
        raise RecordedLevelCalibrationError("calibration source commit/clean evidence가 없습니다")
    clean_source = _validate_clean_source_evidence(
        payload.get("clean_source"),
        source_commit=commit,
    )
    if verify_current_commit:
        try:
            current_source = exact_clean_source_evidence(
                root,
                expected_commit=commit,
                reject_runtime_bytecode=True,
            )
        except SourceTrustError as exc:
            raise RecordedLevelCalibrationError(
                f"calibration 현재 clean exact source 재검증 실패: {exc}"
            ) from exc
        if clean_source != current_source:
            raise RecordedLevelCalibrationError(
                "calibration clean_source evidence가 현재 exact source와 다릅니다"
            )
    if payload.get("welch_recipe") != WELCH_RECIPE:
        raise RecordedLevelCalibrationError("calibration Welch recipe가 exact하지 않습니다")
    shape = payload.get("plant_shape_diagnostic")
    if (
        not isinstance(shape, dict)
        or shape.get("required_ablation_domains")
        != [HISTORICAL_DOMAIN, CURRENT_DOMAIN]
        or shape.get("fit_scope") != "train_split_only_shape_diagnostic"
        or not isinstance(shape.get("relative_error_after_scalar_and_delay"), (int, float))
        or not math.isfinite(float(shape["relative_error_after_scalar_and_delay"]))
        or float(shape["relative_error_after_scalar_and_delay"]) < 0.0
    ):
        raise RecordedLevelCalibrationError(
            "scalar calibration 한계/plant-domain ablation 진단이 없습니다"
        )
    quality = payload.get("quality_gate")
    if (
        not isinstance(quality, dict)
        or quality.get("thresholds") != CALIBRATION_QUALITY_CONTRACT
        or quality.get("pass") is not True
        or quality.get("threshold_policy") != "predeclared_not_result_tuned"
        or not isinstance(quality.get("observed"), dict)
    ):
        raise RecordedLevelCalibrationError("calibration 사전 고정 품질 gate가 없습니다")
    quality_observed = quality["observed"]
    quality_checks = (
        float(quality_observed.get("heldout_split_median_max_abs_db", math.inf))
        <= CALIBRATION_QUALITY_CONTRACT["heldout_split_median_max_abs_db"],
        float(quality_observed.get("all_session_residual_max_abs_db", math.inf))
        <= CALIBRATION_QUALITY_CONTRACT["all_session_residual_max_abs_db"],
        float(quality_observed.get("train_complex_agreement", -math.inf))
        >= CALIBRATION_QUALITY_CONTRACT["train_complex_agreement_min"],
        float(quality_observed.get("train_complex_relative_error", math.inf))
        <= CALIBRATION_QUALITY_CONTRACT["train_complex_relative_error_max"],
        float(quality_observed.get("calibrated_err_abs_peak", math.inf))
        <= CALIBRATION_QUALITY_CONTRACT["calibrated_err_abs_peak_max"],
    )
    if not all(quality_checks):
        raise RecordedLevelCalibrationError("calibration 품질 수치가 고정 threshold를 넘습니다")
    if (
        payload.get("reference_mode") != "digital"
        or payload.get("apply_to") != ["ERR", "d"]
        or payload.get("fit_policy")
        != {
            "allowed_split": "train",
            "heldout_splits_are_diagnostics_only": ["val", "test"],
            "wav_mutation": False,
        }
    ):
        raise RecordedLevelCalibrationError("calibration 적용/fit 정책이 다릅니다")

    def validate_ref(ref: object, label: str, *, verify: bool) -> None:
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "size"}:
            raise RecordedLevelCalibrationError(f"{label} file ref가 exact하지 않습니다")
        relative_value = ref["path"]
        digest = ref["sha256"]
        size = ref["size"]
        if (
            not isinstance(relative_value, str)
            or not relative_value
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise RecordedLevelCalibrationError(
                f"{label} path/size/SHA 형식이 유효하지 않습니다"
            )
        relative = relative_value
        candidate = (root / relative).resolve()
        if _repo_relative(candidate, root) != relative:
            raise RecordedLevelCalibrationError(f"{label} path가 canonical하지 않습니다")
        if verify:
            actual = _file_ref(candidate, repo_root=root, label=label)
            if actual != ref:
                raise RecordedLevelCalibrationError(f"{label} path/size/SHA가 변경됐습니다")

    validate_ref(payload.get("implementation_source"), "calibration implementation", verify=True)
    analysis_contract = payload.get("analysis_contract")
    if not isinstance(analysis_contract, dict):
        raise RecordedLevelCalibrationError("analysis contract identity가 없습니다")
    actual_analysis_sha = _sha256_bytes(_canonical_json(analysis_contract))
    if payload.get("analysis_contract_sha256") != actual_analysis_sha:
        raise RecordedLevelCalibrationError("analysis contract SHA가 다릅니다")
    if analysis_contract.get("implementation_sha256") != payload["implementation_source"]["sha256"]:
        raise RecordedLevelCalibrationError("analysis contract와 implementation SHA가 다릅니다")

    validate_ref(payload.get("recorded_manifest"), "recorded manifest", verify=True)
    validate_ref(payload.get("strict_primary_npz"), "strict primary", verify=True)
    sessions = payload.get("sessions")
    cohorts = payload.get("cohorts")
    if not isinstance(sessions, list) or len(sessions) != 82 or not isinstance(cohorts, dict):
        raise RecordedLevelCalibrationError("calibration sessions/cohorts가 불완전합니다")
    session_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(sessions):
        if not isinstance(item, dict):
            raise RecordedLevelCalibrationError(f"sessions[{index}]가 mapping이 아닙니다")
        session_id = str(item.get("session_id") or "")
        if not session_id or session_id in session_by_id:
            raise RecordedLevelCalibrationError("calibration session ID가 비었거나 중복입니다")
        if item.get("split") not in {"train", "val", "test"}:
            raise RecordedLevelCalibrationError(f"{session_id}: split이 잘못됐습니다")
        if item.get("plant_domain") != HISTORICAL_DOMAIN:
            raise RecordedLevelCalibrationError(f"{session_id}: plant domain이 잘못됐습니다")
        if item.get("cohort") != _cohort(session_id):
            raise RecordedLevelCalibrationError(
                f"{session_id}: timestamp cohort와 receipt cohort가 다릅니다"
            )
        for peak_key in ("raw_err_abs_peak", "calibrated_err_abs_peak"):
            peak = item.get(peak_key)
            if (
                isinstance(peak, bool)
                or not isinstance(peak, (int, float))
                or not math.isfinite(float(peak))
                or float(peak) < 0.0
            ):
                raise RecordedLevelCalibrationError(
                    f"{session_id}: {peak_key}가 유효하지 않습니다"
                )
        validate_ref(
            item.get("source_aligned"),
            f"{session_id} source_aligned",
            verify=verify_bound_audio,
        )
        validate_ref(item.get("mics"), f"{session_id} mics", verify=verify_bound_audio)
        session_by_id[session_id] = item

    gains: dict[str, float] = {}
    domains: dict[str, str] = {}
    expected_cohorts = set(EXPECTED_HISTORICAL_COHORTS.values())
    if set(cohorts) != expected_cohorts:
        raise RecordedLevelCalibrationError("historical cohort 집합이 exact하지 않습니다")
    for cohort, info in cohorts.items():
        if not isinstance(info, dict) or info.get("fit_split") != "train":
            raise RecordedLevelCalibrationError(f"{cohort}: train-only fit이 아닙니다")
        members = sorted(
            sid for sid, item in session_by_id.items() if item.get("cohort") == cohort
        )
        train_ids = sorted(
            sid for sid in members if session_by_id[sid].get("split") == "train"
        )
        if info.get("member_session_ids") != members or info.get("fit_session_ids") != train_ids:
            raise RecordedLevelCalibrationError(
                f"{cohort}: member/fit ID가 receipt sessions의 train split과 다릅니다"
            )
        if info.get("train_fit_count") != len(train_ids):
            raise RecordedLevelCalibrationError(f"{cohort}: train fit count가 다릅니다")
        if any(session_by_id[sid]["split"] != "train" for sid in info["fit_session_ids"]):
            raise RecordedLevelCalibrationError(f"{cohort}: val/test가 fit에 유입됐습니다")
        gain = info.get("err_amplitude_gain")
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(float(gain)) or float(gain) <= 0.0:
            raise RecordedLevelCalibrationError(f"{cohort}: ERR gain이 유효하지 않습니다")
        fitted = info.get("fitted_observed_to_strict_power_ratio_db")
        expected_gain = 10.0 ** (-float(fitted) / 20.0)
        if not math.isclose(float(gain), expected_gain, rel_tol=1e-12, abs_tol=1e-12):
            raise RecordedLevelCalibrationError(f"{cohort}: dB와 amplitude gain이 다릅니다")
        for sid in members:
            expected_peak = float(session_by_id[sid]["raw_err_abs_peak"]) * float(gain)
            calibrated_peak = float(session_by_id[sid]["calibrated_err_abs_peak"])
            if not math.isclose(
                calibrated_peak, expected_peak, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise RecordedLevelCalibrationError(
                    f"{sid}: calibrated ERR peak가 receipt gain과 다릅니다"
                )
            if calibrated_peak > CALIBRATION_QUALITY_CONTRACT[
                "calibrated_err_abs_peak_max"
            ]:
                raise RecordedLevelCalibrationError(
                    f"{sid}: calibrated ERR peak가 0.8 안전 상한 밖입니다"
                )
            gains[sid] = float(gain)
            domains[sid] = HISTORICAL_DOMAIN
        residual_by_split = {
            split: [
                float(session_by_id[sid]["observed_to_strict_power_ratio_db"])
                - float(fitted)
                for sid in members
                if session_by_id[sid]["split"] == split
            ]
            for split in ("train", "val", "test")
        }
        for split in ("val", "test"):
            values = residual_by_split[split]
            if not values or abs(float(np.median(values))) > 1.0:
                raise RecordedLevelCalibrationError(
                    f"{cohort} {split}: heldout median residual이 ±1 dB 밖입니다"
                )
        if max(abs(value) for values in residual_by_split.values() for value in values) > 6.0:
            raise RecordedLevelCalibrationError(
                f"{cohort}: session residual이 보수적 6 dB 상한 밖입니다"
            )
    if set(gains) != set(session_by_id):
        raise RecordedLevelCalibrationError("모든 historical session에 gain이 배정되지 않았습니다")
    return RecordedLevelCalibration(
        path=path.resolve(),
        sha256=actual_sha,
        payload=payload,
        err_gain_by_session=gains,
        plant_domain_by_session=domains,
    )


def require_recorded_level_calibration_config(
    cfg: dict[str, Any], *, repo_root: str | Path
) -> RecordedLevelCalibration | None:
    """recorded_ratio>0인 공식 measured 실행을 run-dir 전에 차단한다."""

    if str(cfg.get("experiment_role") or "") not in {
        "measured_probe",
        "canonical_finetune",
    }:
        return None
    if float(cfg.get("recorded_ratio") or 0.0) <= 0.0:
        return None
    data = cfg.get("data")
    if not isinstance(data, dict):
        raise RecordedLevelCalibrationError("recorded calibration용 data config가 없습니다")
    if str(data.get("reference_mode") or "") != "digital":
        raise RecordedLevelCalibrationError(
            "recorded level calibration은 digital-reference에서만 허용됩니다"
        )
    path = data.get("recorded_level_calibration")
    sha = data.get("recorded_level_calibration_sha256")
    if not path or not sha:
        raise RecordedLevelCalibrationError(
            "recorded_ratio>0 measured 실행에는 recorded_level_calibration path와 "
            "외부 SHA가 필요합니다"
        )
    return validate_recorded_level_calibration_receipt(
        path,
        expected_sha256=str(sha),
        repo_root=repo_root,
        verify_bound_audio=False,
        verify_current_commit=True,
    )
