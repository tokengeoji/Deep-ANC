"""v5 raw를 분석하는 fixture-only offline causal P/S publisher.

실제 live raw authority가 아직 없으므로 이 모듈이 발행하는 envelope는 항상
``canonical_training_eligible=false``다. 오디오 장치를 열지 않는다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import LinearOperator, lsmr

from deep_anc.data.atomic_publish import publish_directory_noreplace
from .fullband_causal_v5 import (
    CONDITION_AUDIT_SUPPORT,
    FS,
    PERIOD,
    ROLES,
    _array_sha256,
    _payload_sha256,
    estimate_common_clock_from_waveforms_v5,
    exact_condition_audit_v5,
    score_candidate_on_role_v5,
)
from .timing import PlantDelays, TrainingTimingContract

ANALYSIS_SCHEMA = "fullband_causal_v5_offline_analysis_v1"
OPERATOR_SCHEMA = "fullband_causal_joint_operator_v5_fixture_only"
AUTHORITY_SCHEMA = "fullband_causal_training_authority_v5_fixture_only"
MAX_FINITE_SUPPORT_UNEXPLAINED_ENERGY_RATIO = 1.0e-4
MAX_TAP_DISAGREEMENT = 0.10
MAX_LINEAR_CUBIC_CLOCK_DIFFERENCE_SAMPLES = 0.006


def _canonical_raw_npz_bytes(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> bytes:
    metadata_bytes = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    stream = io.BytesIO()
    np.savez(
        stream,
        submitted_pcm=arrays["submitted_pcm"],
        captured_pcm=arrays["captured_pcm"],
        callback_frames=arrays["callback_frames"],
        metadata_json_utf8=metadata_bytes,
    )
    return stream.getvalue()


def load_raw_npz_v5(
    path: Path, plan: Mapping[str, Any], *, repository_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """v5 raw NPZ exact key, metadata, 배열 SHA를 다시 검증한다."""

    root = Path(os.path.abspath(repository_root))
    relative = Path(str(plan["publisher_contract"]["raw_session_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("sealed raw path가 lexical repository relative path가 아닙니다")
    expected_path = Path(os.path.abspath(root / relative))
    actual_path = Path(os.path.abspath(path))
    if actual_path != expected_path:
        raise ValueError("raw path가 plan sealed relative path와 다릅니다")
    cursor = root
    if cursor.is_symlink():
        raise ValueError("repository root symlink를 거부합니다")
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"raw path symlink를 거부합니다: {cursor}")
    raw_bytes = actual_path.read_bytes()
    raw_file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
        expected = {"submitted_pcm", "captured_pcm", "callback_frames", "metadata_json_utf8"}
        if set(archive.files) != expected:
            raise ValueError("v5 raw NPZ key 집합이 exact하지 않습니다")
        arrays = {key: np.asarray(archive[key]) for key in expected - {"metadata_json_utf8"}}
        metadata = json.loads(bytes(archive["metadata_json_utf8"]).decode("utf-8"))
    metadata_keys = {
        "schema", "signal_plan_payload_sha256", "submitted_pcm_sha256",
        "captured_pcm_sha256", "callback_frames_sha256", "live_authority_at_plan_time",
        "role", "callback_semantics", "live_xrun_slip_authority",
    }
    if set(metadata) != metadata_keys or metadata.get("schema") != "fullband_causal_raw_capture_v5":
        raise ValueError("v5 raw metadata schema/key 집합이 exact하지 않습니다")
    publisher = plan["publisher_contract"]
    for key in ("role", "callback_semantics", "live_xrun_slip_authority", "live_authority_at_plan_time"):
        if metadata.get(key) != publisher.get(key):
            raise ValueError(f"raw metadata {key}가 plan publisher contract와 다릅니다")
    if metadata.get("live_authority_at_plan_time") is not None:
        raise ValueError("live_authority_at_plan_time은 None이어야 합니다")
    checks = {
        "submitted_pcm_sha256": _array_sha256(arrays["submitted_pcm"]),
        "captured_pcm_sha256": _array_sha256(arrays["captured_pcm"]),
        "callback_frames_sha256": _array_sha256(arrays["callback_frames"]),
    }
    for key, actual in checks.items():
        if metadata.get(key) != actual:
            raise ValueError(f"raw metadata {key} 재계산이 다릅니다")
    if metadata.get("signal_plan_payload_sha256") != plan.get("canonical_payload_sha256"):
        raise ValueError("raw/plan payload SHA가 다릅니다")
    if raw_bytes != _canonical_raw_npz_bytes(arrays, metadata):
        raise ValueError("raw NPZ가 canonical writer bytes와 다릅니다(repackage 거부)")
    return arrays, {**metadata, "raw_file_sha256": raw_file_sha256}


def _role_period(plan: Mapping[str, Any], value: np.ndarray, path: str, role: str) -> np.ndarray:
    row = next(row for row in plan["layout"] if row.get("path") == path and row.get("role") == role)
    return np.asarray(value[row["central_start_frame"] : row["central_stop_frame"]])


def _resample_to_dac(captured: np.ndarray, ratio: float, kind: str) -> np.ndarray:
    q = np.arange(len(captured), dtype=np.float64) / float(ratio)
    valid = q <= len(captured) - 1
    result = np.zeros((int(np.sum(valid)), 2), dtype=np.float64)
    source = np.arange(len(captured), dtype=np.float64)
    for mic in range(2):
        if kind == "linear":
            result[:, mic] = np.interp(q[valid], source, captured[:, mic])
        elif kind == "cubic":
            result[:, mic] = CubicSpline(source, captured[:, mic])(q[valid])
        else:
            raise ValueError("resample kind가 잘못됐습니다")
    return result


def fit_joint_role_v5(
    *, plan: Mapping[str, Any], submitted_pcm: np.ndarray, captured_dac: np.ndarray, role: str
) -> dict[str, Any]:
    """두 time-separated row의 actual two-input spectrum으로 joint LS를 푼다."""

    if role not in ("fit_a", "fit_b"):
        raise ValueError("operator fit은 fit_a/fit_b만 허용합니다")
    x_rows = np.stack(
        [_role_period(plan, submitted_pcm, path, role) for path in ("primary", "secondary")]
    ).astype(np.float64)
    y_rows = np.stack(
        [_role_period(plan, captured_dac, path, role) for path in ("primary", "secondary")]
    ).astype(np.float64)
    x_fft = np.fft.rfft(x_rows, axis=1)
    support = CONDITION_AUDIT_SUPPORT

    def matvec(value: np.ndarray) -> np.ndarray:
        taps = np.asarray(value).reshape(2, support)
        transfer = np.fft.rfft(taps, n=PERIOD, axis=1)
        prediction = np.fft.irfft(
            x_fft[:, :, 0] * transfer[0][None, :]
            + x_fft[:, :, 1] * transfer[1][None, :],
            n=PERIOD,
            axis=1,
        )
        return prediction.reshape(-1)

    def rmatvec(value: np.ndarray) -> np.ndarray:
        residual = np.asarray(value).reshape(2, PERIOD)
        residual_fft = np.fft.rfft(residual, axis=1)
        return np.concatenate(
            [
                np.fft.irfft(
                    np.sum(np.conj(x_fft[:, :, channel]) * residual_fft, axis=0),
                    n=PERIOD,
                )[:support]
                for channel in range(2)
            ]
        )

    operator = LinearOperator(
        (2 * PERIOD, 2 * support), matvec=matvec, rmatvec=rmatvec, dtype=np.float64
    )
    taps = np.zeros((2, 2, support), dtype=np.float64)
    residual_ratios: list[float] = []
    for mic in range(2):
        target = y_rows[:, :, mic].reshape(-1)
        solution = lsmr(operator, target, atol=1e-10, btol=1e-10, maxiter=800)[0]
        taps[mic] = solution.reshape(2, support)
        residual_ratios.append(
            float(np.linalg.norm(target - operator.matvec(solution)) / max(np.linalg.norm(target), 1e-30))
        )
    unexplained_ratio = max(residual_ratios) ** 2
    return {
        "role": role,
        "primary_fir_by_mic": taps[:, 0, :],
        "secondary_fir_by_mic": taps[:, 1, :],
        "maximum_finite_support_unexplained_energy_ratio": float(unexplained_ratio),
        "finite_support_unexplained_energy_passed": bool(
            unexplained_ratio <= MAX_FINITE_SUPPORT_UNEXPLAINED_ENERGY_RATIO
        ),
    }


def analyze_v5_raw_arrays(
    *,
    plan: Mapping[str, Any],
    submitted_pcm: np.ndarray,
    captured_adc_pcm: np.ndarray,
    callback_frames: np.ndarray,
    synthetic_fixture: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    submitted = np.asarray(submitted_pcm)
    captured_adc = np.asarray(captured_adc_pcm)
    callbacks = np.asarray(callback_frames)
    if not synthetic_fixture:
        raise ValueError(
            "actual live raw는 coarse/fractional marker scan과 pre-roll compact-window fit이 "
            "미구현되어 fixture-only analyzer로 발행할 수 없습니다"
        )
    if submitted.dtype != np.int16 or submitted.ndim != 2 or submitted.shape[1] != 2:
        raise ValueError("submitted는 exact int16 [frame,P_S=2]여야 합니다")
    if captured_adc.ndim != 2 or captured_adc.shape[1] != 2 or captured_adc.dtype.kind not in "if":
        raise ValueError("captured는 numeric [frame,ERR_REF=2]여야 합니다")
    if not np.all(np.isfinite(captured_adc)):
        raise ValueError("captured에 non-finite 값이 있습니다")
    if _array_sha256(submitted) != plan.get("actual_submitted_pcm_sha256"):
        raise ValueError("plan/submitted PCM SHA가 다릅니다")
    expected_callbacks = math.ceil(len(captured_adc) / 256)
    if (
        callbacks.ndim != 1
        or callbacks.dtype != np.dtype("<i8")
        or len(callbacks) != expected_callbacks
        or np.any(callbacks != 256)
        or not (0 <= int(np.sum(callbacks)) - len(captured_adc) < 256)
    ):
        raise ValueError("callback frame accounting이 exact 256-block 계약과 다릅니다")
    if int(np.sum(callbacks)) < len(captured_adc):
        raise ValueError("callback frame 합이 raw capture보다 짧습니다")
    condition = exact_condition_audit_v5(plan, submitted)
    if not condition["passed"]:
        raise ValueError("support1024 exact condition gate 실패")
    clock = estimate_common_clock_from_waveforms_v5(
        plan=plan, submitted_pcm=submitted, captured_adc_pcm=captured_adc
    )
    if not clock["passed"]:
        raise ValueError("common-q/change-point gate 실패")
    ratio = float(clock["estimated_rate_ratio"])
    linear_clock = estimate_common_clock_from_waveforms_v5(
        plan=plan,
        submitted_pcm=submitted,
        captured_adc_pcm=captured_adc,
        interpolation_kind="linear",
    )
    interpolation_difference = abs(
        float(linear_clock["estimated_rate_ratio"]) - ratio
    ) * len(submitted)
    if (
        not linear_clock["passed"]
        or interpolation_difference > MAX_LINEAR_CUBIC_CLOCK_DIFFERENCE_SAMPLES
    ):
        raise ValueError("linear/cubic clock-q interpolation sensitivity gate 실패")
    cubic = _resample_to_dac(captured_adc.astype(np.float64), ratio, "cubic")
    if len(cubic) < len(submitted):
        captured_dac = np.pad(cubic, ((0, len(submitted) - len(cubic)), (0, 0)))
    else:
        captured_dac = cubic[: len(submitted)]
    fits = {
        role: fit_joint_role_v5(
            plan=plan, submitted_pcm=submitted, captured_dac=captured_dac, role=role
        )
        for role in ("fit_a", "fit_b")
    }
    if not all(value["finite_support_unexplained_energy_passed"] for value in fits.values()):
        raise ValueError("support1024 finite-support unexplained-energy gate 실패")
    p_a, p_b = fits["fit_a"]["primary_fir_by_mic"], fits["fit_b"]["primary_fir_by_mic"]
    s_a, s_b = fits["fit_a"]["secondary_fir_by_mic"], fits["fit_b"]["secondary_fir_by_mic"]
    disagreement = max(
        float(np.linalg.norm(p_a - p_b) / max(np.linalg.norm(p_a), 1e-30)),
        float(np.linalg.norm(s_a - s_b) / max(np.linalg.norm(s_a), 1e-30)),
    )
    if disagreement > MAX_TAP_DISAGREEMENT:
        raise ValueError("fit_a/fit_b tap stationarity gate 실패")
    primary = (p_a + p_b) * 0.5
    secondary = (s_a + s_b) * 0.5
    scores = {
        role: score_candidate_on_role_v5(
            plan=plan,
            submitted_pcm=submitted,
            captured_pcm=captured_dac,
            primary_fir_by_mic=primary,
            secondary_fir_by_mic=secondary,
            role=role,
        )
        for role in ROLES
    }
    if not all(score["all_paths_microphones_subbands_passed"] for score in scores.values()):
        raise ValueError("P/S×ERR/REF×8대역 fit/cross/terminal holdout gate 실패")
    primary_delay = int(np.argmax(np.abs(primary[0])))
    secondary_delay = int(np.argmax(np.abs(secondary[0])))
    plant_delays = PlantDelays(
        primary_delay_samples=0,
        secondary_delay_samples=0,
        handoff_samples=256,
        sample_rate=FS,
    )
    timing = TrainingTimingContract.derive(
        primary_fir=primary[0], plant_delays=plant_delays
    )
    operator = {
        "primary_fir_by_mic": primary.astype("<f8"),
        "secondary_fir_by_mic": secondary.astype("<f8"),
        "primary_coarse_delay_samples": np.asarray(0, dtype="<i8"),
        "secondary_coarse_delay_samples": np.asarray(0, dtype="<i8"),
        "primary_fractional_delay_samples": np.asarray(0.0, dtype="<f8"),
        "secondary_fractional_delay_samples": np.asarray(0.0, dtype="<f8"),
        "support_samples": np.asarray(CONDITION_AUDIT_SUPPORT, dtype="<i8"),
    }
    analysis: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "status": "FIXTURE_PASS_NOT_LIVE_AUTHORITY" if synthetic_fixture else "BLOCKED_UNREVIEWED_LIVE_RAW",
        "canonical_training_eligible": False,
        "synthetic_fixture": bool(synthetic_fixture),
        "signal_plan_payload_sha256": plan["canonical_payload_sha256"],
        "submitted_pcm_sha256": _array_sha256(submitted),
        "raw_captured_sha256": _array_sha256(captured_adc),
        "raw_container_bound": False,
        "raw_path_relative": None,
        "raw_file_sha256": None,
        "callback_frames_sha256": _array_sha256(callbacks),
        "callback_frames_semantics": "frame_accounting_only_not_xrun_or_slip_authority",
        "live_xrun_slip_authority_available": False,
        "condition_receipt": condition,
        "clock_receipt": clock,
        "linear_interpolation_clock_receipt": linear_clock,
        "linear_cubic_interpolation_sensitivity_samples": interpolation_difference,
        "fit_stationarity_max_relative_tap_difference": disagreement,
        "fit_scores": {role: scores[role] for role in ("fit_a", "fit_b")},
        "terminal_holdout_score": scores["holdout"],
        "holdout_used_for_fit_or_selection": False,
        "plant_delays": plant_delays.model_dump(mode="json"),
        "plant_delays_sha256": _payload_sha256(plant_delays.model_dump(mode="json")),
        "training_timing_contract": timing.model_dump(mode="json"),
        "training_timing_contract_sha256": timing.digest(),
        "fixture_timing_diagnostic": {
            "primary_err_peak_inside_unshifted_fir": primary_delay,
            "secondary_err_peak_inside_unshifted_fir": secondary_delay,
            "zeros_before_fir": 0,
            "all_path_mic_pre_peak_taps_preserved": True,
            "live_coarse_fractional_marker_scan_available": False,
        },
        "live_authority": None,
    }
    analysis["analysis_payload_sha256"] = _payload_sha256(analysis)
    return analysis, operator


def analyze_v5_raw_file(
    *, plan: Mapping[str, Any], raw_path: Path, repository_root: Path, synthetic_fixture: bool
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """동일 raw bytes에서 container SHA와 arrays를 함께 유도한 유일한 publish 경로."""

    arrays, identity = load_raw_npz_v5(raw_path, plan, repository_root=repository_root)
    analysis, operator = analyze_v5_raw_arrays(
        plan=plan,
        submitted_pcm=arrays["submitted_pcm"],
        captured_adc_pcm=arrays["captured_pcm"],
        callback_frames=arrays["callback_frames"],
        synthetic_fixture=synthetic_fixture,
    )
    analysis.pop("analysis_payload_sha256")
    analysis["raw_container_bound"] = True
    analysis["raw_path_relative"] = plan["publisher_contract"]["raw_session_relative_path"]
    analysis["raw_file_sha256"] = identity["raw_file_sha256"]
    analysis["analysis_payload_sha256"] = _payload_sha256(analysis)
    return analysis, operator


def publish_fixture_analysis_v5(
    *, target_directory: Path, analysis: Mapping[str, Any], operator: Mapping[str, np.ndarray]
) -> Path:
    """analysis/operator를 sibling staging 뒤 atomic no-replace로 함께 발행한다."""

    if analysis.get("raw_container_bound") is not True or not analysis.get("raw_file_sha256"):
        raise ValueError("raw container에 결속되지 않은 array-only analysis는 발행할 수 없습니다")
    target = Path(os.path.abspath(target_directory))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"analysis target을 덮어쓸 수 없습니다: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        operator_path = staging / "operator.npz"
        with operator_path.open("xb") as handle:
            np.savez(handle, **operator)
            handle.flush()
            os.fsync(handle.fileno())
        operator_sha = hashlib.sha256(operator_path.read_bytes()).hexdigest()
        authority = {
            "schema": AUTHORITY_SCHEMA,
            "authority": None,
            "status": analysis["status"],
            "canonical_training_eligible": False,
            "analysis": dict(analysis),
            "operator_schema": OPERATOR_SCHEMA,
            "operator_npz_path": "operator.npz",
            "operator_npz_file_sha256": operator_sha,
        }
        authority["evidence_sha256"] = _payload_sha256(authority)
        authority_path = staging / "authority.json"
        with authority_path.open("x", encoding="utf-8") as handle:
            json.dump(authority, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return publish_directory_noreplace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
