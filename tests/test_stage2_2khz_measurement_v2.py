from __future__ import annotations

from functools import lru_cache
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from deep_anc.dsp.broadband_plant_analysis import (
    exact_two_input_periodic_gram_audit,
)
from deep_anc.dsp.stage2_2khz_measurement import (
    PLAYBACK_ROLES,
    ROLE_SEEDS,
    _band_equalized_aperiodic_code,
)
from deep_anc.dsp.stage2_2khz_measurement_v2 import (
    DPSS_APPLIED_LOWER_GUARD_HZ,
    DPSS_REPRESENTATION_BAND_HZ,
    DPSS_REPRESENTATION_GUARD_HZ,
    GRAM_DIMENSION,
    LIVE_SAFETY_STATUS,
    MAX_GRAM_CONDITION,
    MAX_MODEL_ACTUATOR_ABS,
    MAX_SUBMITTED_PEAK_PCM,
    Stage2HoldoutAccessError,
    Stage2HoldoutAccessLedger,
    audit_stage2_v2_live_safe_dpss_gram,
    audit_stage2_v2_shifted_gram,
    build_stage2_bandlimited_dpss_basis,
    build_stage2_v2_live_safe_fallback_plan,
    build_stage2_v2_signal_plan,
    validate_stage2_v2_signal_plan,
)
from deep_anc.train.stage2_2khz_execution import require_stage2_actuator_limit
from deep_anc.dsp.stage2_2khz_analysis_v2 import (
    _actuator_feasibility,
    _holdout_prediction_metrics,
)
from deep_anc.dsp.stage2_2khz_level_contract import (
    build_stage2_physical_operating_level_evidence,
    validate_stage2_physical_operating_level_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _canonical() -> tuple[dict, np.ndarray]:
    plan, pcm = build_stage2_v2_signal_plan()
    return plan, pcm.copy()


@lru_cache(maxsize=1)
def _gram() -> dict:
    plan, pcm = _canonical()
    return audit_stage2_v2_shifted_gram(
        plan, pcm, zeros_by_path=(1_386, 1_245)
    )


def test_exact_24_second_layout_seed_sha_peak_and_live_block() -> None:
    plan, pcm = _canonical()
    assert pcm.dtype == np.int16
    assert pcm.shape == (1_152_000, 2)
    assert plan["signal_seconds"] == 24.0
    assert plan["signal_frames"] == 1_152_000
    assert plan["actual_submitted_pcm"]["peak_pcm"] <= MAX_SUBMITTED_PEAK_PCM
    assert plan["actual_submitted_pcm"]["maximum_normalized_abs"] == (
        MAX_MODEL_ACTUATOR_ABS
    )
    constants = plan["layout_constants"]
    assert constants == {
        "zero_pre_frames": 32_768,
        "pilot_lead_frames": 65_536,
        "pe_slot_count": 6,
        "pe_slot_frames": 65_536,
        "diagnostic_slot_count": 8,
        "diagnostic_slot_frames": 65_536,
        "pilot_tail_frames": 98_304,
        "zero_post_frames": 37_888,
    }
    pe_rows = [row for row in plan["layout"] if row["kind"] == "pe_slot"]
    assert len(pe_rows) == 6
    assert len({row["seed"] for row in pe_rows}) == 6
    assert all(len(row["payload_sha256"]) == 64 for row in pe_rows)
    assert plan["pilot"]["disjoint"] is True
    assert plan["pilot"][
        "time_separated_main_pe_with_continuous_dual_disjoint_pilot"
    ] is True
    assert plan["live_safety"]["status"] == LIVE_SAFETY_STATUS
    assert plan["live_safety"]["audio_execution_allowed_by_this_plan"] is False
    validate_stage2_v2_signal_plan(plan, pcm)
    mutated = pcm.copy()
    mutated[100_000, 0] ^= np.int16(1)
    with pytest.raises(ValueError, match="bytes"):
        validate_stage2_v2_signal_plan(plan, mutated)


def test_actual_int16_shifted_2048_gram_rank_condition_and_crosscheck() -> None:
    receipt = _gram()
    assert receipt["gram_dimension"] == GRAM_DIMENSION == 2_048
    assert receipt["numeric_rank"] == GRAM_DIMENSION
    assert receipt["full_numeric_rank"] is True
    assert receipt["periodic_normal_matrix_gram_condition_number"] <= (
        MAX_GRAM_CONDITION
    )
    assert max(receipt["role_condition_numbers"].values()) <= MAX_GRAM_CONDITION
    assert receipt["quadratic_form_relative_error"] <= 1.0e-10
    assert receipt["normal_vector_relative_error"] <= 1.0e-10
    assert receipt["crosscheck_passed"] is True
    assert receipt["holdout_accessed"] is False
    assert receipt["passed"] is True


def test_diagnostic_slots_have_exact_zero_guards_channel_and_two_x_levels() -> None:
    plan, pcm = _canonical()
    rows = plan["nonlinearity_diagnostics"]["slots"]
    assert len(rows) == 8
    grouped: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    for row in rows:
        active = 0 if row["path"] == "primary" else 1
        inactive = 1 - active
        slot = pcm[row["start_frame"] : row["stop_frame"]]
        assert np.count_nonzero(slot[:, inactive]) == 0
        assert np.count_nonzero(
            pcm[row["guard_lead_start_frame"] : row["guard_lead_stop_frame"]]
        ) == 0
        assert np.count_nonzero(
            pcm[row["guard_tail_start_frame"] : row["guard_tail_stop_frame"]]
        ) == 0
        assert row["analysis_frames"] == 24_000
        assert row["analysis_fft_bin_spacing_hz"] == 2.0
        assert row["fundamental_frequencies_hz"] in ([752, 1248], [1800, 2200])
        assert row["clock_pilot_present"] is False
        active_value = pcm[
            row["active_start_frame"] : row["active_stop_frame"], active
        ]
        grouped.setdefault((row["path"], row["pair_index"]), {})[
            row["level_pcm"]
        ] = active_value
    for levels in grouped.values():
        assert np.array_equal(levels[98].astype(np.int32), 2 * levels[49].astype(np.int32))
        assert int(np.max(np.abs(levels[49].astype(np.int32)))) == 49
        assert int(np.max(np.abs(levels[98].astype(np.int32)))) == 98


def test_old_88_2828_actual_int16_design_fails_unrestricted_1024_condition() -> None:
    period = 32_768
    rows: dict[str, tuple[np.ndarray, ...]] = {}
    for role in ("fit_a", "fit_b"):
        floating = np.column_stack(
            [
                _band_equalized_aperiodic_code(ROLE_SEEDS[role][name], period)
                for name in PLAYBACK_ROLES
            ]
        )
        actual = np.rint(floating * (79.0 / np.max(np.abs(floating)))).astype(
            np.int16
        )
        rows[role] = (actual,)
    receipt = exact_two_input_periodic_gram_audit(
        rows,
        role_order=("fit_a", "fit_b"),
        support_samples=1_024,
        maximum_condition_number=20.0,
    )
    assert receipt["actual_int16_input_required"] is True
    assert receipt["periodic_normal_matrix_gram_condition_number"] > 20.0
    assert receipt["passed"] is False


def test_live_safe_dpss_fallback_is_identifiable_but_not_training_authority() -> None:
    plan, pcm = build_stage2_v2_live_safe_fallback_plan()
    _basis, basis_receipt = build_stage2_bandlimited_dpss_basis()
    assert pcm.shape == (1_152_000, 2)
    assert plan["schema"] == "stage2_2khz_time_separated_lower_guard_dpss_plan_v2"
    assert basis_receipt["schema"] == (
        "stage2_2khz_lower_guard_bandlimited_dpss_basis_v2"
    )
    assert DPSS_REPRESENTATION_GUARD_HZ == 100.0
    assert DPSS_APPLIED_LOWER_GUARD_HZ == 80.0
    assert DPSS_REPRESENTATION_BAND_HZ == (0.0, 2828.4271247462)
    assert basis_receipt["representation_band_hz"] == [0.0, 2828.4271247462]
    assert basis_receipt["upper_representation_extension_hz"] == 0.0
    assert basis_receipt["authority_thresholds_or_excitation_relaxed"] is False
    assert plan["live_safety"]["near_nyquist_targeted_pe_present"] is False
    assert plan["live_safety"]["audio_execution_allowed_by_this_plan"] is False
    assert plan["gram_contract"]["unrestricted_1024tap_fit_allowed"] is False
    receipt = audit_stage2_v2_live_safe_dpss_gram(
        plan, pcm, zeros_by_path=(1_386, 1_245)
    )
    assert receipt["gram_dimension"] == 224
    assert receipt["numeric_rank"] == 224
    assert receipt["projected_normal_matrix_condition_number"] <= 20.0
    assert max(receipt["role_condition_numbers"].values()) <= 20.0
    assert receipt["quadratic_crosscheck_error"] <= 1.0e-10
    assert receipt["numerical_subspace_passed"] is True
    assert receipt["unrestricted_1024tap_authority_claimed"] is False
    assert receipt["training_eval_consumer_basis_binding_implemented"] is False
    assert receipt["canonical_training_eligible"] is False


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"limiter": {}},
        {"limiter": {"limit": 0.2}},
        {"limiter": {"limit": float("nan")}},
        {"limiter": {"limit": True}},
        {"limiter": {"limit": MAX_MODEL_ACTUATOR_ABS / 2.0}},
        {"limiter": {"limit": MAX_MODEL_ACTUATOR_ABS, "other": 1}},
    ],
)
def test_stage2_actuator_admission_fails_closed(config: dict) -> None:
    with pytest.raises(ValueError, match="Stage-2 model"):
        require_stage2_actuator_limit(config)
    assert require_stage2_actuator_limit(
        {"limiter": {"limit": MAX_MODEL_ACTUATOR_ABS}}
    ) == MAX_MODEL_ACTUATOR_ABS


def test_holdout_accessor_rejects_early_access_and_refit_after_access() -> None:
    plan, pcm = _canonical()
    ledger = Stage2HoldoutAccessLedger(plan, pcm)
    with pytest.raises(Stage2HoldoutAccessError, match="freeze 전"):
        ledger.read_holdout_period(path="primary")
    for role in ("fit_a", "fit_b"):
        for path in ("primary", "secondary"):
            period = ledger.read_fit_period(role=role, path=path)
            assert period.shape == (32_768, 2)
            assert period.flags.writeable is False
    ledger.freeze_fit()
    holdout = ledger.read_holdout_period(path="primary")
    assert holdout.shape == (32_768, 2)
    with pytest.raises(Stage2HoldoutAccessError, match="refit"):
        ledger.read_fit_period(role="fit_a", path="primary")
    receipt = ledger.receipt()
    assert receipt["holdout_access_started"] is True
    assert receipt["refit_after_holdout_allowed"] is False
    assert receipt["authority_pass_claimed"] is False


def test_holdout_freeze_requires_every_fit_path() -> None:
    plan, pcm = _canonical()
    ledger = Stage2HoldoutAccessLedger(plan, pcm)
    ledger.read_fit_period(role="fit_a", path="primary")
    with pytest.raises(Stage2HoldoutAccessError, match="모든"):
        ledger.freeze_fit()


def _physical_level_fixture(signal_plan_sha256: str) -> dict:
    names = (
        "meter_raw",
        "meter_receipt",
        "calibration",
        "diagnostic_raw",
        "authorization",
        "ps_raw",
    )
    refs = {
        name: {"path": f"results/fixture/{name}.bin", "sha256": f"{index + 1:064x}"}
        for index, name in enumerate(names)
    }
    return build_stage2_physical_operating_level_evidence(
        signal_plan_sha256=signal_plan_sha256,
        capture_id="synthetic-physical-level-fixture",
        hardware_identity={"fixture": "exact-two-channel-identity"},
        meter_raw_artifact=refs["meter_raw"],
        meter_receipt_artifact=refs["meter_receipt"],
        calibration_evidence_artifact=refs["calibration"],
        diagnostic_raw_artifact=refs["diagnostic_raw"],
        diagnostic_authorization_artifact=refs["authorization"],
        ps_raw_artifact=refs["ps_raw"],
    )


def test_typed_physical_level_rejects_caller_source_peak_bypass() -> None:
    plan, _pcm = _canonical()
    evidence = _physical_level_fixture(plan["canonical_payload_sha256"])
    validated = validate_stage2_physical_operating_level_evidence(evidence)
    assert validated["source_operating_peak_abs"] == 49 / 32768
    forged = dict(evidence)
    forged["source_operating_peak_abs"] = 1.0e-9
    unsigned = {
        key: value for key, value in forged.items() if key != "canonical_payload_sha256"
    }
    forged["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="의미/상수"):
        validate_stage2_physical_operating_level_evidence(forged)


def test_holdout_scale_error_is_not_hidden_by_complex_agreement() -> None:
    observed = np.asarray([1.0 + 1.0j, 0.5 - 0.25j, -0.75j])
    exact = _holdout_prediction_metrics(observed, observed.copy())
    doubled = _holdout_prediction_metrics(observed, 2.0 * observed)
    assert exact["passed"] is True
    assert doubled["complex_agreement"] == pytest.approx(1.0)
    assert doubled["complex_relative_error"] == pytest.approx(1.0)
    assert doubled["magnitude_ratio_error_db"] == pytest.approx(
        20.0 * np.log10(2.0)
    )
    assert doubled["passed"] is False


def test_actuator_feasibility_consumes_typed_level_and_broadband_peak_bound() -> None:
    plan, _pcm = _canonical()
    evidence = _physical_level_fixture(plan["canonical_payload_sha256"])
    primary = np.zeros(1024)
    secondary = np.zeros(1024)
    primary[256] = 0.5
    secondary[256] = 1.0
    receipt = _actuator_feasibility(
        primary, secondary, operating_level_evidence=evidence
    )
    assert receipt["operating_level_evidence_sha256"] == evidence[
        "canonical_payload_sha256"
    ]
    assert receipt["broadband_time_domain_peak_bound"]["passed"] is True
    assert receipt["passed"] is True


def test_diagnostic_failure_publishes_raw_and_never_calls_ps_backend(
    tmp_path: Path,
) -> None:
    # 실제 audio backend 대신 검증된 callback fixture를 주입한다. zero capture는 finite
    # diagnostic FAIL이어야 하고 두 번째 PS capture call은 구조적으로 없어야 한다.
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_audio_duplex_v5 import Backend
    finally:
        sys.path.pop(0)
    from deep_anc.audio_duplex_stage2 import capture_duplex_stage2

    specification = importlib.util.spec_from_file_location(
        "stage2_v2_two_phase_negative", ROOT / "scripts/data/measure_paths_stage2_2khz.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    plan, submitted = build_stage2_v2_live_safe_fallback_plan()
    calls: list[int] = []

    def fixture_capture(_backend, **kwargs):
        frames = len(kwargs["submitted_pcm"])
        calls.append(frames)
        return capture_duplex_stage2(Backend(blocks=frames // 256), **kwargs)

    result = module._run_two_phase_capture(
        repository_root=str(tmp_path),
        plan=plan,
        submitted_pcm=submitted,
        backend=object(),
        devices={"input": 1, "output": 2},
        capture_metadata={"capture_id": "diagnostic-negative-fixture"},
        capture_callable=fixture_capture,
        pre_open_check=None,
    )
    assert result["status"] == "DIAGNOSTIC_BLOCKED_PS_BACKEND_NOT_CALLED"
    assert calls == [plan["live_phase_contract"]["diagnostic_phase_stop_frame"]]
    assert result["diagnostic_receipt"]["passed"] is False
    raw = tmp_path / result["diagnostic_publication"]["path"]
    assert raw.is_file()
    assert not (tmp_path / plan["artifacts"]["ps_phase_raw"]).exists()


def test_ps_backend_only_runs_after_durable_raw_derived_authorization(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_audio_duplex_v5 import Backend
    finally:
        sys.path.pop(0)
    from deep_anc.audio_duplex_stage2 import capture_duplex_stage2

    specification = importlib.util.spec_from_file_location(
        "stage2_v2_two_phase_positive", ROOT / "scripts/data/measure_paths_stage2_2khz.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    plan, submitted = build_stage2_v2_live_safe_fallback_plan()
    calls: list[int] = []

    def linear_fixture_capture(_backend, **kwargs):
        frames = len(kwargs["submitted_pcm"])
        calls.append(frames)
        _, telemetry = capture_duplex_stage2(
            Backend(blocks=frames // 256), **kwargs
        )
        source = kwargs["submitted_pcm"].astype(np.float64) / 32768.0
        captured = np.zeros_like(source)
        for path, delay in enumerate((700, 520)):
            for mic, gain in enumerate(((0.5, 0.3), (0.6, 0.4))[path]):
                captured[delay:, mic] += gain * source[:-delay, path]
        raw = np.clip(
            np.rint(captured * 2147483648.0), -2147483648, 2147483647
        ).astype(np.int32)
        return raw, telemetry

    result = module._run_two_phase_capture(
        repository_root=str(tmp_path),
        plan=plan,
        submitted_pcm=submitted,
        backend=object(),
        devices={"input": 1, "output": 2},
        capture_metadata={"capture_id": "diagnostic-positive-fixture"},
        capture_callable=linear_fixture_capture,
        pre_open_check=None,
    )
    assert result["status"] == "TWO_PHASE_RAW_CAPTURE_PASS_OFFLINE_PS_ANALYSIS_REQUIRED"
    boundary = plan["live_phase_contract"]["diagnostic_phase_stop_frame"]
    assert calls == [boundary, len(submitted) - boundary]
    assert result["diagnostic_receipt"]["passed"] is True
    assert (tmp_path / result["diagnostic_authorization"]["path"]).is_file()
    assert (tmp_path / result["ps_publication"]["path"]).is_file()
    from deep_anc.dsp.stage2_2khz_live_v2 import (
        snapshot_published_stage2_v2_phase,
    )

    # PS raw 자체가 온전해도 embedded authorization bytes가 사라지거나 바뀌면
    # repository-aware offline reload가 fail-closed해야 한다.
    authorization_path = tmp_path / result["diagnostic_authorization"]["path"]
    authorization_path.chmod(0o600)
    authorization_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="authorization"):
        snapshot_published_stage2_v2_phase(
            str(tmp_path),
            result["ps_publication"],
            plan,
            submitted,
            phase="ps",
        )


def test_live_meter_wrapper_rejects_missing_sidecar_even_with_caller_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification = importlib.util.spec_from_file_location(
        "stage2_v2_meter_missing_receipt", ROOT / "scripts/data/measure_paths_stage2_2khz.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    raw = tmp_path / "meter_raw.npz"
    raw.write_bytes(b"self-forged-meter")
    with pytest.raises(FileNotFoundError):
        module._validate_fresh_meter(
            "meter_raw.npz",
            hashlib.sha256(raw.read_bytes()).hexdigest(),
            expected_hardware_identity={},
        )


def test_live_meter_wrapper_propagates_official_601_second_stale_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification = importlib.util.spec_from_file_location(
        "stage2_v2_meter_stale", ROOT / "scripts/data/measure_paths_stage2_2khz.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    raw = tmp_path / "meter_raw.npz"
    raw.write_bytes(b"raw")
    (tmp_path / "meter_raw.receipt.json").write_text("{}", encoding="utf-8")
    calls: list[dict] = []

    def stale(*_args, **kwargs):
        calls.append(kwargs)
        raise ValueError("bootstrap meter raw가 601.0초 지나 freshness 600초를 넘었습니다")

    monkeypatch.setattr(module, "validate_bootstrap_meter_raw", stale)
    with pytest.raises(ValueError, match="601.0초"):
        module._validate_fresh_meter(
            "meter_raw.npz",
            hashlib.sha256(raw.read_bytes()).hexdigest(),
            expected_hardware_identity={"exact": "identity"},
        )
    assert calls == [
        {
            "repository_root": tmp_path,
            "expected_hardware_identity": {"exact": "identity"},
            "require_fresh": True,
        }
    ]
    assert module.BOOTSTRAP_METER_MAX_AGE_SECONDS == 600
