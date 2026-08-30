from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import firwin

from deep_anc.dsp.fullband_causal_v5 import build_plan_v5, synthesize_affine_capture_v5
from deep_anc.dsp.fullband_causal_v5_offline import (
    analyze_v5_raw_file,
    analyze_v5_raw_arrays,
    publish_fixture_analysis_v5,
)


def _fixture() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    plan, submitted = build_plan_v5()
    support = 1024
    base = firwin(129, 12_000, fs=48_000)
    primary = np.zeros((2, support))
    secondary = np.zeros_like(primary)
    primary[0, 10:139] = 0.40 * base
    primary[1, 15:144] = 0.30 * base
    secondary[0, 20:149] = 0.35 * base
    secondary[1, 25:154] = 0.28 * base
    captured = synthesize_affine_capture_v5(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=1.0 + 413.931e-6,
    )
    callbacks = np.full(math.ceil(len(captured) / 256), 256, dtype=np.int64)
    return plan, submitted, captured, callbacks


def test_v5_offline_positive_is_fixture_only_and_atomic(tmp_path: Path) -> None:
    plan, submitted, captured, callbacks = _fixture()
    analysis, operator = analyze_v5_raw_arrays(
        plan=plan,
        submitted_pcm=submitted,
        captured_adc_pcm=captured,
        callback_frames=callbacks,
        synthetic_fixture=True,
    )
    assert analysis["status"] == "FIXTURE_PASS_NOT_LIVE_AUTHORITY"
    assert analysis["canonical_training_eligible"] is False
    assert analysis["live_authority"] is None
    assert analysis["holdout_used_for_fit_or_selection"] is False
    assert analysis["terminal_holdout_score"]["all_paths_microphones_subbands_passed"]
    assert analysis["clock_receipt"]["estimated_ppm"] == pytest.approx(413.931, abs=0.01)
    assert analysis["raw_container_bound"] is False
    assert analysis["live_xrun_slip_authority_available"] is False
    with pytest.raises(ValueError, match="array-only"):
        publish_fixture_analysis_v5(
            target_directory=tmp_path / "unbound", analysis=analysis, operator=operator
        )
    # 실제 positive publish는 canonical raw writer→single-read verified wrapper만 사용한다.
    measure_script = Path("scripts/data/measure_paths_fullband_causal_v5.py")
    spec = importlib.util.spec_from_file_location("v5_measure_for_offline", measure_script)
    assert spec is not None and spec.loader is not None
    measure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measure)
    measure.REPO_ROOT = tmp_path
    if len(captured) < len(submitted):
        captured = np.pad(captured, ((0, len(submitted) - len(captured)), (0, 0)))
    captured_i4 = np.rint(captured * 1_000_000.0).astype("<i4")
    callbacks = np.full(math.ceil(len(captured_i4) / 256), 256, dtype="<i8")
    raw_path = measure._publish_raw_no_replace(
        plan=plan,
        submitted_pcm=submitted,
        captured_pcm=captured_i4,
        callback_frames=callbacks,
    )
    analysis, operator = analyze_v5_raw_file(
        plan=plan, raw_path=raw_path, repository_root=tmp_path, synthetic_fixture=True
    )
    target = publish_fixture_analysis_v5(
        target_directory=tmp_path / "published", analysis=analysis, operator=operator
    )
    authority = json.loads((target / "authority.json").read_text())
    assert authority["authority"] is None
    with pytest.raises(FileExistsError):
        publish_fixture_analysis_v5(
            target_directory=target, analysis=analysis, operator=operator
        )
    with np.load(raw_path, allow_pickle=False) as archive:
        repack = {key: archive[key] for key in archive.files}
    raw_path.unlink()
    np.savez_compressed(raw_path, **repack)
    with pytest.raises(ValueError, match="repackage"):
        analyze_v5_raw_file(
            plan=plan,
            raw_path=raw_path,
            repository_root=tmp_path,
            synthetic_fixture=True,
        )
    wrong_path = tmp_path / "results/fullband_causal_v5/copied.npz"
    wrong_path.write_bytes(raw_path.read_bytes())
    with pytest.raises(ValueError, match="sealed relative path"):
        analyze_v5_raw_file(
            plan=plan,
            raw_path=wrong_path,
            repository_root=tmp_path,
            synthetic_fixture=True,
        )


def test_v5_offline_rejects_tampered_pcm_and_callback_accounting() -> None:
    plan, submitted, captured, callbacks = _fixture()
    tampered = submitted.copy()
    tampered[0, 0] += 1
    with pytest.raises(ValueError, match="SHA"):
        analyze_v5_raw_arrays(
            plan=plan,
            submitted_pcm=tampered,
            captured_adc_pcm=captured,
            callback_frames=callbacks,
            synthetic_fixture=True,
        )
    slipped = callbacks.copy()
    slipped[3] = 255
    with pytest.raises(ValueError, match="accounting"):
        analyze_v5_raw_arrays(
            plan=plan,
            submitted_pcm=submitted,
            captured_adc_pcm=captured,
            callback_frames=slipped,
            synthetic_fixture=True,
        )


def test_v5_offline_rejects_nonstationary_affine_raw() -> None:
    plan, submitted, _, callbacks = _fixture()
    base = firwin(129, 12_000, fs=48_000)
    primary = np.zeros((2, 1024)); secondary = np.zeros_like(primary)
    primary[:, 10:139] = np.asarray([[0.4], [0.3]]) * base
    secondary[:, 20:149] = np.asarray([[0.35], [0.28]]) * base
    captured = synthesize_affine_capture_v5(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=1.0 - 100e-6,
        piecewise_ratio_after_half=1.0 + 300e-6,
    )
    callbacks = np.full(math.ceil(len(captured) / 256), 256, dtype=np.int64)
    with pytest.raises(ValueError, match="common-q"):
        analyze_v5_raw_arrays(
            plan=plan,
            submitted_pcm=submitted,
            captured_adc_pcm=captured,
            callback_frames=callbacks,
            synthetic_fixture=True,
        )


def test_v5_offline_rejects_energy_unexplained_by_support_1024() -> None:
    plan, submitted = build_plan_v5()
    base = firwin(129, 12_000, fs=48_000)
    primary = np.zeros((2, 2048)); secondary = np.zeros_like(primary)
    primary[:, 1300:1429] = np.asarray([[0.40], [0.30]]) * base
    secondary[:, 1400:1529] = np.asarray([[0.35], [0.28]]) * base
    captured = synthesize_affine_capture_v5(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=1.0,
    )
    callbacks = np.full(math.ceil(len(captured) / 256), 256, dtype=np.int64)
    with pytest.raises(ValueError, match="unexplained-energy"):
        analyze_v5_raw_arrays(
            plan=plan,
            submitted_pcm=submitted,
            captured_adc_pcm=captured,
            callback_frames=callbacks,
            synthetic_fixture=True,
        )


def test_v5_offline_rejects_single_highband_bad_terminal_holdout() -> None:
    plan, submitted, captured, _ = _fixture()
    # 별도 drift 없이 재합성해 DAC/ADC row 위치를 동일하게 만든다.
    base = firwin(129, 12_000, fs=48_000)
    primary = np.zeros((2, 1024)); secondary = np.zeros_like(primary)
    primary[:, 10:139] = np.asarray([[0.4], [0.3]]) * base
    secondary[:, 20:149] = np.asarray([[0.35], [0.28]]) * base
    captured = synthesize_affine_capture_v5(
        submitted,
        primary_fir_by_mic=primary,
        secondary_fir_by_mic=secondary,
        rate_ratio=1.0,
    )
    row = next(row for row in plan["layout"] if row.get("role") == "holdout")
    index = np.arange(32768)
    highband_bin = 5461
    captured[row["central_start_frame"] : row["central_stop_frame"], 0] += 25.0 * np.sin(
        2.0 * np.pi * highband_bin * index / 32768.0
    )
    callbacks = np.full(math.ceil(len(captured) / 256), 256, dtype=np.int64)
    with pytest.raises(ValueError, match="8대역"):
        analyze_v5_raw_arrays(
            plan=plan,
            submitted_pcm=submitted,
            captured_adc_pcm=captured,
            callback_frames=callbacks,
            synthetic_fixture=True,
        )


def test_v5_analysis_cli_expected_failure_is_exit2_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = Path("scripts/data/analyze_fullband_causal_v5.py")
    spec = importlib.util.spec_from_file_location("v5_analyze_cli_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.main(
        [
            "--plan-envelope", str(tmp_path / "missing-plan.json"),
            "--raw", str(tmp_path / "missing-raw.npz"),
            "--output-directory", str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "[실패]" in captured.err
    assert "Traceback" not in captured.err
