"""실측 fine-tune 진입/완료 실패-폐쇄 게이트 회귀 테스트."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from deep_anc.config import REPO_ROOT, load_train_config, load_yaml
from deep_anc.data.decoder_audit import (
    DEFAULT_AUDIO_EXTENSIONS,
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_SEGMENT_GRID_DENOMINATOR,
    DEFAULT_SEQUENTIAL_CHUNK_FRAMES,
    MAX_DECODED_PCM_ABS,
    MIN_DECODED_RMS,
    decoder_fingerprint,
)
from deep_anc.data.manifest import read_manifest, write_manifest
from deep_anc.data.public_lineage import (
    PUBLIC_CROSSWALK_POLICY,
    PUBLIC_LINEAGE_SCHEMA,
    canonical_json_sha256,
    validate_public_manifest_lineage,
)
from deep_anc.data.recorded_subband_coverage import (
    RECORDED_SUBBAND_COVERAGE_KIND,
    RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION,
    build_recorded_subband_coverage_contract,
    recorded_subband_coverage_report_path,
    seal_recorded_subband_coverage_report,
)
from deep_anc.eval.trusted_subbands import (
    MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO,
    STRICT_TRUSTED_SUBBAND_SCHEMA,
    STRICT_TRUSTED_SUBBANDS_HZ,
)
from deep_anc.eval.recorded_sampling import (
    CANONICAL_EDGE_TRIM_SECONDS,
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    CANONICAL_SEGMENT_SECONDS,
    RECORDED_SAMPLING_CONTRACT_SCHEMA,
    canonical_feedback_delay_samples,
    canonical_warmup_samples,
    effective_segment_samples,
)
from deep_anc.train import finetune_readiness as readiness
from deep_anc.train.completion_receipt import write_completion_receipt
from deep_anc.train.evaluation_contract import (
    canonical_test_ledger_paths,
    classify_recorded_val_metrics,
    cluster_bootstrap_ci,
    complete_test_evaluation,
    consume_test_capability,
    issue_test_capability,
    seed_neutral_campaign_sha256,
    snapshot_regular_file,
    write_json_exclusive,
)
from deep_anc.train.experiment_contract import stamp_experiment_contract
from deep_anc.dsp.timing import PlantDelays, TrainingTimingContract
from deep_anc.train.finetune_readiness import (
    MAX_RELATIVE_DELAY_SPREAD_SAMPLES,
    achievable_cancellation_ceiling_db,
    audit_finetune_completion,
    audit_finetune_readiness,
    audit_official_path_model,
    require_finetune_readiness,
    required_consistency_for,
    sha256_file,
)


FS = 8_000
INTERLEAVED_FS = 48_000
INTERLEAVED_PERIOD_SECONDS = 0.125
FAMILIES = ("speech", "music", "environment", "machine")
_INTERLEAVED_SOURCE_FIXTURE_ROOT = (
    REPO_ROOT / "results" / ".pytest_finetune_readiness_sources"
)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_interleaved_source_fixtures():
    yield
    shutil.rmtree(_INTERLEAVED_SOURCE_FIXTURE_ROOT, ignore_errors=True)


def _official_path(
    path: Path,
    *,
    channel: str,
    delay: int,
    consistency: float = 0.97,
    amplitude: float = 0.005,
    method: str = "ess",
    interleaved: dict | None = None,
) -> None:
    extra: dict = {}
    artifact_rate = INTERLEAVED_FS if method == "interleaved_multitone" else FS
    fir = np.asarray([0.5, -0.1, 0.02], dtype=np.float32)
    if method == "interleaved_multitone":
        requested = interleaved or {}
        capture_id = str(np.asarray(requested.get("capture_id", "cap-1")).reshape(-1)[0])
        pre_roll = 2
        bulk_delay = int(delay) + pre_roll
        probe_fixture = readiness.build_interleaved_probe(
            sample_rate=artifact_rate,
            period_seconds=INTERLEAVED_PERIOD_SECONDS,
            band_hz=(60.0, 1_650.0),
            amplitude=float(amplitude),
            tone_spacing_hz=None,
        )
        noise_frequencies = (
            probe_fixture.noise_bins * artifact_rate / probe_fixture.period_samples
        ).astype(np.float64)
        cancel_frequencies = (
            probe_fixture.cancel_bins * artifact_rate / probe_fixture.period_samples
        ).astype(np.float64)
        tone_frequencies = (
            noise_frequencies if channel == "noise" else cancel_frequencies
        )
        indices = np.arange(fir.size, dtype=np.float64) + float(delay)
        aligned_transfer = np.exp(
            -2j * np.pi * np.outer(tone_frequencies, indices) / float(artifact_rate)
        ) @ fir.astype(np.float64)
        compact = readiness._compact_transfer_metrics(
            tone_frequencies,
            aligned_transfer,
            fir,
            delay_samples=delay,
            sample_rate=artifact_rate,
            band_hz=(150.0, 1_600.0),
        )
        overall = compact["overall"]
        compact_subbands = compact["subbands"]
        source_token = hashlib.sha256(str(path.parent).encode()).hexdigest()[:16]
        source_dir = _INTERLEAVED_SOURCE_FIXTURE_ROOT / source_token / capture_id
        source_dir.mkdir(parents=True, exist_ok=True)
        raw_source = source_dir / "raw_measurement.npz"
        analysis_source = source_dir / "analysis_results.npz"

        alignment_count = 16
        valid_mask = np.zeros(alignment_count, dtype=np.bool_)
        valid_mask[:15] = True
        clock_indices = np.flatnonzero(valid_mask)
        err_delay_full = np.full(alignment_count, np.nan, dtype=np.float64)
        ref_delay_full = np.full(alignment_count, np.nan, dtype=np.float64)
        err_score_full = np.full(alignment_count, np.nan, dtype=np.float64)
        ref_score_full = np.full(alignment_count, np.nan, dtype=np.float64)
        err_spread_full = np.full(alignment_count, np.nan, dtype=np.float64)
        ref_spread_full = np.full(alignment_count, np.nan, dtype=np.float64)
        common_delay_full = np.full(alignment_count, np.nan, dtype=np.float64)
        q_full = np.full(alignment_count, np.nan, dtype=np.float64)
        delta_full = np.full(alignment_count, np.nan, dtype=np.float64)
        joint_rank_full = np.zeros(alignment_count, dtype=np.int64)
        joint_condition_full = np.full(alignment_count, np.nan, dtype=np.float64)
        residual_full = np.full(alignment_count, np.nan, dtype=np.float64)
        err_delay_full[valid_mask] = 0.21
        ref_delay_full[valid_mask] = 0.19
        err_score_full[valid_mask] = 0.9998
        ref_score_full[valid_mask] = 0.9996
        err_spread_full[valid_mask] = 0.05
        ref_spread_full[valid_mask] = 0.06
        common_delay_full[valid_mask] = (
            err_delay_full[valid_mask] * err_score_full[valid_mask]
            + ref_delay_full[valid_mask] * ref_score_full[valid_mask]
        ) / (err_score_full[valid_mask] + ref_score_full[valid_mask])
        period_samples = int(round(artifact_rate * INTERLEAVED_PERIOD_SECONDS))
        q_full[valid_mask] = period_samples / (
            period_samples + common_delay_full[valid_mask]
        )
        delta_full[valid_mask] = np.abs(
            err_delay_full[valid_mask] - ref_delay_full[valid_mask]
        )
        expected_rank = 2 * (noise_frequencies.size + cancel_frequencies.size)
        joint_rank_full[valid_mask] = expected_rank
        joint_condition_full[valid_mask] = 1.10
        residual_full[valid_mask] = 0.01
        base_tau = np.linspace(-0.05, 0.05, alignment_count, dtype=np.float64)
        common_tau_full = base_tau.copy()
        noise_tau_full = base_tau + 0.10
        cancel_tau_full = base_tau - 0.10
        row_phase = np.arange(alignment_count, dtype=np.float64)[:, None] * 0.01
        noise_phase = np.exp(
            1j * (row_phase + noise_frequencies[None, :] / 10_000.0)
        )
        cancel_phase = np.exp(
            1j * (row_phase + cancel_frequencies[None, :] / 10_000.0)
        )
        noise_transfers = (
            1.0 + noise_frequencies[None, :] / 20_000.0
        ) * noise_phase
        cancel_transfers = (
            0.7 + cancel_frequencies[None, :] / 30_000.0
        ) * cancel_phase

        if not raw_source.exists():
            raw_repeats = alignment_count
            output = np.zeros(
                (raw_repeats * probe_fixture.period_samples, 2), dtype=np.float32
            )
            output[:, 0] = np.tile(probe_fixture.noise_signal, raw_repeats)
            output[:, 1] = np.tile(probe_fixture.cancel_signal, raw_repeats)
            output_pcm = np.rint(
                np.clip(output, -1.0, 1.0) * np.float32(np.iinfo(np.int16).max)
            ).astype(np.int16)
            measurement_index = np.arange(output.shape[0], dtype=np.float64)
            input_raw = np.stack(
                (
                    np.rint(1_000_000.0 * np.sin(0.011 * measurement_index)),
                    np.rint(800_000.0 * np.cos(0.013 * measurement_index)),
                ),
                axis=1,
            ).astype(np.int32)
            preflight_index = np.arange(256, dtype=np.float64)
            preflight_raw = np.stack(
                (
                    np.rint(900_000.0 * np.sin(0.071 * preflight_index)),
                    np.rint(700_000.0 * np.cos(0.053 * preflight_index)),
                ),
                axis=1,
            ).astype(np.int32)
            measurement_report = readiness.analyze_int32_input_probe(input_raw)
            preflight_report = readiness.analyze_int32_input_probe(preflight_raw)
            preflight_report["sample_rate"] = artifact_rate
            raw_metadata = {
                "capture_id": capture_id,
                "method": "interleaved_multitone",
                "raw_capture_schema": readiness.INTERLEAVED_RAW_CAPTURE_SCHEMA,
                "sample_rate": artifact_rate,
                "block_size": 256,
                "latency": "low",
                "channel_map": {
                    logical_name: expected_index
                    for _, (logical_name, expected_index) in (
                        readiness.INTERLEAVED_CHANNEL_MAP_FIELDS.items()
                    )
                },
                "operator_confirmations": {
                    "user_present": True,
                    "volume_minimum": True,
                    "routing_and_geometry": True,
                },
                "amplitude": float(amplitude),
                "period_seconds": INTERLEAVED_PERIOD_SECONDS,
                "warmup_periods": 0,
                "repeats": raw_repeats,
                "lead_in_samples": 0,
                "guard_bins": 1,
                "design_band_hz": [60.0, 1_650.0],
                "channel_band_hz": {
                    drive: [
                        float(value)
                        for value in (
                            probe_fixture.bins_for(drive)[[0, -1]]
                            * artifact_rate
                            / probe_fixture.period_samples
                        )
                    ]
                    for drive in ("noise", "cancel")
                },
                "crest_db": {
                    "noise": float(probe_fixture.crest_db()[0]),
                    "cancel": float(probe_fixture.crest_db()[1]),
                },
                "invalid_reasons": [],
                "warp": {"applied": False},
                "analysis_contract": {
                    "clock_band_hz": list(readiness.INTERLEAVED_CLOCK_BAND_HZ),
                    "clock_min_adjacent_score": readiness.INTERLEAVED_CLOCK_MIN_SCORE,
                    "clock_max_err_ref_delta_samples": (
                        readiness.INTERLEAVED_CLOCK_MAX_ERR_REF_DELTA
                    ),
                    "clock_max_subwindow_spread_samples": (
                        readiness.INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD
                    ),
                    "clock_max_adjacent_change_samples": (
                        readiness.INTERLEAVED_CLOCK_MAX_ADJACENT_CHANGE
                    ),
                    "clock_max_abs_period_delta_samples": (
                        readiness.INTERLEAVED_CLOCK_MAX_ABS_PERIOD_DELTA
                    ),
                    "separation_algorithm": (
                        readiness.INTERLEAVED_SEPARATION_ALGORITHM
                    ),
                    "separation_algorithm_version": (
                        readiness.INTERLEAVED_SEPARATION_ALGORITHM_VERSION
                    ),
                    "max_drift_deviation_samples": 2.0,
                },
                "telemetry": {
                    "completed": True,
                    "captured_frames": int(output.shape[0]),
                    "xrun_count": 0,
                    "unexpected_status_count": 0,
                    "callback_error": None,
                },
                "measurement": measurement_report,
                "preflight": preflight_report,
            }
            np.savez(
                raw_source,
                metadata_json=np.asarray(json.dumps(raw_metadata, sort_keys=True)),
                output=output,
                output_pcm_int16=output_pcm,
                input_raw_int32=input_raw,
                preflight_raw_int32=preflight_raw,
            )
        if not analysis_source.exists():
            np.savez(
                analysis_source,
                noise_transfers=noise_transfers.astype(np.complex128),
                cancel_transfers=cancel_transfers.astype(np.complex128),
                noise_frequencies_hz=noise_frequencies,
                cancel_frequencies_hz=cancel_frequencies,
                noise_cubic_crosscheck_transfers=noise_transfers.astype(np.complex128),
                cancel_cubic_crosscheck_transfers=cancel_transfers.astype(np.complex128),
                clock_valid_mask=valid_mask,
                clock_q_ratio=q_full,
                clock_period_delta_samples=common_delay_full,
                clock_err_delay_samples=err_delay_full,
                clock_ref_delay_samples=ref_delay_full,
                clock_err_score=err_score_full,
                clock_ref_score=ref_score_full,
                clock_err_subwindow_spread_samples=err_spread_full,
                clock_ref_subwindow_spread_samples=ref_spread_full,
                clock_err_ref_delta_samples=delta_full,
                joint_ls_rank=joint_rank_full,
                joint_ls_condition=joint_condition_full,
                joint_ls_reconstruction_relative_error=residual_full,
                common_alignment_tau_samples=common_tau_full,
                noise_provisional_tau_samples=noise_tau_full,
                cancel_provisional_tau_samples=cancel_tau_full,
            )
        raw_relative = raw_source.relative_to(REPO_ROOT)
        analysis_relative = analysis_source.relative_to(REPO_ROOT)
        kept = np.arange(12, dtype=np.int64)
        requested_kept = np.asarray(
            requested.get("kept_repeat_indices", kept)
        )
        if (
            requested_kept.ndim == 1
            and requested_kept.dtype.kind in "iu"
            and requested_kept.size == kept.size
            and np.all((requested_kept >= 0) & (requested_kept < alignment_count))
        ):
            kept = requested_kept.astype(np.int64)
        provisional_kept = (
            noise_tau_full if channel == "noise" else cancel_tau_full
        )[kept]
        common_kept = common_tau_full[kept]
        defaults = {
            "capture_id": np.asarray(capture_id),
            "interleave_guard_bins": np.asarray(1, dtype=np.int64),
            "analysis_period_seconds": np.asarray(INTERLEAVED_PERIOD_SECONDS),
            "tone_count": np.asarray(tone_frequencies.size, dtype=np.int64),
            "tone_snr_median_db": np.asarray(35.0),
            "tone_snr_min_db": np.asarray(14.0),
            "consistency_band_hz": np.asarray([100.0, 1_000.0]),
            "excitation_band_hz": np.asarray([100.0, 1_800.0]),
            "anchor_repeat": np.asarray(3, dtype=np.int64),
            "kept_repeat_indices": kept,
            "alignment_scores": np.full(alignment_count, 0.999, dtype=np.float64),
            # 최악 부대역 게이트가 판정할 배열. 총계 하나로는 약한 대역을 숨길 수
            # 있으므로 게이트가 요구 대역 안 모든 부대역을 따로 본다.
            "band_consistency": np.asarray([0.99, 0.98, 0.97, 0.96]),
            "band_consistency_hz": np.asarray(
                [
                    [150.0, 300.0],
                    [300.0, 600.0],
                    [600.0, 1_000.0],
                    [1_000.0, 1_600.0],
                ]
            ),
            "bulk_delay_samples": np.asarray(bulk_delay, dtype=np.int64),
            "pre_roll_samples": np.asarray(pre_roll, dtype=np.int64),
            "delay_semantics": np.asarray(readiness.INTERLEAVED_DELAY_SEMANTICS),
            "tone_frequencies_hz": tone_frequencies,
            "aligned_mean_transfer_real": aligned_transfer.real,
            "aligned_mean_transfer_imag": aligned_transfer.imag,
            "aligned_mean_transfer_sha256": np.asarray(
                readiness._aligned_transfer_sha256(
                    tone_frequencies, aligned_transfer.real, aligned_transfer.imag
                )
            ),
            "compact_transfer_band_hz": np.asarray([150.0, 1_600.0]),
            "compact_transfer_tone_count": np.asarray(
                overall["tone_count"], dtype=np.int64
            ),
            "compact_transfer_complex_agreement": np.asarray(
                overall["complex_agreement"]
            ),
            "compact_transfer_relative_error": np.asarray(overall["relative_error"]),
            "minimum_compact_transfer_agreement": np.asarray(
                readiness.INTERLEAVED_MIN_COMPACT_TRANSFER_AGREEMENT
            ),
            "maximum_compact_transfer_relative_error": np.asarray(
                readiness.INTERLEAVED_MAX_COMPACT_TRANSFER_RELATIVE_ERROR
            ),
            "compact_transfer_subband_hz": np.asarray(
                [row["band_hz"] for row in compact_subbands]
            ),
            "compact_transfer_subband_tone_count": np.asarray(
                [row["tone_count"] for row in compact_subbands], dtype=np.int64
            ),
            "compact_transfer_subband_complex_agreement": np.asarray(
                [row["complex_agreement"] for row in compact_subbands]
            ),
            "compact_transfer_subband_relative_error": np.asarray(
                [row["relative_error"] for row in compact_subbands]
            ),
            "output_pcm_provenance": np.asarray(
                readiness.INTERLEAVED_OUTPUT_PCM_PROVENANCE
            ),
            "source_raw_npz_path": np.asarray(str(raw_relative)),
            "source_raw_npz_sha256": np.asarray(sha256_file(raw_source)),
            "source_analysis_npz_path": np.asarray(str(analysis_relative)),
            "source_analysis_npz_sha256": np.asarray(sha256_file(analysis_source)),
            "error_mic_channel": np.asarray(0, dtype=np.int64),
            "reference_mic_channel": np.asarray(1, dtype=np.int64),
            "noise_output_channel": np.asarray(0, dtype=np.int64),
            "cancel_output_channel": np.asarray(1, dtype=np.int64),
            "operator_confirmed_volume_minimum": np.asarray(True),
            "operator_confirmed_routing_and_geometry": np.asarray(True),
            "operator_confirmed_user_present": np.asarray(True),
            "separation_algorithm": np.asarray(
                readiness.INTERLEAVED_SEPARATION_ALGORITHM
            ),
            "separation_algorithm_version": np.asarray(
                readiness.INTERLEAVED_SEPARATION_ALGORITHM_VERSION, dtype=np.int64
            ),
            "clock_estimator": np.asarray(readiness.INTERLEAVED_CLOCK_ESTIMATOR),
            "clock_sample_rate": np.asarray(artifact_rate, dtype=np.int64),
            "clock_band_hz": np.asarray(readiness.INTERLEAVED_CLOCK_BAND_HZ),
            "clock_min_adjacent_score": np.asarray(
                readiness.INTERLEAVED_CLOCK_MIN_SCORE
            ),
            "clock_max_err_ref_delta_samples": np.asarray(
                readiness.INTERLEAVED_CLOCK_MAX_ERR_REF_DELTA
            ),
            "clock_max_subwindow_spread_samples": np.asarray(
                readiness.INTERLEAVED_CLOCK_MAX_SUBWINDOW_SPREAD
            ),
            "clock_max_adjacent_change_samples": np.asarray(
                readiness.INTERLEAVED_CLOCK_MAX_ADJACENT_CHANGE
            ),
            "clock_max_abs_period_delta_samples": np.asarray(
                readiness.INTERLEAVED_CLOCK_MAX_ABS_PERIOD_DELTA
            ),
            "clock_max_drift_deviation_samples": np.asarray(2.0),
            "clock_observation_repeat_indices": clock_indices,
            "clock_period_delta_samples": common_delay_full[valid_mask],
            "clock_q_ratio": q_full[valid_mask],
            "clock_err_delay_samples": err_delay_full[valid_mask],
            "clock_ref_delay_samples": ref_delay_full[valid_mask],
            "clock_err_score": err_score_full[valid_mask],
            "clock_ref_score": ref_score_full[valid_mask],
            "clock_err_subwindow_spread_samples": err_spread_full[valid_mask],
            "clock_ref_subwindow_spread_samples": ref_spread_full[valid_mask],
            "clock_err_ref_delta_samples": delta_full[valid_mask],
            "joint_ls_expected_rank": np.asarray(expected_rank, dtype=np.int64),
            "joint_ls_rank": joint_rank_full[valid_mask],
            "joint_ls_condition": joint_condition_full[valid_mask],
            "joint_ls_max_condition": np.asarray(
                readiness.INTERLEAVED_JOINT_LS_MAX_CONDITION
            ),
            "joint_ls_reconstruction_relative_error": residual_full[valid_mask],
            "joint_ls_reconstruction_relative_error_p95": np.asarray(0.01),
            "joint_ls_max_reconstruction_relative_error_p95": np.asarray(
                readiness.INTERLEAVED_JOINT_LS_MAX_RESIDUAL_P95
            ),
            "separation_crosscheck_band_hz": np.asarray(
                readiness.INTERLEAVED_CLOCK_BAND_HZ
            ),
            "separation_crosscheck_complex_agreement": np.asarray(1.0),
            "separation_crosscheck_relative_error": np.asarray(0.0),
            "separation_crosscheck_subband_hz": np.asarray(
                readiness.INTERLEAVED_COMPACT_TRANSFER_SUB_BANDS_HZ
            ),
            "separation_crosscheck_subband_complex_agreement": np.ones(4),
            "separation_crosscheck_subband_relative_error": np.zeros(4),
            "minimum_separation_crosscheck_agreement": np.asarray(
                readiness.INTERLEAVED_SEPARATION_MIN_AGREEMENT
            ),
            "maximum_separation_crosscheck_relative_error": np.asarray(
                readiness.INTERLEAVED_SEPARATION_MAX_RELATIVE_ERROR
            ),
            "repeat_tau_samples": provisional_kept,
            "provisional_repeat_tau_samples": provisional_kept,
            "common_alignment_tau_samples": common_kept,
            "drift_samples_per_period": np.asarray(
                float(np.median(common_delay_full[valid_mask]))
            ),
            "relative_tau_max_abs_samples": np.asarray(0.0),
            "delay_spread_samples": np.asarray(0, dtype=np.int64),
        }
        extra = defaults
    arrays = {
        "fir": fir,
        "delay_samples": np.asarray(delay, dtype=np.int64),
        "sample_rate": np.asarray(artifact_rate, dtype=np.int64),
        "fit_improvement_db": np.asarray(np.nan),
        "coherence_median": np.asarray(consistency),
        "excitation_band_hz": np.asarray([100.0, 1_800.0]),
        "calibration_block_size": np.asarray(256, dtype=np.int64),
        "calibration_latency": np.asarray(
            "low" if method == "interleaved_multitone" else "high"
        ),
        "output_channel": np.asarray(channel),
        "method": np.asarray(method),
        # 유지 반복 수. MIN_KEPT_REPEATS=8 이 하한이다 — 기각을 많이 한 것은 문제가
        # 아니지만(복구 캡처는 48 중 30 을 기각했다) 남은 것이 적으면 한 번의
        # 이상치가 플랜트 형상을 지배한다.
        "repeats": np.asarray(12, dtype=np.int64),
        "amplitude": np.asarray(amplitude),
        "xrun_count": np.asarray(0, dtype=np.int64),
        "delay_spread_samples": np.asarray(1, dtype=np.int64),
        "max_delay_jitter_samples": np.asarray(8, dtype=np.int64),
        **extra,
    }
    # ``interleaved`` 는 interleaved 전용 필드뿐 아니라 공통 필드도 덮어쓸 수 있어야
    # 한다 — 게이트가 "아티팩트가 신고한 허용치"를 믿지 않는지 시험하려면 필요하다.
    arrays.update(
        {key: np.asarray(value) for key, value in (interleaved or {}).items()}
    )
    np.savez(path, **arrays)


def _checkpoint(
    path: Path,
    *,
    cfg: dict,
    step: int,
    best_metric: float = -3.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "cfg": cfg,
            "step": step,
            "best_metric": best_metric,
        },
        path,
    )


GROUPS_PER_FAMILY_PER_SPLIT = 4
"""계열·split 당 그룹 수. ``readiness.min_groups_per_family_per_split`` 과 같아야 한다.

3 으로 내리면 ``recorded_statistical_power`` 가 FAIL 한다 — 즉 이 상수 하나로 검정력
게이트가 살아 있는지 확인할 수 있다.
"""

# P(z) 픽스처의 FIR 은 ``[0.5, -0.1, 0.02]`` 라 최대 tap 이 0 이다. 따라서 유도되는
# 재생→ERR 지연은 ``벌크지연 + 0`` 이고, 실측 세션도 같은 지연을 갖도록 만들어야
# D2 교차검증이 성립한다. 두 값을 **같은 곳에서 유도**하는 것이 이 상수의 목적이다.
PRIMARY_DELAY_SAMPLES = 4
ERR_MIC_DELAY_SAMPLES = PRIMARY_DELAY_SAMPLES
REF_MIC_DELAY_SAMPLES = 2

PRETRAIN_TRUSTED_BAND_HZ = (150.0, 1_600.0)
"""픽스처 init checkpoint 가 학습된 대역.

실제 사고와 같은 축이다: ``runs/pretrain_{base,tiny}_corrected`` 는 [150,600] 으로
학습됐는데 파인튜닝이 유도하는 대역은 [150,1600] 이다. 여기서는 픽스처 S(z) 의
현행 절대목표 대역은 [150,1600] 이며 positive fixture도 그 대역을 그대로 덮어야 한다.
"""


def _recorded_band_noise(frames: int, seed: int) -> np.ndarray:
    """정렬 검사가 통하는 광대역 소스.

    옛 픽스처는 순음이었다. 순음은 상호상관 최대점이 주기마다 반복돼 지연이 다중값이
    되고, 무엇보다 ERR/REF 두 채널을 **같은 신호의 상수배**로 만들어 시간 관계를
    시험하지 않는다. 실측 프로그램 소재는 광대역이므로 픽스처도 광대역이어야 한다.
    """

    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    b, a = butter(4, [120.0 / (FS / 2), 1_200.0 / (FS / 2)], btype="band")
    filtered = lfilter(b, a, rng.standard_normal(frames + 2048))[2048:]
    peak = float(np.max(np.abs(filtered))) or 1.0
    return 0.3 * filtered / peak


def _recorded_manifest(
    root: Path,
    *,
    frames: int = 16_384,
    groups_per_family: int = GROUPS_PER_FAMILY_PER_SPLIT,
    collapse_alignment: bool = False,
    source_delay_samples: int = ERR_MIC_DELAY_SAMPLES,
) -> Path:
    """실측 manifest 픽스처.

    ``collapse_alignment`` 는 2026-08-04 사고(재생/캡처 타임베이스 붕괴)를 주입한다.
    ``source_delay_samples`` 는 D2(실측 지연 vs P(z) 유도값) 교차검증을 흔들 때 쓴다.
    """

    manifest = root / "manifests" / "recorded.jsonl"
    entries = []
    for family_index, family in enumerate(FAMILIES):
        for split_index, split in enumerate(("train", "val", "test")):
            for group_index in range(groups_per_family):
                session_id = f"{family}-{split}-{group_index}"
                session = root / "recorded" / session_id
                session.mkdir(parents=True)
                source = _recorded_band_noise(
                    frames, 1_000 * family_index + 10 * split_index + group_index
                )
                err = 0.7 * np.roll(source, int(source_delay_samples))
                ref = 0.4 * np.roll(source, REF_MIC_DELAY_SAMPLES)
                if collapse_alignment:
                    # ERR/REF 마이크는 그대로 두고 source 만 시간축을 깬다 —
                    # 실측에서 무너진 것이 음향이 아니라 소프트웨어였기 때문이다.
                    rng = np.random.default_rng(7 + group_index)
                    block = max(64, frames // 8)
                    broken = np.zeros(frames, dtype=np.float64)
                    for start in range(0, frames, block):
                        stop = min(frames, start + block)
                        jump = int(rng.integers(-1_000, 1_000))
                        broken[start:stop] = np.roll(source, jump)[start:stop]
                    source = broken
                mics = np.stack([err, ref], axis=1).astype(np.float32)
                sf.write(session / "mics.wav", mics, FS, subtype="FLOAT")
                sf.write(
                    session / "source.wav", source.astype(np.float32), FS, subtype="FLOAT"
                )
                group = f"group-{family}-{split}-{group_index}"
                (session / "session.json").write_text(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "group_id": group,
                            "source_family": family,
                            "sample_rate": FS,
                            "seconds": frames / FS,
                            "program": {"type": "file"},
                        }
                    ),
                    encoding="utf-8",
                )
                entries.append(
                    {
                        "path": str(session),
                        "duration_s": frames / FS,
                        "sample_rate": FS,
                        "channels": 2,
                        "tag": "recorded",
                        "session_id": session_id,
                        "group_id": group,
                        "source_family": family,
                        "split": split,
                    }
                )
    write_manifest(entries, manifest)
    return manifest


def _corpus_fixture(root: Path, *, leak: bool = False) -> tuple[Path, Path]:
    """실측 소스 목록(``sources.csv``)과 합성 소음 manifest 를 만든다.

    ``leak=True`` 면 실측 music 이 쓴 원본이 합성 train 에도 들어간다 — 2026-08-05
    감사가 실측에서 찾은 상태(music 60/60 겹침, 그중 55개가 합성 train)의 축소판이다.
    """

    pool_dir = root / "source_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pool_dir / "sources.csv"
    recorded_clips = {
        "speech": ["spk1-a.flac", "spk1-b.flac"],
        "music": ["music-0001.mp3", "music-0002.mp3"],
        "environment": ["env-1.wav", "env-2.wav"],
        "machine": ["mach-1.wav", "mach-2.wav"],
    }
    rows = ["source_family,session_index,group_id,path,seconds,clips"]
    for family, clips in recorded_clips.items():
        payload = json.dumps(clips).replace('"', '""')
        rows.append(f'{family},0,grp-{family},{family}_000.wav,70.0,"{payload}"')
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest_dir = root / "synth_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    speech_entries = [
        {
            "path": str(root / "raw" / "speech" / "other-1.flac"),
            "duration_s": 5.0,
            "sample_rate": FS,
            "channels": 1,
            "tag": "speech",
            "split": "train",
        }
    ]
    music_paths = ["fresh-a.mp3", "fresh-b.mp3"]
    if leak:
        # 실측이 재생한 바로 그 원본이 합성 train 에도 있다.
        music_paths = ["music-0001.mp3", "fresh-b.mp3"]
    music_entries = [
        {
            "path": str(root / "raw" / "music" / "000" / name),
            "duration_s": 30.0,
            "sample_rate": FS,
            "channels": 2,
            "tag": "music",
            "split": "train",
        }
        for name in music_paths
    ]
    other_entries = {
        tag: [
            {
                "path": str(root / "raw" / tag / f"fixture-{tag}.wav"),
                "duration_s": 5.0,
                "sample_rate": FS,
                "channels": 1,
                "tag": tag,
                "split": "train",
            }
        ]
        for tag in ("demand", "esc50", "machine")
    }
    all_entries = speech_entries + music_entries + [
        entry for entries in other_entries.values() for entry in entries
    ]
    for entry in all_entries:
        raw_path = Path(entry["path"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(f"fixture-audio:{raw_path.name}\n".encode())
        entry["content_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        entry["content_size"] = raw_path.stat().st_size
        entry["lineage_schema"] = PUBLIC_LINEAGE_SCHEMA
        entry["lineage_keys"] = [
            f"fixture:{entry['tag']}:{raw_path.stem.casefold()}"
        ]
        entry["group_id"] = "public-lineage-" + canonical_json_sha256(
            {
                "lineage_keys": entry["lineage_keys"],
                "content_sha256": [entry["content_sha256"]],
            }
        )
    write_manifest(speech_entries, manifest_dir / "speech.jsonl")
    write_manifest(music_entries, manifest_dir / "music.jsonl")
    for tag, entries in other_entries.items():
        write_manifest(entries, manifest_dir / f"{tag}.jsonl")

    # prepare_noise_pool의 세대 sidecar를 축소 재현한다. JSONL들이 서로 다른
    # holdout/config 세대에서 온 혼합 상태면 corpus_disjoint가 PASS하면 안 된다.
    data_config = root / "fixture_data_config.json"
    data_config.write_text(
        '{"source_mix_ratio":{"demand":0.15,"esc50":0.10,"machine":0.25,'
        '"music":0.25,"speech":0.25}}\n'
    )
    lineage_metadata = root / "fixture_public_lineage_metadata.txt"
    lineage_metadata.write_text("fixture authoritative lineage\n")
    metadata_evidence = {
        "path": str(lineage_metadata),
        "sha256": hashlib.sha256(lineage_metadata.read_bytes()).hexdigest(),
        "size": lineage_metadata.stat().st_size,
    }
    holdout_clip = {
        "family": "fixture",
        "clip": "heldout-fixture.wav",
        "content_sha256": hashlib.sha256(b"heldout-fixture").hexdigest(),
        "lineage_keys": ["fixture:holdout:one"],
    }
    holdout_clips = [holdout_clip]
    holdout_lineage = {
        "schema_version": 1,
        "metadata": {
            "librispeech_chapters": {
                "path": "data/raw/speech/LibriSpeech/CHAPTERS.TXT",
                "sha256": "1" * 64,
                "size": 1,
            },
            "fma_tracks": {
                "path": "data/raw/music/fma_metadata/tracks.csv",
                "sha256": "2" * 64,
                "size": 1,
            },
            "esc50": {
                "path": "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
                "sha256": "3" * 64,
                "size": 1,
            },
        },
        "clips": holdout_clips,
        "clips_sha256": canonical_json_sha256(holdout_clips),
    }
    holdout = root / "fixture_recorded_holdout.json"
    holdout.write_text(
        json.dumps(
            {
                "families": {"fixture": ["heldout-fixture.wav"]},
                "clip_lineage": holdout_lineage,
            },
            sort_keys=True,
        )
        + "\n"
    )
    manifests = {}
    for tag in ("demand", "esc50", "machine", "music", "speech"):
        path = manifest_dir / f"{tag}.jsonl"
        manifests[tag] = {
            "file": path.name,
            "entries": sum(1 for line in path.read_text().splitlines() if line),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    entries_by_tag = {
        **other_entries,
        "music": music_entries,
        "speech": speech_entries,
    }
    audit_inventory = []
    for entry in sorted(
        all_entries,
        key=lambda item: Path(str(item["path"])).relative_to(root).as_posix(),
    ):
        raw_path = Path(str(entry["path"]))
        audit_inventory.append(
            {
                "relative_path": raw_path.relative_to(root).as_posix(),
                "content_sha256": entry["content_sha256"],
                "content_size": entry["content_size"],
                "decision": "accept",
            }
        )
    decoder_runtime_fingerprint = decoder_fingerprint()
    decoder_audit = {
        "schema_version": 1,
        "status": "complete",
        "audit_policy": {
            "audio_extensions": sorted(DEFAULT_AUDIO_EXTENSIONS),
            "sequential_chunk_frames": list(DEFAULT_SEQUENTIAL_CHUNK_FRAMES),
            "segment_frames": DEFAULT_SEGMENT_FRAMES,
            "segment_grid_denominator": DEFAULT_SEGMENT_GRID_DENOMINATOR,
            "max_decoded_pcm_abs": MAX_DECODED_PCM_ABS,
            "min_decoded_rms": MIN_DECODED_RMS,
        },
        "decoder_fingerprint": decoder_runtime_fingerprint,
        "decoder_fingerprint_sha256": canonical_json_sha256(decoder_runtime_fingerprint),
        "inventory": audit_inventory,
        "inventory_sha256": canonical_json_sha256(audit_inventory),
        "accepted_inventory_sha256": canonical_json_sha256(
            [
                {
                    "relative_path": row["relative_path"],
                    "content_sha256": row["content_sha256"],
                    "content_size": row["content_size"],
                }
                for row in audit_inventory
            ]
        ),
    }
    decoder_audit_path = manifest_dir / "decoder_audit.json"
    decoder_audit_path.write_text(
        json.dumps(decoder_audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    decoder_audit_binding = {
        "schema_version": 1,
        "file": "decoder_audit.json",
        "sha256": hashlib.sha256(decoder_audit_path.read_bytes()).hexdigest(),
        "size": decoder_audit_path.stat().st_size,
        "inventory_sha256": decoder_audit["inventory_sha256"],
        "accepted_inventory_sha256": decoder_audit["accepted_inventory_sha256"],
        "decoder_fingerprint": decoder_runtime_fingerprint,
        "decoder_fingerprint_sha256": decoder_audit["decoder_fingerprint_sha256"],
    }
    manifest_lineage = validate_public_manifest_lineage(entries_by_tag)
    components = manifest_lineage["components"]
    generation = {
        "schema_version": 4,
        "training_eligible": True,
        "seed": 20260802,
        "data_config": str(data_config),
        "data_config_sha256": hashlib.sha256(data_config.read_bytes()).hexdigest(),
        "holdout": str(holdout),
        "holdout_sha256": hashlib.sha256(holdout.read_bytes()).hexdigest(),
        "raw_roots": [str(root / "raw")],
        "manifests": manifests,
        "decoder_audit": decoder_audit_binding,
        "public_lineage": {
            "schema_version": 1,
            "lineage_schema": PUBLIC_LINEAGE_SCHEMA,
            "metadata": {"fixture": metadata_evidence},
            "component_count": len(components),
            "component_membership_sha256": canonical_json_sha256(components),
            "components": components,
            "manifest_component_count": manifest_lineage["component_count"],
            "manifest_component_membership_sha256": manifest_lineage[
                "component_membership_sha256"
            ],
            "holdout_clips_sha256": holdout_lineage["clips_sha256"],
            "crosswalk_policy": json.loads(json.dumps(PUBLIC_CROSSWALK_POLICY)),
        },
    }
    canonical = (json.dumps(generation, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    generation["build_id"] = hashlib.sha256(canonical).hexdigest()
    generation["created_at"] = "2026-08-26T00:00:00+00:00"
    (manifest_dir / "manifest_generation.json").write_text(
        json.dumps(generation, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    return csv_path, manifest_dir


def _coverage_report_path(cfg: dict) -> Path:
    manifest = Path(cfg["recorded_manifest"])
    snapshot = snapshot_regular_file(manifest)
    contract = build_recorded_subband_coverage_contract(
        manifest_path=snapshot.path,
        manifest_content=snapshot.content,
        data_cfg=cfg["data"],
        model_hop=int(cfg["model"]["hop"]),
    )
    return recorded_subband_coverage_report_path(
        cfg["readiness"]["recorded_subband_coverage_report_dir"], contract
    )


def _write_coverage_fixture(cfg: dict, destination_dir: Path) -> None:
    """WAV FFT와 분리해 readiness report 계약/집계를 시험하는 정상 증거."""

    manifest = Path(cfg["recorded_manifest"])
    manifest_snapshot = snapshot_regular_file(manifest)
    entries = read_manifest(manifest)
    contract = build_recorded_subband_coverage_contract(
        manifest_path=manifest_snapshot.path,
        manifest_content=manifest_snapshot.content,
        data_cfg=cfg["data"],
        model_hop=int(cfg["model"]["hop"]),
    )
    destination = recorded_subband_coverage_report_path(destination_dir, contract)
    split_payloads = {}
    overall_pass = True
    for split in ("train", "val", "test"):
        selected = [entry for entry in entries if entry["split"] == split]
        rows = []
        split_pass = True
        total_segments = 0
        for family in sorted(FAMILIES):
            groups = sorted(
                {
                    str(entry["group_id"])
                    for entry in selected
                    if entry["source_family"] == family
                }
            )
            sessions = sum(1 for entry in selected if entry["source_family"] == family)
            total_segments += sessions
            row_pass = len(groups) >= GROUPS_PER_FAMILY_PER_SPLIT
            split_pass = split_pass and row_pass
            for band in STRICT_TRUSTED_SUBBANDS_HZ:
                rows.append(
                    {
                        "source_family": family,
                        "band_hz": [float(band[0]), float(band[1])],
                        "n_segments": sessions,
                        "n_covered_segments": sessions,
                        "n_covered_groups": len(groups),
                        "covered_group_ids": groups,
                        "density_mean": 1.0,
                        "density_median": 1.0,
                        "density_p10": 1.0,
                        "group_power_pass": row_pass,
                    }
                )
        split_payloads[split] = {
            "n_sessions": len(selected),
            "n_segments": total_segments,
            "group_power_pass": split_pass,
            "rows": rows,
        }
        overall_pass = overall_pass and split_pass
    payload = seal_recorded_subband_coverage_report(
        {
            "schema_version": RECORDED_SUBBAND_COVERAGE_SCHEMA_VERSION,
            "kind": RECORDED_SUBBAND_COVERAGE_KIND,
            **contract,
            "all_requested_splits_pass": overall_pass,
            "splits": split_payloads,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _ready_config(tmp_path: Path, *, manifest: Path | None = None, leak: bool = False) -> dict:
    primary = tmp_path / "primary.npz"
    secondary = tmp_path / "secondary.npz"
    _official_path(primary, channel="noise", delay=PRIMARY_DELAY_SAMPLES)
    _official_path(secondary, channel="cancel", delay=5)
    if manifest is None:
        manifest = _recorded_manifest(tmp_path / "data")
    source_csv, synth_dir = _corpus_fixture(tmp_path / "data", leak=leak)
    model_cfg = {"name": "test-model", "hop": 4}
    timing = TrainingTimingContract.derive(
        primary_fir=np.asarray([0.5, -0.1, 0.02], dtype=np.float32),
        plant_delays=PlantDelays(
            primary_delay_samples=PRIMARY_DELAY_SAMPLES,
            secondary_delay_samples=5,
            handoff_samples=2,
            sample_rate=FS,
        ),
    )
    pretrain_cfg = {
        "model": model_cfg,
        "data": {
            "digital_reference_lead_samples": int(
                timing.digital_reference_lead_samples
            ),
            "training_timing_contract": timing.model_dump(),
        },
        "digital_reference_lead_samples": int(
            timing.digital_reference_lead_samples
        ),
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
        # 어느 대역에서 벌점을 받았는가. trainer 가 resolved cfg 에 저장하는 값이고
        # ``completed_init_checkpoint`` 게이트가 파인튜닝 대역과 대조한다.
        # 현행 절대목표와 같은 [150,1600].
        "trusted_band_hz": list(PRETRAIN_TRUSTED_BAND_HZ),
    }
    init_best = tmp_path / "pretrain" / "ckpt" / "best.pt"
    _checkpoint(init_best, cfg=pretrain_cfg, step=8)
    _checkpoint(init_best.parent / "last.pt", cfg=pretrain_cfg, step=10)

    cfg = {
        "stage": "open_loop",
        "model": model_cfg,
        "data": {
            "sample_rate": FS,
            "segment_seconds": 0.01,
            "reference_mode": "digital",
            "digital_primary_path_mode": "measured",
            "digital_reference_lead_samples": int(
                timing.digital_reference_lead_samples
            ),
            "training_timing_contract": timing.model_dump(),
            # 두 브랜치의 총 선행량을 맞추는 구성 — 출하 설정과 같다.
            # constant 면 합성 D_noise+K 와 실측 잔여+K 가 어긋난다(실측 1460 샘플).
            "recorded_lead_mode": "timeline",
            "closed_loop": {
                "feedback_delay_samples": [4, 8],
                "warmup_seconds": 0.0,
            },
            # 코퍼스 누수 게이트(D1)가 읽는 합성 스트림 구성.
            "noise_manifest_dir": str(synth_dir),
            "source_mix_ratio": {
                "speech": 0.25,
                "music": 0.25,
                "demand": 0.15,
                "esc50": 0.10,
                "machine": 0.25,
            },
        },
        "duct": {
            "acoustics": {"realistic_target_band_hz": [150, 1_600]},
            "secondary_path": {
                "npz": str(secondary),
                "handoff_extra_samples": 2,
            },
            "digital_reference": {
                "primary_path_npz": str(primary),
            },
        },
        "require_measured_primary_path": True,
        "require_init_checkpoint": True,
        "require_recorded_manifest": True,
        "init_ckpt": str(init_best),
        "recorded_manifest": str(manifest),
        "recorded_ratio": 0.7,
        "schedule": {"total_steps": 6},
        "ckpt_dir": str(tmp_path / "finetune"),
        "readiness": {
            "required_path_band_hz": [150, 1_600],
            "min_path_consistency": 0.9,
            "required_recorded_ratio": 0.7,
            "min_recorded_sessions": 12,
            "min_recorded_duration_seconds": 0.1,
            "required_source_families": list(FAMILIES),
            "require_completed_init_checkpoint": True,
            "max_init_best_metric_db": 0.0,
            # --- 신규 게이트가 소비하는 선언 ---
            # 픽스처 일관성 0.97 두 경로 → 상한 12.1 dB. 목표 3 + 여유 3 = 6 이므로 통과.
            "target_cancellation_db": 3.0,
            "cancellation_ceiling_margin_db": 3.0,
            "min_groups_per_family_per_split": GROUPS_PER_FAMILY_PER_SPLIT,
            "recorded_subband_coverage_report_dir": str(
                tmp_path / "recorded_subband_coverage"
            ),
            "recorded_source_pool_csv": str(source_csv),
            # 설계 상한 선언 — **생략도 우회다**(2026-08-06). 선언이 없으면 구속 상한이
            # 플랜트 일관성으로 폴백해 물리적으로 불가능한 목표도 통과한다.
            # 픽스처 플랜트의 재계산 상한: 최악 옥타브 125 Hz 21.03 dB.
            # 21.0 은 그 안쪽이면서, 경계 테스트가 미는 target+margin(약 10.9)보다 크다 —
            # 이 값이 구속하면 다른 게이트의 경계를 시험할 수 없다.
            "measured_design_ceiling_db": 7.590811495963479,
            "measured_design_ceiling_band_hz": [150, 1600],
            "min_delay_crosscheck_sessions": 8,
            "max_measured_delay_mismatch_samples": 8.0,
        },
    }
    _write_coverage_fixture(
        cfg, Path(cfg["readiness"]["recorded_subband_coverage_report_dir"])
    )
    return cfg


def _refresh_timing_contract(cfg: dict) -> TrainingTimingContract:
    primary_path = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary_path = Path(cfg["duct"]["secondary_path"]["npz"])
    with np.load(primary_path, allow_pickle=False) as primary, np.load(
        secondary_path, allow_pickle=False
    ) as secondary:
        contract = TrainingTimingContract.derive(
            primary_fir=np.asarray(primary["fir"], dtype=np.float32),
            plant_delays=PlantDelays(
                primary_delay_samples=int(primary["delay_samples"]),
                secondary_delay_samples=int(secondary["delay_samples"]),
                handoff_samples=int(
                    cfg["duct"]["secondary_path"]["handoff_extra_samples"]
                ),
                sample_rate=int(cfg["data"]["sample_rate"]),
            ),
        )
    cfg["data"]["training_timing_contract"] = contract.model_dump()
    cfg["data"]["digital_reference_lead_samples"] = int(
        contract.digital_reference_lead_samples
    )
    return contract


def _plant_fingerprint_payload(**overrides) -> str:
    """metrics.npz 에 박히는 플랜트 지문. 기본값은 val/test 가 같은 플랜트다."""

    payload = {
        "primary_delay_samples": 4,
        "secondary_delay_samples": 5,
        "handoff_samples": 2,
        "lead_samples": 3,
        "sample_rate": FS,
        "physics_status": "measured_primary_path",
        "optimize_band_hz": [150.0, 1_600.0],
        "secondary_sha256": None,
        "primary_sha256": None,
        "capture_id": None,
        "configured_lead_samples": 3,
    }
    payload.update(overrides)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _g4_metrics(
    path: Path,
    *,
    split: str,
    checkpoint: Path,
    manifest: Path,
    source_pass: bool = True,
    worst_source_db: float = -4.0,
    include_source_fields: bool = True,
    include_modern_fields: bool = True,
    verdict: str | None = None,
    do_no_harm_pass: bool = True,
    power_pass: bool = True,
    ci_pass: bool = True,
    worst_octave_center_hz: float = 500.0,
    worst_octave_worst10_db: float = 3.0,
    fingerprint: str | None = None,
    experiment_contract_sha256: str = "",
    selection_sha256: str = "",
    test_capability_sha256: str = "",
    test_consumed_marker_sha256: str = "",
    timing_contract_sha256: str = "",
    recorded_lead_samples: int = 3,
    recorded_delay_samples: float = 2.0,
    include_strict_subband_fields: bool = True,
    strict_upper_subband_nmse_db: float = -2.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_cfg = checkpoint_state.get("cfg", {})
    checkpoint_data = checkpoint_cfg.get("data", {})
    # persisted canonical G4는 selected manifest의 session/family/group를 그대로
    # 보존해야 한다. manifest fixture의 split별 row와 같은 순서/ID를 사용한다.
    families = tuple(sorted(FAMILIES))
    segment_family = np.repeat(np.asarray(families, dtype=np.str_), 4)
    segment_group = np.asarray(
        [
            f"group-{family}-{split}-{group}"
            for family in families
            for group in range(4)
        ],
        dtype=np.str_,
    )
    segment_session = np.asarray(
        [f"{family}-{split}-{group}" for family in families for group in range(4)],
        dtype=np.str_,
    )

    # raw global/family G4 evidence. speech를 별도 최악 family로 두어 source-level
    # rejection이 평균 global PASS를 우회하지 못하는지 검증한다.
    speech_db = worst_source_db if source_pass else max(worst_source_db, 0.1)
    source_values = {
        family: np.full(4, speech_db if family == "speech" else -4.1)
        for family in families
    }
    per_segment_trusted = np.concatenate(
        [source_values[family] for family in families]
    ).astype(np.float64)
    per_segment_fullband = np.full(16, -2.0, dtype=np.float64)
    per_segment_gap = per_segment_trusted - per_segment_fullband

    def _distribution(values: np.ndarray, *, worst_is_high: bool) -> dict[str, float]:
        ordered = np.sort(np.asarray(values, dtype=np.float64))
        count = max(1, int(np.ceil(ordered.size * 0.1)))
        worst = ordered[-count:] if worst_is_high else ordered[:count]
        return {
            "mean": float(np.mean(ordered)),
            "median": float(np.median(ordered)),
            "worst10": float(np.mean(worst)),
        }

    trusted_stats = _distribution(per_segment_trusted, worst_is_high=True)
    fullband_stats = _distribution(per_segment_fullband, worst_is_high=True)
    source_n_segments = []
    source_n_sessions = []
    source_n_groups = []
    source_trusted_mean = []
    source_trusted_worst10 = []
    source_fullband_mean = []
    source_fullband_worst10 = []
    source_gap_mean = []
    source_ci_lo = []
    source_ci_hi = []
    for family in families:
        family_values = source_values[family]
        family_fullband = np.full(4, -2.0, dtype=np.float64)
        stats = _distribution(family_values, worst_is_high=True)
        full_stats = _distribution(family_fullband, worst_is_high=True)
        lo, hi, _ = cluster_bootstrap_ci(
            family_values,
            np.asarray(
                [f"group-{family}-{split}-{group}" for group in range(4)],
                dtype=np.str_,
            ),
            min_groups=4,
        )
        source_n_segments.append(4)
        source_n_sessions.append(4)
        source_n_groups.append(4)
        source_trusted_mean.append(stats["mean"])
        source_trusted_worst10.append(stats["worst10"])
        source_fullband_mean.append(full_stats["mean"])
        source_fullband_worst10.append(full_stats["worst10"])
        source_gap_mean.append(float(np.mean(family_values - family_fullband)))
        source_ci_lo.append(lo)
        source_ci_hi.append(hi)

    expected_source_pass = bool(
        np.all(np.asarray(source_trusted_mean) < 0.0)
        and np.all(np.asarray(source_trusted_worst10) < 0.0)
    )
    expected_ci_pass = bool(np.all(np.asarray(source_ci_hi) < 0.0))
    worst_source_index = int(np.argmax(np.asarray(source_trusted_mean)))
    expected_worst_source_family = families[worst_source_index]
    expected_worst_source_mean = float(source_trusted_mean[worst_source_index])
    expected_worst_source_worst10 = float(np.max(source_trusted_worst10))
    global_source_ci_lo = np.asarray(source_ci_lo, dtype=np.float64)
    global_source_ci_hi = np.asarray(source_ci_hi, dtype=np.float64)

    # persisted canonical G4에는 octave별 segment raw matrix와 그 재계산 summary가
    # 모두 있어야 한다. 대상 octave만 인자로 조절해 do-no-harm negative case를 만든다.
    octave_centers = np.asarray(
        (125.0, 250.0, 500.0, 1000.0, 1600.0, 2000.0, 4000.0, 8000.0),
        dtype=np.float64,
    )
    per_segment_octave = np.full((16, octave_centers.size), 4.0, dtype=np.float64)
    requested_octave_center = float(worst_octave_center_hz)
    # do-no-harm authority는 trusted 내부 octave가 아니라 대역 밖 octave만 본다.
    # 기존 경계 fixture의 500 Hz 인자는 유지하되, raw canonical evidence에서는
    # 동률 없는 대역 밖 125 Hz 관측점으로 투영한다.
    if 150.0 <= requested_octave_center <= 1600.0:
        requested_octave_center = 125.0
    octave_index = int(np.where(octave_centers == requested_octave_center)[0][0])
    octave_target_db = (
        float(worst_octave_worst10_db)
        if do_no_harm_pass
        else min(float(worst_octave_worst10_db), -1.1)
    )
    per_segment_octave[:, octave_index] = octave_target_db
    octave_mean = np.mean(per_segment_octave, axis=0)
    octave_median = np.median(per_segment_octave, axis=0)
    octave_count = max(1, int(np.ceil(per_segment_octave.shape[0] * 0.1)))
    octave_worst10 = np.mean(
        np.sort(per_segment_octave, axis=0)[:octave_count], axis=0
    )
    octave_trusted = np.asarray(
        (False, True, True, True, True, False, False, False), dtype=np.bool_
    )
    out_of_band = ~octave_trusted
    expected_do_no_harm_pass = bool(not np.any(octave_worst10[out_of_band] <= -1.0))
    worst_octave_index = int(
        np.flatnonzero(out_of_band)[np.argmin(octave_worst10[out_of_band])]
    )

    # power_pass/ci_pass 인자가 False인 legacy negative 호출은 generic family group
    # 수를 위조할 수 없으므로, strict raw coverage를 비워 INCONCLUSIVE를 만든다.
    strict_density_value = 1.0 if power_pass and ci_pass else 0.0
    sampling_hop = int((checkpoint_cfg.get("model") or {}).get("hop", 128))
    sampling_segment_samples = effective_segment_samples(
        sample_rate=INTERLEAVED_FS,
        model_hop=sampling_hop,
        segment_seconds=CANONICAL_SEGMENT_SECONDS,
    )
    sampling_edge_trim = int(
        round(CANONICAL_EDGE_TRIM_SECONDS * INTERLEAVED_FS)
    )
    plant_settle_samples = int(checkpoint_cfg.get("loss_start_sample", 0))
    feedback_delay_samples = (
        canonical_feedback_delay_samples(checkpoint_data)
        if checkpoint_data
        else 0
    )
    sampling_warmup = (
        canonical_warmup_samples(
            checkpoint_data,
            sample_rate=INTERLEAVED_FS,
            plant_settle_samples=plant_settle_samples,
        )
        if checkpoint_data
        and int(checkpoint_data.get("sample_rate", INTERLEAVED_FS)) == INTERLEAVED_FS
        else 0
    )
    payload = {
        "split": np.asarray(split),
        "g4_metric_scope": np.asarray("canonical_recorded_g4"),
        "physics_status": np.asarray("measured_primary_path"),
        "allow_surrogate": np.asarray(False),
        "sample_rate": np.asarray(INTERLEAVED_FS, dtype=np.int64),
        "recorded_sampling_contract_schema": np.asarray(
            RECORDED_SAMPLING_CONTRACT_SCHEMA
        ),
        "recorded_sampling_canonical": np.asarray(True),
        "recorded_sampling_model_hop": np.asarray(
            sampling_hop, dtype=np.int64
        ),
        "recorded_sampling_max_segments_per_session": np.asarray(
            CANONICAL_MAX_SEGMENTS_PER_SESSION, dtype=np.int64
        ),
        "recorded_sampling_segment_seconds": np.asarray(
            CANONICAL_SEGMENT_SECONDS, dtype=np.float64
        ),
        "recorded_sampling_plant_settle_samples": np.asarray(
            plant_settle_samples, dtype=np.int64
        ),
        "segment_samples": np.asarray(
            sampling_segment_samples, dtype=np.int64
        ),
        "metric_samples_per_segment": np.asarray(
            sampling_segment_samples - sampling_warmup, dtype=np.int64
        ),
        "edge_trim_samples": np.asarray(
            sampling_edge_trim, dtype=np.int64
        ),
        "warmup_samples": np.asarray(sampling_warmup, dtype=np.int64),
        "feedback_delay_samples": np.asarray(
            feedback_delay_samples, dtype=np.int64
        ),
        "digital_reference_lead_samples": np.asarray(
            int(
                checkpoint_data.get(
                    "digital_reference_lead_samples", recorded_lead_samples
                )
            ),
            dtype=np.int64,
        ),
        "primary_delay_samples": np.asarray(
            int(
                (checkpoint_data.get("training_timing_contract") or {}).get(
                    "primary_zeros_before_fir_samples", recorded_delay_samples
                )
            ),
            dtype=np.int64,
        ),
        "secondary_delay_samples": np.asarray(
            int(
                (checkpoint_data.get("training_timing_contract") or {}).get(
                    "secondary_delay_samples", 0
                )
            ),
            dtype=np.int64,
        ),
        "secondary_handoff_samples": np.asarray(
            int(
                (checkpoint_data.get("training_timing_contract") or {}).get(
                    "handoff_samples", 0
                )
            ),
            dtype=np.int64,
        ),
        "checkpoint_sha256": np.asarray(sha256_file(checkpoint)),
        "manifest_sha256": np.asarray(sha256_file(manifest)),
        "experiment_contract_sha256": np.asarray(experiment_contract_sha256),
        "model_input_contract_sha256": np.asarray(
            str(checkpoint_cfg.get("model_input_contract_sha256", ""))
        ),
        "selection_sha256": np.asarray(selection_sha256),
        "test_capability_sha256": np.asarray(test_capability_sha256),
        "test_consumed_marker_sha256": np.asarray(
            test_consumed_marker_sha256
        ),
        "g4_trusted_pass": np.asarray(trusted_stats["mean"] < 0.0),
        "g4_fullband_pass": np.asarray(fullband_stats["mean"] <= 0.0),
        "g4_pass": np.asarray(False),
        "nmse_trusted_mean_db": np.asarray(trusted_stats["mean"]),
        "nmse_trusted_median_db": np.asarray(trusted_stats["median"]),
        "nmse_trusted_worst10_mean_db": np.asarray(trusted_stats["worst10"]),
        "nmse_fullband_mean_db": np.asarray(fullband_stats["mean"]),
        "nmse_fullband_median_db": np.asarray(fullband_stats["median"]),
        "nmse_fullband_worst10_mean_db": np.asarray(fullband_stats["worst10"]),
        "nmse_gap_trusted_minus_fullband_mean_db": np.asarray(
            float(np.mean(per_segment_gap))
        ),
        "trusted_band_hz": np.asarray((150.0, 1600.0), dtype=np.float64),
        "source_family": np.asarray(families),
        "n_sessions": np.asarray(16, dtype=np.int64),
        "n_segments": np.asarray(16, dtype=np.int64),
        "n_groups": np.asarray(16, dtype=np.int64),
        "segment_session_id": segment_session,
        "segment_source_family": segment_family,
        "segment_group_id": segment_group,
        "segment_start_sample": np.full(
            16, sampling_edge_trim, dtype=np.int64
        ),
        "per_segment_trusted_db": per_segment_trusted,
        "per_segment_fullband_db": per_segment_fullband,
        "per_segment_gap_db": per_segment_gap,
        "segment_recorded_lead_samples": np.full(
            16, recorded_lead_samples, dtype=np.int64
        ),
        "segment_recorded_delay_samples": np.full(
            16, recorded_delay_samples, dtype=np.float64
        ),
        "segment_timing_contract_sha256": np.asarray(
            [timing_contract_sha256] * 16
        ),
        "segment_source_timeline": np.asarray(["source_aligned.wav"] * 16),
        "source_n_segments": np.asarray(source_n_segments, dtype=np.int64),
        "source_n_sessions": np.asarray(source_n_sessions, dtype=np.int64),
        "source_n_groups": np.asarray(source_n_groups, dtype=np.int64),
        "source_nmse_trusted_mean_db": np.asarray(source_trusted_mean),
        "source_nmse_trusted_worst10_mean_db": np.asarray(source_trusted_worst10),
        "source_nmse_fullband_mean_db": np.asarray(source_fullband_mean),
        "source_nmse_fullband_worst10_mean_db": np.asarray(source_fullband_worst10),
        "source_gap_trusted_minus_fullband_mean_db": np.asarray(source_gap_mean),
        "source_trusted_ci_lo_db": global_source_ci_lo,
        "source_trusted_ci_hi_db": global_source_ci_hi,
        "octave_center_hz": octave_centers,
        "per_segment_octave_attenuation_db": per_segment_octave,
        "octave_attenuation_mean_db": octave_mean,
        "octave_attenuation_median_db": octave_median,
        "octave_attenuation_worst10_mean_db": octave_worst10,
        "octave_trusted": octave_trusted,
        "g4_max_out_of_band_amplification_db": np.asarray(1.0),
        "g4_worst_octave_center_hz": np.asarray(
            float(octave_centers[worst_octave_index])
        ),
        "g4_worst_octave_worst10_db": np.asarray(
            float(octave_worst10[worst_octave_index])
        ),
        "g4_min_groups_per_family": np.asarray(4, dtype=np.int64),
        "g4_underpowered_families": np.asarray([], dtype=np.str_),
        "g4_worst_source_trusted_mean_db": np.asarray(expected_worst_source_mean),
        "g4_worst_source_trusted_worst10_db": np.asarray(
            expected_worst_source_worst10
        ),
        "g4_worst_source_family": np.asarray(expected_worst_source_family),
    }
    if include_source_fields:
        # 기능 2(모든 소리 제거)는 소스별 최악값 판정이다 — 평균만 담은 옛 형식은
        # 게이트가 거부해야 한다(include_source_fields=False 로 그 회귀를 검사한다).
        payload.update(
            g4_source_pass=np.asarray(expected_source_pass),
            g4_worst_source_trusted_mean_db=np.asarray(expected_worst_source_mean),
            g4_worst_source_trusted_worst10_db=np.asarray(
                expected_worst_source_worst10
            ),
            g4_worst_source_family=np.asarray(expected_worst_source_family),
        )
    if include_strict_subband_fields:
        # canonical four-band fixture: family마다 4개 독립 group과 실제 source-energy
        # coverage를 갖는다. upper만 양수로 바꾸면 aggregate G4가 좋아도 completion은
        # fail-closed해야 한다.
        subbands = np.asarray(STRICT_TRUSTED_SUBBANDS_HZ, dtype=np.float64)
        per_segment = np.full(
            (16, len(STRICT_TRUSTED_SUBBANDS_HZ)), -2.0, dtype=np.float64
        )
        per_segment[:, -1] = float(strict_upper_subband_nmse_db)
        source_mean = np.full((len(families), 4), np.nan, dtype=np.float64)
        source_worst10 = np.full((len(families), 4), np.nan, dtype=np.float64)
        source_ci_lo = np.full((len(families), 4), np.nan, dtype=np.float64)
        source_ci_hi = np.full((len(families), 4), np.nan, dtype=np.float64)
        density = np.full((16, 4), strict_density_value, dtype=np.float64)
        coverage = density >= MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO
        source_coverage = np.zeros((len(families), 4), dtype=np.bool_)
        source_power = np.zeros((len(families), 4), dtype=np.bool_)
        source_n_segments = np.zeros((len(families), 4), dtype=np.int64)
        source_n_groups = np.zeros((len(families), 4), dtype=np.int64)
        source_coverage_fraction = np.zeros((len(families), 4), dtype=np.float64)
        source_density_mean = np.zeros((len(families), 4), dtype=np.float64)
        for family_index, family in enumerate(families):
            family_mask = segment_family == family
            for band_index in range(4):
                valid = coverage[family_mask, band_index]
                selected = per_segment[family_mask, band_index][valid]
                selected_groups = segment_group[family_mask][valid]
                source_n_segments[family_index, band_index] = selected.size
                source_n_groups[family_index, band_index] = np.unique(
                    selected_groups
                ).size
                source_coverage_fraction[family_index, band_index] = np.mean(valid)
                source_density_mean[family_index, band_index] = np.mean(
                    density[family_mask, band_index]
                )
                source_coverage[family_index, band_index] = selected.size > 0
                source_power[family_index, band_index] = bool(
                    selected.size > 0 and source_n_groups[family_index, band_index] >= 4
                )
                if selected.size:
                    source_mean[family_index, band_index] = np.mean(selected)
                    count = max(1, int(np.ceil(selected.size * 0.1)))
                    source_worst10[family_index, band_index] = np.mean(
                        np.sort(selected)[-count:]
                    )
                    source_ci_lo[family_index, band_index], source_ci_hi[
                        family_index, band_index
                    ], _ = cluster_bootstrap_ci(
                        selected, selected_groups, min_groups=4
                    )
        source_mean_pass = source_mean < 0.0
        source_worst10_pass = source_worst10 < 0.0
        source_ci_pass = source_ci_hi < 0.0
        source_pass_matrix = (
            source_coverage
            & source_power
            & source_mean_pass
            & source_worst10_pass
            & source_ci_pass
        )
        payload.update(
            strict_trusted_subband_schema=np.asarray(STRICT_TRUSTED_SUBBAND_SCHEMA),
            strict_trusted_subband_min_source_energy_density_ratio=np.asarray(
                MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO
            ),
            trusted_subband_hz=subbands,
            per_segment_trusted_subband_nmse_db=per_segment,
            per_segment_trusted_subband_coverage=coverage,
            per_segment_trusted_subband_source_energy_density_ratio=density,
            source_trusted_subband_n_segments=source_n_segments,
            source_trusted_subband_n_groups=source_n_groups,
            source_trusted_subband_coverage_fraction=source_coverage_fraction,
            source_trusted_subband_source_energy_density_ratio_mean=source_density_mean,
            source_trusted_subband_nmse_mean_db=source_mean,
            source_trusted_subband_nmse_worst10_mean_db=source_worst10,
            source_trusted_subband_ci_lo_db=source_ci_lo,
            source_trusted_subband_ci_hi_db=source_ci_hi,
            source_trusted_subband_coverage_pass=source_coverage,
            source_trusted_subband_power_pass=source_power,
            source_trusted_subband_mean_pass=source_mean_pass,
            source_trusted_subband_worst10_pass=source_worst10_pass,
            source_trusted_subband_ci_pass=source_ci_pass,
            source_trusted_subband_pass=source_pass_matrix,
            g4_trusted_subband_schema_pass=np.asarray(True),
            g4_trusted_subband_coverage_pass=np.asarray(np.all(source_coverage)),
            g4_trusted_subband_power_pass=np.asarray(np.all(source_power)),
            g4_trusted_subband_mean_pass=np.asarray(bool(np.all(source_mean_pass))),
            g4_trusted_subband_worst10_pass=np.asarray(
                bool(np.all(source_worst10_pass))
            ),
            g4_trusted_subband_ci_pass=np.asarray(bool(np.all(source_ci_pass))),
            g4_trusted_subband_pass=np.asarray(bool(np.all(source_pass_matrix))),
            g4_upper_trusted_subband_pass=np.asarray(
                bool(np.all(source_pass_matrix[:, -1]))
            ),
        )
        strict_hard_failure = bool(
            np.any(np.isfinite(source_mean) & (source_mean >= 0.0))
            or np.any(np.isfinite(source_worst10) & (source_worst10 >= 0.0))
        )
        strict_inconclusive = not bool(
            np.all(source_coverage)
            and np.all(source_power)
            and np.all(source_ci_pass)
        )
    else:
        strict_hard_failure = False
        strict_inconclusive = True

    expected_verdict = (
        "FAIL"
        if (
            not bool(trusted_stats["mean"] < 0.0)
            or not bool(fullband_stats["mean"] <= 0.0)
            or not expected_source_pass
            or not expected_do_no_harm_pass
            or strict_hard_failure
        )
        else "INCONCLUSIVE"
        if (not expected_ci_pass or strict_inconclusive)
        else "PASS"
    )
    resolved_verdict = expected_verdict if verdict is None else verdict
    if include_modern_fields:
        # 2026-08-05 신설 판정. 이 필드들이 없는 산출물은 게이트가 거부해야 한다
        # (include_modern_fields=False 로 그 회귀를 검사한다).
        payload.update(
            g4_verdict=np.asarray(resolved_verdict),
            g4_do_no_harm_pass=np.asarray(expected_do_no_harm_pass),
            g4_power_pass=np.asarray(True),
            g4_ci_pass=np.asarray(expected_ci_pass),
            g4_pass=np.asarray(resolved_verdict == "PASS"),
            g4_worst_octave_center_hz=np.asarray(
                float(octave_centers[worst_octave_index])
            ),
            g4_worst_octave_worst10_db=np.asarray(
                float(octave_worst10[worst_octave_index])
            ),
            g4_max_out_of_band_amplification_db=np.asarray(1.0),
            source_trusted_ci_hi_db=global_source_ci_hi,
            plant_fingerprint_json=np.asarray(
                fingerprint if fingerprint is not None else _plant_fingerprint_payload()
            ),
        )
    np.savez_compressed(path, **payload)


def test_official_path_gate_rejects_wrong_channel_and_low_consistency(tmp_path):
    wrong = tmp_path / "wrong.npz"
    _official_path(wrong, channel="cancel", delay=4, consistency=0.8)

    try:
        audit_official_path_model(
            wrong,
            expected_output_channel="noise",
            sample_rate=FS,
            required_band_hz=(100, 1_000),
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - 명시적 실패 메시지를 보기 위한 방어
        raise AssertionError("invalid official path가 통과했습니다")

    assert "output_channel" in message
    assert "일관성" in message


def test_readiness_passes_only_with_official_paths_completed_init_and_full_recorded_qa(
    tmp_path,
):
    cfg = _ready_config(tmp_path)
    report = audit_finetune_readiness(cfg)

    assert report["ok"], report
    assert {item["id"] for item in report["checks"]} == {
        "config_fail_closed_flags",
        "absolute_objective_scope",
        "measured_primary_mode",
        "recorded_mix_ratio",
        "official_secondary_path",
        "official_primary_path",
        "matched_path_measurement_conditions",
        "path_delay_and_lead",
        "completed_init_checkpoint",
        "recorded_dataset_qa",
        # --- 2026-08-05 신설 ---
        # 각각 실측으로 확인된 결함 하나씩에 대응한다. 이 집합에서 항목을 빼면
        # 그 결함을 다시 통과시키는 것이므로 테스트가 즉시 깨진다.
        "recorded_alignment_integrity",   # 결함 2: source→ERR 관계가 존재하는가
        "recorded_statistical_power",     # 결함 4 / D3: 계열당 그룹 ≥ 하한
        "recorded_subband_coverage",      # G2d: family×150–1600Hz target 에너지
        "corpus_disjoint",                # D1: 합성 ∩ 실측 원본 = ∅
        "measured_source_delay_agreement",  # D2: 실측 지연 == P(z) 유도값
        "plant_confidence_ceiling",       # G1c: 목표가 달성 가능 상한 안인가
    }
    assert require_finetune_readiness(cfg)["ok"]

    cfg["data"]["training_timing_contract"][
        "primary_zeros_before_fir_samples"
    ] = PRIMARY_DELAY_SAMPLES + 1
    failed = audit_finetune_readiness(cfg)
    assert not failed["ok"]
    assert not next(
        item for item in failed["checks"] if item["id"] == "path_delay_and_lead"
    )["ok"]


def test_readiness_rejects_timing_invalid_or_legacy_path_metadata(tmp_path):
    cfg = _ready_config(tmp_path)
    legacy = tmp_path / "legacy_secondary.npz"
    np.savez(
        legacy,
        fir=np.ones(4, dtype=np.float32),
        delay_samples=5,
        sample_rate=FS,
        coherence_median=0.4,
        excitation_band_hz=np.asarray([150.0, 600.0]),
    )
    cfg["duct"]["secondary_path"]["npz"] = str(legacy)

    report = audit_finetune_readiness(cfg)
    path_check = next(
        item for item in report["checks"] if item["id"] == "official_secondary_path"
    )

    assert not report["ok"]
    assert not path_check["ok"]
    assert "official ESS 품질 메타데이터" in path_check["message"]


def test_completion_requires_same_checkpoint_and_manifest_sha_for_val_and_test(tmp_path):
    cfg = _ready_config(tmp_path)
    run = tmp_path / "finetune"
    best = run / "ckpt" / "best.pt"
    saved_cfg = {
        **cfg,
        "physics_status": "measured_primary_path",
        "digital_reference_lead_samples": 3,
    }
    _checkpoint(best, cfg=saved_cfg, step=4)
    _checkpoint(best.parent / "last.pt", cfg=saved_cfg, step=6)
    manifest = Path(cfg["recorded_manifest"])
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    _g4_metrics(test_metrics, split="test", checkpoint=best, manifest=manifest)

    passed = audit_finetune_completion(
        cfg,
        checkpoint=best,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    assert passed["ok"], passed
    assert passed["fine_tuning_complete"]

    with np.load(test_metrics, allow_pickle=False) as current:
        arrays = {key: current[key] for key in current.files}
    arrays["checkpoint_sha256"] = np.asarray("0" * 64)
    np.savez_compressed(test_metrics, **arrays)
    failed = audit_finetune_completion(
        cfg,
        checkpoint=best,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    assert not failed["ok"]
    assert not failed["fine_tuning_complete"]
    assert "SHA-256" in next(
        item for item in failed["checks"] if item["id"] == "recorded_test_g4"
    )["message"]


def test_canonical_completion_verifies_selection_capability_marker_metrics_chain(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "source"
    (source_root / "src").mkdir(parents=True)
    (source_root / "src" / "training.py").write_text("VALUE = 1\n")
    manifest = source_root / "data" / "manifests" / "recorded_regrouped.jsonl"
    manifest.parent.mkdir(parents=True)
    # canonical persisted G4 validator는 selected split의 모든 session/family/group를
    # metrics raw metadata와 전단사로 결속한다. capability-chain fixture도 최소한의
    # 유효한 canonical population을 제공해야 한다(실제 오디오 파일은 이 테스트에서
    # 읽지 않는다).
    manifest_entries = []
    for split_name in ("val", "test"):
        for family in sorted(FAMILIES):
            for group in range(4):
                session_id = f"{family}-{split_name}-{group}"
                manifest_entries.append(
                    {
                        "path": str(source_root / "recorded" / session_id),
                        "duration_s": 2.1,
                        "sample_rate": 48_000,
                        "session_id": session_id,
                        "group_id": f"group-{family}-{split_name}-{group}",
                        "source_family": family,
                        "split": split_name,
                    }
                )
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in manifest_entries)
    )
    # canonical config 로더는 RIR bank를 저장소 내부의 실제 regular NPZ로
    # 검증한다. 임시 source commit에도 최소 유효 bank를 넣어, 제품 검증을
    # 실제 workspace artifact로 우회하지 않게 한다.
    rir_bank = source_root / "data" / "rir_bank" / "duct_rirs_v1.npz"
    rir_bank.parent.mkdir(parents=True)
    np.savez_compressed(
        rir_bank,
        p_ref=np.zeros((1, 1), dtype=np.float32),
        p_err=np.zeros((1, 1), dtype=np.float32),
        f_fb=np.zeros((1, 1), dtype=np.float32),
        sample_rate=np.asarray(48_000, dtype=np.int64),
        seed=np.asarray(20260802, dtype=np.int64),
    )
    (source_root / ".gitignore").write_text("/results/\n/recorded/\n")
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "add", "src", "data", ".gitignore"], cwd=source_root, check=True
    )
    subprocess.run(
        [
            "git", "-c", "user.name=tests", "-c",
            "user.email=tests@example.invalid", "commit", "-qm", "source",
        ],
        cwd=source_root,
        check=True,
    )
    import deep_anc.config as config_module
    import deep_anc.data.transfer_contract as transfer_contract_module

    original_resolve_path = config_module._resolve_path

    def _resolve_fixture_rir(path):
        if Path(path).as_posix() == "data/rir_bank/duct_rirs_v1.npz":
            return rir_bank
        return original_resolve_path(path)

    monkeypatch.setattr(config_module, "REPO_ROOT", source_root)
    monkeypatch.setattr(config_module, "_resolve_path", _resolve_fixture_rir)
    monkeypatch.setattr(readiness, "REPO_ROOT", source_root)
    monkeypatch.setattr(
        transfer_contract_module,
        "bind_recorded_transfer_config",
        lambda data, repo_root: data.update(
            transfer_manifest="data/manifests/elice_transfer_manifest.json",
            transfer_manifest_sha256="b" * 64,
            recorded_transfer_aggregate_sha256="c" * 64,
        ),
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
        capture_output=True, text=True,
    ).stdout

    def _completed_canonical_run(run: Path, run_cfg: dict, *, total_steps: int) -> Path:
        best_checkpoint = run / "ckpt" / "best.pt"
        _checkpoint(best_checkpoint, cfg=run_cfg, step=1_000)
        _checkpoint(
            best_checkpoint.parent / "last.pt", cfg=run_cfg, step=total_steps
        )
        (run / "config_snapshot.yaml").write_text(
            yaml.safe_dump(run_cfg, allow_unicode=True, sort_keys=False)
        )
        (run / "git_rev.txt").write_text(commit)
        (run / "pip_freeze.txt").write_text("fixture==1\n")
        (run / "environment.json").write_text(
            json.dumps(
                {
                    "python": "fixture",
                    "torch": "fixture",
                    "cuda_available": False,
                    "device_count": 0,
                    "devices": [],
                    "deterministic_algorithms": True,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cublas_workspace_config": None,
                }
            )
            + "\n"
        )
        write_completion_receipt(best_checkpoint.parent, repo_root=source_root)
        return best_checkpoint

    pretrain_cfg = load_train_config(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        [
            f"data.bootstrap_receipt_sha256={'a' * 64}",
            f"campaign_prerequisite_sha256={'d' * 64}",
        ],
    )
    pretrain_best = _completed_canonical_run(
        source_root / "results" / "runs" / "pretrain",
        pretrain_cfg,
        total_steps=100_000,
    )
    cfg = load_train_config(
        REPO_ROOT / "configs/train_finetune.yaml",
        [
            "data.digital_primary_path_mode=measured",
            f"data.bootstrap_receipt_sha256={'a' * 64}",
            f"init_ckpt={json.dumps(str(pretrain_best))}",
        ],
    )
    timing = TrainingTimingContract.from_data_config(cfg["data"])
    for entry in manifest_entries:
        session_dir = Path(entry["path"])
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session.json").write_text(
            json.dumps(
                {
                    "timeline": {
                        "aligned_lag_median_samples": float(
                            timing.primary_effective_delay_samples
                        )
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
    run = source_root / "results" / "runs" / "canonical"
    best = _completed_canonical_run(run, cfg, total_steps=50_000)

    import deep_anc.train.campaign_prerequisite as campaign_module

    prerequisite_validations: list[int] = []
    monkeypatch.setattr(
        campaign_module,
        "validate_canonical_pretrain_prerequisites",
        lambda init_cfg, *, repo_root: prerequisite_validations.append(
            int(init_cfg["seed"])
        ),
    )

    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    contract_sha = cfg["experiment_contract_sha256"]
    _g4_metrics(
        val_metrics,
        split="val",
        checkpoint=best,
        manifest=manifest,
        experiment_contract_sha256=contract_sha,
        timing_contract_sha256=timing.digest(),
        recorded_lead_samples=timing.digital_reference_lead_samples,
        recorded_delay_samples=timing.primary_effective_delay_samples,
    )
    val_decision = classify_recorded_val_metrics(
        val_metrics.read_bytes(),
        manifest_bytes=manifest.read_bytes(),
        manifest_path=manifest,
        checkpoint_cfg=cfg,
    )
    selection_path = source_root / "results" / "selection.json"
    campaign_sha = seed_neutral_campaign_sha256(cfg)
    selection = {
        "schema_version": 1,
        "selection_split": "val",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "experiment_contract_sha256": contract_sha,
        "model_input_contract_sha256": cfg["model_input_contract_sha256"],
        "seed_neutral_campaign_sha256": campaign_sha,
        "seed": 20260803,
        "decision": val_decision,
        "selected": {
            "checkpoint": str(best),
            "checkpoint_sha256": sha256_file(best),
            "evaluation_dir": str(val_metrics.parent),
            "metrics_sha256": sha256_file(val_metrics),
            "seed": 20260803,
            "seed_neutral_campaign_sha256": campaign_sha,
            "model_input_contract_sha256": cfg[
                "model_input_contract_sha256"
            ],
            "decision": val_decision,
        },
        "candidates": [],
    }
    write_json_exclusive(selection_path, selection)
    capability_path, consumed_path = canonical_test_ledger_paths(
        selection_path, repo_root=source_root
    )
    token = issue_test_capability(
        selection_path=selection_path,
        capability_path=capability_path,
        repo_root=source_root,
    )
    assert prerequisite_validations == [20260803]
    _, _, consumed = consume_test_capability(
        selection_path=selection_path,
        capability_path=capability_path,
        consumed_marker_path=consumed_path,
        token=token,
        checkpoint_path=best,
        manifest_path=manifest,
        repo_root=source_root,
    )
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(
        test_metrics,
        split="test",
        checkpoint=best,
        manifest=manifest,
        experiment_contract_sha256=contract_sha,
        timing_contract_sha256=timing.digest(),
        selection_sha256=consumed["selection_sha256"],
        test_capability_sha256=consumed["capability_sha256"],
        test_consumed_marker_sha256=snapshot_regular_file(consumed_path).sha256,
        recorded_lead_samples=timing.digital_reference_lead_samples,
        recorded_delay_samples=timing.primary_effective_delay_samples,
    )
    (test_metrics.parent / "metrics.md").write_text("fixture\n")
    complete_test_evaluation(
        selection_path=selection_path,
        capability_path=capability_path,
        consumed_marker_path=consumed_path,
        output_dir=test_metrics.parent,
        repo_root=source_root,
    )
    monkeypatch.setattr(
        readiness,
        "audit_finetune_readiness",
        lambda cfg, full_recorded_qa=True: {"ok": True, "checks": []},
    )
    passed = audit_finetune_completion(
        cfg,
        checkpoint=best,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        selection=selection_path,
        test_capability=capability_path,
        test_consumed_marker=consumed_path,
    )
    assert passed["ok"], passed

    selection_path.write_text(selection_path.read_text() + " ")
    attacked = audit_finetune_completion(
        cfg,
        checkpoint=best,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        selection=selection_path,
        test_capability=capability_path,
        test_consumed_marker=consumed_path,
    )
    assert not attacked["ok"]
    chain_gate = next(
        item
        for item in attacked["checks"]
        if item["id"] == "recorded_selection_test_once_chain"
    )
    assert not chain_gate["ok"]


def _completion_setup(tmp_path):
    """완료 감사에 필요한 최소 구성. 위 테스트와 같은 뼈대를 재사용한다."""

    cfg = _ready_config(tmp_path)
    run = Path(cfg["ckpt_dir"])
    best = run / "ckpt" / "best.pt"
    saved_cfg = {
        **cfg,
        "physics_status": "measured_primary_path",
        "digital_reference_lead_samples": 3,
    }
    _checkpoint(best, cfg=saved_cfg, step=4)
    _checkpoint(best.parent / "last.pt", cfg=saved_cfg, step=6)
    return cfg, best, Path(cfg["recorded_manifest"]), run


def test_g4_rejects_model_that_amplifies_one_source_family(tmp_path):
    """기능 2 회귀 방어 — 평균이 좋아도 **최악 소스**가 증폭이면 완료가 아니다.

    이 게이트가 없으면 machine −8 dB / speech +6 dB 가 섞여 평균 −1.75 dB 로 통과한다.
    즉 대화를 6 dB 키우는 모델이 `[COMPLETE] G4 PASS` 로 배포 후보가 된다.
    """

    cfg, best, manifest, run = _completion_setup(tmp_path)
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    # test 쪽만 최악 소스가 증폭(+6 dB)인 상황
    _g4_metrics(
        test_metrics, split="test", checkpoint=best, manifest=manifest,
        source_pass=False, worst_source_db=6.0,
    )

    result = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )
    assert not result["ok"]
    assert not result["fine_tuning_complete"]
    message = next(
        item for item in result["checks"] if item["id"] == "recorded_test_g4"
    )["message"]
    assert "기능2" in message and "speech" in message and "+6.00" in message


def test_g4_rejects_metrics_without_worst_source_fields(tmp_path):
    """최악 소스를 판정하지 않던 **구버전 평가기의 산출물**은 통과시키지 않는다.

    필드가 없으면 조용히 통과시키는 것이 가장 위험하다 — 게이트가 있다고 믿으면서
    실제로는 평균만 보게 된다.
    """

    cfg, best, manifest, run = _completion_setup(tmp_path)
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    _g4_metrics(
        test_metrics, split="test", checkpoint=best, manifest=manifest,
        include_source_fields=False,
    )

    result = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )
    assert not result["ok"]
    assert "g4_source_pass" in next(
        item for item in result["checks"] if item["id"] == "recorded_test_g4"
    )["message"]


# ---------------------------------------------------------------------------
# interleaved_multitone — P/S 동시 측정 방식
#
# 이 방식을 허용하는 것은 게이트를 넓히는 것이 아니다. ESS 가 요구하지 않는 항목
# (guard=1, 분석창 길이, 톤 수, 톤 SNR)을 추가로 요구하고, 무엇보다 두 파일이 같은
# capture 에서 나왔는지를 **파일에 박힌 capture_id 로** 확인한다. 진폭·블록·latency 가
# 우연히 같은 서로 다른 두 측정은 여기서 걸린다.
# ---------------------------------------------------------------------------


def _check(report: dict, name: str) -> dict:
    return next(item for item in report["checks"] if item["id"] == name)


def test_interleaved_method_is_accepted_with_full_metadata(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    report = audit_official_path_model(
        path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
        required_band_hz=(100.0, 1_000.0),
    )
    assert report["method"] == "interleaved_multitone"
    assert report["interleaved"]["capture_id"] == "cap-1"


def test_interleaved_without_extra_metadata_is_rejected(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4)  # ess 필드만
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["method"] = np.asarray("interleaved_multitone")
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="interleaved 측정 메타데이터가 없습니다"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def _rewrite_npz(path: Path, *, drop: set[str] = frozenset(), **updates) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key not in drop}
    arrays.update({key: np.asarray(value) for key, value in updates.items()})
    np.savez(path, **arrays)


def _source_path_from_official(path: Path, field: str) -> Path:
    with np.load(path, allow_pickle=False) as data:
        relative = str(np.asarray(data[field]).reshape(-1)[0])
    return REPO_ROOT / relative


def _rewrite_source_and_refresh_hash(
    official_paths: list[Path], *, source_field: str, sha_field: str, **updates
) -> Path:
    source = _source_path_from_official(official_paths[0], source_field)
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays.update({key: np.asarray(value) for key, value in updates.items()})
    np.savez(source, **arrays)
    digest = sha256_file(source)
    for official in official_paths:
        _rewrite_npz(official, **{sha_field: np.asarray(digest)})
    return source


def test_interleaved_delay_semantics_is_required(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    _rewrite_npz(path, drop={"delay_semantics"})

    with pytest.raises(ValueError, match="delay_semantics"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_effective_bulk_preroll_relation_is_enforced(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    _rewrite_npz(path, bulk_delay_samples=np.int64(999))

    with pytest.raises(ValueError, match="delay 계약 위반"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_source_digest_is_recomputed(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    with np.load(path, allow_pickle=False) as data:
        real = np.asarray(data["aligned_mean_transfer_real"]).copy()
    real[10] += 0.25
    _rewrite_npz(path, aligned_mean_transfer_real=real)

    with pytest.raises(ValueError, match="SHA256"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_fir_is_independently_reaudited(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    _rewrite_npz(path, fir=np.asarray([0.5, 0.4, -0.3], dtype=np.float32))

    with pytest.raises(ValueError, match="compact"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_weak_compact_subband_is_rejected_after_valid_digest(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    with np.load(path, allow_pickle=False) as data:
        frequencies = np.asarray(data["tone_frequencies_hz"]).copy()
        values = np.asarray(data["aligned_mean_transfer_real"]) + 1j * np.asarray(
            data["aligned_mean_transfer_imag"]
        )
    mask = (frequencies >= 1_000.0) & (frequencies <= 1_600.0)
    values[mask] *= np.exp(1j * np.linspace(0.0, np.pi, int(mask.sum())))
    digest = readiness._aligned_transfer_sha256(
        frequencies, values.real, values.imag
    )
    _rewrite_npz(
        path,
        aligned_mean_transfer_real=values.real,
        aligned_mean_transfer_imag=values.imag,
        aligned_mean_transfer_sha256=np.asarray(digest),
    )

    with pytest.raises(ValueError, match="1000-1600Hz compact"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


@pytest.mark.parametrize(
    "override, pattern",
    [
        ({"interleave_guard_bins": 2}, "interleave_guard_bins"),
        ({"analysis_period_seconds": 3.7}, "analysis_period_seconds"),
        ({"tone_count": 32}, "tone_count"),
        ({"tone_snr_median_db": 6.0}, "tone_snr_median_db"),
        ({"capture_id": ""}, "capture_id"),
        ({"consistency_band_hz": [300.0, 500.0]}, "일관성 측정 대역"),
        ({"calibration_block_size": 512}, "block_size"),
        ({"calibration_latency": "high"}, "latency"),
        ({"sample_rate": 8_000}, "sample_rate"),
        ({"error_mic_channel": 1}, "error_mic_channel"),
        (
            {"operator_confirmed_routing_and_geometry": False},
            "operator_confirmed_routing_and_geometry",
        ),
        ({"coherence_median": 0.949999}, "반복 일관성"),
        (
            {"band_consistency": [0.99, 0.99, 0.949999, 0.99]},
            "600-1000Hz 일관성",
        ),
    ],
)
def test_interleaved_quality_fields_are_enforced(tmp_path, override, pattern):
    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4,
        method="interleaved_multitone", interleaved=override,
    )
    with pytest.raises(ValueError, match=pattern):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"repeats": 11}, "repeats=11"),
        ({"kept_repeat_indices": [0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
         "strictly sorted unique"),
        ({"kept_repeat_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16]},
         "alignment_scores 범위"),
        ({"anchor_repeat": 15}, "anchor_repeat가 kept_repeat_indices"),
        ({"alignment_scores": np.ones((4, 4))}, "alignment_scores는"),
    ],
)
def test_interleaved_repeat_provenance_is_enforced(tmp_path, override, pattern):
    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4,
        method="interleaved_multitone", interleaved=override,
    )

    with pytest.raises(ValueError, match=pattern):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"output_pcm_provenance": "derived_not_observed"}, "output_pcm_provenance"),
        ({"clock_sample_rate": INTERLEAVED_FS + 1}, "clock_sample_rate"),
        ({"clock_max_drift_deviation_samples": 2.1}, "drift_deviation"),
        ({"clock_q_ratio": np.ones(15)}, "clock_q_ratio"),
        ({"clock_err_score": np.full(15, 0.90)}, "correlation score"),
        ({"clock_err_subwindow_spread_samples": np.full(15, -0.01)}, "음수"),
        ({"clock_observation_repeat_indices": np.arange(1, 16)}, "원본 반복 범위"),
        ({"joint_ls_condition": np.full(15, 0.9)}, "condition은 1 미만"),
        ({"joint_ls_condition": np.full(15, 1.3)}, "condition이 공식 상한"),
        ({"joint_ls_reconstruction_relative_error": np.full(15, -0.01)}, "residual은 음수"),
        ({"separation_crosscheck_complex_agreement": 0.998}, "crosscheck"),
        ({"relative_tau_max_abs_samples": 1.01}, "relative_tau_max_abs"),
        ({"repeat_tau_samples": np.linspace(0.0, 1.0, 12)}, "repeat_tau_samples"),
        ({"clock_min_adjacent_score": 0.90}, "공식 고정값"),
    ],
)
def test_interleaved_fractional_separation_tamper_fails_closed(
    tmp_path, override, pattern
):
    path = tmp_path / "primary.npz"
    _official_path(
        path,
        channel="noise",
        delay=4,
        method="interleaved_multitone",
        interleaved=override,
    )

    with pytest.raises(ValueError, match=pattern):
        audit_official_path_model(
            path,
            expected_output_channel="noise",
            sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_source_raw_sha_is_recomputed(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    raw_source = _source_path_from_official(path, "source_raw_npz_path")
    raw_source.write_bytes(raw_source.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="source_raw_npz_sha256 불일치"):
        audit_official_path_model(
            path,
            expected_output_channel="noise",
            sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_source_analysis_version_name_is_tightly_anchored(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    _rewrite_npz(
        path,
        source_analysis_npz_path=np.asarray(
            "results/session/analysis_results.reanalysis_untrusted.npz"
        ),
    )

    with pytest.raises(ValueError, match="source_analysis_npz_path basename"):
        audit_official_path_model(
            path,
            expected_output_channel="noise",
            sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_raw_channel_map_must_match_official_and_fixed_routing(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    raw_source = _source_path_from_official(path, "source_raw_npz_path")
    with np.load(raw_source, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
        metadata = json.loads(str(data["metadata_json"]))
    metadata["channel_map"]["error_mic"] = 1
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(raw_source, **arrays)
    _rewrite_npz(path, source_raw_npz_sha256=np.asarray(sha256_file(raw_source)))

    with pytest.raises(ValueError, match="source raw channel_map"):
        audit_official_path_model(
            path,
            expected_output_channel="noise",
            sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_dead_measurement_raw_is_rejected_even_with_matching_report(
    tmp_path,
):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    raw_source = _source_path_from_official(path, "source_raw_npz_path")
    with np.load(raw_source, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
        metadata = json.loads(str(data["metadata_json"]))
    dead = np.zeros_like(arrays["input_raw_int32"])
    metadata["measurement"] = readiness.analyze_int32_input_probe(dead)
    arrays["input_raw_int32"] = dead
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(raw_source, **arrays)
    _rewrite_npz(path, source_raw_npz_sha256=np.asarray(sha256_file(raw_source)))

    with pytest.raises(ValueError, match="measurement ERR/REF channel이 유효하지"):
        audit_official_path_model(
            path,
            expected_output_channel="noise",
            sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_raw_playback_must_equal_reconstructed_probe(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    raw_source = _source_path_from_official(path, "source_raw_npz_path")
    with np.load(raw_source, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    output = np.asarray(arrays["output"]).copy()
    output[0, 0] += np.float32(1e-4)
    arrays["output"] = output
    arrays["output_pcm_int16"] = np.rint(
        np.clip(output, -1.0, 1.0) * np.float32(np.iinfo(np.int16).max)
    ).astype(np.int16)
    np.savez(raw_source, **arrays)
    _rewrite_npz(path, source_raw_npz_sha256=np.asarray(sha256_file(raw_source)))

    with pytest.raises(ValueError, match="probe 재구성과 다릅니다"):
        audit_official_path_model(
            path,
            expected_output_channel="noise",
            sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_interleaved_source_clock_valid_mask_is_independently_recomputed(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    source = _source_path_from_official(path, "source_analysis_npz_path")
    with np.load(source, allow_pickle=False) as data:
        mask = np.asarray(data["clock_valid_mask"]).copy()
    mask[0] = False
    _rewrite_source_and_refresh_hash(
        [path],
        source_field="source_analysis_npz_path",
        sha_field="source_analysis_npz_sha256",
        clock_valid_mask=mask,
    )

    with pytest.raises(ValueError, match="clock_valid_mask.*재계산"):
        audit_official_path_model(
            path,
            expected_output_channel="noise",
            sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_readiness_clock_mask_uses_same_final_median_fixed_point_contract():
    common = np.asarray([0.0] * 5 + [2.0] + [4.0] * 5 + [np.nan])
    base_valid = np.isfinite(common)
    adjacent = np.full(common.size, np.nan, dtype=np.float64)
    adjacent[1:-1] = np.abs(np.diff(common[:-1]))

    with pytest.raises(ValueError, match="final-median fixed-point.*8개 미만"):
        readiness._fixed_point_clock_valid_mask(
            base_valid=base_valid,
            common_delay_samples=common,
            adjacent_change_samples=adjacent,
            max_drift_deviation_samples=2.0,
            max_adjacent_change_samples=0.5,
            min_valid_periods=8,
        )


def test_interleaved_direct_official_passes_fixed_point_clock_reaudit(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")

    report = audit_official_path_model(
        path,
        expected_output_channel="noise",
        sample_rate=INTERLEAVED_FS,
        required_band_hz=(100.0, 1_000.0),
    )

    assert report["method"] == "interleaved_multitone"


# ---------------------------------------------------------------------------
# 2026-08-05 결함 1 회귀 — 아래 세 게이트가 없으면 33%(형상 기준 50%) 틀린 S(z) 가
# 다시 파인튜닝까지 들어간다. 셋 다 실제로 통과했던 값으로 테스트한다.
# ---------------------------------------------------------------------------


def test_artifact_cannot_declare_its_own_delay_jitter_allowance(tmp_path):
    """허용치를 아티팩트에서 읽으면 게이트가 자기증명이 된다.

    출하본은 delay_spread 32 를 **같은 NPZ 안의** max_delay_jitter 48 과 비교해
    통과했다. 측정 시 --max-delay-jitter-ms 를 키우면 검사가 사라지는 구조였다.
    """

    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={"delay_spread_samples": 32, "max_delay_jitter_samples": 48},
    )
    with pytest.raises(ValueError, match="상대 τ spread 32 > 허용 3"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_missing_sub_band_consistency_is_rejected(tmp_path):
    """최악 부대역을 검증할 수 없는 아티팩트는 official 이 될 수 없다."""

    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="interleaved_multitone")
    with np.load(path) as data:
        arrays = {
            key: data[key] for key in data.files
            if key not in {"band_consistency", "band_consistency_hz"}
        }
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="band_consistency"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_weak_sub_band_is_rejected_even_when_the_total_passes(tmp_path):
    """총계는 에너지 가중이라 약한 대역을 숨긴다 — 최악값이 판정 기준이다."""

    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, consistency=0.99,
        method="interleaved_multitone",
        # 총계 0.99 는 통과하지만 600-1000Hz 부대역만 0.73 이다(출하본 실측값).
        interleaved={"band_consistency": [0.99, 0.99, 0.73, 0.99]},
    )
    with pytest.raises(ValueError, match="부대역 600-1000Hz 일관성 0.7300"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_canonical_sub_band_is_judged_even_outside_requested_band(tmp_path):
    """설정으로 required band를 좁혀 canonical 4개 중 하나를 숨길 수 없다."""

    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={"band_consistency": [0.99, 0.99, 0.99, 0.40]},
    )
    with pytest.raises(ValueError, match="canonical 부대역 1000-1600Hz"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 600.0),
        )


def test_all_canonical_sub_bands_pass_at_the_required_boundary(tmp_path):
    """canonical 4개 부대역이 하한 0.9406에 정확히 붙어도 통과한다."""

    path = tmp_path / "primary.npz"
    _official_path(
        path,
        channel="noise",
        delay=4,
        method="interleaved_multitone",
        consistency=0.95,
        interleaved={
            "consistency_band_hz": [150.0, 1_600.0],
            "band_consistency": [0.95, 0.95, 0.95, 0.95],
        },
    )
    model = audit_official_path_model(
        path,
        expected_output_channel="noise",
        sample_rate=INTERLEAVED_FS,
        required_band_hz=(150.0, 1_600.0),
        min_consistency=0.95,
    )
    assert model["consistency"] == pytest.approx(0.95)


@pytest.mark.parametrize(
    "params, pattern",
    [
        ({"min_alignment_score": 0.5}, "min_alignment_score=0.5 < 0.95"),
        ({"max_relative_tau_samples": 48.0}, "max_relative_tau_samples=48.0 > 3.0"),
        ({"max_drift_deviation_samples": 25.0}, "max_drift_deviation_samples"),
        ({"min_kept_repeats": 3}, "min_kept_repeats=3 < 8"),
        ({"min_alignment_score": None}, "재분석 파라미터 min_alignment_score 가 없습니다"),
    ],
)
def test_reanalysis_parameter_envelope_is_enforced(tmp_path, params, pattern):
    """재분석 도구가 게이트를 약화한 값으로 푼 결과는 official 이 될 수 없다."""

    envelope = {
        "min_alignment_score": 0.95,
        "max_relative_tau_samples": 3.0,
        "max_drift_deviation_samples": 2.0,
        "min_kept_repeats": 8,
    }
    for key, value in params.items():
        if value is None:
            envelope.pop(key)
        else:
            envelope[key] = value
    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={"reanalysis_params_json": json.dumps(envelope, sort_keys=True)},
    )
    with pytest.raises(ValueError, match=pattern):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def test_reanalysis_inside_the_envelope_is_accepted(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(
        path, channel="noise", delay=4, method="interleaved_multitone",
        interleaved={
            "reanalysis_params_json": json.dumps(
                {
                    "min_alignment_score": 0.95,
                    "max_relative_tau_samples": 3.0,
                    "max_drift_deviation_samples": 2.0,
                    "min_kept_repeats": 8,
                },
                sort_keys=True,
            )
        },
    )
    report = audit_official_path_model(
        path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
        required_band_hz=(100.0, 1_000.0),
    )
    assert report["interleaved"]["reanalysis_params"]["min_kept_repeats"] == 8


def test_legacy_shipped_interleaved_artifacts_fail_closed():
    """새 delay/source 계약이 없는 과거 interleaved NPZ는 재사용할 수 없다."""

    duct = load_yaml(REPO_ROOT / "configs/duct.yaml")
    finetune = load_yaml(REPO_ROOT / "configs/train_finetune.yaml")
    band = tuple(float(v) for v in finetune["readiness"]["required_path_band_hz"])
    # 현재 duct.yaml은 strict capture를 가리킨다. 이 테스트는 이름 그대로
    # 과거에 배송된 legacy artifact가 계속 fail-closed인지 확인한다.
    for key, channel in (
        ("assets/measured/primary_path_il.npz", "noise"),
        ("assets/measured/secondary_path_il.npz", "cancel"),
    ):
        with pytest.raises(ValueError, match="interleaved 측정 메타데이터가 없습니다"):
            audit_official_path_model(
                REPO_ROOT / key, expected_output_channel=channel,
                sample_rate=48_000, required_band_hz=band,
                min_consistency=float(finetune["readiness"]["min_path_consistency"]),
            )


def test_unknown_method_is_still_rejected(tmp_path):
    path = tmp_path / "primary.npz"
    _official_path(path, channel="noise", delay=4, method="white_noise")
    with pytest.raises(ValueError, match="허용 method"):
        audit_official_path_model(
            path, expected_output_channel="noise", sample_rate=INTERLEAVED_FS,
            required_band_hz=(100.0, 1_000.0),
        )


def _official_current_interleaved_path(
    path: Path,
    *,
    channel: str,
    delay: int,
    interleaved: dict | None = None,
) -> None:
    """Current fine-tune의 절대 대역 150–1600 Hz를 덮는 pair fixture."""

    metadata = {"consistency_band_hz": [150.0, 1_600.0]}
    metadata.update(interleaved or {})
    _official_path(
        path,
        channel=channel,
        delay=delay,
        method="interleaved_multitone",
        interleaved=metadata,
    )


def test_interleaved_pair_from_different_captures_fails_matched_conditions(tmp_path):
    """조건 값이 전부 같아도 **다른 capture** 면 통과하면 안 된다.

    다른 재생에서 나왔다면 그 사이의 클록 wander 가 두 경로의 상대 지연에 실린다.
    lead = S_delay + handoff − P_delay 가 바로 그 값이므로, 이걸 놓치면 파인튜닝이
    틀린 lead 로 조용히 진행된다.
    """

    cfg = _ready_config(tmp_path)
    cfg["data"]["sample_rate"] = INTERLEAVED_FS
    primary = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    primary.unlink()
    secondary.unlink()
    _official_current_interleaved_path(
        primary, channel="noise", delay=4, interleaved={"capture_id": "cap-A"}
    )
    _official_current_interleaved_path(
        secondary, channel="cancel", delay=5, interleaved={"capture_id": "cap-B"}
    )
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert not matched["ok"]
    assert "capture_id 불일치" in matched["message"]


def test_interleaved_pair_from_one_capture_passes_matched_conditions(tmp_path):
    cfg = _ready_config(tmp_path)
    cfg["data"]["sample_rate"] = INTERLEAVED_FS
    primary = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    primary.unlink()
    secondary.unlink()
    _official_current_interleaved_path(primary, channel="noise", delay=4)
    _official_current_interleaved_path(secondary, channel="cancel", delay=5)
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert matched["ok"], matched["message"]


def test_interleaved_pair_common_tau_is_recomputed_from_channel_scores(tmp_path):
    cfg = _ready_config(tmp_path)
    cfg["data"]["sample_rate"] = INTERLEAVED_FS
    primary = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    primary.unlink()
    secondary.unlink()
    _official_current_interleaved_path(primary, channel="noise", delay=4)
    _official_current_interleaved_path(secondary, channel="cancel", delay=5)
    source = _source_path_from_official(primary, "source_analysis_npz_path")
    with np.load(source, allow_pickle=False) as data:
        shifted_full = np.asarray(data["common_alignment_tau_samples"]) + 0.02
    _rewrite_source_and_refresh_hash(
        [primary, secondary],
        source_field="source_analysis_npz_path",
        sha_field="source_analysis_npz_sha256",
        common_alignment_tau_samples=shifted_full,
    )
    with np.load(primary, allow_pickle=False) as data:
        kept = np.asarray(data["kept_repeat_indices"], dtype=np.int64)
    shifted_kept = shifted_full[kept]
    _rewrite_npz(primary, common_alignment_tau_samples=shifted_kept)
    _rewrite_npz(secondary, common_alignment_tau_samples=shifted_kept)

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert not matched["ok"]
    assert "score 가중평균" in matched["message"]


@pytest.mark.parametrize(
    ("override", "pattern"),
    [
        ({"relative_tau_max_abs_samples": 0.5}, "relative_tau_max_abs_samples 저장값"),
        ({"delay_spread_samples": 1}, "delay_spread_samples 저장값"),
    ],
)
def test_interleaved_pair_relative_tau_scalars_are_independently_recomputed(
    tmp_path, override, pattern
):
    cfg = _ready_config(tmp_path)
    cfg["data"]["sample_rate"] = INTERLEAVED_FS
    primary = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    primary.unlink()
    secondary.unlink()
    _official_current_interleaved_path(
        primary, channel="noise", delay=4, interleaved=override
    )
    _official_current_interleaved_path(
        secondary, channel="cancel", delay=5, interleaved=override
    )

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert not matched["ok"]
    assert pattern in matched["message"]


@pytest.mark.parametrize(
    ("secondary_override", "pattern"),
    [
        ({"anchor_repeat": 4}, "anchor_repeat 불일치"),
        (
            {"kept_repeat_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]},
            "kept_repeat_indices 불일치",
        ),
    ],
)
def test_interleaved_pair_requires_same_anchor_and_kept_repeats(
    tmp_path, secondary_override, pattern
):
    cfg = _ready_config(tmp_path)
    cfg["data"]["sample_rate"] = INTERLEAVED_FS
    primary = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    primary.unlink()
    secondary.unlink()
    _official_current_interleaved_path(primary, channel="noise", delay=4)
    _official_current_interleaved_path(
        secondary, channel="cancel", delay=5, interleaved=secondary_override
    )

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert not matched["ok"]
    assert pattern in matched["message"]


def test_mixed_methods_are_rejected(tmp_path):
    cfg = _ready_config(tmp_path)
    primary = Path(cfg["duct"]["digital_reference"]["primary_path_npz"])
    secondary = Path(cfg["duct"]["secondary_path"]["npz"])
    # interleaved physical artifacts are fixed at 48 kHz. Keep the ESS side valid at
    # the same runtime rate so this test reaches the intended method-mismatch gate.
    _rewrite_npz(primary, sample_rate=np.asarray(INTERLEAVED_FS, dtype=np.int64))
    cfg["data"]["sample_rate"] = INTERLEAVED_FS
    secondary.unlink()
    _official_current_interleaved_path(secondary, channel="cancel", delay=5)
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    matched = _check(report, "matched_path_measurement_conditions")
    assert not matched["ok"]
    assert "측정 방식 불일치" in matched["message"]


def test_init_lead_mismatch_is_rejected_by_default(tmp_path):
    """기본값은 정확히 일치다 — 허용치는 설정에 명시해야만 열린다."""

    cfg = _ready_config(tmp_path)
    cfg["duct"]["secondary_path"]["handoff_extra_samples"] = 3
    _refresh_timing_contract(cfg)  # official contract lead=4, checkpoint lead=3
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    check = _check(report, "completed_init_checkpoint")
    assert not check["ok"]
    assert "lead 불일치" in check["message"]


def test_init_lead_mismatch_within_declared_tolerance_passes(tmp_path):
    cfg = _ready_config(tmp_path)
    cfg["duct"]["secondary_path"]["handoff_extra_samples"] = 3
    _refresh_timing_contract(cfg)
    cfg["readiness"]["max_init_lead_mismatch_samples"] = 2
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    assert _check(report, "completed_init_checkpoint")["ok"]


def test_gross_lead_mismatch_is_rejected_despite_tolerance(tmp_path):
    """허용치는 유계다 — 실물 규모의 사고(lead 113 vs 0)는 통과하면 안 된다.

    픽스처의 lead 는 3 이라 checkpoint 를 0 으로 두면 차이가 3 뿐이다. 허용치 16 을
    넘는 차이를 만들어야 "유계성"을 실제로 검사한다.
    """

    cfg = _ready_config(tmp_path)
    init = Path(cfg["init_ckpt"])
    pretrain_cfg = {
        "model": cfg["model"],
        "data": {"digital_reference_lead_samples": 50},
        "digital_reference_lead_samples": 50,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
        "trusted_band_hz": list(PRETRAIN_TRUSTED_BAND_HZ),
    }
    _checkpoint(init, cfg=pretrain_cfg, step=8)
    _checkpoint(init.parent / "last.pt", cfg=pretrain_cfg, step=10)
    cfg["readiness"]["max_init_lead_mismatch_samples"] = 16
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    check = _check(report, "completed_init_checkpoint")
    assert not check["ok"]
    assert "lead 불일치" in check["message"]


def test_trainer_and_gate_share_one_lead_tolerance():
    """같은 규칙을 두 곳에 구현하면 갈라진다 — 둘 다 같은 설정 키를 읽어야 한다.

    실제로 readiness 만 고쳤다가 trainer 의 별도 검사에서 학습이 막혔다.
    """

    import inspect

    from deep_anc.train import trainer as trainer_module

    source = inspect.getsource(trainer_module.Trainer.__init__)
    assert "max_init_lead_mismatch_samples" in source, (
        "trainer 가 readiness 와 다른 기준으로 init lead 를 검사하고 있다"
    )


# ======================================================================================
# 신설 게이트 5종의 negative fixture
#
# 규칙: **게이트마다 그것을 FAIL 시키는 테스트가 짝으로 있어야 한다.** 이것이 없어서
# 게이트 9개가 전부 PASS 인 채로 무용지물이라는 것을 아무도 몰랐다.
# ======================================================================================
def _gate(report: dict, gate_id: str) -> dict:
    for item in report["checks"]:
        if item["id"] == gate_id:
            return item
    raise AssertionError(f"게이트를 찾을 수 없습니다: {gate_id} ({[c['id'] for c in report['checks']]})")


# ---- G2b 학습 데이터 정렬 (결함 2) ---------------------------------------------------
def test_readiness_rejects_collapsed_source_err_timebase(tmp_path):
    """재생/캡처 타임베이스가 깨진 실측 데이터로는 학습을 시작할 수 없다.

    2026-08-04 실측: coh²(source→ERR) 0.021~0.126 인데 QA 는 80/80 PASS 였다.
    """

    manifest = _recorded_manifest(tmp_path / "data", collapse_alignment=True)
    cfg = _ready_config(tmp_path, manifest=manifest)

    report = audit_finetune_readiness(cfg)

    assert not report["ok"]
    gate = _gate(report, "recorded_alignment_integrity")
    assert not gate["ok"]
    assert "결맞음" in gate["message"] or "재녹음" in gate["message"]


def test_readiness_does_not_count_a_skipped_qa_as_alignment_evidence(tmp_path):
    """QA 를 건너뛴 실행은 정렬 게이트를 **통과하지 못한다**.

    "측정하지 않았다"를 "측정해서 통과했다"와 같게 취급하는 것이 이 저장소에서
    반복된 실패다.
    """

    cfg = _ready_config(tmp_path)
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)

    gate = _gate(report, "recorded_alignment_integrity")
    assert not gate["ok"]
    assert "측정하지 않은" in gate["message"]


# ---- G2c 통계적 검정력 (결함 4 / D3) -------------------------------------------------
def test_readiness_rejects_underpowered_val_and_test_groups(tmp_path):
    """계열당 그룹이 1–2개면 G4 판정이 성립하지 않으므로 진입을 막는다.

    실측 상태: val machine 1그룹, test environment 1그룹, test machine 1그룹.
    """

    manifest = _recorded_manifest(tmp_path / "data", groups_per_family=2)
    cfg = _ready_config(tmp_path, manifest=manifest)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "recorded_statistical_power")
    assert not gate["ok"]
    assert "cluster bootstrap" in gate["message"]
    # 어느 split/계열이 부족한지 기계 판독 가능해야 한다.
    weak = {(item[0], item[1]) for item in gate["details"]["weak"]}
    assert ("val", "machine") in weak and ("test", "speech") in weak


def test_readiness_rejects_missing_recorded_subband_coverage_report(tmp_path):
    cfg = _ready_config(tmp_path)
    _coverage_report_path(cfg).unlink()

    gate = _gate(audit_finetune_readiness(cfg, full_recorded_qa=False), "recorded_subband_coverage")
    assert not gate["ok"]
    assert "snapshot" in gate["message"] or "열 수 없습니다" in gate["message"]


def test_readiness_rejects_stale_recorded_subband_coverage_manifest(tmp_path):
    cfg = _ready_config(tmp_path)
    manifest = Path(cfg["recorded_manifest"])
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    gate = _gate(audit_finetune_readiness(cfg, full_recorded_qa=False), "recorded_subband_coverage")
    assert not gate["ok"]
    assert "snapshot" in gate["message"] or "열 수 없습니다" in gate["message"]


def test_readiness_rejects_forged_recorded_subband_coverage_aggregate(tmp_path):
    cfg = _ready_config(tmp_path, manifest=_recorded_manifest(tmp_path / "data", groups_per_family=2))
    report_path = _coverage_report_path(cfg)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["splits"]["val"]["rows"][0]["group_power_pass"] = True
    payload["splits"]["val"]["group_power_pass"] = True
    payload["all_requested_splits_pass"] = True
    payload = seal_recorded_subband_coverage_report(payload)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    gate = _gate(audit_finetune_readiness(cfg, full_recorded_qa=False), "recorded_subband_coverage")
    assert not gate["ok"]
    assert "위조" in gate["message"]


@pytest.mark.parametrize("role", ["canonical_finetune", "measured_probe"])
def test_canonical_coverage_requires_external_bootstrap_receipt_binding(
    tmp_path, role
):
    cfg = _ready_config(tmp_path)
    cfg["experiment_role"] = role
    audit = readiness._Audit("fixture")
    readiness._audit_recorded_subband_coverage(
        audit,
        cfg,
        cfg["readiness"],
        manifest_path=Path(cfg["recorded_manifest"]),
        transfer_snapshot=None,
    )
    assert not audit.checks[0]["ok"]
    assert "bootstrap receipt" in audit.checks[0]["message"]


@pytest.mark.parametrize("role", ["canonical_finetune", "measured_probe"])
def test_recorded_trust_roles_require_transfer_and_canonical_lineage(
    tmp_path, monkeypatch, role
):
    cfg = _ready_config(tmp_path)
    cfg["experiment_role"] = role
    calls = {"transfer": 0, "lineage": 0}

    def reject_transfer(data_cfg, *, repo_root):
        calls["transfer"] += 1
        raise ValueError("fixture transfer sentinel")

    def reject_lineage(manifest_path, data_cfg, transfer_snapshot=None):
        calls["lineage"] += 1
        raise ValueError("fixture lineage sentinel")

    monkeypatch.setattr(readiness, "validate_recorded_training_snapshot", reject_transfer)
    monkeypatch.setattr(readiness, "_canonical_recorded_lineage_snapshot", reject_lineage)

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)

    assert calls == {"transfer": 1, "lineage": 1}
    transfer_gate = _gate(report, "recorded_transfer_snapshot")
    manifest_gate = _gate(report, "recorded_dataset_qa")
    assert not transfer_gate["ok"] and "transfer sentinel" in transfer_gate["message"]
    assert not manifest_gate["ok"] and "lineage sentinel" in manifest_gate["message"]


@pytest.mark.parametrize("bind_resolved_config", [True, False])
def test_readiness_cross_binds_and_reports_transfer_level_calibration(
    tmp_path, monkeypatch, bind_resolved_config
):
    cfg = _ready_config(tmp_path)
    cfg["experiment_role"] = "measured_probe"
    calibration_path = (
        tmp_path / "data/manifests/recorded_level_calibration/fixture.json"
    )
    calibration_path.parent.mkdir(parents=True)
    calibration_path.write_text("{}\n", encoding="utf-8")
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    transfer = SimpleNamespace(
        transfer_manifest=SimpleNamespace(sha256="1" * 64),
        recorded_aggregate_sha256="2" * 64,
        recorded_level_calibration=SimpleNamespace(
            path=calibration_path,
            sha256=calibration_sha,
        ),
        recorded_generation=None,
        recorded_subband_coverage_receipt=None,
        recorded_subband_coverage_report=None,
    )

    def validate_transfer(data_cfg, *, repo_root):
        assert repo_root == tmp_path
        if bind_resolved_config:
            data_cfg["recorded_level_calibration"] = (
                calibration_path.relative_to(tmp_path).as_posix()
            )
            data_cfg["recorded_level_calibration_sha256"] = calibration_sha
        return transfer

    monkeypatch.setattr(readiness, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        readiness, "validate_recorded_training_snapshot", validate_transfer
    )
    monkeypatch.setattr(
        readiness,
        "_canonical_recorded_lineage_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("fixture lineage")),
    )

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    gate = _gate(report, "recorded_transfer_snapshot")

    assert gate["ok"] is bind_resolved_config
    if bind_resolved_config:
        assert gate["details"]["recorded_level_calibration_path"] == (
            calibration_path.relative_to(tmp_path).as_posix()
        )
        assert gate["details"]["recorded_level_calibration_sha256"] == calibration_sha
    else:
        assert "transfer-검증 snapshot과 다릅니다" in gate["message"]


# ---- D1 코퍼스 누수 -------------------------------------------------------------------
def test_readiness_rejects_corpus_leak_between_synthetic_and_recorded(tmp_path):
    """합성 학습 스트림과 실측이 같은 원본 오디오를 쓰면 FAIL 한다.

    2026-08-05 실측: music 60/60(100%)이 겹치고 55개가 합성 train 에 있었다.
    같은 곡에서 두 브랜치가 반대 방향 gradient 를 주고, **music 만 개선되지 않았다.**
    """

    cfg = _ready_config(tmp_path, leak=True)

    report = audit_finetune_readiness(cfg)

    assert not report["ok"]
    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "music" in gate["message"]
    families = gate["details"]["families"]
    assert families["music"]["overlap_clips"] == 1
    assert families["speech"]["overlap_clips"] == 0
    # 겹친 원본이 합성의 어느 split 에 있었는지까지 나와야 한다.
    assert families["music"]["overlap_by_synthetic_split"] == {"train": 1}


def test_readiness_refuses_to_claim_disjoint_corpora_it_cannot_see(tmp_path):
    """합성 manifest 가 없으면 "겹치지 않는다"고 주장하지 않는다 (실패 폐쇄)."""

    cfg = _ready_config(tmp_path)
    cfg["data"]["noise_manifest_dir"] = str(tmp_path / "nonexistent")

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "판정할 수 없습니다" in gate["message"]


def test_readiness_refuses_when_only_some_declared_tags_have_a_manifest(tmp_path):
    """태그 **일부만** manifest 가 있으면 "서로소" 라고 말하지 않는다.

    2026-08-06 통합 검증에서 실제로 재현된 fail-open 이다. data/manifests 에
    esc50.jsonl 하나만 있는 상태에서 이 게이트가 "실측 691개와 합성 1587개가 서로소"
    로 PASS 했는데, D1 이 실제로 찾은 누수는 music 60/60(100%)이고 music.jsonl 이
    없어 비교 대상에 아예 들어가지 않았다. 없는 태그는 조용히 사라지지 않고
    synth_dataset 이 합성원으로 폴백하므로, 선언한 혼합비와 다른 데이터로 돌기까지 한다.
    """

    cfg = _ready_config(tmp_path)
    manifest_dir = Path(cfg["data"]["noise_manifest_dir"])
    # speech 만 남기고 music manifest 를 지운다 — 선언은 그대로 speech/music 이다.
    (manifest_dir / "music.jsonl").unlink()

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "music" in gate["message"]
    assert "manifest 가 없는" in gate["message"]


def test_readiness_rejects_manifest_bytes_from_a_mixed_generation(tmp_path):
    """sidecar 이후 JSONL 하나만 바뀌면 누수가 0이어도 학습을 시작하지 않는다."""

    cfg = _ready_config(tmp_path)
    manifest = Path(cfg["data"]["noise_manifest_dir"]) / "speech.jsonl"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "speech SHA 불일치" in gate["message"]


def test_corpus_gate_rechecks_external_addition_raw_sha_against_public_manifests(
    tmp_path, monkeypatch
):
    """parent holdout 밖 external raw도 synthetic 6종과 직접 다시 비교한다."""

    source_csv = tmp_path / "sources.csv"
    source_csv.write_text(
        "source_family,path,clips\n"
        'machine,machine_000.wav,"[""parent-only.wav""]"\n',
        encoding="utf-8",
    )
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "machine.jsonl").write_text("fixture\n", encoding="utf-8")
    raw_sha = hashlib.sha256(b"external ESC raw").hexdigest()
    synthetic_entry = {
        "path": str(tmp_path / "raw/renamed-machine.wav"),
        "content_sha256": raw_sha,
        "lineage_keys": ["esc50:source:external"],
        "group_id": "public-lineage-external",
        "split": "train",
    }
    generation_exclusion = {
        "schema_version": 1,
        "generation_id": "highband-v1",
        "generation": {
            "path": "data/manifests/recorded_generations/highband-v1/generation.json",
            "sha256": "1" * 64,
            "size": 1,
        },
        "source_plan": {
            "path": "data/source_plans/recorded_additions/highband-v1.csv",
            "sha256": "2" * 64,
            "size": 1,
        },
        "identity_count": 1,
        "identities": [
            {
                "source_row_number": 2,
                "source_kind": "external_exact_composite",
                "source_family": "machine",
                "source_path": "data/source_plans/recorded_additions/x.wav",
                "source_file_sha256": hashlib.sha256(b"composite").hexdigest(),
                "raw_member_path": "data/raw/noise/esc50/external.wav",
                "raw_member_sha256": raw_sha,
                "raw_member_lineage_key": "esc50:source:external",
                "authority_components": [
                    "clip_identity:external.wav",
                    "esc50_identity:esc50:source:external",
                ],
            }
        ],
        "identities_sha256": "3" * 64,
    }
    monkeypatch.setattr(
        readiness,
        "validate_manifest_generation",
        lambda *args, **kwargs: {
            "build_id": "4" * 64,
            "public_lineage": {},
            "_validated_entries": {"machine": [synthetic_entry]},
            "_validated_recorded_generation_exclusion": generation_exclusion,
            "recorded_generation_exclusion": generation_exclusion,
        },
    )
    audit = readiness._Audit("fixture")
    readiness._audit_corpus_leak(
        audit,
        {"recorded_source_pool_csv": str(source_csv)},
        {
            "noise_manifest_dir": str(manifest_dir),
            "source_mix_ratio": {"machine": 1.0},
        },
        entries=[],
    )

    assert len(audit.checks) == 1
    assert not audit.checks[0]["ok"]
    assert "recorded generation additions" in audit.checks[0]["message"]
    assert "content_sha256" in audit.checks[0]["message"]


def test_corpus_gate_requires_generation_exclusion_for_schema_v2_transfer(
    tmp_path, monkeypatch
):
    source_csv = tmp_path / "sources.csv"
    source_csv.write_text(
        "source_family,path,clips\n"
        'machine,machine_000.wav,"[""parent-only.wav""]"\n',
        encoding="utf-8",
    )
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "machine.jsonl").write_text("fixture\n", encoding="utf-8")
    synthetic_entry = {
        "path": str(tmp_path / "raw/fresh.wav"),
        "content_sha256": hashlib.sha256(b"fresh").hexdigest(),
        "lineage_keys": ["fixture:fresh"],
        "group_id": "public-lineage-fresh",
        "split": "train",
    }
    monkeypatch.setattr(readiness, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        readiness,
        "validate_manifest_generation",
        lambda *args, **kwargs: {
            "build_id": "4" * 64,
            "public_lineage": {},
            "_validated_entries": {"machine": [synthetic_entry]},
            "_validated_recorded_generation_exclusion": None,
        },
    )
    transfer_generation = tmp_path / "generation.json"
    transfer_generation.write_text("{}\n", encoding="utf-8")
    transfer = SimpleNamespace(
        recorded_generation=SimpleNamespace(
            path=transfer_generation,
            sha256=hashlib.sha256(transfer_generation.read_bytes()).hexdigest(),
            size=transfer_generation.stat().st_size,
        ),
        recorded_generation_summary={},
    )
    audit = readiness._Audit("fixture")
    readiness._audit_corpus_leak(
        audit,
        {"recorded_source_pool_csv": str(source_csv)},
        {
            "noise_manifest_dir": str(manifest_dir),
            "source_mix_ratio": {"machine": 1.0},
        },
        entries=[],
        transfer_snapshot=transfer,
    )

    assert len(audit.checks) == 1
    assert not audit.checks[0]["ok"]
    assert "schema v2" in audit.checks[0]["message"]
    assert "recorded_generation_exclusion" in audit.checks[0]["message"]


# ---- G1c 달성 가능 상한 ---------------------------------------------------------------
def test_readiness_rejects_a_target_above_the_achievable_ceiling(tmp_path):
    """플랜트가 허용하는 상한을 넘는 목표는 **학습 시작 전에** 막는다."""

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["target_cancellation_db"] = 20.0

    report = audit_finetune_readiness(cfg)

    assert not report["ok"]
    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert "재측정" in gate["message"]
    # 현행 fixture는 재계산한 정규방정식 상한(7.59 dB)이 γ 상한보다
    # 더 작다. 더 느슨한 상한을 기대하는 예전 fixture가 아니라 실제 구속 조건을 본다.
    assert gate["details"]["binding_constraint"] == "정규방정식 설계 상한"


def test_readiness_uses_the_tighter_of_the_two_ceilings(tmp_path):
    """γ 상한이 낙관적일 때 **정규방정식 설계 상한**이 구속해야 한다.

    복구된 플랜트에서 γ 상한은 약 28 dB 인데 직접 계산한 설계 상한은 6.53 dB 다.
    낙관적인 상한 하나만 믿는 것이 이 저장소에서 반복된 사고의 형태이므로, 두 값 중
    **작은 쪽**으로 판정하는지 못 박아 둔다.
    """

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["target_cancellation_db"] = 5.0
    # 설계 상한은 아티팩트에서 다시 계산되므로 임의의 낮은 숫자를
    # 적어 자기증명하지 않는다. fixture의 재계산값 그대로 선언한다.
    cfg["readiness"]["measured_design_ceiling_db"] = 7.590811495963479
    cfg["readiness"]["measured_design_ceiling_band_hz"] = [150, 1_600]

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert gate["details"]["binding_constraint"] == "정규방정식 설계 상한"
    assert gate["details"]["binding_ceiling_db"] == pytest.approx(7.590811495963479)
    assert gate["details"]["gamma_ceiling_db"] > 10.0


def test_readiness_rejects_a_design_ceiling_measured_on_a_narrower_band(tmp_path):
    """좁은 대역에서 푼 상한으로 넓은 대역 요구를 통과시키지 못한다.

    실제로 있었던 오판정이다: 설정이 6.53 dB 를 선언했고 그것은 150-600Hz 값인데
    required_path_band_hz 는 [150, 1600] 이었다. 같은 플랜트를 요구 대역 전체에서
    다시 풀면 4.6 dB 라 목표 3.0 + 여유 3.0 을 통과할 수 없었는데, 대역 표시가 없는
    float 하나여서 아무도 대조하지 않았다. 상한이 낙관적인 방향으로 틀리면 그 오판정은
    항상 "어려운 대역을 방치한다" 쪽으로 나온다 — 절대목표 1 과 정면으로 충돌한다.
    """

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["required_path_band_hz"] = [100, 1_000]
    cfg["readiness"]["target_cancellation_db"] = 3.0
    # 상한 자체는 넉넉하다. 그런데 그것을 잰 대역이 요구 대역의 절반뿐이다.
    cfg["readiness"]["measured_design_ceiling_db"] = 30.0
    cfg["readiness"]["measured_design_ceiling_band_hz"] = [100, 500]

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert "덮지 못합니다" in gate["message"]
    assert gate["details"]["design_ceiling_band_hz"] == [100.0, 500.0]
    assert gate["details"]["required_path_band_hz"] == [100.0, 1_000.0]


def test_readiness_rejects_a_design_ceiling_declared_without_its_band(tmp_path):
    """대역 없는 상한 선언은 그 자체로 거부한다 (짝 fixture)."""

    cfg = _ready_config(tmp_path)
    cfg["readiness"]["target_cancellation_db"] = 3.0
    cfg["readiness"]["measured_design_ceiling_db"] = 30.0
    cfg["readiness"].pop("measured_design_ceiling_band_hz", None)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert "measured_design_ceiling_band_hz" in gate["message"]


def test_readiness_requires_the_target_to_be_declared_at_all(tmp_path):
    """목표를 선언하지 않으면 min_path_consistency 가 근거를 잃는다 — 그것도 FAIL 이다."""

    cfg = _ready_config(tmp_path)
    del cfg["readiness"]["target_cancellation_db"]

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "plant_confidence_ceiling")
    assert not gate["ok"]
    assert "target_cancellation_db" in gate["message"]


# ---- 상한 계산기 자체의 검증 -------------------------------------------------------------
def test_achievable_ceiling_matches_the_measured_plants():
    """유도식이 실측값을 재현하는지 확인한다. 주석의 숫자가 곧 회귀 기준이다."""

    # 출하본(오염된 S(z)) — 150-600Hz 와 전대역
    assert round(achievable_cancellation_ceiling_db(0.9556, 0.9730), 2) == 11.30
    assert round(achievable_cancellation_ceiling_db(0.7812, 0.9200), 2) == 4.35
    # 클린 재분석본
    assert round(achievable_cancellation_ceiling_db(0.993, 0.993), 2) == 18.51
    assert round(achievable_cancellation_ceiling_db(0.968, 0.979), 2) == 12.64
    # 단일 경로 기준 역함수
    assert round(required_consistency_for(12.0), 4) == 0.9406
    # 게이트 임계 0.90 이 암묵적으로 뜻하던 상한
    assert round(achievable_cancellation_ceiling_db(0.90), 2) == 9.54


def test_achievable_ceiling_is_monotonic_and_rejects_impossible_gamma():
    """γ 가 좋아지면 상한도 좋아지고, 물리적으로 불가능한 γ 는 거부된다."""

    values = [achievable_cancellation_ceiling_db(g) for g in (0.6, 0.8, 0.95, 0.999)]
    assert values == sorted(values)
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            achievable_cancellation_ceiling_db(bad)


def test_required_consistency_round_trips_through_the_ceiling():
    """``required_consistency_for`` 는 단일 경로 상한의 역함수여야 한다."""

    for target in (3.0, 6.0, 12.0, 20.0):
        gamma = required_consistency_for(target)
        assert abs(achievable_cancellation_ceiling_db(gamma) - target) < 1e-9


# ======================================================================================
# 완료 게이트 — G4 강화(결함 3·4) 와 G5 플랜트 동일성(결함 5)의 negative fixture
# ======================================================================================
def _g4_completion_setup(tmp_path: Path, **metrics_kwargs):
    """완료 판정에 필요한 checkpoint/metrics 한 벌을 만든다."""

    cfg = _ready_config(tmp_path)
    run = tmp_path / "finetune"
    best = run / "ckpt" / "best.pt"
    saved_cfg = {
        **cfg,
        "physics_status": "measured_primary_path",
        "digital_reference_lead_samples": 3,
    }
    _checkpoint(best, cfg=saved_cfg, step=4)
    _checkpoint(best.parent / "last.pt", cfg=saved_cfg, step=6)
    manifest = Path(cfg["recorded_manifest"])
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    val_kwargs = dict(metrics_kwargs)
    test_kwargs = dict(metrics_kwargs)
    test_kwargs.pop("val_only", None)
    for key in list(val_kwargs):
        if key.startswith("test_"):
            test_kwargs[key[len("test_"):]] = val_kwargs.pop(key)
    test_kwargs = {k: v for k, v in test_kwargs.items() if not k.startswith("test_")}
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest, **val_kwargs)
    _g4_metrics(
        test_metrics, split="test", checkpoint=best, manifest=manifest, **test_kwargs
    )
    return cfg, best, val_metrics, test_metrics


def test_completion_rejects_metrics_from_an_evaluator_without_do_no_harm(tmp_path):
    """대역 밖 do-no-harm·검정력·지문을 판정하지 않는 구버전 산출물은 거부된다.

    이 관례는 ``g4_source_pass`` 에서 이미 쓰였고 옳았다 — 옛 형식을 통과시키면
    새 게이트가 있으나 마나다.
    """

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path, include_modern_fields=False
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "recorded_val_g4")
    assert not gate["ok"]
    assert "구버전" in gate["message"]
    assert "g4_do_no_harm_pass" in gate["message"]


def test_completion_rejects_legacy_metrics_without_strict_trusted_subbands(tmp_path):
    """구형 150–1600 평균-only metrics는 canonical completion 근거가 아니다."""

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path, include_strict_subband_fields=False
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "recorded_val_g4")
    assert not gate["ok"]
    assert "strict trusted 150–1600Hz" in gate["message"]


def test_completion_rejects_aggregate_pass_when_upper_trusted_subband_fails(tmp_path):
    """전체 trusted 평균/G4 bool을 위조해도 1000–1600 Hz raw가 양수면 실패한다."""

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path, strict_upper_subband_nmse_db=0.25
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "recorded_val_g4")
    assert not gate["ok"]
    assert "upper 1000–1600Hz" in gate["message"]


def test_completion_rejects_out_of_band_amplification(tmp_path):
    """신뢰 대역이 좋아도 대역 밖을 키웠다면 완료가 아니다 (절대목표 1).

    실측 반증: tone300 이 fullband +5.95 dB 로 기준을 만족하면서 8 kHz 를
    −21.56 dB 증폭했다.
    """

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path,
        verdict="FAIL",
        do_no_harm_pass=False,
        worst_octave_center_hz=8_000.0,
        worst_octave_worst10_db=-21.56,
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "recorded_val_g4")
    assert "do-no-harm 실패" in gate["message"]
    assert "8000Hz" in gate["message"].replace(" ", "")


def test_completion_does_not_accept_an_inconclusive_g4(tmp_path):
    """판정 불가는 완료가 아니다. 표본 부족을 PASS 로 흘려보내지 않는다."""

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path, verdict="INCONCLUSIVE", power_pass=False, ci_pass=False
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "recorded_val_g4")
    assert "INCONCLUSIVE" in gate["message"]
    # 새 raw strict 계약에서는 generic family group 수를 위조하지 않고, strict
    # coverage 부족으로 판정 불가를 재현한다. 어느 경우든 완료로 승격되어서는 안 된다.
    assert "trusted upper 1000–1600Hz" in gate["message"]


def test_completion_rejects_val_and_test_from_different_plants(tmp_path):
    """val 과 test 가 서로 다른 플랜트에서 평가됐다면 나란히 놓을 수 없다 (G5).

    2026-08-04 사고: 전 = S 지연 1342 / lead 109 / surrogate, 후 = 1465 / 113 /
    measured 를 비교해 "1.30 dB 개선" 이라고 적었다.
    """

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path,
        test_fingerprint=_plant_fingerprint_payload(
            secondary_delay_samples=1_465, lead_samples=113, configured_lead_samples=113
        ),
    )

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert not report["ok"]
    gate = _gate(report, "plant_identity_for_comparison")
    assert not gate["ok"]
    assert "서로 다른 플랜트" in gate["message"]
    assert "secondary_delay_samples" in gate["message"]


def test_completion_accepts_val_and_test_from_the_same_plant(tmp_path):
    """같은 플랜트끼리는 통과해야 한다 — 게이트가 무조건 거부하는 게 아님을 증명."""

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(tmp_path)

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    assert report["ok"], report
    gate = _gate(report, "plant_identity_for_comparison")
    assert gate["ok"]
    assert gate["details"]["digest"]


# ---- 플랜트 아티팩트 강화: 유지 반복 수와 P−S 상대 τ 궤적 ------------------------------
def test_official_path_gate_rejects_too_few_kept_repeats(tmp_path):
    """반복을 적게 남긴 플랜트는 official 이 될 수 없다.

    기각을 많이 한 것은 문제가 아니다 — 2026-08-05 복구 캡처는 48 중 30 을 기각했고
    그것이 옳은 조치였다. 문제는 **남은 것이 적을 때**다. 이전 하한 3 은 한 번의
    이상치가 플랜트 형상을 지배하는 것을 허용했다.
    """

    thin = tmp_path / "thin.npz"
    _official_path(thin, channel="cancel", delay=5)
    with np.load(thin, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["repeats"] = np.asarray(3, dtype=np.int64)
    np.savez(thin, **arrays)

    with pytest.raises(ValueError, match="유지된 반복"):
        audit_official_path_model(
            thin,
            expected_output_channel="cancel",
            sample_rate=FS,
            required_band_hz=(100, 1_000),
        )


def test_matched_conditions_reads_the_stored_relative_tau_trajectory(tmp_path):
    """게이트가 **저장돼 있던 궤적을 실제로 읽는지** 확인한다.

    2026-08-04 사고의 핵심은 증거가 없었던 게 아니라 게이트가 파일 안의 증거를 한 번도
    열어보지 않았다는 것이다. 여기서는 그 궤적에 실제 사고 값을 넣고 거부되는지 본다.

    실측 P−S 상대 τ (출하 아티팩트): 반복 0–10 은 ~1.2, 반복 11–15 는 ~32.
    스칼라 delay_spread_samples 는 range 라 이 이봉 구조를 32 라는 한 숫자로 뭉갠다.
    """

    shipped_relative = np.asarray(
        [0.0, 1.20, 1.13, 1.09, 1.09, 1.29, 1.41, 1.47, 1.14, 1.48, 1.36,
         32.11, 32.18, 31.75, 30.26, 29.06]
    )
    tau_secondary = np.zeros_like(shipped_relative)

    cfg = _ready_config(tmp_path)
    for key, tau in (
        ("digital_reference", shipped_relative),
        ("secondary_path", tau_secondary),
    ):
        path = (
            cfg["duct"]["digital_reference"]["primary_path_npz"]
            if key == "digital_reference"
            else cfg["duct"]["secondary_path"]["npz"]
        )
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["repeat_tau_samples"] = tau
        # 스칼라 요약은 **정상값으로 남겨 둔다** — 궤적을 읽지 않으면 통과해야 한다.
        arrays["delay_spread_samples"] = np.asarray(1, dtype=np.int64)
        np.savez(path, **arrays)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "matched_path_measurement_conditions")
    assert not gate["ok"], "스칼라 요약만 보면 통과하지만 궤적을 보면 프레임 슬립이다"
    assert "프레임 슬립" in gate["message"]
    assert "[11, 12, 13, 14, 15]" in gate["message"]


def test_matched_conditions_accepts_a_constant_relative_tau(tmp_path):
    """상수 궤적은 통과해야 한다 — 무조건 거부하는 검사가 아님을 증명."""

    cfg = _ready_config(tmp_path)
    drift = np.linspace(0.0, 40.0, 12)  # 두 채널에 **공통**으로 실린 드리프트
    for path, offset in (
        (cfg["duct"]["digital_reference"]["primary_path_npz"], 1.4),
        (cfg["duct"]["secondary_path"]["npz"], 0.0),
    ):
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["repeat_tau_samples"] = drift + offset
        np.savez(path, **arrays)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "matched_path_measurement_conditions")
    assert gate["ok"], gate
    # 공통 드리프트는 상대 τ 에서 상쇄된다 — 그것이 동시 측정의 요점이다.
    assert gate["details"]["relative_tau"]["max_deviation_samples"] < 3.0


def test_group_floor_has_a_single_source():
    """계열당 그룹 하한이 **한 곳에서만** 정의되는지 강제한다 (발생기 A).

    진입 게이트와 G4 평가기가 각자 숫자를 들고 있으면 언젠가 한쪽만 바뀌고, 그러면
    "진입은 통과했는데 완료는 판정 불가" 라는 해석 불가능한 상태가 된다. 이 저장소의
    사고는 대부분 그 모양이었다(lead 109 vs 113 등).
    """

    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY
    from deep_anc.train.finetune_readiness import (
        MIN_GROUPS_PER_FAMILY_PER_SPLIT,
        _min_groups_per_family_default,
    )

    assert _min_groups_per_family_default() == MIN_GROUPS_PER_FAMILY
    assert MIN_GROUPS_PER_FAMILY_PER_SPLIT == MIN_GROUPS_PER_FAMILY
    # 설정이 값을 주지 않아도 게이트가 같은 하한을 쓴다.
    assert load_yaml(REPO_ROOT / "configs/train_finetune.yaml")["readiness"][
        "min_groups_per_family_per_split"
    ] == MIN_GROUPS_PER_FAMILY


# ======================================================================================
# 오발동 반증 — **정상 산출물을 경계까지 몰아도 게이트가 울리지 않는가** (군집 B 나머지 절반)
#
# 2026-08-06 반증 #13: 이 저장소의 메타 테스트는 "발동시키는 fixture 가 있는가" 만
# 강제했다. 모든 게이트의 반응은 차단(학습 중단 / mute = 상쇄 0 dB)인데, "정상 입력에서
# 안 울리는가" 를 운용 범위 끝까지 몰아본 게이트가 하나도 없었다. 그래서 실제로
# 재정렬에 성공한 세션 9개 중 4개(44%)를 QA 게이트가 떨어뜨리고 있었는데도 아무도
# 몰랐다. 아래 테스트들이 그 반쪽이다 — 전부 **한계 위의 정상값**을 넣는다.
# ======================================================================================
def _boundary_config(tmp_path: Path) -> dict:
    """모든 진입 게이트를 **한계에 붙여** 통과시키는 설정.

    여유를 주지 않는다. 여기서 하나라도 한 눈금 더 나빠지면 그 게이트가 FAIL 해야
    하고(그 짝은 각 negative fixture 가 본다), 지금 이대로는 전부 PASS 해야 한다.
    """

    # D2 교차검증을 허용 오차 8.0 샘플의 **90% 지점**(7 샘플 어긋남)에서 돌린다.
    manifest = _recorded_manifest(
        tmp_path / "data", source_delay_samples=ERR_MIC_DELAY_SAMPLES + 7
    )
    cfg = _ready_config(tmp_path, manifest=manifest)
    entries = read_manifest(Path(cfg["recorded_manifest"]))
    sessions = len(entries)
    duration = sum(float(item["duration_s"]) for item in entries)

    # 픽스처 아티팩트의 실제 값. 게이트 임계를 여기에 **정확히** 맞춘다.
    with np.load(tmp_path / "secondary.npz", allow_pickle=False) as data:
        consistency = float(data["coherence_median"])
        excitation = [float(v) for v in np.asarray(data["excitation_band_hz"]).reshape(-1)]

    # 상대 τ spread 를 허용 최대값에 정확히 맞춘다 (3 = 코드 상수).
    for name in ("primary.npz", "secondary.npz"):
        path = tmp_path / name
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["delay_spread_samples"] = np.asarray(
            MAX_RELATIVE_DELAY_SPREAD_SAMPLES, dtype=np.int64
        )
        np.savez(path, **arrays)

    ceiling = min(
        achievable_cancellation_ceiling_db(consistency, consistency),
        float(cfg["readiness"]["measured_design_ceiling_db"]),
    )
    margin = 3.0
    cfg["readiness"].update(
        {
            # 대역: 사용자 절대목표의 **양 끝**을 그대로 요구한다.
            "required_path_band_hz": [150.0, 1_600.0],
            "min_path_consistency": consistency,          # 여유 0
            "required_recorded_ratio": float(cfg["recorded_ratio"]),
            "min_recorded_sessions": sessions,            # 여유 0
            "min_recorded_duration_seconds": duration,    # 여유 0
            "min_groups_per_family_per_split": GROUPS_PER_FAMILY_PER_SPLIT,
            "min_delay_crosscheck_sessions": sessions,    # 여유 0
            "max_measured_delay_mismatch_samples": 8.0,   # 실제 어긋남 7 = 한계의 90%
            # 목표 + 여유가 상한의 90% 지점에 오도록 잡는다.
            "target_cancellation_db": 0.9 * ceiling - margin,
            "cancellation_ceiling_margin_db": margin,
        }
    )
    return cfg


def test_every_entry_gate_passes_at_its_declared_boundary(tmp_path):
    """진입 게이트 전부가 **한계에 붙은 정상 데이터**에서 PASS 한다.

    몰아본 경계 (전부 여유 0 또는 한계의 90%):
      · 세션 수 = 최소 세션 수 (48)             · 분량 = 최소 분량
      · 반복 일관성 = min_path_consistency      · 요구 대역 = 절대목표 양 끝 150/1600Hz
      · P−S 상대 τ spread = 허용 최대 3 샘플    · 계열당 그룹 = 하한 4
      · source→ERR 지연 어긋남 7 = 허용 8.0 의 90%
      · 목표 + 여유 = 달성 가능 상한의 90%
    """

    from deep_anc.ops.gate_registry import CANONICAL_FINETUNE_READINESS_GATE_IDS

    cfg = _boundary_config(tmp_path)

    report = audit_finetune_readiness(cfg)

    failed = [item for item in report["checks"] if not item["ok"]]
    assert failed == [], failed
    assert report["ok"], report
    # canonical 분모는 정확히 17개다. 이 non-trust-role fixture는 외부 transfer
    # snapshot gate 1개만 조건부로 제외하고 나머지 16개를 모두 실행한다.
    authority = set(CANONICAL_FINETUNE_READINESS_GATE_IDS)
    assert len(authority) == 17
    assert {item["id"] for item in report["checks"]} == authority - {
        "recorded_transfer_snapshot"
    }


def test_completion_gates_pass_at_the_minimum_sample_boundary(tmp_path):
    """완료 게이트 전부가 **최소 표본**의 정상 산출물에서 PASS 한다.

    몰아본 경계: family×strict 부대역마다 독립 그룹 4개(통계 하한 정확히),
    최악 계열 −0.01 dB(개선이라 말할 수 있는 최소값), 최악 옥타브 0.01 dB.
    """

    cfg, best, val_metrics, test_metrics = _g4_completion_setup(
        tmp_path,
        worst_source_db=-0.01,
        worst_octave_worst10_db=0.01,
    )
    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )

    failed = [item for item in report["checks"] if not item["ok"]]
    assert failed == [], failed
    assert report["fine_tuning_complete"]


def test_official_path_delay_spread_passes_at_the_allowed_maximum(tmp_path):
    """허용 최대 spread(3 샘플) 정확히 위에서는 통과한다 — 한 샘플 더 가면 FAIL."""

    cfg = _ready_config(tmp_path)
    for name in ("primary.npz", "secondary.npz"):
        path = tmp_path / name
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["delay_spread_samples"] = np.asarray(
            MAX_RELATIVE_DELAY_SPREAD_SAMPLES, dtype=np.int64
        )
        np.savez(path, **arrays)

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    assert _check(report, "official_secondary_path")["ok"]
    assert _check(report, "official_primary_path")["ok"]

    # 한 샘플 더: 같은 게이트가 거부한다.
    path = tmp_path / "secondary.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["delay_spread_samples"] = np.asarray(
        MAX_RELATIVE_DELAY_SPREAD_SAMPLES + 1, dtype=np.int64
    )
    np.savez(path, **arrays)
    worse = audit_finetune_readiness(cfg, full_recorded_qa=False)
    assert not _check(worse, "official_secondary_path")["ok"]


# ---- 결함: init checkpoint 의 **대역 축** (2026-08-06) ---------------------------------
def test_init_checkpoint_trained_on_a_narrower_band_is_rejected(tmp_path):
    """좁은 대역으로 학습한 checkpoint 는 고역 증폭기다 — 게이트가 대역 축을 본다.

    실제 상태: ``runs/pretrain_{base,tiny}_corrected`` 는 cfg.trusted_band_hz
    [150, 600] 인데 현재 설정이 유도하는 대역은 [150, 1600] 이다. 벌점이 없던
    600-1600Hz 를 그 모델은 적극 증폭한다([150,800] 설정 실측 +27.01 dB).
    lead 축만 보던 게이트는 이것을 통과시켰다.
    """

    cfg = _ready_config(tmp_path)
    init = Path(cfg["init_ckpt"])
    narrow = {
        "model": cfg["model"],
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
        # 위쪽 절반을 못 본 채로 학습됐다.
        "trusted_band_hz": [PRETRAIN_TRUSTED_BAND_HZ[0], 600.0],
    }
    _checkpoint(init, cfg=narrow, step=8)
    _checkpoint(init.parent / "last.pt", cfg=narrow, step=10)

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)

    check = _check(report, "completed_init_checkpoint")
    assert not check["ok"]
    assert "대역" in check["message"]
    assert "+27.01 dB" in check["message"]


def test_init_checkpoint_without_a_recorded_band_is_rejected(tmp_path):
    """대역을 기록하지 않은 checkpoint 는 '모른다' 이지 '괜찮다' 가 아니다."""

    cfg = _ready_config(tmp_path)
    init = Path(cfg["init_ckpt"])
    unlabelled = {
        "model": cfg["model"],
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
    }
    _checkpoint(init, cfg=unlabelled, step=8)
    _checkpoint(init.parent / "last.pt", cfg=unlabelled, step=10)

    report = audit_finetune_readiness(cfg, full_recorded_qa=False)

    check = _check(report, "completed_init_checkpoint")
    assert not check["ok"]
    assert "trusted_band_hz" in check["message"]


def test_init_checkpoint_band_exactly_equal_to_the_finetune_band_passes(tmp_path):
    """경계: checkpoint 대역이 파인튜닝 대역과 **정확히 같을 때** 통과한다.

    150.0/1600.0 Hz 양 끝이 한 눈금도 어긋나지 않은 상태 — 여유 0 이다.
    더 넓은 대역에서 온 것도 통과해야 한다(벌점을 받아 본 구간이 더 넓다).
    """

    cfg = _ready_config(tmp_path)
    report = audit_finetune_readiness(cfg, full_recorded_qa=False)
    check = _check(report, "completed_init_checkpoint")
    assert check["ok"], check
    assert check["details"]["checkpoint"]["trusted_band_hz"] == [150.0, 1_600.0]
    assert check["details"]["checkpoint"]["expected_trusted_band_hz"] == [150.0, 1_600.0]

    init = Path(cfg["init_ckpt"])
    wider = {
        "model": cfg["model"],
        "data": {"digital_reference_lead_samples": 3},
        "digital_reference_lead_samples": 3,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "schedule": {"total_steps": 10},
        "trusted_band_hz": [80.0, 1_600.0],
    }
    _checkpoint(init, cfg=wider, step=8)
    _checkpoint(init.parent / "last.pt", cfg=wider, step=10)
    assert _check(
        audit_finetune_readiness(cfg, full_recorded_qa=False),
        "completed_init_checkpoint",
    )["ok"]


@pytest.mark.parametrize(
    (
        "role",
        "required_init_role",
        "require_receipt",
        "expected_completion_step",
    ),
    [
        ("measured_probe", "loss_pilot", False, 20_000),
        ("canonical_finetune", "canonical_pretrain", True, None),
    ],
)
def test_readiness_separates_operational_completion_from_completion_receipt(
    tmp_path,
    monkeypatch,
    role,
    required_init_role,
    require_receipt,
    expected_completion_step,
):
    cfg = _ready_config(tmp_path)
    cfg.update(
        experiment_role=role,
        required_init_experiment_role=required_init_role,
        require_init_completion_receipt=require_receipt,
    )
    observed = {}

    def _audit_init(*_args, **kwargs):
        observed.update(kwargs)
        return {"fixture": True}

    monkeypatch.setattr(readiness, "audit_init_checkpoint", _audit_init)
    audit_finetune_readiness(cfg, full_recorded_qa=False)

    assert observed["require_completed"] is True
    assert observed["require_completion_receipt"] is require_receipt
    assert observed["expected_completion_step"] == expected_completion_step
    assert observed["required_experiment_role"] == required_init_role


def test_the_shipped_init_checkpoints_are_high_frequency_amplifiers():
    """출하 상태 확인 — 배포 후보 init checkpoint 가 실제로 이 게이트에 걸린다.

    runs/ 는 .gitignore 대상이라 이 기기에만 있다. 없으면 건너뛴다(다른 환경에서
    빨간불이 되지 않게). 있으면 **[150,600] 으로 학습된 사실**을 못박는다.
    """

    from deep_anc.dsp.timing import FrequencyBand
    from deep_anc.train.finetune_readiness import _checkpoint_optimize_band

    found = 0
    for name in ("pretrain_base_corrected", "pretrain_tiny_corrected"):
        path = REPO_ROOT / "runs" / name / "ckpt" / "best.pt"
        if not path.is_file():
            continue
        found += 1
        state = torch.load(path, map_location="cpu", weights_only=False)
        band = _checkpoint_optimize_band(state)
        assert band is not None
        assert band.as_tuple() == (150.0, 600.0), (name, band)
        # 현재 설정이 요구하는 대역을 덮지 못한다 = 게이트가 거부해야 한다.
        assert not band.covers(FrequencyBand(lo_hz=150.0, hi_hz=1_600.0))
    if found == 0:
        pytest.skip("runs/ 산출물이 없는 트리입니다 (.gitignore 대상)")


def test_readiness_alignment_thresholds_actually_reach_the_qa():
    """readiness 가 선언한 정렬 임계가 **실제로 QA 에 도달하는지** 못 박는다.

    ⚠ 2026-08-06 통합 검증이 잡은 결함의 회귀 테스트. 키 목록이 옛 4개로
    하드코딩돼 있어서 새 키를 선언하면 경고 한 줄 없이 버려졌다. HANDOFF 가
    지시한 "키 이름을 새 것으로 갈아라"를 따르는 순간 설정이 무력화되고,
    그 사실을 알리던 폐기 안내문마저 사라지는 조합이었다.

    목록을 손으로 베끼는 것이 원인이므로 QA 에서 유도하는지도 함께 본다.
    """

    from deep_anc.data.recorded_qa import (
        _ALIGNMENT_OVERRIDE_KEYS,
        settings_from_data_config,
    )
    from deep_anc.train.finetune_readiness import (
        _alignment_cfg_keys,
        _alignment_overrides,
    )

    # QA 가 받는 키는 전부 통로를 지나갈 수 있어야 한다 (복사본이 아니라 유도).
    assert set(_ALIGNMENT_OVERRIDE_KEYS) <= _alignment_cfg_keys()

    data_cfg = {
        "sample_rate": 48_000,
        "segment_seconds": 1.0,
        "digital_reference_lead_samples": 116,
    }

    # 1) 새 키가 실제로 반영된다 — 이것이 이전에 조용히 버려지던 경로다.
    # 값은 **강화 방향**이어야 통과한다. 2026-08-06 에 지터 상한이 8.0 에서
    # 대역 유도값 3.41 로 강화되면서 옛 4.0 은 이제 완화 방향이 됐다.
    declared = {
        "max_source_err_delay_robust_std_samples": 3.0,
        "max_source_err_delay_p95_p5_samples": 24.0,
        "min_source_err_delay_window_ratio": 0.90,
    }
    settings = settings_from_data_config(
        data_cfg, alignment_overrides=_alignment_overrides(declared)
    )
    assert settings.max_source_err_delay_robust_std_samples == 3.0
    assert settings.max_source_err_delay_p95_p5_samples == 24.0
    assert settings.min_source_err_delay_window_ratio == 0.90

    # 2) 폐기 키는 조용히 무시되지 않고 안내로 남는다.
    legacy = settings_from_data_config(
        data_cfg,
        alignment_overrides=_alignment_overrides(
            {
                "max_source_err_delay_std_samples": 64,
                "max_source_err_delay_range_samples": 256,
            }
        ),
    )
    assert len(legacy.deprecated_threshold_notes) == 2
    assert all("폐기" in note for note in legacy.deprecated_threshold_notes)

    # 3) 완화 방향은 통로를 지나가되 QA 가 거절한다 (조용히 통과하지 않는다).
    with pytest.raises(ValueError, match="강화 방향"):
        settings_from_data_config(
            data_cfg,
            alignment_overrides=_alignment_overrides(
                {"min_source_err_delay_window_ratio": 0.50}
            ),
        )


def test_the_two_training_branches_must_give_the_same_total_advance():
    """D2 재설계 — **두 브랜치가 모델에게 같은 과제를 주는가.**

    2026-08-07. 옛 판정은 실측 세션의 **절대** source→ERR 지연을 P(z) 유도값과
    대조했는데, 그 관측량은 재현되지 않는다 — 82세션 산포가 1425.6~1614.2(폭 189 샘플)로
    허용치 64 의 3배다. 설정 차이가 아니고(둘 다 latency=low, block=256) 추정기 차이도
    아니다(같은 추정기로도 248 샘플).

    걱정 자체는 옳았다. 그것을 **재현되는 양**으로 다시 세운다 — 총 선행량이다::

        합성 = D_noise + K
        실측 = d_recorded + K'

    실측(2026-08-06): ``recorded_lead_mode=constant`` 이면 합성 1718 vs 실측 258 로
    **1460 샘플(30.4 ms)** 어긋난다. ``timeline`` 이면 0.1 샘플이 된다.
    """

    from deep_anc.train.finetune_readiness import _Audit, _audit_measured_source_delay

    primary = {"path": "assets/measured/primary_path_il.npz", "delay_samples": 1_602}
    primary_fir = np.zeros(248, dtype=np.float32)
    primary_fir[-1] = 1.0
    timing = TrainingTimingContract.derive(
        primary_fir=primary_fir,
        plant_delays=PlantDelays(
            primary_delay_samples=1_602,
            secondary_delay_samples=1_462,
            handoff_samples=256,
            sample_rate=48_000,
        ),
    )

    def _run(residual: float, lead_mode: str, sessions: int = 10) -> dict:
        audit = _Audit("t")
        report = {
            "sessions": [
                {
                    "alignment": {
                        "source_err_delay_median_samples": residual,
                        "raw_source_err_delay_median_samples": 1_849.0,
                    }
                }
                for _ in range(sessions)
            ]
        }
        data_cfg = {
            "sample_rate": 48_000,
            "digital_reference_lead_samples": int(
                timing.digital_reference_lead_samples
            ),
            "training_timing_contract": timing.model_dump(),
            "recorded_lead_mode": lead_mode,
        }
        _audit_measured_source_delay(
            audit,
            {},
            primary,
            report,
            data_cfg,
            secondary={"delay_samples": 1_462},
            duct_cfg={"secondary_path": {"handoff_extra_samples": 256}},
            timing_contract=timing,
        )
        return next(
            item
            for item in audit.report()["checks"]
            if item["id"] == "measured_source_delay_agreement"
        )

    # (a) constant 모드 — 실측 잔여 142.5 면 두 브랜치가 1460 샘플 어긋난다. FAIL.
    constant = _run(142.5, "constant")
    assert not constant["ok"]
    assert "총 선행량" in constant["message"]
    assert "1706" in constant["message"], constant["message"]
    assert "timeline" in constant["message"]  # 처방을 함께 말한다

    # (b) timeline 모드 — 같은 데이터에서 통과한다. 즉 게이트가 항상 실패하지 않는다.
    timeline = _run(142.5, "timeline")
    assert timeline["ok"], timeline
    assert timeline["details"]["synthetic_advance_samples"] == 1_965.0
    assert timeline["details"]["recorded_advance_samples"] == pytest.approx(1_965.0, abs=1.0)

    # (c) 표본이 모자라면 판정하지 않는다 — 없는 것을 통과로 세지 않는다.
    thin = _run(142.5, "timeline", sessions=3)
    assert not thin["ok"]
    assert "표본" in thin["message"]


def test_the_gate_refuses_when_the_synthetic_advance_is_unknown():
    """TrainingTimingContract가 없으면 두 브랜치를 맞출 수 없다 — 추측하지 않는다."""

    from deep_anc.train.finetune_readiness import _Audit, _audit_measured_source_delay

    audit = _Audit("t")
    report = {
        "sessions": [
            {"alignment": {"source_err_delay_median_samples": 142.5}} for _ in range(10)
        ]
    }
    _audit_measured_source_delay(
        audit,
        {},
        {"path": "assets/measured/primary_path_il.npz", "delay_samples": 1_602},
        report,
        {"digital_reference_lead_samples": 116},
        secondary={"delay_samples": 1_462},
        duct_cfg={"secondary_path": {"handoff_extra_samples": 256}},
    )
    gate = next(
        item
        for item in audit.report()["checks"]
        if item["id"] == "measured_source_delay_agreement"
    )
    assert not gate["ok"]
    assert "TrainingTimingContract" in gate["message"]


def test_readiness_rejects_a_pool_the_sessions_did_not_actually_play(tmp_path):
    """설정이 선언한 소스풀과 **세션이 실제로 재생한 풀**이 다르면 FAIL 한다.

    2026-08-06 감사가 재현한 fail-open. ``recorded_source_pool_csv`` 가 v1 을 가리키는데
    재녹음을 v2 로 하면, 누수 게이트가 **v1 클립끼리 비교해 PASS 하면서 v2 누수를 100%
    통과**시킨다. v1 은 machine 이 8 그룹뿐이라 분할 하한(9)을 만족할 수 없어 재녹음은
    v2 로 할 수밖에 없으므로, 이 상태는 우연이 아니라 **예정된 경로**였다.
    """

    cfg = _ready_config(tmp_path)
    sessions = tmp_path / "recorded_sessions"
    for name, played in (
        ("s0", "data/source_pool_v2/music/music_000.wav"),
        ("s1", "data/source_pool_v2/speech/speech_000.wav"),
    ):
        d = sessions / name
        d.mkdir(parents=True)
        (d / "session.json").write_text(
            json.dumps({"session_id": name, "program": {"type": "file", "file": played}}),
            encoding="utf-8",
        )
    cfg["readiness"]["recorded_session_root"] = str(sessions)

    report = audit_finetune_readiness(cfg)

    gate = _gate(report, "corpus_disjoint")
    assert not gate["ok"]
    assert "세션이 실제로 재생한 풀" in gate["message"]
    assert "source_pool_v2" in gate["message"]
