from __future__ import annotations

import hashlib
import io
import json
import math
import wave

import numpy as np
import pytest

from deep_anc.data import recording_source_conditioning as conditioning


def _wav_bytes(samples: np.ndarray) -> bytes:
    pcm = np.clip(np.rint(np.asarray(samples) * 32767.0), -32768, 32767).astype(
        "<i2"
    )
    handle = io.BytesIO()
    with wave.open(handle, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(48_000)
        writer.writeframes(pcm.tobytes())
    return handle.getvalue()


def _two_band_source() -> np.ndarray:
    t = np.arange(720_000, dtype=np.float64) / 48_000.0
    return 0.5 * np.sin(2.0 * np.pi * 300.0 * t) + 0.5 * np.sin(
        2.0 * np.pi * 1_000.0 * t
    )


def _condition(source: np.ndarray) -> conditioning.ConditionedSourceResult:
    return conditioning.condition_source_at_cap(
        source_bytes=_wav_bytes(source),
        source_path="data/fixture.wav",
        start_seconds=0.0,
        strict_primary_fir=np.asarray([1.0, 0.0]),
        strict_primary_path="assets/measured/fixture.npz",
        strict_primary_sha256="a" * 64,
        amplitude_millionths=5_788,
        lineage={
            "source_family": "fixture",
            "group_id": "fixture-group",
            "lineage_key": "fixture-lineage",
            "split": "train",
        },
    )


def test_cap_aware_identity_pass_is_deterministic_and_training_only() -> None:
    first = _condition(_two_band_source())
    second = _condition(_two_band_source())

    assert first.wav_bytes == second.wav_bytes
    assert first.receipt == second.receipt
    assert first.receipt["status"] == "PASS_COVERAGE_TRAINING_ONLY"
    assert first.receipt["role"] == conditioning.CONDITIONING_ROLE
    assert first.receipt["natural_unprocessed_evaluation_eligible"] is False
    assert first.receipt["thresholds_relaxed"] is False
    assert '"playback_amplitude"' not in json.dumps(
        first.receipt, sort_keys=True
    )
    assert first.receipt["selected_recipe"]["active_frame_compaction"] is False
    assert first.receipt["selected_recipe"]["four_band_gain_db"] == [0.0] * 4
    assert first.receipt["selected_recipe"]["leveler_drive"] == 0.0
    assert first.receipt["exact_cap_audit"]["passed"] is True
    assert all(first.receipt["exact_cap_audit"]["gates"].values())
    assert first.wav_bytes is not None
    assert first.receipt["derived_wav"]["sha256"] == hashlib.sha256(
        first.wav_bytes
    ).hexdigest()

    unsealed = dict(first.receipt)
    seal = unsealed.pop("receipt_sha256")
    expected = hashlib.sha256(
        json.dumps(
            unsealed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert seal == expected


def test_missing_origin_bands_require_lineage_clean_replacement() -> None:
    t = np.arange(720_000, dtype=np.float64) / 48_000.0
    result = _condition(np.sin(2.0 * np.pi * 300.0 * t))

    assert result.wav_bytes is None
    assert result.receipt["status"] == "BLOCKED_REPLACEMENT_REQUIRED"
    assert result.receipt["origin_band_admission"]["passed"] is False
    assert result.receipt["blocker_reasons"] == [
        "origin_band_energy_at_or_below_quantization_guard"
    ]
    requirement = result.receipt["minimum_replacement_requirement"]
    assert requirement["same_source_family"] is True
    assert requirement["lineage_disjoint"] is True


def test_exact_cap_audit_keeps_existing_snr_and_peak_rms_gates() -> None:
    result = _condition(_two_band_source())
    assert result.wav_bytes is not None
    audit = conditioning.audit_derived_wav_at_cap(
        result.wav_bytes,
        strict_primary_fir=np.asarray([1.0, 0.0]),
        amplitude_millionths=5_788,
    )

    expected_snr = 10.0 * math.log10(0.90 / 0.10)
    assert audit["minimum_strict_primary_snr_db"] == pytest.approx(expected_snr)
    assert audit["adc_peak_hard_ceiling"] == 0.5
    assert audit["adc_rms_hard_ceiling"] == 0.5
    preflight = audit["cap_aware_rendered_source_preflight"]
    assert preflight["schema"] == conditioning.CAP_AWARE_PREFLIGHT_SCHEMA
    assert preflight["commanded_amplitude_millionths"] == 5_788
    assert preflight["commanded_amplitude_linear"] == 0.005788
    assert '"playback_amplitude"' not in json.dumps(audit, sort_keys=True)
    assert audit["passed"] is True


def test_legacy_point_zero_six_cannot_be_forged_as_current_command() -> None:
    result = _condition(_two_band_source())
    assert result.wav_bytes is not None
    audit = conditioning.audit_derived_wav_at_cap(
        result.wav_bytes,
        strict_primary_fir=np.asarray([1.0, 0.0]),
        amplitude_millionths=5_788,
    )
    forged = json.loads(
        json.dumps(audit["cap_aware_rendered_source_preflight"])
    )
    forged["commanded_amplitude_linear"] = 0.06
    with pytest.raises(conditioning.RecordingSourceConditioningError):
        conditioning.validate_cap_aware_source_preflight(forged)


def test_no_replace_refuses_different_bytes(tmp_path) -> None:
    target = tmp_path / "derived.wav"
    conditioning.publish_no_replace(target, b"first")
    conditioning.publish_no_replace(target, b"first")
    with pytest.raises(conditioning.RecordingSourceConditioningError):
        conditioning.publish_no_replace(target, b"different")
    assert target.read_bytes() == b"first"


@pytest.mark.parametrize("value", [0, 6_001, True])
def test_cap_outside_probe_range_is_rejected(value) -> None:
    with pytest.raises(conditioning.RecordingSourceConditioningError):
        conditioning.audit_derived_wav_at_cap(
            _wav_bytes(_two_band_source()),
            strict_primary_fir=np.asarray([1.0, 0.0]),
            amplitude_millionths=value,
        )
