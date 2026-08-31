from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import signal as os_signal
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal

from deep_anc.audio_io import analyze_int32_input_probe
from deep_anc.data import recording_gain_linearity as linearity


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hardware(root: Path) -> Path:
    path = root / "configs/hardware.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """audio:
  sample_rate: 48000
  block_size: 256
  latency: low
  input: {card: APE, pcm: 1, channels: 2, dtype: int32}
  output: {card: Audio, pcm: 0, channels: 2, dtype: int16}
channels:
  error_mic: 0
  reference_mic: 1
  noise_out: 0
  cancel_out: 1
""",
        encoding="utf-8",
    )
    return path


def _plan(root: Path) -> tuple[dict, np.ndarray, str, str]:
    _hardware(root)
    payload, pcm = linearity.build_gain_linearity_plan(
        repo_root=root,
        hardware_path="configs/hardware.yaml",
        source_commit="a" * 40,
        physical_fingerprint={"schema": "fixture", "cards": ["APE", "Audio"]},
    )
    plan_path = root / "results/linearity/plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload, pcm, "results/linearity/plan.json", _sha(plan_path)


def _raw(
    root: Path,
    payload: dict,
    pcm: np.ndarray,
    plan_path: str,
    plan_sha: str,
    *,
    unsafe_gain: float | None = None,
) -> tuple[str, str]:
    source = pcm[:, 0].astype(np.float64) / 32767.0
    recorded = np.zeros((pcm.shape[0], 2), dtype=np.float64)
    firs = []
    for gain, echo, delay in ((4.0, 0.30, 150), (6.0, -0.20, 210)):
        fir = np.zeros(256, dtype=np.float64)
        fir[0], fir[25] = gain, echo
        firs.append((fir, delay))
    if unsafe_gain is not None:
        firs = [(np.asarray([unsafe_gain], dtype=np.float64), 150)] * 2
    for group in payload["capture_groups"]:
        start = int(group["start_frame"])
        stop = int(group["stop_frame"])
        for channel, (fir, delay) in enumerate(firs):
            response = signal.fftconvolve(source[start:stop], fir)
            count = min(response.size, stop - start - delay)
            recorded[start + delay : start + delay + count, channel] = response[:count]
    recorded_i32 = np.rint(
        np.clip(recorded, -0.999999, 0.999999) * float(2**31)
    ).astype(np.int32)
    telemetry = []
    callback_arrays = {}
    for group_index, group in enumerate(payload["capture_groups"]):
        frames = int(group["stop_frame"]) - int(group["start_frame"])
        starts = np.arange(0, frames, 256, dtype=np.int64)
        counts = np.full(starts.shape, 256, dtype=np.int64)
        clock = starts.astype(np.float64) / 48_000.0 + 1.0 + group_index * 10.0
        callback_evidence, callback_raw = linearity.callback_time_info_evidence(
            {
                "callback_start_frames": starts,
                "callback_frame_counts": counts,
                "input_buffer_adc_time": clock,
                "output_buffer_dac_time": clock + 0.001,
                "callback_current_time": clock + 0.002,
            },
            group_index=group_index,
            expected_frames=frames,
        )
        callback_arrays.update(callback_raw)
        telemetry.append({
            "level_millionths": group["level_millionths"],
            "start_frame": group["start_frame"],
            "stop_frame": group["stop_frame"],
            "completed": True,
            "callback_count": int(starts.size),
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_status_count": 0,
            "priming_output_count": 0,
            "statuses": [],
            "termination_signal": None,
            "captured_frames": frames,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
            "nominal_output_seconds": frames / 48_000.0,
            "hard_max_output_seconds": frames / 48_000.0 + 1.0,
            "absolute_deadline_monotonic": 100.0,
            "absolute_deadline_exceeded": False,
            "absolute_deadline_abort_error": None,
            "output_elapsed_seconds": frames / 48_000.0,
            "live_campaign_elapsed_seconds": 3.0 + (group_index + 1) * 6.5,
            "callback_time_info": callback_evidence,
        })
    execution = {
        "repository_commit": payload["source_commit"],
        "repository_branch": "DETACHED",
        "repository_dirty": False,
        "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
        "script_file_sha256": "c" * 64,
    }
    rng = np.random.default_rng(20260831)
    preflight = rng.integers(
        -2_000_000, 2_000_001, size=(120_000, 2), dtype=np.int32
    )
    preflight_report = analyze_int32_input_probe(
        preflight, min_rms_dbfs=-80.0, max_clip_ratio=0.005
    )
    preflight_report.update(
        {"device": 7, "sample_rate": 48_000, "settle_seconds": 0.5}
    )
    metadata = {
        "raw_capture_schema": linearity.GAIN_LINEARITY_RAW_SCHEMA,
        "status": "RAW_COMPLETE_NOT_ANALYSED",
        "source_commit": payload["source_commit"],
        "repository_execution": execution,
        "hardware": payload["hardware"],
        "plan": {
            "path": plan_path,
            "sha256": plan_sha,
            "payload_sha256": payload["plan_payload_sha256"],
            "pcm_sha256": payload["output"]["pcm_sha256"],
        },
        "operator_confirmations": dict(linearity.EXACT_OPERATOR_CONFIRMATIONS),
        "preflight": preflight_report,
        "segment_telemetry": telemetry,
        "safety_stop": None,
        "invalid_reasons": [],
        "analysis_status": "NOT_RUN_RAW_FIRST",
        "capture_exception": None,
    }
    raw_path = root / "results/linearity/raw_measurement.npz"
    with raw_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            submitted_output_pcm_int16=pcm,
            input_raw_int32=recorded_i32,
            preflight_raw_int32=preflight,
            **callback_arrays,
        )
    return "results/linearity/raw_measurement.npz", _sha(raw_path)


def _publication_args(root: Path, raw_relative: str) -> dict[str, str]:
    """Synthetic raw fixture에 strict 3-leaf publication anchor를 붙인다."""

    raw_path = root / raw_relative
    with np.load(raw_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    session = raw_path.parent
    sidecar = session / "metadata.json"
    sidecar.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    def held_ref(path: Path) -> dict:
        status = path.stat()
        return {
            "path": path.relative_to(root).as_posix(),
            "size": status.st_size,
            "sha256": _sha(path),
            "device": status.st_dev,
            "inode": status.st_ino,
        }

    payload = linearity.build_gain_linearity_capture_publication_payload(
        canonical_session_path=session.relative_to(root).as_posix(),
        raw_ref=held_ref(raw_path),
        metadata_ref=held_ref(sidecar),
        metadata=metadata,
    )
    publication = session / "capture_publication.json"
    publication.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "publication_path": publication.relative_to(root).as_posix(),
        "expected_publication_sha256": _sha(publication),
    }


def _rewrite_raw(
    root: Path,
    raw_relative: str,
    *,
    mutate_metadata=None,
    mutate_arrays=None,
    preflight: np.ndarray | None = None,
) -> str:
    path = root / raw_relative
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata_json"
        }
        saved_preflight = arrays["preflight_raw_int32"]
    if mutate_metadata is not None:
        mutate_metadata(metadata)
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    if preflight is None:
        preflight = saved_preflight
    arrays["preflight_raw_int32"] = preflight
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )
    return _sha(path)


def test_plan_is_four_level_four_stream_ns_only_and_predictively_bounded(tmp_path: Path):
    payload, pcm, _path, _sha256 = _plan(tmp_path)

    assert payload["contract"]["levels_millionths"] == [3000, 4000, 5000, 6000]
    assert payload["duration"]["output_open_seconds"] == 26.0
    assert payload["duration"]["stream_open_count"] == 4
    assert payload["duration"]["live_campaign_hard_deadline_seconds"] == 37.0
    assert 15.5 < payload["duration"]["nominal_active_seconds"] < 15.7
    assert payload["duration"]["exact_nonzero_pcm_seconds"] == pytest.approx(
        np.count_nonzero(pcm[:, 0]) / 48_000.0
    )
    assert pcm.shape == (26 * 48_000, 2)
    assert np.count_nonzero(pcm[:, 1]) == 0
    assert len(payload["capture_groups"]) == 4
    assert len(payload["clock_pilot_layout"]) == (
        len(payload["capture_groups"]) * len(linearity.CLOCK_PILOT_OFFSETS_SECONDS)
    )
    for index, group in enumerate(payload["capture_groups"]):
        assert np.count_nonzero(
            pcm[group["start_frame"] : group["stimulus_start_frame"]]
        ) == 0
        expected = [
            list(linearity.IMD_PAIRS_HZ[(index + offset) % 3])
            for offset in range(3)
        ]
        assert group["imd_pair_order_hz"] == expected
        active = [
            (row["start_frame"], row["active_stop_frame"])
            for row in payload["layout"]
            if row["level_millionths"] == group["level_millionths"]
        ]
        pilot_count = len(linearity.CLOCK_PILOT_OFFSETS_SECONDS)
        for pilot in payload["clock_pilot_layout"][
            index * pilot_count : (index + 1) * pilot_count
        ]:
            assert all(
                not (
                    pilot["start_frame"] < active_stop
                    and active_start < pilot["stop_frame"]
                )
                for active_start, active_stop in active
            )
            assert len(pilot["noise_ch0_pcm_sha256"]) == 64

    safe = linearity.next_level_stop_decision(
        observed_peak=0.10, current_millionths=3000, next_millionths=4000
    )
    predictive = linearity.next_level_stop_decision(
        observed_peak=0.38, current_millionths=5000, next_millionths=6000
    )
    hard = linearity.next_level_stop_decision(
        observed_peak=0.50, current_millionths=6000, next_millionths=None
    )
    assert safe["stop"] is False
    assert predictive["stop"] is True
    assert "predictive_next_level_peak" in predictive["reasons"]
    assert "adc_absolute_peak_ceiling" in hard["reasons"]


@pytest.mark.parametrize(
    "stale_schema",
    ("recording_gain_linearity_plan/v2", "recording_gain_linearity_plan/v3_gain012"),
)
def test_old_probe_plan_schemas_cannot_be_resealed_as_v3_authority(
    tmp_path: Path, stale_schema: str
):
    payload, _pcm, _path, _sha256 = _plan(tmp_path)
    payload["schema"] = stale_schema
    payload.pop("plan_payload_sha256")
    payload["plan_payload_sha256"] = linearity._seal(payload)
    with pytest.raises(linearity.RecordingGainLinearityError, match="schema"):
        linearity.validate_gain_linearity_plan_payload(payload)


@pytest.mark.parametrize("validation_branch", ("DETACHED", "dev"))
def test_receipt_fits_err_ref_operator_and_uses_independent_6k_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_branch: str,
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": validation_branch,
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )

    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )

    assert receipt["status"] == "PASS", receipt["failure_reasons"]
    operator = receipt["analysis"]["safety_operator"]
    assert operator["role"] == "source_gain_prediction_only_not_anc_plant_authority"
    assert receipt["analysis"]["safety_operator_is_anc_plant_authority"] is False
    for name in ("err", "ref"):
        channel = operator["channels"][name]
        assert channel["passed"] is True
        assert channel["holdout"]["level_millionths"] == 6000
        assert channel["holdout"]["passed"] is True
        assert all(row["passed"] for row in channel["holdout"]["roundtrip"])


def test_receipt_fails_closed_on_adc_certification_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(
        tmp_path, payload, pcm, plan_path, plan_sha, unsafe_gain=80.0
    )
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    assert receipt["status"] == "FAIL"
    assert any("certification_peak" in reason for reason in receipt["failure_reasons"])


def test_missing_publication_remains_forensic_but_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
    )
    assert receipt["status"] == "FAIL"
    assert "capture_publication_missing" in receipt["failure_reasons"]
    assert receipt["analysis"]["failure_before_metrics"] is False
    assert receipt["analysis"]["rows"]


def test_strict_publication_receipt_roundtrip_rebuilds_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    publication_args = _publication_args(tmp_path, raw_path)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    receipt_path, receipt_sha, receipt = linearity.issue_gain_linearity_receipt(
        repo_root=tmp_path,
        output_path="results/linearity/receipt.json",
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **publication_args,
    )
    assert receipt["status"] == "PASS", receipt["failure_reasons"]
    checked = linearity.validate_gain_linearity_receipt(
        repo_root=tmp_path,
        receipt_path=receipt_path.relative_to(tmp_path).as_posix(),
        expected_sha256=receipt_sha,
    )
    assert checked["passed"] is True
    assert checked["payload"] == receipt
    assert checked["capture_publication"] == receipt["capture_publication"]
    metadata_path = tmp_path / "results/linearity/metadata.json"
    assert checked["capture_metadata"] == {
        "path": metadata_path.relative_to(tmp_path).as_posix(),
        "size": metadata_path.stat().st_size,
        "sha256": _sha(metadata_path),
    }


@pytest.mark.parametrize("target", ("metadata", "publication"))
def test_publication_or_sidecar_tamper_is_forensic_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    publication_args = _publication_args(tmp_path, raw_path)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    path = (
        tmp_path / "results/linearity/metadata.json"
        if target == "metadata"
        else tmp_path / publication_args["publication_path"]
    )
    path.write_bytes(path.read_bytes() + b" ")
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **publication_args,
    )
    assert receipt["status"] == "FAIL"
    assert any(
        reason.startswith("capture_publication_invalid:")
        for reason in receipt["failure_reasons"]
    )
    assert receipt["analysis"]["failure_before_metrics"] is False


def test_recovery_copy_cannot_be_promoted_without_publication_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    recovery = tmp_path / ".deep_anc_live_recovery_fixture_gainprobe_v3_raw"
    recovery.write_bytes((tmp_path / raw_path).read_bytes())
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=recovery.name,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
    )
    assert receipt["status"] == "FAIL"
    assert "raw_recovery_only" in receipt["failure_reasons"]
    assert "capture_publication_missing" in receipt["failure_reasons"]
    assert receipt["analysis"]["failure_before_metrics"] is False


def test_cli_dry_run_never_imports_sounddevice_or_opens_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module_name = "measure_recording_gain_linearity_dry_run_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts/data/measure_recording_gain_linearity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    hardware = _hardware(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_environment",
        lambda _path: {
            "hardware_file": hardware,
            "config": {},
            "audio": {},
            "channel_map": {},
            "fingerprint": {"schema": "fixture"},
            "identity": {},
        },
    )
    monkeypatch.setattr(
        module,
        "_raw_session_for_plan",
        lambda _plan, _requested: tmp_path / "results/raw-session",
    )
    calls = []
    original_import = module.importlib.import_module

    def guarded_import(name, *args, **kwargs):
        calls.append(name)
        if name == "sounddevice":
            raise AssertionError("dry-run imported sounddevice")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(module.importlib, "import_module", guarded_import)
    monkeypatch.setattr(
        module,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": "a" * 40,
            "repository_branch": "dev",
            "repository_dirty": False,
            "script_path": module.SCRIPT_RELATIVE_PATH,
            "script_file_sha256": "b" * 64,
        },
    )
    result = module.main(
        [
            "--dry-run",
            "--expected-commit",
            "a" * 40,
            "--hardware",
            "configs/hardware.yaml",
            "--output",
            "results/plan.json",
        ]
    )
    assert result == 0
    assert calls == []
    assert (tmp_path / "results/plan.json").is_file()


def test_cli_dry_run_rejects_wrong_checkout_before_plan_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module_name = "measure_recording_gain_linearity_wrong_commit_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts/data/measure_recording_gain_linearity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_environment",
        lambda _path: {
            "hardware_file": tmp_path / "configs/hardware.yaml",
            "config": {},
            "audio": {},
            "channel_map": {},
            "fingerprint": {"schema": "fixture"},
            "identity": {},
        },
    )
    monkeypatch.setattr(
        module,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": "b" * 40,
            "repository_branch": "dev",
            "repository_dirty": False,
            "script_path": module.SCRIPT_RELATIVE_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    result = module.main(
        [
            "--dry-run",
            "--expected-commit",
            "a" * 40,
            "--hardware",
            "configs/hardware.yaml",
            "--output",
            "results/plan.json",
        ]
    )
    assert result == 2
    assert not (tmp_path / "results/plan.json").exists()


def test_receipt_rejects_zero_preflight_and_missing_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, _raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    raw_sha = _rewrite_raw(
        tmp_path,
        raw_path,
        mutate_metadata=lambda metadata: metadata["operator_confirmations"].pop(
            "bounded_gain_probe"
        ),
        preflight=np.zeros((120_000, 2), dtype=np.int32),
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    assert receipt["status"] == "FAIL"
    assert "operator_confirmations" in receipt["failure_reasons"]
    assert "preflight_channel_invalid" in receipt["failure_reasons"]


def test_receipt_rejects_missing_callback_timebase_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, _ = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    raw_sha = _rewrite_raw(
        tmp_path,
        raw_path,
        mutate_metadata=lambda metadata: metadata["segment_telemetry"][0].__setitem__(
            "callback_time_info", None
        ),
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    assert receipt["status"] == "FAIL"
    assert "callback_timebase_witness_invalid" in receipt["failure_reasons"]


def test_callback_time_witness_does_not_treat_dac_queue_slope_as_clock_q():
    """PortAudio DAC queue timestamp slope는 pilot clock-q authority가 아니다."""

    starts = np.arange(0, 48_000, 256, dtype=np.int64)
    counts = np.full(starts.shape, 256, dtype=np.int64)
    nominal = starts.astype(np.float64) / 48_000.0 + 100.0
    evidence, _arrays = linearity.callback_time_info_evidence(
        {
            "callback_start_frames": starts,
            "callback_frame_counts": counts,
            "input_buffer_adc_time": nominal,
            # 실제 Jetson outputBufferDacTime에는 driver queue 예측 변동이 있어
            # 회귀 기울기가 ±1000 ppm을 넘을 수 있어도 strict-monotonic이다.
            "output_buffer_dac_time": 100.1 + starts.astype(np.float64) / 47_800.0,
            "callback_current_time": nominal + 0.2,
        },
        group_index=0,
        expected_frames=48_000,
    )
    summary = evidence["summary"]
    assert summary["valid"] is True
    assert summary["software_frame_gap_count"] == 0
    assert summary["hardware_sample_slip_authority"] is False
    assert (
        abs(summary["informational_fit_rate_ppm"]["output_buffer_dac_time"])
        > linearity.CLOCK_MAX_ABS_PPM
    )
    assert summary["role"].endswith("not_clock_q_authority")


def test_callback_time_witness_still_rejects_nonmonotonic_timestamp():
    starts = np.arange(0, 1024, 256, dtype=np.int64)
    counts = np.full(starts.shape, 256, dtype=np.int64)
    nominal = starts.astype(np.float64) / 48_000.0 + 100.0
    output = nominal + 0.1
    output[2] = output[1]
    with pytest.raises(linearity.RecordingGainLinearityError, match="finite monotonic"):
        linearity.callback_time_info_evidence(
            {
                "callback_start_frames": starts,
                "callback_frame_counts": counts,
                "input_buffer_adc_time": nominal,
                "output_buffer_dac_time": output,
                "callback_current_time": nominal + 0.2,
            },
            group_index=0,
            expected_frames=1024,
        )


def test_callback_time_witness_still_rejects_sample_slip():
    starts = np.asarray([0, 256, 513, 768], dtype=np.int64)
    counts = np.full(starts.shape, 256, dtype=np.int64)
    nominal = np.arange(starts.size, dtype=np.float64) / 100.0 + 100.0
    with pytest.raises(
        linearity.RecordingGainLinearityError, match="coverage/slip"
    ):
        linearity.callback_time_info_evidence(
            {
                "callback_start_frames": starts,
                "callback_frame_counts": counts,
                "input_buffer_adc_time": nominal,
                "output_buffer_dac_time": nominal + 0.1,
                "callback_current_time": nominal + 0.2,
            },
            group_index=0,
            expected_frames=1024,
        )


def test_receipt_rejects_wrong_peak_or_unobservable_clock_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, _ = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    pilot_starts = payload["capture_groups"][0]["clock_pilot_start_frames"]

    def erase_pilot_responses(arrays):
        recorded = arrays["input_raw_int32"].copy()
        for start in pilot_starts:
            recorded[
                start : start + linearity.CLOCK_PILOT_FRAMES + linearity.MAX_DELAY_SAMPLES
            ] = 0
        arrays["input_raw_int32"] = recorded

    raw_sha = _rewrite_raw(
        tmp_path, raw_path, mutate_arrays=erase_pilot_responses
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    assert receipt["status"] == "FAIL"
    assert any("pilot_correlation" in reason for reason in receipt["failure_reasons"])


def test_receipt_dynamic_supported_cap_never_exceeds_independent_6k_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(
        tmp_path, payload, pcm, plan_path, plan_sha, unsafe_gain=55.0
    )
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    assert receipt["status"] == "PASS", receipt["failure_reasons"]
    assert receipt["analysis"]["tested_max_amplitude_millionths"] == 6000
    assert 1 <= receipt["analysis"]["supported_max_amplitude_millionths"] < 6000
    for channel in receipt["analysis"]["safety_operator"]["channels"].values():
        assert channel["residual_bound"]["valid_through_amplitude_millionths"] == 6000


def test_distortion_below_noise_does_not_add_distortion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, _ = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    target = next(
        row
        for row in payload["layout"]
        if row["kind"] == "IMD" and row["pair_hz"] == [3600.0, 4400.0]
    )

    def erase_imd(arrays):
        recorded = arrays["input_raw_int32"].copy()
        recorded[target["start_frame"] : target["stop_frame"]] = 0
        arrays["input_raw_int32"] = recorded

    raw_sha = _rewrite_raw(tmp_path, raw_path, mutate_arrays=erase_imd)
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    # 이 fixture는 해당 IMD slot 안의 clock pilot도 지우므로 별도의 clock/operator
    # 사유로는 FAIL이다. 다만 관측 불가능한 distortion 자체가 failure reason을
    # 만들면 안 된다.
    assert receipt["status"] == "FAIL"
    assert not any(
        "_thd_" in reason or "_imd_" in reason
        for reason in receipt["failure_reasons"]
    )
    rows = [
        row
        for row in receipt["analysis"]["rows"]
        if row["kind"] == "IMD" and row["pair_hz"] == [3600.0, 4400.0]
    ]
    assert any(
        channel["verdict"] == "INCONCLUSIVE"
        for row in rows
        for channel in row["channels"].values()
    )


def test_unobservable_distortion_is_not_certified_but_operator_can_bound_adc_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        linearity,
        "_distortion_metrics",
        lambda *_args, **_kwargs: {
            # Raw ratio alone is over the nominal gate, but the noise floor
            # makes it unobservable. It must remain INCONCLUSIVE rather than
            # fail the deliberately peak-safety-only receipt.
            "thd_dbc": -20.0,
            "imd_dbc": -19.0,
            "fundamental_snr_db": 20.0,
            "observable": False,
            "verdict": "INCONCLUSIVE",
        },
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    assert receipt["status"] == "PASS", receipt["failure_reasons"]
    analysis = receipt["analysis"]
    assert analysis["distortion_certified"] is False
    assert analysis["physical_authority_scope"] == (
        linearity.GAIN_LINEARITY_AUTHORITY_SCOPE
    )
    assert analysis["distortion_observability"][
        "observable_channel_row_count"
    ] == 0
    assert analysis["distortion_observability"][
        "inconclusive_is_not_thd_pass"
    ] is True


def test_observable_distortion_over_gate_fails_peak_safety_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        linearity,
        "_distortion_metrics",
        lambda *_args, **_kwargs: {
            "thd_dbc": -20.0,
            "imd_dbc": -19.0,
            "fundamental_snr_db": 60.0,
            "observable": True,
            "verdict": "FAIL",
        },
    )
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    assert receipt["status"] == "FAIL"
    assert any("_thd_" in reason for reason in receipt["failure_reasons"])
    assert any("_imd_" in reason for reason in receipt["failure_reasons"])
    assert receipt["analysis"]["distortion_observability"][
        "observable_channel_row_count"
    ] == 24


def test_repeated_pilots_recover_group_common_400ppm_without_erasing_relative_delay(
    tmp_path: Path
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, _ = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    with np.load(tmp_path / raw_path, allow_pickle=False) as archive:
        baseline = np.asarray(archive["input_raw_int32"], dtype=np.float64) / float(2**31)
    warped = np.zeros_like(baseline)
    q = 1.0 + 400.0e-6
    for group in payload["capture_groups"]:
        start, stop = int(group["start_frame"]), int(group["stop_frame"])
        output_axis = np.arange(start, stop, dtype=np.float64)
        source_query = start + (output_axis - start) / q
        for channel in range(2):
            warped[start:stop, channel] = np.interp(
                source_query,
                output_axis,
                baseline[start:stop, channel],
            )
    alignments, reasons = linearity._group_clock_alignment(
        payload=payload,
        source_float=pcm[:, 0].astype(np.float64) / 32767.0,
        input_float=warped,
    )
    assert reasons == []
    for item in alignments.values():
        assert (item["common_q_ratio"] - 1.0) * 1.0e6 == pytest.approx(400.0, abs=5.0)
        assert item["all_groups_relative_delay_spread_samples"] <= 3.0
        relative = item["relative_delay_samples"]
        assert relative[0] == pytest.approx(relative[1], abs=0.1)


def _plant_colored_capture(payload: dict, pcm: np.ndarray) -> np.ndarray:
    source = pcm[:, 0].astype(np.float64) / 32767.0
    recorded = np.zeros((pcm.shape[0], 2), dtype=np.float64)
    filters = (
        (signal.firwin(257, [900.0, 6_500.0], pass_zero=False, fs=48_000) * 5.0, 150),
        (signal.firwin(257, [500.0, 4_500.0], pass_zero=False, fs=48_000) * 5.0, 210),
    )
    for group in payload["capture_groups"]:
        start, stop = int(group["start_frame"]), int(group["stop_frame"])
        for channel, (fir, delay) in enumerate(filters):
            response = signal.fftconvolve(source[start:stop], fir)
            count = min(response.size, stop - start - delay)
            recorded[start + delay : start + delay + count, channel] = response[:count]
    return recorded


def test_plant_colored_repeat_responses_recover_q_when_source_template_corr_is_low(
    tmp_path: Path,
):
    payload, pcm, _plan_path, _plan_sha = _plan(tmp_path)
    baseline = _plant_colored_capture(payload, pcm)
    warped = np.zeros_like(baseline)
    q = 1.0 + 400.0e-6
    for group in payload["capture_groups"]:
        start, stop = int(group["start_frame"]), int(group["stop_frame"])
        observed_axis = np.arange(start, stop, dtype=np.float64)
        source_query = start + (observed_axis - start) / q
        for channel in range(2):
            warped[start:stop, channel] = np.interp(
                source_query,
                observed_axis,
                baseline[start:stop, channel],
            )
    alignments, reasons = linearity._group_clock_alignment(
        payload=payload,
        source_float=pcm[:, 0].astype(np.float64) / 32767.0,
        input_float=warped,
    )
    assert reasons == []
    for alignment in alignments.values():
        assert (alignment["common_q_ratio"] - 1.0) * 1.0e6 == pytest.approx(
            400.0, abs=1.0
        )
        assert any(
            channel[
                "source_template_first_normalised_correlation_informational"
            ]
            < linearity.CLOCK_PILOT_MIN_NORMALISED_CORRELATION
            for channel in alignment["channels"].values()
        )
        assert all(
            min(channel["repeat_response_normalised_correlation"])
            >= linearity.CLOCK_PILOT_MIN_NORMALISED_CORRELATION
            for channel in alignment["channels"].values()
        )


def test_five_pilot_trajectory_rejects_one_sample_midstream_step(tmp_path: Path):
    payload, pcm, _plan_path, _plan_sha = _plan(tmp_path)
    baseline = _plant_colored_capture(payload, pcm)
    stepped = baseline.copy()
    for group in payload["capture_groups"]:
        start, stop = int(group["start_frame"]), int(group["stop_frame"])
        cut = int(group["stimulus_start_frame"]) + 48_000
        stepped[cut : stop - 1] = baseline[cut + 1 : stop]
        stepped[stop - 1] = 0.0
    _alignments, reasons = linearity._group_clock_alignment(
        payload=payload,
        source_float=pcm[:, 0].astype(np.float64) / 32767.0,
        input_float=stepped,
    )
    assert any(reason.endswith("clock_trajectory") for reason in reasons)


@pytest.mark.parametrize(
    ("target", "reason_prefix"),
    (
        ("_group_clock_alignment", "clock_alignment_build:"),
        ("_build_safety_operators", "safety_operator_build:"),
    ),
)
def test_analysis_failure_is_sealed_as_fail_receipt_instead_of_keyerror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    reason_prefix: str,
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    monkeypatch.setattr(
        linearity,
        "repository_execution_identity",
        lambda *_args: {
            "repository_commit": payload["source_commit"],
            "repository_branch": "DETACHED",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": "c" * 64,
        },
    )

    def fail(*_args, **_kwargs):
        raise linearity.RecordingGainLinearityError("forced physical analysis failure")

    monkeypatch.setattr(linearity, target, fail)
    receipt = linearity.build_gain_linearity_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **_publication_args(tmp_path, raw_path),
    )
    assert receipt["status"] == "FAIL"
    assert receipt["analysis"]["failure_before_metrics"] is True
    assert receipt["analysis"]["rows"] == []
    assert any(reason.startswith(reason_prefix) for reason in receipt["failure_reasons"])


def test_distortion_observability_uses_same_cardinality_integrated_noise_masks():
    frames = int(round(linearity.IMD_ANALYSIS_SECONDS * linearity.SAMPLE_RATE))
    time_axis = np.arange(frames, dtype=np.float64) / linearity.SAMPLE_RATE
    pair = (1_800.0, 2_200.0)
    measured = np.sin(2.0 * np.pi * pair[0] * time_axis) + np.sin(
        2.0 * np.pi * pair[1] * time_axis
    )
    products = {
        abs(pair[1] - pair[0]),
        pair[0] + pair[1],
        abs(2 * pair[0] - pair[1]),
        abs(2 * pair[1] - pair[0]),
        2 * pair[0] + pair[1],
        pair[0] + 2 * pair[1],
    }
    # 각 product bin은 -42 dBc보다 낮지만 여섯 product의 integrated bound는
    # 약 -34.2 dBc다. Single-bin noise gate는 이를 잘못 observable로 만들었다.
    per_product_amplitude = math.sqrt(2.0 * 10.0 ** (-42.0 / 10.0))
    preflight = sum(
        per_product_amplitude * np.sin(2.0 * np.pi * frequency * time_axis)
        for frequency in products
    )
    metrics = linearity._distortion_metrics(
        measured,
        pair,
        preflight_values=preflight,
    )
    assert metrics["thd_noise_bin_count"] == 12
    assert metrics["imd_noise_bin_count"] == 18
    assert metrics["imd_matched_noise_dbc"] == pytest.approx(-34.2185, abs=0.02)
    assert metrics["imd_noise_margin_below_gate_db"] < 10.0
    assert metrics["observable"] is False
    assert metrics["verdict"] == "INCONCLUSIVE"


def test_strict_artifact_coloration_uses_absolute_bound_for_negligible_highbands(
    tmp_path: Path,
):
    payload, pcm, _plan_path, _plan_sha = _plan(tmp_path)
    source = pcm[:, 0].astype(np.float64) / 32767.0
    recorded = np.zeros((pcm.shape[0], 2), dtype=np.float64)
    artifacts = (
        ROOT / "assets/measured/primary_path_il_strict_5dc06fdd.npz",
        ROOT / "assets/measured/secondary_path_il_strict_5dc06fdd.npz",
    )
    for channel, artifact in enumerate(artifacts):
        with np.load(artifact, allow_pickle=False) as archive:
            fir = np.asarray(archive["fir"], dtype=np.float64)
            delay = int(archive["delay_samples"])
        for group in payload["capture_groups"]:
            start, stop = int(group["start_frame"]), int(group["stop_frame"])
            response = signal.fftconvolve(source[start:stop], fir)
            count = min(response.size, stop - start - delay)
            recorded[start + delay : start + delay + count, channel] = response[:count]
    alignment, clock_reasons = linearity._group_clock_alignment(
        payload=payload,
        source_float=source,
        input_float=recorded,
    )
    assert clock_reasons == []
    operator, operator_reasons = linearity._build_safety_operators(
        payload=payload,
        source_float=source,
        input_float=recorded,
        clock_alignment=alignment,
    )
    assert operator_reasons == []
    for channel in operator["channels"].values():
        assert channel["passed"] is True
        rows = channel["holdout"]["roundtrip"]
        assert rows[0]["relative_gate_applicable"] is True
        assert rows[0]["passed"] is True
        for row in rows[-2:]:
            assert row["target_norm_ratio_to_fullband"] < 0.01
            assert row["relative_gate_applicable"] is False
            assert row["absolute_residual_bound_role"] is True
            assert row["passed"] is True


def test_cli_parser_has_one_plan_sha_option_and_help_is_renderable():
    module_name = "measure_recording_gain_linearity_parser_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts/data/measure_recording_gain_linearity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    parser = module.build_parser()
    parser.format_help()
    assert sum(
        "--plan-sha256" in action.option_strings for action in parser._actions
    ) == 1


def test_cli_ref_witness_reanalysis_separates_capture_and_analyzer_commits_without_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module_name = "measure_recording_gain_linearity_reanalysis_cli_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts/data/measure_recording_gain_linearity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    capture_commit = "a" * 40
    analyzer_commit = "b" * 40
    monkeypatch.setattr(
        module,
        "load_gain_linearity_plan",
        lambda **_kwargs: {
            "payload": {"source_commit": capture_commit},
            "file": {"path": "results/plan.json", "sha256": "c" * 64},
        },
    )
    monkeypatch.setattr(
        module,
        "repository_execution_identity",
        lambda *_args: {"repository_commit": analyzer_commit},
    )
    issued = []

    def _issue(**kwargs):
        issued.append(kwargs)
        return (
            tmp_path / "results/new-v5.json",
            "d" * 64,
            {"status": "PASS", "failure_reasons": []},
        )

    monkeypatch.setattr(module, "issue_gain_linearity_reanalysis_receipt", _issue)
    imported = []
    original_import = module.importlib.import_module

    def _guarded_import(name, *args, **kwargs):
        imported.append(name)
        if name == "sounddevice":
            raise AssertionError("reanalysis imported sounddevice")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(module.importlib, "import_module", _guarded_import)
    result = module.main(
        [
            "--reanalyze-ref-witness",
            "--expected-commit",
            analyzer_commit,
            "--expected-capture-commit",
            capture_commit,
            "--raw",
            "results/raw.npz",
            "--raw-sha256",
            "e" * 64,
            "--plan",
            "results/plan.json",
            "--plan-sha256",
            "c" * 64,
            "--publication",
            "results/publication.json",
            "--publication-sha256",
            "f" * 64,
            "--receipt-out",
            "results/new-v5.json",
        ]
    )
    assert result == 0
    assert len(issued) == 1
    assert imported == []


def test_global_live_deadline_reserves_next_output_before_open():
    module_name = "measure_recording_gain_linearity_deadline_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts/data/measure_recording_gain_linearity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    assert module._global_live_deadline_elapsed(
        campaign_started=100.0,
        campaign_deadline=137.0,
        now=129.5,
        reserve_seconds=7.5,
        stage="last_group_pre_open",
    ) == pytest.approx(29.5)
    with pytest.raises(TimeoutError, match="last_group_pre_open"):
        module._global_live_deadline_elapsed(
            campaign_started=100.0,
            campaign_deadline=137.0,
            now=129.500001,
            reserve_seconds=7.5,
            stage="last_group_pre_open",
        )


@pytest.mark.parametrize(
    ("failure_kind", "expected_calls", "expected_reason"),
    (
        ("last_group_plan_mutation", 4, "capture_exception:"),
        ("first_group_submitted_mismatch", 1, "submitted_pcm_not_exact_segment"),
        ("first_group_callback_duplicate", 1, "callback_time_witness:"),
        ("first_group_session_splice", 1, "capture_exception:"),
        ("first_group_raw_leaf_preclaim", 1, "capture_exception:"),
        ("first_group_results_ancestor_splice", 1, "capture_exception:"),
        ("signal_between_groups", 1, "capture_exception:"),
        ("signal_after_output", 4, None),
    ),
)
def test_live_integrity_failure_stops_before_higher_level_and_saves_invalid_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_calls: int,
    expected_reason: str | None,
):
    module_name = "measure_recording_gain_linearity_last_group_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts/data/measure_recording_gain_linearity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    payload, _pcm, plan_path, plan_sha = _plan(tmp_path)
    hardware = tmp_path / "configs/hardware.yaml"
    execution = {
        "repository_commit": "a" * 40,
        "repository_branch": "DETACHED",
        "repository_dirty": False,
        "script_path": module.SCRIPT_RELATIVE_PATH,
        "script_file_sha256": "c" * 64,
    }
    environment = {
        "hardware_file": hardware,
        "config": {},
        "audio": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "output": {"card": "Audio", "pcm": 0},
        },
        "channel_map": {"error_mic": 0, "reference_mic": 1},
        "fingerprint": payload["hardware"]["physical_fingerprint"],
        "identity": {},
    }
    rng = np.random.default_rng(99)
    preflight = rng.integers(-2_000_000, 2_000_001, (120_000, 2), dtype=np.int32)
    report = analyze_int32_input_probe(preflight)
    report.update({"device": 7, "sample_rate": 48_000, "settle_seconds": 0.5})
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "repository_execution_identity", lambda *_a: execution)
    monkeypatch.setattr(module, "_environment", lambda *_a: environment)
    monkeypatch.setattr(module, "assert_live_pcm_clock_preconditions", lambda *_a: None)
    monkeypatch.setattr(module, "resolve_alsa_portaudio_device", lambda *_a: 8)
    monkeypatch.setattr(module.broadband, "validate_fresh_raw_session_target", lambda *_a: None)
    monkeypatch.setattr(
        module,
        "capture_measurement_preflight_raw",
        lambda *_a, **_k: (
            preflight,
            {
                "passed": True,
                "resolved_input_device": 7,
                "sample_rate_hz": 48_000,
            },
        ),
    )

    @contextmanager
    def lock(*_args, **_kwargs):
        yield

    calls = 0

    def capture(_sd, *, output_float, pre_open_check, **_kwargs):
        nonlocal calls
        calls += 1
        pre_open_check()
        submitted = np.rint(output_float * np.float32(32767.0)).astype(np.int16)
        recorded = np.zeros(submitted.shape, dtype=np.int32)
        if failure_kind == "first_group_submitted_mismatch" and calls == 1:
            submitted = submitted.copy()
            submitted[0, 0] = np.int16(int(submitted[0, 0]) + 1)
        if failure_kind == "last_group_plan_mutation" and calls == 4:
            (tmp_path / plan_path).write_bytes((tmp_path / plan_path).read_bytes() + b" ")
        if failure_kind == "first_group_session_splice" and calls == 1:
            session = tmp_path / "results/raw-session"
            detached = tmp_path / "results/raw-session-detached"
            replacement = tmp_path / "results/raw-session"
            session.rename(detached)
            replacement.mkdir()
        if failure_kind == "first_group_raw_leaf_preclaim" and calls == 1:
            (tmp_path / "results/raw-session/raw_measurement.npz").write_bytes(
                b"attacker-owned-leaf"
            )
        if failure_kind == "first_group_results_ancestor_splice" and calls == 1:
            results = tmp_path / "results"
            results.rename(tmp_path / "results-detached")
            results.mkdir()
        starts = np.arange(0, len(submitted), 256, dtype=np.int64)
        counts = np.full(starts.shape, 256, dtype=np.int64)
        clock = starts.astype(np.float64) / 48_000.0 + calls * 10.0
        if failure_kind == "first_group_callback_duplicate" and calls == 1:
            clock[1] = clock[0]
        telemetry = {
            "completed": True,
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_status_count": 0,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
            "callback_time_info": {
                "callback_start_frames": starts,
                "callback_frame_counts": counts,
                "input_buffer_adc_time": clock,
                "output_buffer_dac_time": clock + 0.001,
                "callback_current_time": clock + 0.002,
            },
        }
        cleanup_handoff = _kwargs.get("on_output_cleanup_complete")
        assert callable(cleanup_handoff)
        cleanup_handoff()
        if failure_kind == "signal_between_groups" and calls == 1:
            os.kill(os.getpid(), int(os_signal.SIGTERM))
        return recorded, submitted, telemetry

    if failure_kind == "signal_after_output":
        def signal_between_notice_and_commit(capture_callable):
            result = capture_callable()
            # capture_callable의 finally가 queue-only 상태를 설정한 직후를 공격한다.
            os.kill(os.getpid(), int(os_signal.SIGTERM))
            return result

        monkeypatch.setattr(
            module.mpi,
            "capture_with_speaker_release_notice",
            signal_between_notice_and_commit,
        )

    result = module.execute_live_capture(
        hardware_path="configs/hardware.yaml",
        expected_commit="a" * 40,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        raw_session_dir=tmp_path / "results/raw-session",
        confirmations=dict(linearity.EXACT_OPERATOR_CONFIRMATIONS),
        sounddevice_module=object(),
        capture_function=capture,
        audio_lock_factory=lock,
    )
    assert calls == expected_calls
    if failure_kind == "signal_after_output":
        assert result["valid_raw"] is True
        assert result["deferred_termination_signal"] == int(os_signal.SIGTERM)
        assert result["paths"]["raw"].is_file()
        assert result["paths"]["metadata"].is_file()
        assert result["paths"]["publication"].is_file()
        return
    assert result["valid_raw"] is False
    assert expected_reason is not None
    assert any(
        reason.startswith(expected_reason)
        for reason in result["metadata"]["invalid_reasons"]
    )
    if failure_kind in {
        "last_group_plan_mutation",
        "first_group_session_splice",
            "first_group_raw_leaf_preclaim",
            "first_group_results_ancestor_splice",
            "signal_between_groups",
        }:
        assert result["metadata"]["capture_exception"] is not None
    else:
        assert result["metadata"]["capture_exception"] is None
        assert result["metadata"]["safety_stop"]["stop"] is True
    assert result["paths"]["raw"].is_file()
    if failure_kind == "first_group_session_splice":
        replacement = tmp_path / "results/raw-session"
        detached = tmp_path / "results/raw-session-detached"
        assert list(replacement.iterdir()) == []
        assert (detached / "raw_measurement.npz").is_file()
        assert "raw_session_binding_changed" in result["metadata"]["invalid_reasons"]
        assert result["raw_ref"]["recovery_path"] is not None
    if failure_kind == "first_group_raw_leaf_preclaim":
        attacker = tmp_path / "results/raw-session/raw_measurement.npz"
        assert attacker.read_bytes() == b"attacker-owned-leaf"
        assert result["raw_ref"]["final_published"] is False
        assert result["paths"]["raw"] != attacker
        assert result["paths"]["raw"].parent == tmp_path
    if failure_kind == "first_group_results_ancestor_splice":
        assert list((tmp_path / "results").iterdir()) == []
        assert (
            tmp_path / "results-detached/raw-session/raw_measurement.npz"
        ).is_file()
        assert result["paths"]["raw"].parent == tmp_path
        assert result["raw_ref"]["recovery_path"] is not None


@pytest.mark.parametrize(
    "signum", (os_signal.SIGINT, os_signal.SIGTERM, os_signal.SIGHUP)
)
def test_gainprobe_raw_commit_defers_signal_until_same_inode_sha_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signum: os_signal.Signals,
):
    module_name = f"measure_recording_gain_linearity_signal_{int(signum)}"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts/data/measure_recording_gain_linearity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    guard = module.RepositoryDirectoryGuard.create_fresh(
        tmp_path, "results/raw-signal", label="signal fixture"
    )
    original_savez = module.np.savez_compressed

    def signal_during_compression(*args, **kwargs):
        os.kill(os.getpid(), int(signum))
        return original_savez(*args, **kwargs)

    monkeypatch.setattr(module.np, "savez_compressed", signal_during_compression)
    try:
        result = module._commit_raw_capture_held(
            session_guard=guard,
            metadata={"status": "fixture"},
            arrays={"input_raw_int32": np.arange(32, dtype=np.int32)},
        )
    finally:
        guard.close()

    raw_path = result["paths"]["raw"]
    assert result["deferred_termination_signal"] == int(signum)
    assert raw_path.is_file()
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == result["raw_ref"][
        "sha256"
    ]
    assert result["raw_ref"]["recovery_path"] is not None


@pytest.mark.parametrize(
    ("preclaimed_leaf", "error_key"),
    (
        ("metadata.json", "metadata_error"),
        ("capture_publication.json", "publication_error"),
    ),
)
def test_late_sidecar_or_publication_preclaim_preserves_raw_but_blocks_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preclaimed_leaf: str,
    error_key: str,
):
    module_name = f"measure_recording_gain_linearity_preclaim_{preclaimed_leaf}"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts/data/measure_recording_gain_linearity.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    guard = module.RepositoryDirectoryGuard.create_fresh(
        tmp_path, "results/raw-preclaim", label="preclaim fixture"
    )
    attacker = tmp_path / "results/raw-preclaim" / preclaimed_leaf
    attacker.write_bytes(b"attacker-owned-leaf")
    execution = {
        "repository_commit": "a" * 40,
        "repository_branch": "dev",
        "repository_dirty": False,
        "script_path": module.SCRIPT_RELATIVE_PATH,
        "script_file_sha256": "c" * 64,
    }
    try:
        result = module._commit_raw_capture_held(
            session_guard=guard,
            metadata={
                "source_commit": "a" * 40,
                "repository_execution": execution,
                "plan": {"path": "results/plan.json", "sha256": "d" * 64},
            },
            arrays={"input_raw_int32": np.arange(32, dtype=np.int32)},
        )
    finally:
        guard.close()
    assert attacker.read_bytes() == b"attacker-owned-leaf"
    assert result["paths"]["raw"].is_file()
    assert result["raw_ref"]["final_published"] is True
    assert result[error_key] is not None
    assert result["publication_ref"] is None


def test_ref_witness_reanalysis_separates_capture_and_analysis_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(tmp_path, payload, pcm, plan_path, plan_sha)
    publication = _publication_args(tmp_path, raw_path)
    analysis_execution = {
        "repository_commit": "b" * 40,
        "repository_branch": "dev",
        "repository_dirty": False,
        "script_path": linearity.GAIN_LINEARITY_ANALYZER_PATH,
        "script_file_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        linearity, "_analysis_execution_identity", lambda _root: analysis_execution
    )
    monkeypatch.setattr(
        linearity, "_verify_historical_capture_execution", lambda *_args: None
    )
    monkeypatch.setattr(
        linearity,
        "_group_clock_alignment",
        lambda **_kwargs: pytest.fail("재분석이 affine clock 경로를 호출했습니다"),
    )

    receipt = linearity.build_gain_linearity_reanalysis_receipt(
        repo_root=tmp_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha,
        plan_path=plan_path,
        expected_plan_sha256=plan_sha,
        **publication,
    )

    assert receipt["schema"] == linearity.GAIN_LINEARITY_REANALYSIS_RECEIPT_SCHEMA
    assert receipt["status"] == "PASS"
    assert receipt["source_commit"] == "b" * 40
    assert receipt["capture_provenance"]["source_commit"] == "a" * 40
    assert receipt["analysis_provenance"] == analysis_execution
    assert receipt["analysis"]["clock_alignment_method"] == linearity.TIMELINE_METHOD
    assert all(
        row["affine_q_used"] is False
        and row["valid_windows"] >= linearity.MIN_STREAM_DELAY_VALID_WINDOWS
        for row in receipt["analysis"]["clock_alignment"]
    )
    envelope = receipt["analysis"]["safety_operator"]
    assert envelope["schema"] == linearity.GAIN_LINEARITY_PEAK_ENVELOPE_SCHEMA
    assert envelope["complex_operator_thresholds_relaxed"] is False
    assert envelope["complex_operator_used_as_authority"] is False


def test_reanalysis_receipt_uses_new_noreplace_path_and_preserves_old_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    old = tmp_path / "results/old_fail_receipt.json"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"immutable-old-fail\n")
    monkeypatch.setattr(
        linearity,
        "build_gain_linearity_reanalysis_receipt",
        lambda **_kwargs: {"schema": "fixture", "status": "PASS"},
    )
    kwargs = {
        "repo_root": tmp_path,
        "output_path": "results/new_reanalysis_receipt.json",
        "raw_path": "results/raw.npz",
        "expected_raw_sha256": "a" * 64,
        "plan_path": "results/plan.json",
        "expected_plan_sha256": "b" * 64,
        "publication_path": "results/capture_publication.json",
        "expected_publication_sha256": "c" * 64,
    }
    target, _digest, _payload = (
        linearity.issue_gain_linearity_reanalysis_receipt(**kwargs)
    )
    assert target.name == "new_reanalysis_receipt.json"
    assert old.read_bytes() == b"immutable-old-fail\n"
    with pytest.raises(linearity.RecordingGainLinearityError, match="no-replace"):
        linearity.issue_gain_linearity_reanalysis_receipt(**kwargs)


def test_reanalysis_rejects_replace_or_dirty_analyzer_before_historical_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    historical_called = False

    def reject_analyzer(_root):
        raise RuntimeError("git replacement object가 있는 checkout")

    def historical_should_not_run(*_args):
        nonlocal historical_called
        historical_called = True

    monkeypatch.setattr(linearity, "_analysis_execution_identity", reject_analyzer)
    monkeypatch.setattr(
        linearity, "_verify_historical_capture_execution", historical_should_not_run
    )
    with pytest.raises(RuntimeError, match="replacement object"):
        linearity.build_gain_linearity_reanalysis_receipt(
            repo_root=tmp_path,
            raw_path="results/raw.npz",
            expected_raw_sha256="a" * 64,
            plan_path="results/plan.json",
            expected_plan_sha256="b" * 64,
            publication_path="results/capture_publication.json",
            expected_publication_sha256="c" * 64,
        )
    assert historical_called is False


def test_historical_capture_blob_lookup_disables_replace_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    script_bytes = b"historical capture executable\n"
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs.get("env")
        return SimpleNamespace(stdout=script_bytes)

    monkeypatch.setattr(linearity.subprocess, "run", fake_run)
    linearity._verify_historical_capture_execution(
        tmp_path,
        {
            "repository_commit": "a" * 40,
            "repository_branch": "dev",
            "repository_dirty": False,
            "script_path": linearity.GAIN_LINEARITY_SCRIPT_PATH,
            "script_file_sha256": hashlib.sha256(script_bytes).hexdigest(),
        },
    )

    assert observed["command"][-1] == (
        "a" * 40 + ":" + linearity.GAIN_LINEARITY_SCRIPT_PATH
    )
    assert isinstance(observed["env"], dict)
    assert observed["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
