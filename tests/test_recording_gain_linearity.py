from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

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
    for row in payload["layout"]:
        start = int(row["start_frame"])
        stop = int(row["stop_frame"])
        active = int(row["active_frames"])
        for channel, (fir, delay) in enumerate(firs):
            response = signal.fftconvolve(source[start : start + active], fir)
            count = min(response.size, stop - start - delay)
            recorded[start + delay : start + delay + count, channel] = response[:count]
    recorded_i32 = np.rint(
        np.clip(recorded, -0.999999, 0.999999) * float(2**31)
    ).astype(np.int32)
    telemetry = [
        {
            "level_millionths": group["level_millionths"],
            "start_frame": group["start_frame"],
            "stop_frame": group["stop_frame"],
            "completed": True,
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_status_count": 0,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
        }
        for group in payload["capture_groups"]
    ]
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
    raw_path = root / "results/linearity/raw.npz"
    with raw_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            submitted_output_pcm_int16=pcm,
            input_raw_int32=recorded_i32,
            preflight_raw_int32=preflight,
        )
    return "results/linearity/raw.npz", _sha(raw_path)


def _rewrite_raw(
    root: Path,
    raw_relative: str,
    *,
    mutate_metadata=None,
    preflight: np.ndarray | None = None,
) -> str:
    path = root / raw_relative
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        submitted = np.asarray(archive["submitted_output_pcm_int16"])
        recorded = np.asarray(archive["input_raw_int32"])
        saved_preflight = np.asarray(archive["preflight_raw_int32"])
    if mutate_metadata is not None:
        mutate_metadata(metadata)
    if preflight is None:
        preflight = saved_preflight
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            submitted_output_pcm_int16=submitted,
            input_raw_int32=recorded,
            preflight_raw_int32=preflight,
        )
    return _sha(path)


def test_plan_is_four_level_four_stream_ns_only_and_predictively_bounded(tmp_path: Path):
    payload, pcm, _path, _sha256 = _plan(tmp_path)

    assert payload["contract"]["levels_millionths"] == [3000, 6000, 9000, 12000]
    assert payload["duration"]["output_open_seconds"] == 24.0
    assert payload["duration"]["stream_open_count"] == 4
    assert payload["duration"]["connected_upper_seconds"] == 35.0
    assert 14.0 < payload["duration"]["audible_nonzero_seconds"] < 15.0
    assert pcm.shape == (24 * 48_000, 2)
    assert np.count_nonzero(pcm[:, 1]) == 0
    assert len(payload["capture_groups"]) == 4

    safe = linearity.next_level_stop_decision(
        observed_peak=0.10, current_millionths=3000, next_millionths=6000
    )
    predictive = linearity.next_level_stop_decision(
        observed_peak=0.30, current_millionths=6000, next_millionths=9000
    )
    hard = linearity.next_level_stop_decision(
        observed_peak=0.50, current_millionths=12000, next_millionths=None
    )
    assert safe["stop"] is False
    assert predictive["stop"] is True
    assert "predictive_next_level_peak" in predictive["reasons"]
    assert "adc_absolute_peak_ceiling" in hard["reasons"]


def test_receipt_fits_err_ref_operator_and_uses_independent_12k_holdout(
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

    assert receipt["status"] == "PASS", receipt["failure_reasons"]
    operator = receipt["analysis"]["safety_operator"]
    assert operator["role"] == "source_gain_prediction_only_not_anc_plant_authority"
    assert receipt["analysis"]["safety_operator_is_anc_plant_authority"] is False
    for name in ("err", "ref"):
        channel = operator["channels"][name]
        assert channel["passed"] is True
        assert channel["holdout"]["level_millionths"] == 12000
        assert channel["holdout"]["passed"] is True
        assert all(row["passed"] for row in channel["holdout"]["roundtrip"])


def test_receipt_fails_closed_on_adc_certification_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload, pcm, plan_path, plan_sha = _plan(tmp_path)
    raw_path, raw_sha = _raw(
        tmp_path, payload, pcm, plan_path, plan_sha, unsafe_gain=40.0
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
    )
    assert receipt["status"] == "FAIL"
    assert any("certification_peak" in reason for reason in receipt["failure_reasons"])


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
    )
    assert receipt["status"] == "FAIL"
    assert "operator_confirmations" in receipt["failure_reasons"]
    assert "preflight_channel_invalid" in receipt["failure_reasons"]


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


def test_live_last_group_plan_mutation_is_saved_as_invalid_raw_without_audio_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        if calls == 4:
            (tmp_path / plan_path).write_bytes((tmp_path / plan_path).read_bytes() + b" ")
        telemetry = {
            "completed": True,
            "xrun_count": 0,
            "unexpected_status_count": 0,
            "callback_status_count": 0,
            "callback_error": None,
            "stream_abort_error": None,
            "stream_close_error": None,
            "output_stop_confirmed": True,
        }
        return recorded, submitted, telemetry

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
    assert calls == 4
    assert result["valid_raw"] is False
    assert any(
        reason.startswith("capture_exception:")
        for reason in result["metadata"]["invalid_reasons"]
    )
    assert result["metadata"]["capture_exception"] is not None
    assert result["paths"]["raw"].is_file()
