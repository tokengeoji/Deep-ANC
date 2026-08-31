from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from deep_anc.audio_duplex_stage2 import capture_output_master_stage2
from deep_anc.dsp.stage2_2khz_diagnostic_clock import DIAGNOSTIC_CLOCK_SCHEMA
from deep_anc.dsp.stage2_2khz_measurement_v2 import (
    _payload_sha256,
    build_stage2_v2_live_safe_fallback_plan,
)
from deep_anc.dsp.stage2_2khz_output_master_diagnostic import (
    OUTPUT_MASTER_CLOCK_RECEIPT_SCHEMA,
    OUTPUT_MASTER_PARTIAL_RAW_SCHEMA,
    OUTPUT_MASTER_RAW_SCHEMA,
    OutputMasterDiagnosticCaptureError,
    capture_publish_reload_analyse_output_master_diagnostic,
    output_master_session_targets,
    publish_output_master_raw_no_replace,
)


ROOT = Path(__file__).resolve().parents[1]


class Stop(Exception):
    pass


class Abort(Exception):
    pass


class FakeSplitBackend:
    CallbackStop = Stop
    CallbackAbort = Abort

    def __init__(self, *, bad_output_callback: int | None = None) -> None:
        self.bad_output_callback = bad_output_callback
        self.input_kwargs: dict = {}
        self.output_kwargs: dict = {}
        self.input_index = 0
        self.output_index = 0
        self.input_stopped = False
        self.calls: list[object] = []

    def _pump_input(self, blocks: int) -> None:
        callback = self.input_kwargs["callback"]
        for _ in range(blocks):
            if self.input_stopped:
                return
            index = self.input_index
            # 실제 signal response가 아니라 transport/reload 검사용 finite raw다.
            data = np.full((256, 2), index + 1, dtype="<i4")
            seconds = index * 256.25 / 48_000.0
            self.input_index += 1
            try:
                callback(
                    data,
                    256,
                    {
                        "inputBufferAdcTime": 10.0 + seconds,
                        "currentTime": 20.0 + seconds,
                    },
                    None,
                )
            except Stop:
                self.input_stopped = True
            except Abort:
                self.input_stopped = True

    def InputStream(self, **kwargs):
        self.input_kwargs = kwargs
        outer = self

        class Stream:
            def start(self):
                outer.calls.append("input_start")
                outer._pump_input(16)

            def stop(self, *, ignore_errors):
                outer.calls.append(("input_stop", ignore_errors))

            def abort(self, *, ignore_errors):
                outer.calls.append(("input_abort", ignore_errors))

            def close(self, *, ignore_errors):
                outer.calls.append(("input_close", ignore_errors))

        return Stream()

    def OutputStream(self, **kwargs):
        self.output_kwargs = kwargs
        outer = self

        class Stream:
            def start(self):
                outer.calls.append("output_start")
                callback = kwargs["callback"]
                planned_blocks = 557_056 // 256
                for _ in range(planned_blocks):
                    index = outer.output_index
                    outer.output_index += 1
                    frames = 128 if index == outer.bad_output_callback else 256
                    sink = np.full((256, 2), 9, dtype="<i2")
                    seconds = index * 256.0 / 48_000.0
                    try:
                        callback(
                            sink,
                            frames,
                            {
                                "outputBufferDacTime": 30.0 + seconds,
                                "currentTime": 40.0 + seconds,
                            },
                            None,
                        )
                    except Stop:
                        break
                    except Abort:
                        break
                    outer._pump_input(1)

            def stop(self, *, ignore_errors):
                outer.calls.append(("output_stop", ignore_errors))
                outer._pump_input(32)

            def abort(self, *, ignore_errors):
                outer.calls.append(("output_abort", ignore_errors))

            def close(self, *, ignore_errors):
                outer.calls.append(("output_close", ignore_errors))

        return Stream()


def _clock_stub(plan, submitted, captured):
    receipt = {
        "schema": DIAGNOSTIC_CLOCK_SCHEMA,
        "signal_plan_sha256": plan["canonical_payload_sha256"],
        "submitted_phase_frames": len(submitted),
        "captured_frames": len(captured),
        "passed": True,
        "diagnostic_linearity_may_run": True,
        "ps_phase_may_start": False,
    }
    receipt["canonical_payload_sha256"] = _payload_sha256(receipt)
    return receipt


def _run(tmp_path: Path, backend: FakeSplitBackend, session: str):
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    return capture_publish_reload_analyse_output_master_diagnostic(
        str(tmp_path),
        session,
        plan,
        full,
        backend=backend,
        devices={"input": 5, "output": 24},
        capture_metadata={"capture_id": "a" * 32},
        capture_callable=capture_output_master_stage2,
        pre_open_check=lambda: None,
        watchdog_grace_seconds=0.2,
        on_output_closed=None,
        clock_estimator=_clock_stub,
    )


def test_fake_backend_publishes_variable_input_raw_then_reloads_before_clock(
    tmp_path: Path,
) -> None:
    session = "results/stage2_2khz_output_master_diagnostic/session_a"
    result = _run(tmp_path, FakeSplitBackend(), session)

    assert result["status"] == (
        "PASS_OUTPUT_MASTER_CLOCK_DIAGNOSTIC_PS_STILL_FORBIDDEN"
    )
    assert result["ps_backend_calls_allowed"] == 0
    assert result["ps_phase_may_start"] is False
    assert result["plant_identification_eligible"] is False
    assert result["canonical_training_eligible"] is False
    targets = output_master_session_targets(session)
    assert result["raw_publication"]["path"] == targets["raw"]
    assert result["clock_publication"]["path"] == targets["clock_receipt"]

    with np.load(tmp_path / targets["raw"], allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        submitted = np.asarray(archive["submitted_pcm"])
        captured = np.asarray(archive["captured_pcm"])
        assert metadata["schema"] == OUTPUT_MASTER_RAW_SCHEMA
        assert metadata["ps_phase_may_start"] is False
        assert metadata["clock_authority_granted"] is False
        assert captured.shape[0] != submitted.shape[0]
        assert archive["telemetry_input_callback_sequence"].shape[0] != archive[
            "telemetry_output_callback_sequence"
        ].shape[0]
        assert np.all(archive["telemetry_submitted_valid_mask"])
        assert np.all(archive["telemetry_capture_valid_mask"])

    clock = json.loads((tmp_path / targets["clock_receipt"]).read_text())
    assert clock["schema"] == OUTPUT_MASTER_CLOCK_RECEIPT_SCHEMA
    assert clock["passed"] is True
    assert clock["ps_phase_may_start"] is False
    assert clock["plant_identification_eligible"] is False
    assert clock["canonical_training_eligible"] is False
    assert clock["raw_artifact"]["sha256"] == result["raw_publication"]["sha256"]


def test_success_raw_is_no_replace(tmp_path: Path) -> None:
    session = "results/stage2_2khz_output_master_diagnostic/session_b"
    backend = FakeSplitBackend()
    result = _run(tmp_path, backend, session)
    plan, full = build_stage2_v2_live_safe_fallback_plan()
    raw_path = tmp_path / result["raw_publication"]["path"]
    before = raw_path.read_bytes()
    with np.load(io.BytesIO(before), allow_pickle=False) as archive:
        captured = np.asarray(archive["captured_pcm"]).copy()
        metadata = json.loads(str(archive["metadata_json"].item()))
        scalar = metadata["telemetry_scalar"]
        telemetry = {
            **scalar,
            **{
                name.removeprefix("telemetry_"): np.asarray(archive[name]).copy()
                for name in archive.files
                if name.startswith("telemetry_")
            },
            "actual_submitted_pcm": np.asarray(archive["submitted_pcm"]).copy(),
        }
    with pytest.raises(FileExistsError):
        publish_output_master_raw_no_replace(
            str(tmp_path),
            session,
            plan,
            full,
            captured_pcm=captured,
            telemetry=telemetry,
            capture_metadata={"capture_id": "b" * 32},
        )
    assert raw_path.read_bytes() == before


def test_transport_failure_preserves_partial_raw_and_forbids_retry(
    tmp_path: Path,
) -> None:
    session = "results/stage2_2khz_output_master_diagnostic/session_c"
    with pytest.raises(OutputMasterDiagnosticCaptureError) as caught:
        _run(tmp_path, FakeSplitBackend(bad_output_callback=1), session)

    publication = caught.value.partial_publication
    assert publication["path"] == output_master_session_targets(session)["partial_raw"]
    with np.load(tmp_path / publication["path"], allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata["schema"] == OUTPUT_MASTER_PARTIAL_RAW_SCHEMA
        assert metadata["partial_capture_never_promotable"] is True
        assert metadata["automatic_retry_allowed"] is False
        assert metadata["ps_phase_may_start"] is False
        assert metadata["canonical_training_eligible"] is False
        assert np.count_nonzero(archive["telemetry_submitted_valid_mask"]) == 256
        assert np.count_nonzero(archive["actual_submitted_pcm"][256:]) == 0
    assert not (
        tmp_path / output_master_session_targets(session)["clock_receipt"]
    ).exists()


def test_cli_dry_run_never_imports_sounddevice(monkeypatch, capsys) -> None:
    path = ROOT / "scripts/data/capture_stage2_output_master_diagnostic.py"
    spec = importlib.util.spec_from_file_location(
        "stage2_output_master_diagnostic_cli_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    imports: list[str] = []

    def forbidden_import(name: str):
        imports.append(name)
        raise AssertionError("dry-run은 backend를 import하면 안 됩니다")

    monkeypatch.setattr(module.importlib, "import_module", forbidden_import)
    assert module.main(["--dry-run"]) == 0
    stdout = capsys.readouterr().out
    assert "output_stream=11.605333s" in stdout
    assert "nonzero_output=5.537417s frames=265796 peak_pcm=98" in stdout
    assert "sounddevice import/open=0" in stdout
    assert imports == []
