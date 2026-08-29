from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from deep_anc.audio_duplex_s32_v10 import (
    S32DuplexCaptureFailure,
    TELEMETRY_SCHEMA,
    capture_planned_s32_duplex,
)
from deep_anc.audio_duplex_v5 import STATUS_INPUT_OVERFLOW


REPO_ROOT = Path(__file__).resolve().parents[1]


class Stop(Exception):
    pass


class Abort(Exception):
    pass


class Overflow:
    input_overflow = True

    def __bool__(self) -> bool:
        return True


class Unexpected:
    def __bool__(self) -> bool:
        return True


class Priming:
    priming_output = True

    def __bool__(self) -> bool:
        return True


class TamperS32(np.ndarray):
    """nonzero block write만 1 LSB 변조하는 fake output buffer."""

    def __setitem__(self, key, value):  # noqa: ANN001
        super().__setitem__(key, value)
        if np.any(np.asarray(value)):
            raw = self.view(np.ndarray).reshape(-1)
            raw[0] = np.int32(raw[0] + 1)


class Backend:
    CallbackStop = Stop
    CallbackAbort = Abort

    def __init__(self, **values):  # noqa: ANN003
        self.__dict__.update(values)
        self.calls: list[object] = []
        self.outputs: list[np.ndarray] = []
        self.kwargs: dict[str, object] = {}

    def Stream(self, **kwargs):  # noqa: ANN003, ANN202
        self.kwargs = kwargs
        self.calls.append("stream_constructor")
        if getattr(self, "constructor_error", False):
            raise RuntimeError("constructor failed")
        outer = self

        class Stream:
            def start(self):
                outer.calls.append("start")
                if getattr(outer, "start_error", False):
                    raise RuntimeError("start failed")
                for index in range(getattr(outer, "blocks", 4)):
                    frames = getattr(outer, "frames", 256)
                    source = np.full((frames, 2), index + 1, dtype=getattr(outer, "in_dtype", "<i4"))
                    sink = np.full(getattr(outer, "out_shape", (frames, 2)), 77, dtype=getattr(outer, "out_dtype", "<i4"))
                    if getattr(outer, "tamper_assignment", False):
                        sink = sink.view(TamperS32)
                    outer.outputs.append(sink)
                    time = {
                        "inputBufferAdcTime": index + 0.1,
                        "outputBufferDacTime": index + 0.2,
                        "currentTime": index + 0.3,
                    }
                    status = getattr(outer, "forced_status", None)
                    if status is None and getattr(outer, "xrun", False) and index == 1:
                        status = Overflow()
                    try:
                        kwargs["callback"](source, sink, frames, time, status)
                    except (Stop, Abort):
                        break

            def stop(self, *, ignore_errors):
                outer.calls.append(("stop", ignore_errors))
                if getattr(outer, "stop_error", False):
                    raise RuntimeError("stop failed")

            def abort(self, *, ignore_errors):
                outer.calls.append(("abort", ignore_errors))
                if getattr(outer, "abort_error", False):
                    raise RuntimeError("abort failed")

            def close(self, *, ignore_errors):
                outer.calls.append(("close", ignore_errors))
                if getattr(outer, "close_error", False):
                    raise RuntimeError("close failed")

        return Stream()


def _planned() -> np.ndarray:
    return np.arange(1024 * 2, dtype="<i4").reshape(1024, 2) - 1000


def _run(backend: Backend, **kwargs):  # noqa: ANN003, ANN201
    return capture_planned_s32_duplex(
        backend,
        planned_pcm=_planned(),
        input_device=5,
        output_device=4,
        **kwargs,
    )


def test_normal_callback_zeroes_then_assigns_exact_planned_s32() -> None:
    backend = Backend()
    captured, telemetry = _run(backend)
    expected = _planned()
    assert backend.kwargs["dtype"] == ("int32", "int32")
    assert backend.kwargs["device"] == (5, 4)
    assert backend.kwargs["dither_off"] is True
    assert backend.kwargs["prime_output_buffers_using_stream_callback"] is False
    assert all(np.array_equal(block, expected[index * 256 : (index + 1) * 256]) for index, block in enumerate(backend.outputs))
    assert captured.dtype == np.dtype("<i4")
    assert np.array_equal(telemetry["actual_submitted_pcm"], expected)
    assert telemetry["schema"] == TELEMETRY_SCHEMA
    assert telemetry["authority"] == "application_buffer_planned_s32_only_no_physical_sample_identity"
    assert telemetry["hardware_sample_slip_authority"] is False
    assert telemetry["physical_output_authority"] is False
    assert telemetry["electrical_output_authority"] is False
    assert telemetry["acoustic_output_authority"] is False
    assert telemetry["callback_zero_attempt_count"] == 4
    assert telemetry["callback_zero_confirmed_count"] == 4
    assert telemetry["callback_planned_assignment_attempt_count"] == 4
    assert telemetry["callback_planned_assignment_confirmed_count"] == 4
    assert telemetry["actual_matches_planned_application_buffer"] is True
    assert telemetry["actual_submitted_pcm_hash_eligible"] is True
    assert telemetry["normal_stop_completed"] is True
    assert telemetry["xrun_count"] == 0


@pytest.mark.parametrize(
    "name,value",
    (
        ("planned_pcm", np.zeros((255, 2), dtype="<i4")),
        ("planned_pcm", np.zeros((256, 2), dtype="<i2")),
        ("planned_pcm", np.zeros((256,), dtype="<i4")),
        ("input_device", True),
        ("input_device", -1),
        ("output_device", "4"),
    ),
)
def test_invalid_contract_never_constructs_stream(name: str, value: object) -> None:
    backend = Backend()
    arguments: dict[str, object] = {
        "planned_pcm": _planned(),
        "input_device": 5,
        "output_device": 4,
    }
    arguments[name] = value
    with pytest.raises((TypeError, ValueError)):
        capture_planned_s32_duplex(backend, **arguments)  # type: ignore[arg-type]
    assert backend.calls == []
    assert backend.outputs == []


def test_preopen_failure_never_constructs_stream() -> None:
    backend = Backend()
    ticks = iter((1.0, 2.0, 3.0))
    with pytest.raises(S32DuplexCaptureFailure, match="barrier") as caught:
        _run(
            backend,
            pre_open_check=lambda: (_ for _ in ()).throw(RuntimeError("barrier missing")),
            monotonic=lambda: next(ticks),
        )
    assert backend.calls == []
    assert caught.value.telemetry["submitted_frames"] == 0


def test_bad_frame_or_output_dtype_is_zeroed_before_abort() -> None:
    for backend, match in (
        (Backend(frames=128), "exact 256"),
        (Backend(out_dtype="<i2"), "output은 exact"),
        (Backend(in_dtype="<i2"), "input은 exact"),
    ):
        with pytest.raises(S32DuplexCaptureFailure, match=match) as caught:
            _run(backend)
        assert np.count_nonzero(backend.outputs[0]) == 0
        assert caught.value.telemetry["submitted_frames"] == 0
        assert caught.value.telemetry["callback_zero_attempt_count"] >= 1
        assert ("abort", False) in backend.calls
        assert ("close", False) in backend.calls


@pytest.mark.parametrize("status", (Overflow(), Unexpected(), Priming()))
def test_invalid_callback_status_is_zeroed_before_any_planned_assignment(status: object) -> None:
    backend = Backend(forced_status=status)
    with pytest.raises(S32DuplexCaptureFailure, match="canonical-invalid") as caught:
        _run(backend)
    telemetry = caught.value.telemetry
    assert np.count_nonzero(backend.outputs[0]) == 0
    assert telemetry["callback_planned_assignment_attempt_count"] == 0
    assert telemetry["submitted_frames"] == 0
    assert telemetry["actual_submitted_pcm_hash_eligible"] is False


def test_xrun_fails_closed_without_pcm_hash_authority() -> None:
    backend = Backend(xrun=True)
    with pytest.raises(S32DuplexCaptureFailure, match="canonical-invalid") as caught:
        _run(backend)
    telemetry = caught.value.telemetry
    assert telemetry["xrun_count"] == 1
    assert int(telemetry["callback_status_bitmask"][1]) & STATUS_INPUT_OVERFLOW
    assert telemetry["actual_submitted_pcm_hash_eligible"] is False
    assert telemetry["actual_submitted_pcm_sha256"] is None
    assert np.array_equal(caught.value.actual_submitted_pcm[:256], _planned()[:256])
    assert not np.any(caught.value.actual_submitted_pcm[256:])
    assert np.count_nonzero(backend.outputs[1]) == 0
    assert telemetry["callback_planned_assignment_attempt_count"] == 1
    assert telemetry["on_stream_started_is_observational_not_nonzero_admission"] is True


def test_partial_capture_cannot_authorize_planned_pcm_hash() -> None:
    backend = Backend(blocks=1)
    ticks = iter((0.0, 0.0, 0.0, 10.0, 11.0))
    with pytest.raises(S32DuplexCaptureFailure, match="watchdog") as caught:
        _run(backend, watchdog_grace_seconds=0.1, monotonic=lambda: next(ticks), sleep=lambda _s: None)
    telemetry = caught.value.telemetry
    assert telemetry["captured_frames"] == 256
    assert telemetry["submitted_frames"] == 256
    assert telemetry["full_frame_accounting_valid"] is False
    assert telemetry["actual_submitted_pcm_hash_eligible"] is False
    assert not np.any(caught.value.submitted_valid_mask[256:])


def test_owned_plan_is_immune_to_caller_mutation_during_preopen() -> None:
    backend = Backend()
    planned = _planned()
    original = planned.copy()
    _captured, telemetry = capture_planned_s32_duplex(
        backend,
        planned_pcm=planned,
        input_device=5,
        output_device=4,
        pre_open_check=lambda: planned.__setitem__(slice(None), 0),
    )
    assert np.array_equal(planned, np.zeros_like(planned))
    assert np.array_equal(telemetry["actual_submitted_pcm"], original)


def test_assignment_mismatch_rezeros_and_marks_output_state_uncertain() -> None:
    backend = Backend(tamper_assignment=True)
    with pytest.raises(S32DuplexCaptureFailure, match="assignment") as caught:
        _run(backend)
    telemetry = caught.value.telemetry
    assert np.count_nonzero(backend.outputs[0]) == 0
    assert telemetry["callback_planned_assignment_attempt_count"] == 1
    assert telemetry["callback_planned_assignment_confirmed_count"] == 0
    assert telemetry["possible_nonzero_output_after_failed_assignment"] is True
    assert telemetry["submitted_frames"] == 0


def test_close_notice_is_called_after_close_and_faults_are_preserved() -> None:
    backend = Backend(stop_error=True, abort_error=True, close_error=True)
    notices: list[bool] = []
    with pytest.raises(S32DuplexCaptureFailure) as caught:
        _run(backend, on_output_closed=lambda closed: notices.append(closed))
    assert notices == [False]
    assert all(item in str(caught.value) for item in ("stop failed", "abort failed", "close failed"))
    assert backend.calls[-1] == ("close", False)


def test_v10_transport_source_imports_no_audio_backend() -> None:
    source = (REPO_ROOT / "src/deep_anc/audio_duplex_s32_v10.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert "sounddevice" not in imported
    assert "subprocess" not in imported
