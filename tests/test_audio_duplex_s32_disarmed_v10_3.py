from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from deep_anc.audio_duplex_s32_disarmed_v10_3 import (
    S32DisarmedDuplexCaptureFailure,
    TELEMETRY_SCHEMA,
    capture_disarmed_planned_s32_duplex,
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


class TamperS32(np.ndarray):
    def __setitem__(self, key, value):  # noqa: ANN001
        super().__setitem__(key, value)
        if np.any(np.asarray(value)):
            raw = self.view(np.ndarray).reshape(-1)
            raw[0] = np.int32(raw[0] + 1)


class ThreadedBackend:
    """start() 내부의 pre-arm callback과 arm 후 callback을 분리한 backend fixture."""

    CallbackStop = Stop
    CallbackAbort = Abort

    def __init__(self, **values):  # noqa: ANN003
        self.__dict__.update(values)
        self.calls: list[object] = []
        self.outputs: list[np.ndarray] = []
        self.kwargs: dict[str, object] = {}
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

    def _emit(self, callback, index: int, *, prearm: bool) -> bool:  # noqa: ANN001
        frames = int(getattr(self, "frames", 256))
        source = np.full((frames, 2), index + 1, dtype=getattr(self, "in_dtype", "<i4"))
        sink = np.full(
            getattr(self, "out_shape", (frames, 2)),
            77,
            dtype=getattr(self, "out_dtype", "<i4"),
        )
        if getattr(self, "tamper_assignment", False) and not prearm:
            sink = sink.view(TamperS32)
        self.outputs.append(sink)
        stamp = float(index)
        time_info = {
            "inputBufferAdcTime": stamp + 0.1,
            "outputBufferDacTime": stamp + 0.2,
            "currentTime": stamp + 0.3,
        }
        status = getattr(self, "prearm_status", None) if prearm else getattr(self, "planned_status", None)
        try:
            callback(source, sink, frames, time_info, status)
        except (Stop, Abort):
            return False
        return True

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
                # This call is synchronous inside Stream.start, so it is a direct
                # regression fixture for "on_stream_started is too late".
                if not outer._emit(kwargs["callback"], 0, prearm=True):
                    return

                def worker() -> None:
                    # Supervisor performs post_start_pre_arm_check immediately after
                    # start returns.  The small delay models the next hardware period.
                    time.sleep(0.01)
                    for index in range(1, 1 + getattr(outer, "planned_blocks", 4)):
                        if outer.stop_event.is_set():
                            return
                        if not outer._emit(kwargs["callback"], index, prearm=False):
                            return

                outer.worker = threading.Thread(target=worker, daemon=True)
                outer.worker.start()

            def stop(self, *, ignore_errors):
                outer.calls.append(("stop", ignore_errors))
                if outer.worker is not None:
                    outer.worker.join(timeout=1.0)
                if getattr(outer, "stop_error", False):
                    raise RuntimeError("stop failed")

            def abort(self, *, ignore_errors):
                outer.calls.append(("abort", ignore_errors))
                outer.stop_event.set()
                if outer.worker is not None:
                    outer.worker.join(timeout=1.0)
                if getattr(outer, "abort_error", False):
                    raise RuntimeError("abort failed")

            def close(self, *, ignore_errors):
                outer.calls.append(("close", ignore_errors))
                if getattr(outer, "close_error", False):
                    raise RuntimeError("close failed")

        return Stream()


def _planned() -> np.ndarray:
    return np.arange(1024 * 2, dtype="<i4").reshape(1024, 2) - 1000


def _run(backend: ThreadedBackend, **kwargs):  # noqa: ANN003, ANN201
    planned = kwargs.pop("planned_pcm", _planned())
    return capture_disarmed_planned_s32_duplex(
        backend,
        planned_pcm=planned,
        input_device=5,
        output_device=4,
        post_start_pre_arm_check=kwargs.pop("post_start_pre_arm_check", lambda: None),
        **kwargs,
    )


def test_prearm_callback_is_zero_before_hw_check_then_plan_is_exact_after_arm() -> None:
    backend = ThreadedBackend()
    seen: list[str] = []

    def arm_check() -> None:
        seen.append("check")
        assert backend.outputs and np.count_nonzero(backend.outputs[0]) == 0
        assert len(backend.outputs) == 1

    captured, telemetry = _run(backend, post_start_pre_arm_check=arm_check)
    expected = _planned()
    assert seen == ["check"]
    assert np.count_nonzero(backend.outputs[0]) == 0
    assert all(
        np.array_equal(block, expected[(index - 1) * 256 : index * 256])
        for index, block in enumerate(backend.outputs[1:], start=1)
    )
    assert captured.dtype == np.dtype("<i4")
    assert np.array_equal(telemetry["actual_submitted_pcm"], expected)
    assert telemetry["schema"] == TELEMETRY_SCHEMA
    assert telemetry["post_start_pre_arm_check_required"] is True
    assert telemetry["post_start_pre_arm_check_passed"] is True
    assert telemetry["prearm_callback_count"] == 1
    assert telemetry["prearm_output_zero_observed"] is True
    assert telemetry["callback_planned_assignment_attempt_count"] == 4
    assert telemetry["actual_matches_planned_application_buffer"] is True
    assert telemetry["actual_submitted_pcm_hash_eligible"] is True
    assert telemetry["normal_stop_completed"] is True
    assert telemetry["physical_output_authority"] is False
    assert telemetry["electrical_output_authority"] is False


def test_prearm_check_failure_never_assigns_nonzero_pcm() -> None:
    backend = ThreadedBackend()
    with pytest.raises(S32DisarmedDuplexCaptureFailure, match="hw params") as caught:
        _run(
            backend,
            post_start_pre_arm_check=lambda: (_ for _ in ()).throw(RuntimeError("hw params missing")),
        )
    telemetry = caught.value.telemetry
    assert all(np.count_nonzero(output) == 0 for output in backend.outputs)
    assert telemetry["post_start_pre_arm_check_passed"] is False
    assert telemetry["callback_planned_assignment_attempt_count"] == 0
    assert telemetry["submitted_frames"] == 0
    assert telemetry["actual_submitted_pcm_hash_eligible"] is False
    assert ("abort", False) in backend.calls
    assert ("close", False) in backend.calls


@pytest.mark.parametrize("status", (Overflow(), Unexpected()))
def test_invalid_prearm_status_is_zero_and_blocks_arm(status: object) -> None:
    backend = ThreadedBackend(prearm_status=status)
    with pytest.raises(S32DisarmedDuplexCaptureFailure, match="canonical-invalid") as caught:
        _run(backend)
    telemetry = caught.value.telemetry
    assert np.count_nonzero(backend.outputs[0]) == 0
    assert telemetry["post_start_pre_arm_check_passed"] is False
    assert telemetry["callback_planned_assignment_attempt_count"] == 0
    assert telemetry["submitted_frames"] == 0
    assert telemetry["xrun_count"] == (1 if isinstance(status, Overflow) else 0)


def test_invalid_planned_status_rezeros_and_drops_hash_authority() -> None:
    backend = ThreadedBackend(planned_status=Overflow())
    with pytest.raises(S32DisarmedDuplexCaptureFailure, match="canonical-invalid") as caught:
        _run(backend)
    telemetry = caught.value.telemetry
    assert np.count_nonzero(backend.outputs[0]) == 0
    assert np.count_nonzero(backend.outputs[1]) == 0
    assert telemetry["post_start_pre_arm_check_passed"] is True
    assert telemetry["callback_planned_assignment_attempt_count"] == 0
    assert telemetry["submitted_frames"] == 0
    assert telemetry["actual_submitted_pcm_hash_eligible"] is False
    assert int(telemetry["planned_callback_status_bitmask"][0]) & STATUS_INPUT_OVERFLOW


def test_assignment_mismatch_rezeros_and_marks_state_uncertain() -> None:
    backend = ThreadedBackend(tamper_assignment=True)
    with pytest.raises(S32DisarmedDuplexCaptureFailure, match="assignment") as caught:
        _run(backend)
    telemetry = caught.value.telemetry
    assert np.count_nonzero(backend.outputs[0]) == 0
    assert np.count_nonzero(backend.outputs[1]) == 0
    assert telemetry["callback_planned_assignment_attempt_count"] == 1
    assert telemetry["callback_planned_assignment_confirmed_count"] == 0
    assert telemetry["possible_nonzero_output_after_failed_assignment"] is True
    assert telemetry["submitted_frames"] == 0


def test_invalid_arguments_never_construct_stream() -> None:
    backend = ThreadedBackend()
    with pytest.raises(TypeError, match="mandatory"):
        capture_disarmed_planned_s32_duplex(
            backend,
            planned_pcm=_planned(),
            input_device=5,
            output_device=4,
            post_start_pre_arm_check=None,  # type: ignore[arg-type]
        )
    assert backend.calls == []
    with pytest.raises(ValueError, match="256"):
        _run(backend, planned_pcm=np.zeros((255, 2), dtype="<i4"))
    assert backend.calls == []


def test_close_notice_runs_after_close_even_when_close_fails() -> None:
    backend = ThreadedBackend(close_error=True)
    notices: list[bool] = []
    with pytest.raises(S32DisarmedDuplexCaptureFailure, match="close failed"):
        _run(backend, on_output_closed=lambda closed: notices.append(closed))
    assert notices == [False]
    assert backend.calls[-1] == ("close", False)


def test_disarmed_transport_source_imports_no_audio_backend() -> None:
    source = (REPO_ROOT / "src/deep_anc/audio_duplex_s32_disarmed_v10_3.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert "sounddevice" not in imported
    assert "subprocess" not in imported
