from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import deep_anc.dsp.stage2_2khz_rt5640_s32_capture as capture_adapter
from deep_anc.dsp.stage2_2khz_rt5640_s32_capture import (
    BLOCKED_STATUS,
    CAPTURE_SCAFFOLD_SCHEMA,
    RAW_SCHEMA,
    Stage2Rt5640S32CaptureBlocked,
    assert_stage2_rt5640_s32_live_capture_blocked,
    build_stage2_rt5640_s32_capture_dry_run_receipt,
    execute_stage2_rt5640_s32_disarmed_capture,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/jetson/capture_stage2_2khz_rt5640_s32.py"
MODULE = REPO_ROOT / "src/deep_anc/dsp/stage2_2khz_rt5640_s32_capture.py"


class FakeBackend:
    """live gate가 stream constructor보다 먼저 막는지 보는 no-audio backend."""

    def __init__(self) -> None:
        self.stream_calls = 0

    def Stream(self, **_kwargs):  # noqa: ANN003, N802
        self.stream_calls += 1
        raise AssertionError("current signal-only plan이 fake Stream까지 도달했습니다")


def test_dry_run_receipt_is_same_card_s32_only_and_fail_closed() -> None:
    receipt = build_stage2_rt5640_s32_capture_dry_run_receipt()

    assert receipt["schema"] == CAPTURE_SCAFFOLD_SCHEMA
    assert receipt["status"] == BLOCKED_STATUS
    assert receipt["dry_run"] is True
    assert receipt["audio_backend_imported"] is False
    assert receipt["alsa_or_pcm_opened"] is False
    assert receipt["speaker_output"] is False
    assert receipt["raw_written"] is False
    assert receipt["live_capture_may_open"] is False
    assert receipt["hardware_audio"]["input"] == {
        "card": "APE",
        "pcm": 1,
        "channels": 2,
        "format": "S32_LE",
        "route": "I2S2_ADMAIF2_ERR_REF",
    }
    assert receipt["hardware_audio"]["output"] == {
        "card": "APE",
        "pcm": 0,
        "channels": 2,
        "format": "S32_LE",
        "route": "ADMAIF1_I2S1_RT5640_J511",
    }
    assert receipt["hardware_audio"]["sample_rate_hz"] == 48_000
    assert receipt["hardware_audio"]["block_size"] == 256
    assert receipt["planned_signal"]["planned_s32_dtype"] == "<i4"
    assert receipt["planned_signal"]["low_16_bits_must_be_zero"] is True
    assert receipt["planned_signal"]["source_audio_execution_allowed"] is False
    assert receipt["planned_transport_provenance"]["same_card"] is True
    assert receipt["planned_transport_provenance"]["same_clock_domain"] == "APE_PLL_A_SHARED"
    assert receipt["planned_transport_provenance"]["native_format"] == "S32_LE"
    assert receipt["raw_first_publication_plan"]["schema"] == RAW_SCHEMA
    assert receipt["raw_first_publication_plan"]["dry_run_must_not_write"] is True
    assert receipt["disarmed_primitive"]["pre_arm_output_exact_zero_required"] is True
    assert receipt["disarmed_primitive"]["current_plan_can_reach_primitive"] is False
    assert receipt["authority"]["canonical_training_eligible"] is False
    assert len(receipt["canonical_payload_sha256"]) == 64


def test_current_signal_only_plan_blocks_before_backend_or_argument_validation() -> None:
    backend = FakeBackend()

    with pytest.raises(Stage2Rt5640S32CaptureBlocked, match=BLOCKED_STATUS):
        execute_stage2_rt5640_s32_disarmed_capture(
            backend,
            input_device="not-an-int",  # type: ignore[arg-type]
            output_device="not-an-int",  # type: ignore[arg-type]
            post_start_pre_arm_check=None,  # type: ignore[arg-type]
        )
    assert backend.stream_calls == 0

    with pytest.raises(Stage2Rt5640S32CaptureBlocked, match=BLOCKED_STATUS):
        assert_stage2_rt5640_s32_live_capture_blocked()


def test_current_signal_only_plan_never_invokes_disarmed_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def forbidden_primitive(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append((args, kwargs))
        raise AssertionError("signal-only plan이 disarmed primitive에 도달했습니다")

    monkeypatch.setattr(capture_adapter, "capture_disarmed_planned_s32_duplex", forbidden_primitive)
    with pytest.raises(Stage2Rt5640S32CaptureBlocked, match=BLOCKED_STATUS):
        capture_adapter.execute_stage2_rt5640_s32_disarmed_capture(
            FakeBackend(),
            input_device=1,
            output_device=0,
            post_start_pre_arm_check=lambda: None,
        )
    assert calls == []


def test_dry_run_receipt_writes_no_cwd_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    build_stage2_rt5640_s32_capture_dry_run_receipt()
    assert list(tmp_path.iterdir()) == []


def test_capture_module_does_not_import_or_open_an_audio_backend() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert "sounddevice" not in imported
    assert "subprocess" not in imported
    assert "os" not in imported
    assert "np.savez" not in source
    assert ".write(" not in source


def test_default_cli_is_no_audio_and_direct_import_uses_checkout_src() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "[PASS]" in completed.stdout
    assert "APE PCM1/S32_LE" in completed.stdout
    assert "APE PCM0/S32_LE" in completed.stdout
    assert "audio_backend_import=0; ALSA_PCM_open=0; speaker_output=0; raw_write=0" in completed.stdout


def test_execute_live_cli_is_blocked_before_audio() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--execute-live"],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "[BLOCKED_BEFORE_AUDIO]" in completed.stderr
    assert BLOCKED_STATUS in completed.stderr
    assert "audio_backend_import=0; ALSA_PCM_open=0; speaker_output=0; raw_write=0" in completed.stderr
