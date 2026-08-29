"""저레벨 출력-마이크 채널 경로 진단의 안전 경계와 분석 테스트."""

import builtins
import json

import numpy as np
import pytest

from scripts.bench import measure_channel_paths as paths


def _hardware() -> dict:
    return {
        "audio": {
            "sample_rate": 48_000,
            "block_size": 256,
            "latency": "low",
            "input": {
                "card": "APE",
                "pcm": 1,
                "channels": 2,
                "dtype": "int32",
            },
            "output": {
                "card": "Audio",
                "pcm": 0,
                "channels": 2,
                "dtype": "int16",
            },
        },
        "channels": {
            "error_mic": 0,
            "reference_mic": 1,
            "noise_out": 0,
            "cancel_out": 1,
        },
    }


def test_program_drives_exactly_one_channel_with_silent_boundaries():
    output, bounds = paths.build_output_program(
        sample_rate=48_000,
        frequency=300.0,
        amplitude=0.005,
        output_channel=0,
        pre_seconds=0.5,
        tone_seconds=1.0,
        post_seconds=0.5,
    )

    assert output.dtype == np.int16
    assert np.all(output[bounds["pre"][0] : bounds["pre"][1]] == 0)
    assert np.all(output[bounds["post"][0] : bounds["post"][1]] == 0)
    assert np.all(output[:, 1] == 0)
    assert np.any(output[bounds["tone"][0] : bounds["tone"][1], 0] != 0)
    assert np.max(np.abs(output[:, 0].astype(np.int32))) <= round(0.005 * 32767)


def test_amplitude_has_hard_safety_ceiling():
    with pytest.raises(ValueError, match="0.020"):
        paths.validate_probe_settings(
            sample_rate=48_000,
            frequency=300.0,
            amplitude=0.02001,
            block_size=512,
            pre_seconds=1.0,
            tone_seconds=2.0,
            post_seconds=1.0,
        )


def test_analysis_detects_tone_coupling_without_calling_it_performance():
    fs = 8_000
    pre = fs
    tone = 2 * fs
    post = fs
    total = pre + tone + post
    rng = np.random.default_rng(12)
    floating = rng.normal(0.0, 2.0e-5, size=(total, 2))
    t = np.arange(tone, dtype=np.float64) / fs
    floating[pre : pre + tone, 0] += 0.01 * np.sin(2 * np.pi * 300.0 * t)
    raw = np.rint(np.clip(floating, -1.0, 1.0) * (2**31)).astype(np.int32)
    bounds = {"pre": (0, pre), "tone": (pre, pre + tone), "post": (pre + tone, total)}

    report = paths.analyze_path_capture(
        raw,
        bounds=bounds,
        frequency=300.0,
        sample_rate=fs,
        status_report={"status_blocks": 0},
    )

    assert report[0]["coupling_detected"] is True
    assert report[0]["tone_bin_change_db"] > 30.0
    assert report[0]["tone_steady"]["clip_ratio"] == 0.0
    assert report[1]["coupling_detected"] is False


def test_callback_status_invalidates_coupling_verdict():
    fs = 8_000
    bounds = {"pre": (0, fs), "tone": (fs, 3 * fs), "post": (3 * fs, 4 * fs)}
    raw = np.zeros((4 * fs, 2), dtype=np.int32)
    t = np.arange(2 * fs) / fs
    raw[fs : 3 * fs, 0] = np.rint(
        0.01 * np.sin(2 * np.pi * 300.0 * t) * (2**31)
    ).astype(np.int32)

    report = paths.analyze_path_capture(
        raw,
        bounds=bounds,
        frequency=300.0,
        sample_rate=fs,
        status_report={"status_blocks": 1, "xrun_blocks": 1},
    )
    assert report[0]["tone_bin_change_db"] > 50.0
    assert report[0]["coupling_detected"] is False


def test_finally_flushes_both_outputs_when_second_probe_fails(monkeypatch):
    calls = []

    def fake_run(_sd, **kwargs):
        calls.append("run")
        if len(calls) == 2:
            raise RuntimeError("simulated callback failure")
        frames = kwargs["output_pcm"].shape[0]
        return np.zeros((frames, 2), dtype=np.int32), {}, 0.1

    def fake_flush(_sd, **_kwargs):
        calls.append("flush")
        return {"both_channels_zero": True}

    monkeypatch.setattr(paths, "run_output_probe", fake_run)
    monkeypatch.setattr(paths, "flush_output_silence", fake_flush)
    program = np.zeros((1024, 2), dtype=np.int16)

    with pytest.raises(RuntimeError, match="simulated"):
        paths.collect_channel_paths(
            object(),
            input_device=1,
            output_device=2,
            sample_rate=48_000,
            block_size=512,
            latency="high",
            programs=[("noise_out", 0, program), ("cancel_out", 1, program)],
        )
    assert calls == ["run", "run", "flush"]


def test_cli_requires_user_presence_and_minimum_volume():
    assert paths.main([]) == 2


def test_dry_run_requires_no_confirmations_and_never_imports_or_opens_audio(
    tmp_path, monkeypatch, capsys
):
    """무음 계획 경로는 YAML/PCM만 다루며 audio primitive에 닿지 않는다."""
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "load_yaml", lambda _path: _hardware())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run에서 live audio primitive를 호출하면 안 됩니다")

    monkeypatch.setattr(paths, "assert_live_pcm_clock_preconditions", forbidden)
    monkeypatch.setattr(paths, "assert_measurement_preconditions", forbidden)
    monkeypatch.setattr(paths, "capture_input_probe", forbidden)
    monkeypatch.setattr(paths, "resolve_alsa_portaudio_device", forbidden)
    monkeypatch.setattr(paths, "collect_channel_paths", forbidden)
    monkeypatch.setattr(paths, "_save_results", forbidden)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("dry-run에서 sounddevice import를 하면 안 됩니다")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = paths.main(
        [
            "--dry-run",
            "--frequency",
            "2000",
            "--out-prefix",
            "results/channel_paths/dry_run_only",
        ]
    )

    assert result == 0
    stdout = capsys.readouterr().out
    assert stdout.startswith("[DRY-RUN PASS]")
    report = json.loads(stdout.partition("\n")[2])
    assert report["frequency_hz"] == pytest.approx(2000.0)
    assert report["confirmation_policy"]["required_for_dry_run"] is False
    assert report["dry_run_guarantees"] == {
        "alsa_or_portaudio_device_opened": False,
        "artifact_written": False,
        "raw_microphone_capture_created": False,
        "sounddevice_imported": False,
        "speaker_output": False,
    }
    assert report["duration_plan"]["input_preflight_seconds"] == pytest.approx(2.0)
    assert report["duration_plan"]["audible_seconds"] == pytest.approx(4.0)
    assert report["duration_plan"]["flush_stream_seconds"] == pytest.approx(0.256)
    assert report["duration_plan"]["total_expected_device_occupancy_seconds"] == pytest.approx(
        10.256
    )
    assert report["duration_plan"]["output_streams"] == [
        {
            "name": "noise_out",
            "output_channel": 0,
            "frames": 192_000,
            "seconds": 4.0,
            "pre_silence_frames": 48_000,
            "tone_frames": 96_000,
            "post_silence_frames": 48_000,
            "pre_silence_seconds": 1.0,
            "audible_tone_seconds": 2.0,
            "post_silence_seconds": 1.0,
        },
        {
            "name": "cancel_out",
            "output_channel": 1,
            "frames": 192_000,
            "seconds": 4.0,
            "pre_silence_frames": 48_000,
            "tone_frames": 96_000,
            "post_silence_frames": 48_000,
            "pre_silence_seconds": 1.0,
            "audible_tone_seconds": 2.0,
            "post_silence_seconds": 1.0,
        },
    ]
    assert not (tmp_path / "results/channel_paths/dry_run_only.npz").exists()
    assert not (tmp_path / "results/channel_paths/dry_run_only.json").exists()


def test_dry_run_checks_no_replace_paths_without_touching_audio(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "load_yaml", lambda _path: _hardware())
    existing = tmp_path / "results" / "channel_paths" / "already_used.npz"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"immutable existing measurement")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run no-replace 검사에서 audio primitive를 호출하면 안 됩니다")

    monkeypatch.setattr(paths, "assert_live_pcm_clock_preconditions", forbidden)
    monkeypatch.setattr(paths, "capture_input_probe", forbidden)
    monkeypatch.setattr(paths, "resolve_alsa_portaudio_device", forbidden)

    assert (
        paths.main(
            [
                "--dry-run",
                "--out-prefix",
                "results/channel_paths/already_used",
            ]
        )
        == 2
    )
    assert "덮어쓰지 않습니다" in capsys.readouterr().err
    assert existing.read_bytes() == b"immutable existing measurement"
    assert not (tmp_path / "results/channel_paths/already_used.json").exists()


def _forbid_audio_primitives(monkeypatch):
    """경로 거부가 실제 audio import/open보다 앞서는지 검사하는 공용 sentinel."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("진단 output 경로 거부 전에 audio primitive를 호출하면 안 됩니다")

    for name in (
        "assert_live_pcm_clock_preconditions",
        "assert_measurement_preconditions",
        "capture_input_probe",
        "resolve_alsa_portaudio_device",
        "collect_channel_paths",
        "_save_results",
    ):
        monkeypatch.setattr(paths, name, forbidden)


@pytest.mark.parametrize(
    ("mode_args", "out_prefix"),
    [
        (["--dry-run"], "assets/measured/diagnostic_escape"),
        (
            ["--confirm-user-present-volume-minimum", "--confirm-speaker"],
            "results/channel_paths/../../assets/measured/diagnostic_escape",
        ),
    ],
)
def test_output_prefix_escape_is_rejected_before_any_audio_access(
    tmp_path, monkeypatch, capsys, mode_args, out_prefix
):
    """dry-run/live 모두 realpath가 diagnostic root 밖이면 장치를 건드리지 않는다."""

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "load_yaml", lambda _path: _hardware())
    _forbid_audio_primitives(monkeypatch)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("경로 거부 전에 sounddevice import를 하면 안 됩니다")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert paths.main([*mode_args, "--out-prefix", out_prefix]) == 2
    assert "results/channel_paths" in capsys.readouterr().err
    assert not list(tmp_path.rglob("*.npz"))
    assert not list(tmp_path.rglob("*.json"))


def test_dry_run_rejects_symlinked_output_prefix_escape_before_audio_access(
    tmp_path, monkeypatch, capsys
):
    """repo 안의 symlink라도 realpath가 diagnostic root 밖이면 허용하지 않는다."""

    root = tmp_path / "results" / "channel_paths"
    root.mkdir(parents=True)
    outside = tmp_path / "assets" / "measured"
    outside.mkdir(parents=True)
    (root / "escape").symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "load_yaml", lambda _path: _hardware())
    _forbid_audio_primitives(monkeypatch)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("symlink 경로 거부 전에 sounddevice import를 하면 안 됩니다")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert (
        paths.main(
            [
                "--dry-run",
                "--out-prefix",
                "results/channel_paths/escape/diagnostic",
            ]
        )
        == 2
    )
    assert "results/channel_paths" in capsys.readouterr().err
    assert not list(outside.iterdir())


def test_output_prefix_normalizes_inside_diagnostic_root_and_preserves_suffixes(
    tmp_path, monkeypatch
):
    """안전한 ``..`` 정규화와 기존 .npz/.json prefix 호환성을 유지한다."""

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    npz_path, json_path = paths._output_paths(
        "results/channel_paths/../channel_paths/nested/capture.npz"
    )

    expected = tmp_path / "results" / "channel_paths" / "nested" / "capture"
    assert npz_path == expected.with_suffix(".npz")
    assert json_path == expected.with_suffix(".json")

    default_npz, default_json = paths._output_paths(None)
    assert default_npz.parent == tmp_path / "results" / "channel_paths"
    assert default_npz.stem == default_json.stem
    assert default_npz.stem.startswith("channel_paths_")


@pytest.mark.parametrize("target_kind", ["outside_repo", "other_repo_directory"])
def test_output_root_symlink_escape_is_rejected(tmp_path, monkeypatch, target_kind):
    """root 자체가 외부 또는 다른 repo 역할로 symlink되면 기본 경로도 막는다."""

    results = tmp_path / "results"
    results.mkdir()
    if target_kind == "outside_repo":
        target = tmp_path.parent
    else:
        target = tmp_path / "assets" / "measured"
        target.mkdir(parents=True)
    (results / "channel_paths").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)

    with pytest.raises(ValueError, match="전용 diagnostic root"):
        paths._output_paths(None)
