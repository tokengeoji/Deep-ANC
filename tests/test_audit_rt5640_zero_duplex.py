from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path("scripts/jetson/audit_rt5640_zero_duplex.py")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "config": str(Path("configs/hardware_jetson_rt5640_zero_smoke.yaml").resolve()),
        "expected_commit": "c" * 40,
        "confirm_j511_disconnected": True,
        "confirm_amplifier_power_off": True,
        "confirm_amplifier_input_disconnected": True,
        "confirm_ab13x_amplifier_disconnected": True,
        "confirm_user_present": True,
        "execute_live": True,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _completed(command, *, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


_VOLATILE_NAMES = {
    1132: "Lane1 Ratio Int",
    1133: "Lane1 Ratio Frac",
    1134: "Lane2 Ratio Int",
    1135: "Lane2 Ratio Frac",
    1136: "Lane3 Ratio Int",
    1137: "Lane3 Ratio Frac",
    1138: "Lane4 Ratio Int",
    1139: "Lane4 Ratio Frac",
    1140: "Lane5 Ratio Int",
    1141: "Lane5 Ratio Frac",
    1142: "Lane6 Ratio Int",
    1143: "Lane6 Ratio Frac",
}


def _alsactl_fixture(*, volatile_base: int = 100, normal_value: int = 0) -> bytes:
    lines = ["state.APE {\n"]
    for offset, (numid, name) in enumerate(_VOLATILE_NAMES.items()):
        lines.extend(
            [
                f"\tcontrol.{numid} {{\n",
                "\t\tiface MIXER\n",
                f"\t\tname '{name}'\n",
                f"\t\tvalue {volatile_base + offset}\n",
                "\t\tcomment {\n",
                "\t\t\taccess 'read volatile'\n",
                "\t\t\ttype INTEGER\n",
                "\t\t\tcount 1\n",
                "\t\t\trange '0 - -1'\n",
                "\t\t}\n",
                "\t}\n",
            ]
        )
    lines.extend(
        [
            "\tcontrol.1200 {\n",
            "\t\tiface MIXER\n",
            "\t\tname 'Stable Control'\n",
            f"\t\tvalue {normal_value}\n",
            "\t\tcomment {\n",
            "\t\t\taccess 'read write'\n",
            "\t\t\ttype INTEGER\n",
            "\t\t\tcount 1\n",
            "\t\t}\n",
            "\t}\n",
            "}\n",
        ]
    )
    return "".join(lines).encode()


def _amixer_fixture(*, volatile_base: int = 100, normal_value: int = 0) -> bytes:
    lines = []
    for offset, (numid, name) in enumerate(_VOLATILE_NAMES.items()):
        lines.extend(
            [
                f"numid={numid},iface=MIXER,name='{name}'\n",
                "  ; type=INTEGER,access=r--v----,values=1,min=0,max=-1,step=0\n",
                f"  : values={volatile_base + offset}\n",
            ]
        )
    lines.extend(
        [
            "numid=1200,iface=MIXER,name='Stable Control'\n",
            "  ; type=INTEGER,access=rw------,values=1,min=0,max=10,step=0\n",
            f"  : values={normal_value}\n",
        ]
    )
    return "".join(lines).encode()


def test_default_mode_is_device_free_dry_run(monkeypatch, capsys) -> None:
    module = _load_script("rt5640_adapter_default_dry")
    calls: list[str] = []
    fake = {
        "config_binding": {"file_sha256": "1" * 64},
        "script": {"file_sha256": "2" * 64},
        "plan": {"canonical_payload_sha256": "3" * 64},
        "generation": module.REPO_ROOT / "results/rt5640_zero_duplex/v1",
    }
    monkeypatch.setattr(module, "_static_contract", lambda _args, *, live: calls.append(f"static:{live}") or fake)
    monkeypatch.setattr(
        module,
        "_sounddevice_import",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run backend import")),
    )
    assert module.main([]) == 0
    assert calls == ["static:False"]
    output = capsys.readouterr().out
    assert "60.000초" in output
    assert "audible signal: 0초" in output


@pytest.mark.parametrize("missing", [*[
    "confirm_j511_disconnected",
    "confirm_amplifier_power_off",
    "confirm_amplifier_input_disconnected",
    "confirm_ab13x_amplifier_disconnected",
    "confirm_user_present",
], "expected_commit"])
def test_live_requires_every_exact_confirmation_before_claim(missing: str) -> None:
    module = _load_script(f"rt5640_adapter_missing_{missing}")
    kwargs = {missing: False if missing != "expected_commit" else None}
    with pytest.raises(RuntimeError, match="확인 플래그|expected-commit"):
        module._require_live_flags(_args(**kwargs))


def test_current_worktree_binding_bootstraps_repo_src_without_pythonpath(tmp_path, monkeypatch) -> None:
    module = _load_script("rt5640_adapter_no_pythonpath")
    wrong = tmp_path / "src"
    wrong.mkdir()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(module.sys, "path", [str(wrong), *module.sys.path])

    binding = module._assert_current_worktree_binding()

    assert binding["expected_src"] == str(REPO_ROOT / "src")
    assert binding["pythonpath"] == ""
    assert module._resolve_pythonpath_entry(module.sys.path[0]) == REPO_ROOT / "src"


def test_current_worktree_binding_rejects_preloaded_foreign_package(tmp_path, monkeypatch) -> None:
    module = _load_script("rt5640_adapter_preloaded_foreign_package")
    foreign_package = tmp_path / "foreign_deep_anc"
    foreign_package.mkdir()
    foreign_init = foreign_package / "__init__.py"
    foreign_init.write_text("__version__ = 'foreign'\n", encoding="utf-8")
    fake_module = type(sys)("deep_anc")
    fake_module.__file__ = str(foreign_init)
    monkeypatch.setitem(module.sys.modules, "deep_anc", fake_module)

    with pytest.raises(RuntimeError, match="preload된 deep_anc import가 current worktree 밖"):
        module._assert_current_worktree_binding()


def test_cli_dry_run_from_repo_root_needs_no_pythonpath(monkeypatch) -> None:
    """실제 operator 명령은 source bootstrap 뒤 backend 없이 성공해야 한다."""

    python = REPO_ROOT / ".venv/bin/python"
    assert python.is_file(), "Jetson project venv가 필요합니다"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONPROFILEIMPORTTIME"] = "1"
    result = subprocess.run(
        [str(python), "-B", str(SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[DRY-RUN PASS]" in result.stdout
    assert "sounddevice import/장치 open/system mutation 없음" in result.stdout
    assert "sounddevice" not in result.stderr.lower(), result.stderr


def test_parse_hw_params_requires_s32_48k_two_channel_block_256() -> None:
    module = _load_script("rt5640_adapter_hw_params")
    text = """access: RW_INTERLEAVED
format: S32_LE
subformat: STD
channels: 2
rate: 48000 (48000/1)
period_size: 256
buffer_size: 512
"""
    parsed = module._parse_hw_params(text)
    assert parsed["format"] == "S32_LE"
    assert parsed["rate"] == 48_000
    assert parsed["channels"] == 2
    assert parsed["period_size"] == 256

    with pytest.raises(RuntimeError, match="format"):
        module._parse_hw_params(text.replace("S32_LE", "S16_LE"))
    with pytest.raises(RuntimeError, match="period_size"):
        module._parse_hw_params(text.replace("period_size: 256", "period_size: 128"))


def _seed_fake_alsa(tmp_path: Path) -> tuple[Path, Path]:
    proc = tmp_path / "proc_asound"
    dev = tmp_path / "dev_snd"
    (proc / "card1/pcm0p/sub0").mkdir(parents=True)
    (proc / "card1/pcm1c/sub0").mkdir(parents=True)
    (proc / "cards").write_text(
        " 1 [APE            ]: tegra-ape - NVIDIA Jetson AGX Orin APE\n",
        encoding="utf-8",
    )
    for endpoint in (proc / "card1/pcm0p/sub0", proc / "card1/pcm1c/sub0"):
        (endpoint / "status").write_text("closed\n", encoding="utf-8")
    dev.mkdir()
    (dev / "pcmC1D0p").touch()
    (dev / "pcmC1D1c").touch()
    return proc, dev


def _pulse_cards() -> str:
    return """Card #2
    Name: alsa_card.platform-sound
    Properties:
        alsa.card = "1"
    Active Profile: off
"""


def _route_cget(command: list[str]) -> str:
    control = command[-1].split("=", 1)[1]
    expected = {
        "I2S1 Mux": (1, "ADMAIF1"),
        "ADMAIF1 Mux": (17, "I2S1"),
        "ADMAIF2 Mux": (18, "I2S2"),
        "I2S2 Mux": (2, "ADMAIF2"),
    }
    number, item = expected[control]
    return f"; Item #{number} '{item}'\n: values={number}\n"


def test_strict_pcm_fuser_and_pulse_gate_uses_actual_nodes(tmp_path, monkeypatch) -> None:
    module = _load_script("rt5640_adapter_pcm_gate")
    proc, dev = _seed_fake_alsa(tmp_path)
    monkeypatch.setattr(module, "PROC_ASOUND_ROOT", proc)
    monkeypatch.setattr(module, "DEV_SND_ROOT", dev)
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        command = list(command)
        commands.append(command)
        if command[0] == "fuser":
            return _completed(command, returncode=1)
        if command[-2:] == ["list", "cards"]:
            return _completed(command, stdout=_pulse_cards())
        if command[0] == "amixer" and "cget" in command:
            return _completed(command, stdout=_route_cget(command))
        return _completed(command, stdout="")

    monkeypatch.setattr(module, "_run_command", run)
    report = module.assert_live_pcm_clock_preconditions()
    assert report["alsa_card_index"] == 1
    assert report["all_system_pcm_closed"]["status_count"] == 2
    fuser = next(command for command in commands if command[0] == "fuser")
    assert str(dev / "pcmC1D0p") in fuser
    assert str(dev / "pcmC1D1c") in fuser
    assert report["pulse"]["active_profile"] == "off"
    assert report["pulse"]["mutation_performed"] is False


def test_pcm_busy_or_fuser_owner_fails_closed(tmp_path, monkeypatch) -> None:
    module = _load_script("rt5640_adapter_pcm_busy")
    proc, dev = _seed_fake_alsa(tmp_path)
    monkeypatch.setattr(module, "PROC_ASOUND_ROOT", proc)
    monkeypatch.setattr(module, "DEV_SND_ROOT", dev)
    (proc / "card1/pcm1c/sub0/status").write_text("state: RUNNING\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_run_command",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("busy proc 뒤 command 실행")),
    )
    with pytest.raises(RuntimeError, match="점유 중"):
        module.assert_live_pcm_clock_preconditions()

    (proc / "card1/pcm1c/sub0/status").write_text("closed\n", encoding="utf-8")

    def owner(command, **_kwargs):
        if command[0] == "fuser":
            return _completed(command, returncode=0, stderr="1234")
        return _completed(command, stdout=_pulse_cards())

    monkeypatch.setattr(module, "_run_command", owner)
    with pytest.raises(RuntimeError, match="owner"):
        module.assert_live_pcm_clock_preconditions()


def test_pulse_non_off_is_rejected_without_mutation(tmp_path, monkeypatch) -> None:
    module = _load_script("rt5640_adapter_pulse_on")
    proc, dev = _seed_fake_alsa(tmp_path)
    monkeypatch.setattr(module, "PROC_ASOUND_ROOT", proc)
    monkeypatch.setattr(module, "DEV_SND_ROOT", dev)

    def run(command, **_kwargs):
        if command[0] == "fuser":
            return _completed(command, returncode=1)
        return _completed(command, stdout=_pulse_cards().replace("Active Profile: off", "Active Profile: output:analog-stereo"))

    monkeypatch.setattr(module, "_run_command", run)
    with pytest.raises(RuntimeError, match="exact off|자동 변경하지 않습니다"):
        module.assert_live_pcm_clock_preconditions()


def test_portaudio_mapping_is_exact_and_does_not_confuse_pcm1_with_pcm10() -> None:
    module = _load_script("rt5640_adapter_exact_portaudio")

    def device(name, inputs=16, outputs=16):
        return {
            "name": name,
            "hostapi": 0,
            "max_input_channels": inputs,
            "max_output_channels": outputs,
            "default_samplerate": 44_100.0,
        }

    class Backend:
        def __init__(self):
            self.checked: list[str] = []

        def query_devices(self):
            return [
                device("APE (hw:1,0)"),
                device("APE (hw:1,1)"),
                device("APE (hw:1,10)"),
            ]

        def query_hostapis(self, _index):
            return {"name": "ALSA"}

        def check_input_settings(self, **_kwargs):
            self.checked.append("input")

        def check_output_settings(self, **_kwargs):
            self.checked.append("output")

    backend = Backend()
    result = module._resolve_exact_portaudio_devices(backend, 1)
    assert result["input"]["index"] == 1
    assert result["output"]["index"] == 0
    assert backend.checked == ["input", "output"]


def test_command_runner_pins_c_locale(monkeypatch) -> None:
    module = _load_script("rt5640_adapter_locale")
    observed = {}

    def run(command, **kwargs):
        observed.update(kwargs)
        return _completed(command)

    monkeypatch.setattr(module.subprocess, "run", run)
    module._run_command(["true"])
    assert observed["env"]["LC_ALL"] == "C"
    assert observed["env"]["LANG"] == "C"


def test_same_duplex_raw_probe_discards_startup_and_rejects_dead_input() -> None:
    module = _load_script("rt5640_adapter_same_raw_probe")
    rng = np.random.default_rng(7)
    raw = np.zeros((96_000, 2), dtype="<i4")
    raw[48_000:] = rng.integers(-4_000_000, 4_000_001, size=(48_000, 2), dtype=np.int32)
    report = module._analyze_simultaneous_input(raw)
    assert report["passed"] is True
    assert report["settle_frames_discarded"] == 48_000
    assert report["source"] == "same_simultaneous_duplex_raw"

    with pytest.raises(RuntimeError, match="dead/stuck/railed"):
        module._analyze_simultaneous_input(np.zeros_like(raw))


def _snapshot_files(
    module,
    generation: Path,
    label: str,
    *,
    volatile_base: int = 100,
    normal_value: int = 0,
):
    state = _alsactl_fixture(
        volatile_base=volatile_base, normal_value=normal_value
    )
    mixer = _amixer_fixture(
        volatile_base=volatile_base, normal_value=normal_value
    )
    state_canonical, state_controls = module._normalize_alsactl_volatile_values(state)
    mixer_canonical, mixer_controls = module._normalize_amixer_volatile_values(mixer)
    module._crosscheck_volatile_identities(state_controls, mixer_controls)
    (generation / f"ape_{label}.state").write_bytes(state)
    (generation / f"amixer_{label}.txt").write_bytes(mixer)
    (generation / f"ape_{label}.volatile_normalized.state").write_bytes(
        state_canonical
    )
    (generation / f"amixer_{label}.volatile_normalized.txt").write_bytes(
        mixer_canonical
    )
    volatile_controls = {
        str(numid): {
            "numid": numid,
            "name": _VOLATILE_NAMES[numid],
            "iface": "MIXER",
            "type": "INTEGER",
            "count": 1,
            "alsactl_access": "read volatile",
            "amixer_access": "r--v----",
            "alsactl_raw_value": state_controls[numid]["raw_value"],
            "amixer_raw_value": mixer_controls[numid]["raw_value"],
        }
        for numid in sorted(_VOLATILE_NAMES)
    }
    return {
        "schema": module.SNAPSHOT_SCHEMA,
        "label": label,
        "alsactl": {
            "path": f"ape_{label}.state",
            "sha256": module._sha256_bytes(state),
            "size": len(state),
            "volatile_normalized_path": f"ape_{label}.volatile_normalized.state",
            "volatile_normalized_sha256": module._sha256_bytes(state_canonical),
            "volatile_normalized_size": len(state_canonical),
        },
        "amixer": {
            "path": f"amixer_{label}.txt",
            "sha256": module._sha256_bytes(mixer),
            "size": len(mixer),
            "volatile_normalized_path": f"amixer_{label}.volatile_normalized.txt",
            "volatile_normalized_sha256": module._sha256_bytes(mixer_canonical),
            "volatile_normalized_size": len(mixer_canonical),
        },
        "volatile_controls": volatile_controls,
        "volatile_allowlist_exact": True,
        "cross_parser_identity_exact": True,
        "normalization_scope": "exact_12_value_lines_only",
        "system_mutation_performed": False,
    }


def test_snapshot_mismatch_is_state_uncertain_and_never_restores(tmp_path, monkeypatch) -> None:
    module = _load_script("rt5640_adapter_snapshot_mismatch")
    before = _snapshot_files(module, tmp_path, "before")
    after = _snapshot_files(module, tmp_path, "after", normal_value=1)
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "_run_command", lambda command, **_k: commands.append(list(command)))
    with pytest.raises(module.StateUncertainError, match="자동 restore하지 않습니다"):
        module._compare_snapshots(before, after, tmp_path)
    assert commands == []


def test_exact_12_volatile_value_changes_pass_with_raw_values_preserved(tmp_path) -> None:
    module = _load_script("rt5640_adapter_volatile_12_pass")
    before = _snapshot_files(module, tmp_path, "before", volatile_base=100)
    after = _snapshot_files(module, tmp_path, "after", volatile_base=900)
    result = module._compare_snapshots(before, after, tmp_path)
    assert result["alsactl_raw_byte_exact"] is False
    assert result["amixer_raw_byte_exact"] is False
    assert result["alsactl_volatile_normalized_byte_exact"] is True
    assert result["amixer_volatile_normalized_byte_exact"] is True
    assert before["volatile_controls"]["1132"]["alsactl_raw_value"] == 100
    assert after["volatile_controls"]["1132"]["alsactl_raw_value"] == 900


def test_thirteenth_volatile_control_is_rejected() -> None:
    module = _load_script("rt5640_adapter_volatile_13_reject")
    base = _alsactl_fixture().decode()
    alsa = (
        base[:-2]
        + "\tcontrol.1201 {\n"
        "\t\tiface MIXER\n"
        "\t\tname 'Unexpected Volatile'\n"
        "\t\tvalue 1\n"
        "\t\tcomment {\n"
        "\t\t\taccess 'read volatile'\n"
        "\t\t\ttype INTEGER\n"
        "\t\t\tcount 1\n"
        "\t\t}\n"
        "\t}\n"
        "}\n"
    ).encode()
    with pytest.raises(RuntimeError, match=r"extra=\[1201\]"):
        module._normalize_alsactl_volatile_values(alsa)

    amixer = _amixer_fixture() + (
        b"numid=1201,iface=MIXER,name='Unexpected Volatile'\n"
        b"  ; type=INTEGER,access=r--v----,values=1,min=0,max=1,step=0\n"
        b"  : values=1\n"
    )
    with pytest.raises(RuntimeError, match=r"extra=\[1201\]"):
        module._normalize_amixer_volatile_values(amixer)


@pytest.mark.parametrize(
    ("kind", "mutated", "match"),
    [
        (
            "alsactl",
            _alsactl_fixture().replace(b"Lane1 Ratio Int", b"Lane1 Ratio Wrong", 1),
            "identity/metadata",
        ),
        (
            "amixer",
            _amixer_fixture().replace(b"type=INTEGER", b"type=BOOLEAN", 1),
            "identity/metadata",
        ),
    ],
)
def test_volatile_metadata_change_is_rejected(kind, mutated, match) -> None:
    module = _load_script(f"rt5640_adapter_volatile_metadata_{kind}")
    parser = (
        module._normalize_alsactl_volatile_values
        if kind == "alsactl"
        else module._normalize_amixer_volatile_values
    )
    with pytest.raises(RuntimeError, match=match):
        parser(mutated)


def test_read_only_snapshot_uses_store_and_contents_but_never_restore_or_cset(
    tmp_path, monkeypatch
) -> None:
    module = _load_script("rt5640_adapter_read_only_snapshot")
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        command = list(command)
        commands.append(command)
        if command[0] == "alsactl":
            state = Path(command[command.index("-f") + 1])
            state.write_bytes(_alsactl_fixture())
            return _completed(command)
        if command[0] == "amixer":
            return _completed(command, stdout=_amixer_fixture().decode())
        raise AssertionError(command)

    monkeypatch.setattr(module, "_run_command", run)
    snapshot = module._snapshot_read_only(tmp_path, label="before")
    assert snapshot["system_mutation_performed"] is False
    flat = [item for command in commands for item in command]
    assert "store" in flat
    assert "contents" in flat
    assert "restore" not in flat
    assert "cset" not in flat


def test_generation_claim_is_no_replace_and_failed_second_claim_writes_nothing(tmp_path) -> None:
    module = _load_script("rt5640_adapter_generation_claim")
    monkey_root = tmp_path / "repo"
    monkey_root.mkdir()
    module.REPO_ROOT = monkey_root
    generation = monkey_root / "results/rt5640_zero_duplex/v1"
    module._claim_generation(generation, {"schema": "claim", "value": 1})
    existing = sorted(path.name for path in generation.iterdir())
    with pytest.raises(FileExistsError):
        module._claim_generation(generation, {"schema": "claim", "value": 2})
    assert sorted(path.name for path in generation.iterdir()) == existing


def test_terminal_success_and_failure_receipts_can_never_coexist(tmp_path) -> None:
    module = _load_script("rt5640_adapter_terminal_exclusion")
    module._publish_terminal_receipt(
        tmp_path, success=True, value={"schema": "success"}
    )
    with pytest.raises(RuntimeError, match="공존 금지"):
        module._publish_terminal_receipt(
            tmp_path, success=False, value={"schema": "failure"}
        )
    assert (tmp_path / "receipt.json").is_file()
    assert not (tmp_path / "failure.json").exists()


def test_success_receipt_keeps_physical_counters_unknown_and_authority_false(
    tmp_path, monkeypatch
) -> None:
    module = _load_script("rt5640_adapter_live_success")
    repo = tmp_path / "repo"
    repo.mkdir()
    generation = repo / "results/rt5640_zero_duplex/v1"
    module.REPO_ROOT = repo
    frames = 256
    plan = {
        "frame_count": frames,
        "callback_count": 1,
        "watchdog_grace_seconds": 2.0,
        "canonical_payload_sha256": "1" * 64,
        "planned_pcm_sha256": "2" * 64,
    }
    planned = np.zeros((frames, 2), dtype="<i4")

    class CaptureFailure(RuntimeError):
        pass

    class AudioAPI:
        ZeroDuplexCaptureFailure = CaptureFailure

        @staticmethod
        def capture_zero_duplex(_backend, **kwargs):
            kwargs["pre_open_check"]()
            kwargs["on_stream_started"]()
            captured = np.arange(frames * 2, dtype="<i4").reshape(frames, 2)
            telemetry = {
                "actual_submitted_pcm": np.zeros_like(captured),
                "capture_valid_mask": np.ones(frames, dtype=np.bool_),
                "submitted_valid_mask": np.ones(frames, dtype=np.bool_),
                "callback_sequence": np.asarray([0], dtype="<i8"),
                "callback_start_frames": np.asarray([0], dtype="<i8"),
                "callback_frame_counts": np.asarray([256], dtype="<i8"),
                "callback_status_bitmask": np.asarray([0], dtype="<u4"),
                "input_buffer_adc_time": np.asarray([1.0], dtype="<f8"),
                "output_buffer_dac_time": np.asarray([1.1], dtype="<f8"),
                "callback_current_time": np.asarray([1.2], dtype="<f8"),
            }
            return captured, telemetry

    class ContractAPI:
        @staticmethod
        def capture_telemetry_to_contract(**_kwargs):
            return {"passed": True}

        @staticmethod
        def build_zero_duplex_receipt(**_kwargs):
            return {
                "status": module.EXPECTED_AUTHORITY_CEILING,
                "authority": {},
                "telemetry_receipt": {"passed": True},
            }

    static = {
        "generation": generation,
        "config": {"schema": "rt5640_zero_duplex_hardware_v1"},
        "config_binding": {"file_sha256": "3" * 64},
        "python_binding": {"expected_src": str(repo / "src")},
        "git": {"commit": "c" * 40, "branch": "work/v8", "dirty": False},
        "script": {"file_sha256": "4" * 64},
        "tools": {"git": {"file_sha256": "5" * 64}},
        "physical_confirmations": {
            "confirm_j511_disconnected": True,
            "confirm_amplifier_power_off": True,
            "confirm_amplifier_input_disconnected": True,
            "confirm_ab13x_amplifier_disconnected": True,
            "confirm_user_present": True,
        },
        "plan": plan,
        "planned_pcm": planned,
        "audio_api": AudioAPI,
        "contract_api": ContractAPI,
    }
    monkeypatch.setattr(module, "_static_contract", lambda *_a, **_k: static)
    lock_events: list[str] = []

    @contextmanager
    def lock():
        lock_events.append("acquire")
        try:
            yield {"path": "/run/user/1000/test.lock", "pid": os.getpid()}
        finally:
            lock_events.append("release")

    monkeypatch.setattr(module, "_machine_global_audio_lock", lock)
    monkeypatch.setattr(module, "_repository_live_audio_lock", lock)
    pre = {
        "alsa_card_index": 1,
        "all_system_pcm_closed": {"status_count": 2},
        "fuser": {"owners": []},
        "pulse": {"active_profile": "off"},
    }
    monkeypatch.setattr(module, "assert_live_pcm_clock_preconditions", lambda **_k: pre)
    monkeypatch.setattr(
        module,
        "_collect_alsa_physical_fingerprint",
        lambda _config: {"payload_sha256": "6" * 64, "shared_clock_authority": False},
    )

    def snapshot(gen, *, label):
        if label == "after":
            assert (gen / "raw_capture.npz").is_file(), "raw must precede post snapshot"
        return _snapshot_files(module, gen, label)

    monkeypatch.setattr(module, "_snapshot_read_only", snapshot)
    monkeypatch.setattr(module, "_sounddevice_import", lambda: object())
    monkeypatch.setattr(module, "_sounddevice_fingerprint", lambda _sd: {"version": "test"})
    monkeypatch.setattr(
        module,
        "_resolve_exact_portaudio_devices",
        lambda _sd, _card: {"input": {"index": 11}, "output": {"index": 10}},
    )
    monkeypatch.setattr(
        module,
        "_read_stream_proc",
        lambda _card: {
            "authority": {
                "physical_sample_drop_count": None,
                "physical_sample_add_count": None,
                "hardware_deadline_miss_count": None,
                "hardware_sample_slip_authority": False,
            }
        },
    )
    monkeypatch.setattr(
        module,
        "_analyze_simultaneous_input",
        lambda _raw: {"passed": True, "source": "same_simultaneous_duplex_raw"},
    )
    monkeypatch.setattr(
        module,
        "_revalidate_live_static",
        lambda *_a, **_k: {"passed": True},
    )
    original_terminal_publish = module._publish_terminal_receipt

    def terminal_publish(gen, *, success, value):
        if success:
            assert lock_events.count("release") == 2
        return original_terminal_publish(gen, success=success, value=value)

    monkeypatch.setattr(module, "_publish_terminal_receipt", terminal_publish)
    assert module._execute_live(_args()) == 0
    receipt = json.loads((generation / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["physical_counters"] == {
        "add_sample_count": None,
        "deadline_miss_count": None,
        "drop_sample_count": None,
        "hardware_sample_slip_authority": False,
    }
    assert receipt["authority"]["shared_clock_authority_pass"] is False
    assert receipt["authority"]["plant_identification_pass"] is False
    assert all(receipt["static_binding"]["physical_confirmations"].values())
    assert receipt["read_only_snapshots"]["system_mutation_performed"] is False
    assert receipt["read_only_snapshots"]["automatic_restore_performed"] is False
    assert receipt["transaction_finalization"] == {
        "machine_global_lock_released_before_success_publication": True,
        "repository_audio_lock_released_before_success_publication": True,
        "signal_handlers_restored_before_success_publication": True,
    }
    assert (generation / "raw_capture.npz").is_file()
    assert not (generation / "failure.json").exists()
