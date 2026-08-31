from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from deep_anc.dsp.stage2_2khz_actual_ps_plan import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    load_stage2_actual_ps_static_config,
)
from deep_anc.dsp.stage2_2khz_rt5640_s32 import load_stage2_rt5640_s32_static_contract
from deep_anc.dsp.rt5640_stage2_s32_preflight import (
    ACTUAL_CONFIG_PROVENANCE_STATUS,
    J511_DISCONNECTED_STATUS,
    PASS_STATUS,
    ReadOnlyCommandResult,
    assert_rt5640_stage2_s32_preflight,
    collect_rt5640_stage2_s32_preflight,
    receipt_to_jsonable,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/jetson/preflight_stage2_2khz_rt5640_s32.py"
MODULE = REPO_ROOT / "src/deep_anc/dsp/rt5640_stage2_s32_preflight.py"


def _jack_output(value: int) -> str:
    return "\n".join(
        (
            "numid=1156,iface=MIXER,name='CVB-RT Jack-state'",
            "  ; type=ENUMERATED,access=r--v----,values=1,items=4",
            "  ; Item #0 'None'",
            "  ; Item #1 'HP'",
            "  ; Item #2 'MIC'",
            "  ; Item #3 'HS'",
            f"  : values={value}",
            "",
        )
    )


def _route_output(name: str, expected: str, index: int) -> str:
    return "\n".join(
        (
            f"numid={index},iface=MIXER,name='{name}'",
            "  ; type=ENUMERATED,access=rw------,values=1,items=2",
            f"  ; Item #0 'Other'",
            f"  ; Item #1 '{expected}'",
            "  : values=1",
            "",
        )
    )


def _make_proc_and_dev(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "proc_asound"
    proc.mkdir()
    (proc / "cards").write_text(" 1 [APE            ]: APE - APE\n", encoding="utf-8")
    for suffix in ("pcm0p", "pcm1c"):
        status = proc / "card1" / suffix / "sub0" / "status"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text("closed\n", encoding="utf-8")
    dev = tmp_path / "dev_snd"
    dev.mkdir()
    (dev / "pcmC1D0p").touch()
    (dev / "pcmC1D1c").touch()
    return proc, dev


def _runner(*, jack_value: int = 1, fuser_returncode: int = 1):
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...]) -> ReadOnlyCommandResult:
        calls.append(command)
        if command[:4] == ("amixer", "-c", "APE", "cget"):
            control = command[4]
            if "CVB-RT Jack-state" in control:
                return ReadOnlyCommandResult(0, _jack_output(jack_value), "")
            route = control.removeprefix("name=")
            routes = {
                "I2S1 Mux": ("ADMAIF1", 1),
                "ADMAIF1 Mux": ("I2S1", 17),
                "ADMAIF2 Mux": ("I2S2", 18),
                "I2S2 Mux": ("ADMAIF2", 2),
            }
            expected, index = routes[route]
            return ReadOnlyCommandResult(0, _route_output(route, expected, index), "")
        if command[0] == "fuser":
            return ReadOnlyCommandResult(fuser_returncode, "", "")
        raise AssertionError(f"unexpected command: {command}")

    return run, calls


def _actual_ps_config() -> dict[str, object]:
    return deepcopy(load_stage2_actual_ps_static_config(repository_root=REPO_ROOT))


def test_default_loader_binds_actual_ps_config_provenance_and_fake_runner_is_read_only(
    tmp_path: Path,
) -> None:
    proc, dev = _make_proc_and_dev(tmp_path)
    runner, calls = _runner(jack_value=1)

    receipt = collect_rt5640_stage2_s32_preflight(
        proc_asound_root=proc, dev_snd_root=dev, runner=runner
    )

    assert receipt["status"] == PASS_STATUS
    assert receipt["passed"] is True
    assert receipt["j511"]["observed_states"] == ("HP", "HP", "HP")
    assert receipt["pcm_occupancy"]["all_pcm_substreams_closed"] is True
    assert "static_transport" not in receipt
    assert receipt["actual_ps_config"]["config_path"].endswith(DEFAULT_CONFIG_RELATIVE_PATH)
    assert receipt["actual_ps_config"]["prohibited_transports"] == {
        "usb_ab13x_selected": False,
        "output_master_split_clock_selected": False,
        "bandlimited_fallback_selected": False,
        "s16_selected": False,
        "contract_forbids_usb_ab13x": True,
        "contract_forbids_output_master_split_clock": True,
        "contract_forbids_bandlimited_fallback": True,
        "contract_forbids_s16": True,
    }
    assert receipt["actual_ps_config"]["forbidden_source_or_receipt_origins"] == {
        "usb_ab13x": True,
        "output_master_split_clock": True,
        "bandlimited_fallback": True,
        "s16_transport": True,
        "legacy_relabel_or_promotion": True,
    }
    assert receipt["actual_ps_config"]["authority"]["plan_preparation_only"] is True
    assert receipt["actual_ps_config"]["authority"]["physical_ps_authority"] is False
    assert receipt["authority"]["same_card_s32_actual_config_provenance_pass"] is True
    assert receipt["audio_backend_imported"] is False
    assert receipt["alsa_pcm_opened"] is False
    assert receipt["speaker_output"] is False
    assert sum("CVB-RT Jack-state" in command[-1] for command in calls) == 3
    with pytest.raises(TypeError):
        receipt["status"] = "forged"  # type: ignore[index]
    assert receipt_to_jsonable(receipt)["status"] == PASS_STATUS


def test_j511_none_is_a_fail_closed_block_but_other_read_only_observations_are_retained(
    tmp_path: Path,
) -> None:
    proc, dev = _make_proc_and_dev(tmp_path)
    runner, calls = _runner(jack_value=0)

    receipt = collect_rt5640_stage2_s32_preflight(
        proc_asound_root=proc, dev_snd_root=dev, runner=runner, actual_ps_config_loader=_actual_ps_config
    )

    assert receipt["passed"] is False
    assert receipt["status"] == J511_DISCONNECTED_STATUS
    assert receipt["j511"]["observed_states"] == ("None", "None", "None")
    assert receipt["ape_routes"] is not None
    assert receipt["pcm_occupancy"] is not None
    assert sum("CVB-RT Jack-state" in command[-1] for command in calls) == 3
    with pytest.raises(RuntimeError, match=J511_DISCONNECTED_STATUS):
        assert_rt5640_stage2_s32_preflight(
            proc_asound_root=proc,
            dev_snd_root=dev,
            runner=runner,
            actual_ps_config_loader=_actual_ps_config,
        )


def test_busy_pcm_or_untrusted_actual_ps_config_cannot_pass(tmp_path: Path) -> None:
    proc, dev = _make_proc_and_dev(tmp_path)
    busy = proc / "card1" / "pcm0p" / "sub0" / "status"
    busy.write_text("running\n", encoding="utf-8")
    runner, _calls = _runner(jack_value=1)
    receipt = collect_rt5640_stage2_s32_preflight(
        proc_asound_root=proc, dev_snd_root=dev, runner=runner, actual_ps_config_loader=_actual_ps_config
    )
    assert receipt["passed"] is False
    assert receipt["status"] == "BLOCKED_PCM_STREAM_OCCUPIED"

    proc, dev = _make_proc_and_dev(tmp_path / "fuser_owner")
    owner_runner, _calls = _runner(jack_value=1, fuser_returncode=0)
    receipt = collect_rt5640_stage2_s32_preflight(
        proc_asound_root=proc,
        dev_snd_root=dev,
        runner=owner_runner,
        actual_ps_config_loader=_actual_ps_config,
    )
    assert receipt["passed"] is False
    assert receipt["status"] == "BLOCKED_PCM_STREAM_OCCUPIED"

    def usb_contract() -> dict[str, object]:
        bad = _actual_ps_config()
        bad["hardware_audio"]["output"]["card"] = "Audio"  # type: ignore[index]
        return bad

    receipt = collect_rt5640_stage2_s32_preflight(
        proc_asound_root=proc, dev_snd_root=dev, runner=runner, actual_ps_config_loader=usb_contract
    )
    assert receipt["passed"] is False
    assert receipt["status"] == ACTUAL_CONFIG_PROVENANCE_STATUS


def test_old_fallback_static_config_receipt_is_explicitly_rejected_by_actual_preflight(
    tmp_path: Path,
) -> None:
    proc, dev = _make_proc_and_dev(tmp_path)
    runner, _calls = _runner(jack_value=1)

    def legacy_fallback_config() -> dict[str, object]:
        return load_stage2_rt5640_s32_static_contract(repository_root=REPO_ROOT)

    receipt = collect_rt5640_stage2_s32_preflight(
        proc_asound_root=proc,
        dev_snd_root=dev,
        runner=runner,
        actual_ps_config_loader=legacy_fallback_config,
    )

    assert receipt["passed"] is False
    assert receipt["status"] == ACTUAL_CONFIG_PROVENANCE_STATUS
    assert receipt["actual_ps_config"] is None
    assert any("구형 fallback" in error for error in receipt["errors"])


def test_module_and_default_cli_are_read_only_and_write_no_cwd_artifact(tmp_path: Path) -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert "sounddevice" not in imported
    assert "pactl" not in MODULE.read_text(encoding="utf-8")
    assert "load_stage2_actual_ps_static_config" in MODULE.read_text(encoding="utf-8")
    assert "load_stage2_rt5640_s32_static_contract" not in MODULE.read_text(encoding="utf-8")
    before = sorted(path.name for path in tmp_path.iterdir())
    completed = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--dry-run", "--json"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode in {0, 2}
    assert "audio_backend_import=0; ALSA_PCM_open=0; speaker_output=0; raw_write=0" in completed.stdout
    assert sorted(path.name for path in tmp_path.iterdir()) == before
