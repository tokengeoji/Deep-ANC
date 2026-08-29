from __future__ import annotations

import pytest

from deep_anc.realtime.rt5640_jack import (
    JACK_CONTROL_NAME,
    assert_rt5640_jack_state,
    parse_jack_state_amixer,
    read_rt5640_jack_state,
)


def _amixer(value: int) -> str:
    return "\n".join(
        (
            f"numid=1156,iface=MIXER,name='{JACK_CONTROL_NAME}'",
            "  ; type=ENUMERATED,access=rw------,values=1,items=4",
            "  ; Item #0 'None'",
            "  ; Item #1 'HP'",
            "  ; Item #2 'MIC'",
            "  ; Item #3 'HS'",
            f"  : values={value}",
        )
    )


@pytest.mark.parametrize("value,expected", [(0, "None"), (1, "HP"), (2, "MIC"), (3, "HS")])
def test_parse_rt5640_jack_state(value: int, expected: str) -> None:
    assert parse_jack_state_amixer(_amixer(value)) == expected


@pytest.mark.parametrize(
    "changed",
    (
        "",
        _amixer(4),
        _amixer(0).replace("Item #3 'HS'", "Item #3 'UNKNOWN'"),
        _amixer(0).replace(f"name='{JACK_CONTROL_NAME}'", "name='other'"),
        _amixer(0).replace("  : values=0", "  : values=0\n  : values=0"),
    ),
)
def test_parse_rt5640_jack_state_fails_closed(changed: str) -> None:
    with pytest.raises(ValueError):
        parse_jack_state_amixer(changed)


def test_read_rt5640_jack_state_uses_exact_control() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> str:
        commands.append(command)
        return _amixer(1)

    assert read_rt5640_jack_state(runner=runner) == "HP"
    assert commands == [["amixer", "-c", "APE", "cget", f"name={JACK_CONTROL_NAME}"]]


def test_assert_rt5640_jack_state_records_limits() -> None:
    report = assert_rt5640_jack_state(
        "None", samples=3, runner=lambda _command: _amixer(0)
    )
    assert report["passed"] is True
    assert report["observed_states"] == ["None", "None", "None"]
    assert report["authority"] == {
        "j511_plug_detected": False,
        "j511_unplugged_detected": True,
        "amplifier_end_connected": False,
        "amplifier_power_state": False,
        "electrical_output_witness": False,
        "acoustic_output_witness": False,
    }


def test_assert_rt5640_jack_state_rejects_unstable_or_invalid_request() -> None:
    values = iter((_amixer(0), _amixer(1)))
    with pytest.raises(RuntimeError, match="불일치"):
        assert_rt5640_jack_state("None", samples=2, runner=lambda _command: next(values))
    with pytest.raises(ValueError, match="기대값"):
        assert_rt5640_jack_state("LINE", runner=lambda _command: _amixer(0))
    with pytest.raises(ValueError, match="exact int"):
        assert_rt5640_jack_state("None", samples=True, runner=lambda _command: _amixer(0))
