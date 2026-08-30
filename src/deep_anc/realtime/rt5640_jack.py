"""RT5640 J511 connector state를 읽기 전용으로 판정한다.

이 모듈은 ALSA mixer의 ``CVB-RT Jack-state`` control만 읽는다.  따라서 Jetson
J511 잭에 plug가 감지됐는지는 확인할 수 있지만, 케이블 반대편이 앰프에 꽂혔는지,
앰프 전원이 켜졌는지, 실제 아날로그 전압이나 음향 출력이 존재하는지는 확인하지
못한다. 그 한계는 receipt에도 명시적으로 남긴다.
"""

from __future__ import annotations

from collections.abc import Callable
import re
import subprocess
import time
from typing import Any


JACK_CONTROL_NAME = "CVB-RT Jack-state"
JACK_STATES = frozenset({"None", "HP", "MIC", "HS"})
JACK_STATE_SCHEMA = "rt5640_j511_jack_state_v1"
_ITEM_RE = re.compile(r"^\s*;\s*Item\s+#(?P<index>[0-9]+)\s+'(?P<name>[^']+)'\s*$")
_VALUE_RE = re.compile(r"^\s*:\s*values=(?P<value>[^\s]+)\s*$")


def parse_jack_state_amixer(output: str) -> str:
    """``amixer cget`` 출력에서 enum state를 fail-closed로 해석한다."""

    if not isinstance(output, str) or not output.strip():
        raise ValueError("RT5640 jack-state amixer 출력이 비었습니다")
    if f"name='{JACK_CONTROL_NAME}'" not in output:
        raise ValueError("RT5640 Jack-state control identity가 다릅니다")

    items: dict[int, str] = {}
    values: list[str] = []
    for line in output.splitlines():
        item = _ITEM_RE.match(line)
        if item:
            index = int(item.group("index"))
            name = item.group("name")
            if index in items or name not in JACK_STATES:
                raise ValueError("RT5640 Jack-state enum item이 불명확합니다")
            items[index] = name
            continue
        value = _VALUE_RE.match(line)
        if value:
            values.append(value.group("value"))

    if (
        set(items.values()) != JACK_STATES
        or set(items) != set(range(len(JACK_STATES)))
        or len(items) != len(JACK_STATES)
    ):
        raise ValueError("RT5640 Jack-state enum 집합이 다릅니다")
    if len(values) != 1 or not values[0].isdigit():
        raise ValueError("RT5640 Jack-state value가 exact integer 하나가 아닙니다")
    index = int(values[0])
    if index not in items:
        raise ValueError("RT5640 Jack-state value가 enum 범위 밖입니다")
    return items[index]


def _default_runner(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"RT5640 Jack-state 조회 실패 (exit={completed.returncode}): {stderr}"
        )
    return completed.stdout


def read_rt5640_jack_state(
    *,
    card_id: str = "APE",
    runner: Callable[[list[str]], str] = _default_runner,
) -> str:
    """현재 J511 plug 감지 상태를 반환한다. ALSA mixer를 변경하지 않는다."""

    if not isinstance(card_id, str) or not card_id.strip():
        raise ValueError("ALSA card_id가 비었습니다")
    output = runner(["amixer", "-c", card_id.strip(), "cget", f"name={JACK_CONTROL_NAME}"])
    return parse_jack_state_amixer(output)


def assert_rt5640_jack_state(
    expected: str,
    *,
    card_id: str = "APE",
    samples: int = 3,
    interval_seconds: float = 0.0,
    runner: Callable[[list[str]], str] = _default_runner,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """여러 read-only sample이 같은 expected plug 상태인지 확인한다.

    ``None``은 J511 plug가 감지되지 않았다는 뜻이고, ``HP``/``HS``는 실제 측정 전에
    line/headset plug가 감지됐다는 뜻이다. 앰프 쪽 상태에 대한 권한은 항상 false다.
    """

    if expected not in JACK_STATES:
        raise ValueError(f"지원하지 않는 RT5640 Jack-state 기대값: {expected!r}")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples는 1 이상의 exact int여야 합니다")
    interval = float(interval_seconds)
    if interval < 0.0:
        raise ValueError("interval_seconds는 음수가 될 수 없습니다")

    observed: list[str] = []
    for index in range(samples):
        observed.append(read_rt5640_jack_state(card_id=card_id, runner=runner))
        if index + 1 < samples and interval:
            sleep(interval)
    if any(value != expected for value in observed):
        raise RuntimeError(
            "RT5640 J511 Jack-state 불일치: "
            f"expected={expected!r}, observed={observed!r}"
        )
    return {
        "schema": JACK_STATE_SCHEMA,
        "card_id": card_id.strip(),
        "control": JACK_CONTROL_NAME,
        "expected_state": expected,
        "observed_states": observed,
        "samples": samples,
        "passed": True,
        "authority": {
            "j511_plug_detected": expected != "None",
            "j511_unplugged_detected": expected == "None",
            "amplifier_end_connected": False,
            "amplifier_power_state": False,
            "electrical_output_witness": False,
            "acoustic_output_witness": False,
        },
    }


__all__ = [
    "JACK_CONTROL_NAME",
    "JACK_STATE_SCHEMA",
    "JACK_STATES",
    "assert_rt5640_jack_state",
    "parse_jack_state_amixer",
    "read_rt5640_jack_state",
]
