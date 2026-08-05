"""폐기된 설정 키가 **조용히 살아남는 자리**를 저장소 전체에서 막는다 (발생기).

2026-08-06 반증 #17/#20 이 실제로 잡아낸 것
------------------------------------------
``scripts/bench/diagnose_training_overfit.py`` 의 ``--nmse-only`` 는
``lambda_mrstft`` / ``lambda_pow`` / ``lambda_clip`` **세 개를 리터럴로** 0 으로 만들었다.
그 사이에

  · ``lambda_clip`` 은 폐기 키가 되어 0 으로 써도 아무 효과가 없어졌고,
  · 새 항 3개(``lambda_dnh`` 0.12 / ``lambda_frame`` 0.5 / ``lambda_sat`` 1.0)가 생겼는데
    목록은 그대로였다.

결과: "NMSE 만" 이라고 이름 붙은 모드가 NMSE 를 분리하지 못했고, 그 사실을 아무 테스트도
보지 않았다. 이 저장소가 없애겠다고 선언한 발생기 — **죽은 설정이 조용히 무시돼 다음
사람을 속인다** — 가 그대로 재생산된 자리다.

그래서 이 파일은 증상 하나가 아니라 **자리**를 막는다.
  1. 폐기 키 목록을 손으로 적지 않는다 — 소유 모듈(``losses/config.py``)에게 물어본다.
  2. 그 키가 설정 파일이나 코드에서 **값으로 쓰이는** 곳이 저장소에 하나도 없어야 한다.
  3. 손실 항 스위치를 리터럴 목록으로 들고 있는 코드가 없어야 한다.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from deep_anc.config import REPO_ROOT
from deep_anc.losses.config import LossConfig


_DEAD_KEY = re.compile(r"loss\.([A-Za-z_][A-Za-z0-9_]*)=")

# 이 두 파일은 "폐기됐다" 는 **사실 자체를 소유**하므로 키 이름이 나타나는 것이 정상이다.
_OWNERS = {
    "src/deep_anc/losses/config.py",
    "tests/test_anc_loss.py",
    "tests/test_gate_dead_config_keys.py",
    "scripts/bench/diagnose_training_overfit.py",   # 폐기 키를 **거부**하는 코드
}

_SCANNED = ("src", "scripts", "configs")


def deprecated_loss_keys() -> frozenset[str]:
    """``LossConfig`` 가 DeprecationWarning 을 내는 키 = 폐기 키. 단일 출처에 물어본다."""

    dead: set[str] = set()
    for name in LossConfig.model_fields:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                LossConfig.parse({name: 0.0})
            except (TypeError, ValueError):
                continue
        for item in caught:
            if issubclass(item.category, DeprecationWarning):
                dead.update(_DEAD_KEY.findall(str(item.message)))
    return frozenset(dead)


def _files() -> list[Path]:
    found: list[Path] = []
    for top in _SCANNED:
        root = REPO_ROOT / top
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix in (".py", ".yaml", ".yml") and "__pycache__" not in path.parts:
                found.append(path)
    return found


def test_the_deprecated_key_set_is_not_empty():
    """이 검사가 아무것도 안 보고 통과하는 상태를 막는다."""

    dead = deprecated_loss_keys()
    assert dead, "폐기 키를 하나도 찾지 못했습니다 — 탐지 방식이 깨졌습니다"
    assert "lambda_clip" in dead


def test_no_file_assigns_a_deprecated_loss_key():
    """폐기 키가 **값으로 쓰이는** 자리가 저장소에 하나도 없어야 한다.

    주석에 남기는 것은 허용된다(왜 없어졌는지 설명해야 하므로). 금지되는 것은
    ``cfg["loss"]["lambda_clip"] = 0.0`` 이나 YAML 의 ``lambda_clip: 1.0`` 처럼
    **값이 대입되는** 형태다 — 그것이 다음 사람을 속인다.
    """

    dead = deprecated_loss_keys()
    patterns = {
        name: re.compile(
            rf"""(?:\[\s*["']{name}["']\s*\]\s*=)"""     # cfg["loss"]["lambda_clip"] =
            rf"""|(?:^\s*{name}\s*:\s*[^\s#])"""          # YAML: lambda_clip: 1.0
            rf"""|(?:["']{name}["']\s*:\s*[^\s#])""",     # dict 리터럴 안
            re.MULTILINE,
        )
        for name in dead
    }
    offenders: list[str] = []
    for path in _files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in _OWNERS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{rel}:{line} loss.{name}")
    assert offenders == [], (
        "폐기된 손실 키에 값을 넣는 자리가 남아 있습니다 (조용히 무시된다): "
        + ", ".join(offenders)
    )


def test_nmse_isolation_is_derived_not_hand_listed():
    """``--nmse-only`` 가 끄는 항을 **리터럴 목록**으로 들고 있지 않은지 본다.

    목록을 손으로 적으면 항이 하나 추가되는 순간 격리가 조용히 깨진다 — 실제로 깨졌다.
    """

    import importlib.util

    path = REPO_ROOT / "scripts/bench/diagnose_training_overfit.py"
    spec = importlib.util.spec_from_file_location("_diagnose_overfit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    live = {
        name
        for name in LossConfig.model_fields
        if name.startswith("lambda_")
    } - deprecated_loss_keys()
    assert len(live) >= 5, live

    isolated = module.isolate_nmse({"lambda_mrstft": 1.0})
    parsed = LossConfig.parse(isolated)
    still_on = {
        name: float(getattr(parsed, name))
        for name in live
        if float(getattr(parsed, name) or 0.0) != 0.0
    }
    assert still_on == {}, still_on

    # 폐기 키는 0 으로 덮지 않고 **거부**한다 (덮으면 죽은 키를 새로 심는 셈이다).
    for name in deprecated_loss_keys():
        assert name not in isolated or isolated[name] is None
