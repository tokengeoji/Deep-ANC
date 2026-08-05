"""메타 테스트 — **실패를 증명한 적 없는 게이트를 금지한다** (발생기 B).

이것이 이번 작업의 핵심 산출물이다. 2026-08-04 사고에서 게이트 9개가 전부 PASS 였는데
전부 무용지물이었다. 그 사고가 다시 일어나지 않게 하는 유일한 구조적 방어는
"게이트를 세어 보고, 각각에 대해 **그것을 FAIL 시키는 fixture 가 존재하는지** 검사하는
테스트" 다. 없으면 이 파일이 실패한다.

여기서 검사하는 것:
  1. 소스에 있는 게이트가 전부 선언돼 있는가 (선언 누락 = 짝 없는 게이트)
  2. 선언된 negative fixture 가 실제로 존재하는 pytest 노드인가
  3. 그 fixture 가 **실패를 단언**하는가 (pytest.raises / not ok / 거부 …)
  4. 게이트 선언 자체가 negative fixture 없이는 만들어지지 않는가
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from deep_anc.config import REPO_ROOT
from deep_anc.ops.gate_registry import (
    GATES,
    GateDeclaration,
    declared_gate_ids,
    discover_audit_gate_ids,
    gate,
    gates_for_owner,
)


# 게이트 id 를 문자열 리터럴로 들고 있어 스캔이 가능한 소스.
_SCANNED_SOURCES = ("src/deep_anc/train/finetune_readiness.py",)

# negative fixture 가 "실패를 단언한다"고 인정하는 흔적. 하나도 없으면 그 테스트는
# 게이트가 거부하는 것을 보고 있지 않다는 뜻이다.
_FAILURE_MARKERS = (
    "pytest.raises",
    "not report",
    "not check",
    "not result",
    "not passed",
    "is False",
    "== False",
    "assert not ",
    "raise AssertionError",
    "errors",
    "invalid",
)

# 기각 단언이 도메인 특유의 형태(예: ``flatnonzero(bad) == [11..15]``)라 위 마커로는
# 잡히지 않는 경우의 보조 판정 — **테스트 이름**이 무엇을 보는지 말해야 한다.
_REJECTION_NAME_MARKERS = (
    "reject",
    "refus",
    "fail",
    "flag",
    "detect",
    "leak",
    "mismatch",
    "forgery",
    "slip",
    "amplif",
    "broken",
    "collapse",
    "disagree",
    "without",
    "loosening",
)


def _test_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _resolve(node_id: str) -> tuple[Path, ast.FunctionDef]:
    file_part, _, test_part = node_id.partition("::")
    name = test_part.split("[", 1)[0]
    path = REPO_ROOT / file_part
    assert path.exists(), f"negative fixture 파일이 없습니다: {node_id}"
    functions = _test_functions(path)
    assert name in functions, (
        f"negative fixture 테스트가 없습니다: {node_id} — 게이트를 선언했으면 "
        "그것을 FAIL 시키는 테스트가 반드시 있어야 합니다"
    )
    return path, functions[name]


# ----------------------------------------------------------------------------------
def test_every_gate_in_the_source_is_declared():
    """소스에서 발견된 게이트 id 가 전부 레지스트리에 있는가.

    게이트를 새로 만들고 선언을 빠뜨리면 여기서 실패한다 — 그리고 선언을 하려면
    negative fixture 를 먼저 써야 한다. 그것이 이 구조의 요점이다.
    """

    declared = declared_gate_ids()
    missing: dict[str, set[str]] = {}
    for rel in _SCANNED_SOURCES:
        found = discover_audit_gate_ids(REPO_ROOT / rel)
        gap = found - declared
        if gap:
            missing[rel] = gap
    assert missing == {}, (
        f"선언되지 않은 게이트가 있습니다: {missing} — "
        "deep_anc/ops/gate_registry.py 에 negative fixture 와 함께 선언하세요"
    )


def test_every_declared_gate_has_a_failing_fixture():
    """**핵심 단언.** 모든 게이트가 자기를 FAIL 시키는 테스트를 짝으로 갖는다."""

    unpaired: list[str] = []
    for declaration in GATES:
        path, function = _resolve(declaration.negative_fixture)
        source = ast.get_source_segment(path.read_text(encoding="utf-8"), function) or ""
        asserts = any(isinstance(node, ast.Assert) for node in ast.walk(function))
        named = any(marker in function.name for marker in _REJECTION_NAME_MARKERS)
        if not any(marker in source for marker in _FAILURE_MARKERS) and not (
            asserts and named
        ):
            unpaired.append(
                f"{declaration.gate_id} → {declaration.negative_fixture} "
                "(실패를 단언하지 않습니다)"
            )
    assert unpaired == [], (
        "다음 게이트의 fixture 가 '거부한다'를 확인하지 않습니다: " + ", ".join(unpaired)
    )


def test_positive_fixtures_exist_when_declared():
    """오기각 방지 — 정상 입력이 통과하는 것도 확인된 게이트가 있어야 한다."""

    for declaration in GATES:
        if declaration.positive_fixture is not None:
            _resolve(declaration.positive_fixture)


def test_gate_ids_are_unique_and_owners_exist():
    ids = [item.gate_id for item in GATES]
    assert len(ids) == len(set(ids)), "중복 게이트 id 가 있습니다"
    for declaration in GATES:
        assert (REPO_ROOT / declaration.owner).exists(), (
            f"{declaration.gate_id}: owner 파일이 없습니다 — {declaration.owner}"
        )


def test_a_gate_cannot_be_declared_without_a_negative_fixture():
    """레지스트리 모델 자체가 짝 없는 게이트를 거부하는가 (이 방어의 근본)."""

    common = dict(
        gate_id="example_gate",
        owner="src/deep_anc/train/finetune_readiness.py",
        what_it_asserts="예시",
    )
    with pytest.raises(Exception):
        GateDeclaration(**common)  # negative_fixture 없음
    with pytest.raises(Exception):
        GateDeclaration(**common, negative_fixture="")
    with pytest.raises(Exception):
        GateDeclaration(**common, negative_fixture="언젠가 쓸 예정")
    # 형식이 맞으면 만들어진다.
    ok = GateDeclaration(
        **common,
        negative_fixture="tests/test_gate_registry.py::test_gate_ids_are_unique_and_owners_exist",
    )
    assert ok.gate_id == "example_gate"


def test_the_gates_found_in_the_20260804_incident_are_all_covered():
    """사고 당시 'PASS 였지만 무용지물' 이었던 게이트가 전부 짝을 갖는가.

    이 목록이 이 작업의 계약이다. 여기 있는 게이트 중 하나라도 negative fixture 를
    잃으면 테스트가 실패한다.
    """

    incident_gates = (
        "official_secondary_path",
        "official_primary_path",
        "official_path_delay_spread",
        "official_path_sub_band_consistency",
        "matched_path_measurement_conditions",
        "path_delay_and_lead",
        "recorded_dataset_qa",
        "recorded_val_g4",
        "recorded_test_g4",
        "measurement_relative_tau_outliers",
        "invariant_relative_tau_constancy",
        "invariant_stream_coherence",
        "invariant_plant_fingerprint_match",
        "invariant_lead_agreement",
    )
    for gate_id in incident_gates:
        declaration = gate(gate_id)
        _resolve(declaration.negative_fixture)


def test_registry_covers_every_gate_owning_module():
    """게이트를 소유한 모듈이 하나도 빠지지 않았는가 (등록 누락 방지)."""

    owners = {declaration.owner for declaration in GATES}
    expected = {
        "src/deep_anc/train/finetune_readiness.py",
        "src/deep_anc/data/recorded_qa.py",
        "src/deep_anc/dsp/interleaved_probe.py",
        "src/deep_anc/dsp/invariants.py",
        "src/deep_anc/dsp/timing.py",
        "src/deep_anc/realtime/run_realtime.py",
        "scripts/data/reanalyse_paths_interleaved.py",
    }
    assert expected <= owners
    for owner in expected:
        assert gates_for_owner(owner), f"{owner} 에 선언된 게이트가 없습니다"


def test_the_registry_is_not_silently_shrinking():
    """게이트 수의 하한. 게이트를 지우면서 선언만 지우는 일을 막는다."""

    assert len(GATES) >= 30, f"선언된 게이트가 {len(GATES)}개로 줄었습니다"
