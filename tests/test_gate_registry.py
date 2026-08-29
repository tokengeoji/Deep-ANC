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
    CANONICAL_FINETUNE_READINESS_GATE_IDS,
    GATES,
    GateDeclaration,
    declared_gate_ids,
    discover_audit_gate_ids,
    gate,
    gates_for_owner,
)


# 게이트 id를 문자열 literal 또는 같은 scope의 단순 문자열 변수로 들고 있어
# AST 스캔이 가능한 소스.
_SCANNED_SOURCES = ("src/deep_anc/train/finetune_readiness.py",)
# ⚠ 미선언 탐지는 이 **한 파일**에서만 돈다. ``discover_audit_gate_ids`` 가 찾는 것은
# 정적 문자열 또는 단순 문자열 변수로 호출한 ``audit.fail/pass_``뿐이고, 나머지 15개
# 소유 파일은 게이트를
# ``raise ValueError`` / ``errors.append`` / 스크립트 종료코드로 표현하기 때문이다.
# 즉 "게이트는 짝 없이 존재할 수 없다"는 **선언된 게이트에 대해서만** 참이고, 새
# 게이트를 선언 없이 만드는 경로는 열려 있다 — 2026-08-06 에 실제로 그렇게 만들어진
# 게이트가 있었다(prepare_noise_pool 종료코드 1, 지금은 선언됨).
#
# 남은 크기를 숫자로 적어 둔다(2026-08-06 실측): ``return 1``/``sys.exit(1)`` 을 가진
# scripts/ 파일 16개 중 게이트가 선언된 것은 3개(prepare_noise_pool, record_duct,
# reanalyse_paths_interleaved)뿐이고 **12개가 미선언**이다. 그중 상당수는 인자 검증
# 이라 게이트가 아니므로 기계적으로 강제하면 소음이 된다 — 사람이 하나씩 판정해야
# 하고, 그 전까지 이 축은 사람의 기억에 의존한다는 사실을 여기 남긴다.

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


def test_variable_bound_audit_gate_ids_are_discovered(tmp_path):
    """변수에 묶인 새 gate가 literal-only scan을 피해갈 수 없어야 한다."""

    source = tmp_path / "variable_gate.py"
    source.write_text(
        "def check(audit):\n"
        "    gate_id = 'variable_bound_gate'\n"
        "    audit.fail(gate_id, 'fixture')\n"
        "    audit.pass_(gate_id, 'fixture')\n",
        encoding="utf-8",
    )
    assert discover_audit_gate_ids(source) == {"variable_bound_gate"}


def test_canonical_finetune_readiness_authority_is_exactly_17_declared_gates():
    """canonical 진입 점수의 분모와 gate identity를 함께 고정한다."""

    expected = {
        "config_fail_closed_flags",
        "recorded_transfer_snapshot",
        "absolute_objective_scope",
        "measured_primary_mode",
        "recorded_mix_ratio",
        "official_secondary_path",
        "official_primary_path",
        "matched_path_measurement_conditions",
        "path_delay_and_lead",
        "completed_init_checkpoint",
        "recorded_dataset_qa",
        "recorded_alignment_integrity",
        "recorded_statistical_power",
        "recorded_subband_coverage",
        "corpus_disjoint",
        "measured_source_delay_agreement",
        "plant_confidence_ceiling",
    }
    authority = tuple(CANONICAL_FINETUNE_READINESS_GATE_IDS)
    assert len(authority) == 17
    assert len(set(authority)) == 17
    assert set(authority) == expected
    assert expected <= declared_gate_ids()
    assert all(
        gate(gate_id).owner == "src/deep_anc/train/finetune_readiness.py"
        for gate_id in authority
    )
    discovered = discover_audit_gate_ids(REPO_ROOT / _SCANNED_SOURCES[0])
    assert expected <= discovered


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


_PASS_MARKERS = (
    '["ok"]',
    "assert report",
    "assert budget",
    "assert track",
    "assert capture",
    "== []",
    "is True",
    "== 116",
    "== 256",
    ").ok",
    "].ok",
    "assert check",
    "assert passed",
    "verdict is None",
    "fired == []",
    "failures == []",
    "pytest.approx",
    "np.isfinite",
    "np.allclose",
    "assert parsed",
    "assert not ",
    "assert ",
)

# "예외가 없어야 한다" 형태(단언 없이 호출만 하는 정상 짝)는 위 마커로 잡히지 않는다.
# 그때는 **테스트 이름**이 무엇을 보는지 말해야 한다 — negative 쪽 규칙과 대칭이다.
_PASS_NAME_MARKERS = (
    "allowed",
    "accepts",
    "passes",
    "survives",
    "silent",
    "preferred",
    "derives",
    "recovered",
    "restored",
    "round_trips",
    "does_not",
    "is_not",
)


def test_every_declared_gate_has_a_positive_fixture():
    """**핵심 단언 2.** 모든 게이트가 "정상 입력에서 안 울린다" 는 짝을 갖는다.

    2026-08-06 반증 #13: 이 저장소의 게이트는 반응이 전부 차단이다 — readiness 는
    학습을 막고 런타임 워치독은 mute(상쇄 0 dB) 한다. 그런데 강제된 것은 발동 짝뿐이라
    **오발동을 반증한 게이트가 0개**였고, 실제로 정상 세션 9개 중 4개를 떨어뜨리는
    게이트가 살아 있었다. 짝이 없으면 여기서 실패한다.
    """

    unpaired: list[str] = []
    for declaration in GATES:
        if not declaration.positive_fixture:
            unpaired.append(f"{declaration.gate_id} (positive_fixture 없음)")
            continue
        _resolve(declaration.positive_fixture)
    assert unpaired == [], (
        "다음 게이트에 '정상 입력에서 발동하지 않는다' 짝이 없습니다: "
        + ", ".join(unpaired)
    )


def test_positive_fixtures_assert_a_pass_at_a_declared_boundary():
    """정상 짝이 (a) 통과를 단언하고 (b) **선언한 경계 숫자를 실제로 넣는지** 본다.

    ``positive_probe`` 에 적힌 숫자 중 하나가 fixture 소스에 나타나야 한다. 이것이
    없으면 "정상값으로 한 번 돌려봤다" 와 "한계까지 몰아봤다" 를 구분할 방법이 없다.
    """

    problems: list[str] = []
    for declaration in GATES:
        path, function = _resolve(declaration.positive_fixture)
        source = ast.get_source_segment(path.read_text(encoding="utf-8"), function) or ""
        named = any(marker in function.name for marker in _PASS_NAME_MARKERS)
        if not any(marker in source for marker in _PASS_MARKERS) and not named:
            problems.append(
                f"{declaration.gate_id} → {declaration.positive_fixture} "
                "(통과를 단언하지 않습니다)"
            )
            continue
        numbers = declaration.probe_numbers()
        if not any(number in source for number in numbers):
            problems.append(
                f"{declaration.gate_id} → positive_probe 의 숫자 {numbers} 가 "
                f"{declaration.positive_fixture} 소스에 없습니다"
            )
    assert problems == [], "; ".join(problems)


def test_a_gate_cannot_be_declared_without_a_positive_fixture():
    """레지스트리 모델이 정상 짝 없는 게이트를 **런타임에** 거부하는가."""

    common = dict(
        gate_id="example_gate",
        owner="src/deep_anc/train/finetune_readiness.py",
        what_it_asserts="예시",
        negative_fixture=(
            "tests/test_gate_registry.py::test_gate_ids_are_unique_and_owners_exist"
        ),
    )
    with pytest.raises(Exception):
        GateDeclaration(**common)  # positive_fixture 없음
    with pytest.raises(Exception):
        GateDeclaration(**common, positive_fixture="", positive_probe="한계의 90%")
    with pytest.raises(Exception):
        GateDeclaration(
            **common,
            positive_fixture=(
                "tests/test_gate_registry.py::test_gate_ids_are_unique_and_owners_exist"
            ),
            positive_probe="정상값으로 돌려봤다",   # 숫자 없음 = 경계를 안 적었다
        )
    ok = GateDeclaration(
        **common,
        positive_fixture=(
            "tests/test_gate_registry.py::test_gate_ids_are_unique_and_owners_exist"
        ),
        positive_probe="한계의 90% 지점",
    )
    assert ok.positive_probe.endswith("지점")


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
        positive_fixture=(
            "tests/test_gate_registry.py::test_gate_ids_are_unique_and_owners_exist"
        ),
        positive_probe="한계의 90% 지점",
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
