"""게이트 선언 — **실패를 증명한 적 없는 게이트를 금지한다.**

왜 이 모듈이 있는가
------------------
2026-08-05 결함 군집 B: 확인된 결함 18건 중 5건이 "게이트가 통과를 주장하는데 그
주장이 **반증된 적이 없다**" 는 하나의 발생기에서 나왔다.

* 게이트 9개가 전부 PASS 인데 플랜트가 형상 기준 50% 틀려 있었다.
* recorded QA 가 80/80 PASS 인데 학습 데이터의 시간축이 붕괴해 있었다.
* G4 가 옥타브 감쇠를 **계산해서 저장까지 하면서** 판정에는 쓰지 않았다.
* 고역 15–22 dB 증폭이 검출되지 않았다.

공통점: 전부 **좋아 보이는 데이터에서 한 번 통과시켜 보고 끝**이었다. 나쁜 데이터에서
FAIL 하는 것을 확인한 게이트가 하나도 없었다.

그래서 이 모듈은 게이트를 "선언"으로 만들고, ``negative_fixture`` 를 **필수 필드**로
요구한다. 짝이 없는 게이트는 선언 자체가 만들어지지 않는다. ``tests/test_gate_registry.py``
가 (a) 소스에서 발견된 게이트가 전부 선언돼 있는지, (b) 선언된 negative fixture 가 실제로
존재하고 **실패를 단언하는지** 검사한다.

2026-08-06 — 군집 B 의 나머지 절반: **오발동**
---------------------------------------------
위 구조는 절반만 고쳤다. 강제된 것은 "발동시키는 fixture 가 있는가" 뿐이었고,
**"정상 입력에서 발동하지 않는가" 를 실제 운용 범위 전체로 몰아본 게이트가 하나도
없었다.** 이 저장소의 모든 게이트는 반응이 차단이다 — 학습을 막거나(readiness),
상쇄를 0 dB 로 만든다(런타임 mute = 절대목표 2 의 최악값). 즉 오발동 한 번의 대가가
결함 한 건의 대가와 같다. 그런데 오발동 쪽 반증은 존재한 적이 없었다.

그래서 ``positive_fixture`` 도 **필수**가 됐고, 그것이 정상이라고 주장하는 입력이
어디까지 몰린 것인지를 ``positive_probe`` 로 **숫자와 함께** 적어야 한다. 메타 테스트가
그 숫자가 실제 fixture 소스에 나타나는지까지 검사하므로, "정상 하나 통과시켜 봤다"
수준의 짝은 선언을 통과하지 못한다.

게이트를 새로 만들 때
--------------------
1. 게이트를 구현한다.
2. **그것을 FAIL 시키는 테스트를 쓴다.**
3. **정상 산출물을 한계 근처(예: 한계의 90%, 목표 대역 양 끝, 최소 세션 수)까지 몰아
   PASS 하는 테스트를 쓴다.**
4. 여기에 :class:`GateDeclaration` 을 추가한다.
2·3 중 하나라도 건너뛰면 4를 쓸 수 없고, 4를 건너뛰면 메타 테스트가 실패한다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "CANONICAL_FINETUNE_READINESS_GATE_IDS",
    "GATES",
    "GateDeclaration",
    "declared_gate_ids",
    "discover_audit_gate_ids",
    "gate",
    "gates_for_owner",
]


_NODE_ID = re.compile(r"^tests/[A-Za-z0-9_]+\.py::[A-Za-z0-9_]+(\[[^\]]+\])?$")
_GATE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class GateDeclaration(BaseModel):
    """게이트 하나의 선언. ``negative_fixture`` 가 **없으면 만들 수 없다.**"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_id: str
    owner: str
    """게이트를 구현한 저장소 상대 경로."""

    what_it_asserts: str
    """무엇이 참이어야 통과하는가. 한 문장."""

    negative_fixture: str
    """**이 게이트를 FAIL 시키는** pytest 노드 id. 필수 필드다.

    "통과하는 것을 봤다"는 게이트가 작동한다는 증거가 아니다. 실측으로 확인된 사실:
    게이트 9개 전부 PASS 였는데 전부 무용지물이었다.
    """

    positive_fixture: str
    """**이 게이트가 정상 입력에서 발동하지 않는 것을** 보이는 pytest 노드 id. 필수다.

    negative 짝만 강제하면 "울리기는 하는데 아무 데서나 울리는" 게이트가 만들어진다.
    이 저장소에서 게이트의 반응은 전부 차단이므로 오발동은 결함과 같은 값을 갖는다.
    """

    positive_probe: str
    """정상 fixture 가 **어디까지 몰아봤는지**. 숫자가 반드시 들어가야 한다.

    "정상 신호로 한 번 돌려봤다" 는 오기각 방지의 증거가 아니다. 요구는 운용 범위의
    **경계**다 — 한계의 90% 지점, 목표 대역 최저·최고 주파수, 최소 세션 수, 허용
    spread 의 최대값 같은 것. 메타 테스트가 여기 적힌 숫자 중 하나가 실제 fixture
    소스에 나타나는지 검사하므로, 이 문장은 문서가 아니라 fixture 와 묶인 주장이다.
    """

    discoverable_id: bool = True
    """게이트 id 가 소스에 문자열 리터럴로 나타나는가.

    ``recorded_{split}_g4`` 처럼 f-string 으로 만들어지는 id 는 소스 스캔으로 발견되지
    않으므로 False 로 선언하고, 메타 테스트가 그 사실을 알고 넘어간다.
    """

    @model_validator(mode="after")
    def _validate(self) -> "GateDeclaration":
        if not _GATE_ID.match(self.gate_id):
            raise ValueError(f"gate_id 는 snake_case 여야 합니다: {self.gate_id!r}")
        if not self.owner or not self.owner.endswith(".py"):
            raise ValueError(f"owner 는 저장소 상대 .py 경로여야 합니다: {self.owner!r}")
        if not self.what_it_asserts.strip():
            raise ValueError(f"{self.gate_id}: what_it_asserts 가 비었습니다")
        if not _NODE_ID.match(self.negative_fixture):
            raise ValueError(
                f"{self.gate_id}: negative_fixture 는 'tests/<file>.py::<test>' 형식의 "
                f"pytest 노드 id 여야 합니다: {self.negative_fixture!r} — "
                "게이트를 FAIL 시키는 fixture 없이는 게이트를 선언할 수 없습니다"
            )
        if not _NODE_ID.match(self.positive_fixture):
            raise ValueError(
                f"{self.gate_id}: positive_fixture 는 'tests/<file>.py::<test>' 형식의 "
                f"pytest 노드 id 여야 합니다: {self.positive_fixture!r} — "
                "정상 입력에서 발동하지 않는 것을 보인 fixture 없이는 게이트를 "
                "선언할 수 없습니다 (오발동은 결함과 같은 값을 갖는다)"
            )
        if not _NUMBER.search(self.positive_probe):
            raise ValueError(
                f"{self.gate_id}: positive_probe 에 숫자가 없습니다: "
                f"{self.positive_probe!r} — 정상 fixture 가 **어디까지** 몰린 것인지"
                "(한계의 90%, 대역 양 끝, 최소 세션 수 …)를 숫자로 적어야 합니다"
            )
        return self

    def probe_numbers(self) -> tuple[str, ...]:
        """``positive_probe`` 에 적힌 숫자 토큰. 메타 테스트가 fixture 소스와 대조한다."""

        return tuple(_NUMBER.findall(self.positive_probe))


_READINESS = "src/deep_anc/train/finetune_readiness.py"
_QA = "src/deep_anc/data/recorded_qa.py"
_PROBE = "src/deep_anc/dsp/interleaved_probe.py"
_INVARIANTS = "src/deep_anc/dsp/invariants.py"
_TIMING = "src/deep_anc/dsp/timing.py"
_REANALYSE = "scripts/data/reanalyse_paths_interleaved.py"
_RUNTIME = "src/deep_anc/realtime/run_realtime.py"
_SAFETY = "src/deep_anc/realtime/safety.py"
_RECORD = "scripts/data/record_duct.py"
_NOISE_POOL = "scripts/data/prepare_noise_pool.py"
_TIMELINE = "src/deep_anc/data/timeline.py"
_RECORDED_DATASET = "src/deep_anc/data/recorded_dataset.py"

_LOSS = "src/deep_anc/losses/anc_loss.py"
_LOSS_CFG = "src/deep_anc/losses/config.py"
_SYNTH_DATASET = "src/deep_anc/data/synth_dataset.py"
_EVAL_RECORDED = "src/deep_anc/eval/recorded.py"

_NEG = "tests/test_gate_negative_fixtures.py"
_RTSAFE = "tests/test_realtime_safety.py"
_LOSS_TESTS = "tests/test_anc_loss.py"
_START_TESTS = "tests/test_loss_start_sample.py"


# canonical fine-tune을 시작하기 전에 집계하는 leaf gate의 단일 authority다.
# ``readiness`` 집계 gate와 completion gate는 포함하지 않는다.
CANONICAL_FINETUNE_READINESS_GATE_IDS: tuple[str, ...] = (
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
)


GATES: tuple[GateDeclaration, ...] = (
    # ---------------- 파인튜닝 진입 게이트 (audit_finetune_readiness) ----------------
    GateDeclaration(
        gate_id="config_fail_closed_flags",
        owner=_READINESS,
        what_it_asserts="필수 fail-closed 설정 3종이 켜져 있다",
        negative_fixture=f"{_NEG}::test_config_fail_closed_flags_gate_fails_when_a_flag_is_off",
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe=(
            "canonical 진입 leaf authority 분모 17개: non-trust 경계 fixture 16개와 "
            "transfer 별도 정상 fixture 1개"
        ),
    ),
    GateDeclaration(
        gate_id="absolute_objective_scope",
        owner=_READINESS,
        what_it_asserts=(
            "150–1600Hz와 speech/music/environment/machine 네 계열을 설정으로 "
            "축소할 수 없다"
        ),
        negative_fixture=(
            "tests/test_canonical_finetune_guardrails_doc.py::"
            "test_absolute_objective_requires_all_four_families_and_environment_pool_mapping"
        ),
        positive_fixture=(
            "tests/test_canonical_finetune_guardrails_doc.py::"
            "test_absolute_objective_requires_all_four_families_and_environment_pool_mapping"
        ),
        positive_probe="절대목표 150–1600Hz와 source family 4개를 정확히 유지",
    ),
    GateDeclaration(
        gate_id="measured_primary_mode",
        owner=_READINESS,
        what_it_asserts="digital-ref 파인튜닝이 실측 P(z) 모드로 돈다",
        negative_fixture=f"{_NEG}::test_measured_primary_mode_gate_fails_on_surrogate",
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe=(
            "실측 P(z) 모드 그대로, canonical 진입 leaf authority 17개 중 "
            "non-trust 경계 fixture 16개가 PASS"
        ),
    ),
    GateDeclaration(
        gate_id="recorded_mix_ratio",
        owner=_READINESS,
        what_it_asserts="실측 데이터 혼합비가 요구 범위 안이다",
        negative_fixture=f"{_NEG}::test_recorded_mix_ratio_gate_fails_when_recorded_share_is_too_small",
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="혼합비를 요구값과 정확히 같게 둔 세션 48개 경계 설정",
    ),
    GateDeclaration(
        gate_id="official_secondary_path",
        owner=_READINESS,
        what_it_asserts="S(z) 아티팩트가 official 품질 메타데이터와 일관성 기준을 만족한다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_timing_invalid_or_legacy_path_metadata"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="일관성 = min_path_consistency 정확히, 절대목표 양 끝 150/1600Hz",
    ),
    GateDeclaration(
        gate_id="official_primary_path",
        owner=_READINESS,
        what_it_asserts="P(z) 아티팩트가 official 품질 기준과 출력 채널 규약을 만족한다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_official_path_gate_rejects_wrong_channel_and_low_consistency"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="요구 대역 상단 1600Hz 를 정확히 요구한 상태에서 통과",
    ),
    GateDeclaration(
        gate_id="matched_path_measurement_conditions",
        owner=_READINESS,
        what_it_asserts="P 와 S 가 같은 캡처·같은 측정 조건에서 나왔다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_interleaved_pair_from_different_captures_fails_matched_conditions"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_interleaved_pair_from_one_capture_passes_matched_conditions"
        ),
        positive_probe="같은 capture_id, P 지연 4 샘플 정합",
    ),
    GateDeclaration(
        gate_id="path_delay_and_lead",
        owner=_READINESS,
        what_it_asserts="설정 lead 가 측정 P/S 지연에서 유도되는 값과 정확히 같다",
        negative_fixture=f"{_NEG}::test_path_delay_and_lead_gate_fails_when_lead_is_hand_written",
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="lead 유도값과 정확히 일치 + 상대 τ spread 허용 최대 3 샘플",
    ),
    GateDeclaration(
        gate_id="completed_init_checkpoint",
        owner=_READINESS,
        what_it_asserts="init checkpoint 가 완주했고 lead·물리 모드가 허용 범위다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::test_init_lead_mismatch_is_rejected_by_default"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_init_checkpoint_band_exactly_equal_to_the_finetune_band_passes"
        ),
        positive_probe="checkpoint 학습 대역이 파인튜닝 대역과 정확히 같은 100.0–1000.0Hz (여유 0)",
    ),
    GateDeclaration(
        gate_id="recorded_dataset_qa",
        owner=_READINESS,
        what_it_asserts="실측 manifest 의 모든 세션이 QA 를 통과하고 커버리지가 충족된다",
        negative_fixture=f"{_NEG}::test_recorded_dataset_qa_gate_fails_on_a_broken_session",
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="세션 48개 = 최소 세션 수, 분량 = 최소 분량 (여유 0)",
    ),
    GateDeclaration(
        gate_id="readiness",
        owner=_READINESS,
        what_it_asserts="진입 게이트 전부가 통과했다 (집계)",
        negative_fixture=f"{_NEG}::test_readiness_aggregate_gate_fails_when_any_child_fails",
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe=(
            "집계 분모는 canonical 진입 leaf authority 17개이며 non-trust 경계 "
            "fixture 16개와 transfer 별도 정상 fixture 1개로 검증"
        ),
    ),
    # ---- 2026-08-05 신설: 확인된 결함 하나에 게이트 하나 ----
    GateDeclaration(
        gate_id="recorded_alignment_integrity",
        owner=_READINESS,
        what_it_asserts="학습 데이터에 source→ERR 시간 관계가 실제로 존재한다 (결함 2)",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_collapsed_source_err_timebase"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="세션 48개 전수 QA 를 최소 요구치에 붙여 통과",
    ),
    GateDeclaration(
        gate_id="recorded_statistical_power",
        owner=_READINESS,
        what_it_asserts="val/test 의 계열당 그룹이 CI 를 정의할 수 있는 하한 이상이다 (D3)",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_underpowered_val_and_test_groups"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="계열당 그룹 = 하한 4 정확히",
    ),
    GateDeclaration(
        gate_id="recorded_subband_coverage",
        owner=_READINESS,
        what_it_asserts=(
            "현재 manifest/timing에 결속된 train/val/test family×strict 부대역 target "
            "coverage가 독립 그룹 하한을 만족한다"
        ),
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_forged_recorded_subband_coverage_aggregate"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_passes_only_with_official_paths_completed_init_and_full_recorded_qa"
        ),
        positive_probe="family×부대역 독립 그룹 = 하한 4 정확히",
    ),
    GateDeclaration(
        gate_id="corpus_disjoint",
        owner=_READINESS,
        what_it_asserts="합성 학습 스트림과 실측이 같은 원본 오디오를 쓰지 않는다 (D1)",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_corpus_leak_between_synthetic_and_recorded"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="선언 태그 전부 manifest 존재, 세션 48개 경계 설정에서 겹침 없음",
    ),
    GateDeclaration(
        gate_id="measured_source_delay_agreement",
        owner=_READINESS,
        what_it_asserts=(
            "합성 브랜치와 실측 브랜치가 모델에게 주는 **총 선행량**이 같다 (D2) —"
            " 다르면 같은 모델이 두 브랜치에서 서로 다른 예측 과제를 배운다"
        ),
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_the_two_training_branches_must_give_the_same_total_advance"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_the_two_training_branches_must_give_the_same_total_advance"
        ),
        positive_probe=(
            "timeline 모드에서 합성 1718 vs 실측 1718 (차이 0 샘플) — 같은 데이터에서 "
            "constant 는 1459 샘플 어긋나 FAIL 한다"
        ),
    ),
    GateDeclaration(
        gate_id="plant_confidence_ceiling",
        owner=_READINESS,
        what_it_asserts="선언한 목표가 이 플랜트의 달성 가능 상한 안에 있다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_a_target_above_the_achievable_ceiling"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_every_entry_gate_passes_at_its_declared_boundary"
        ),
        positive_probe="목표 + 여유 = 달성 가능 상한의 90% 지점",
    ),
    GateDeclaration(
        gate_id="plant_identity_for_comparison",
        owner=_READINESS,
        what_it_asserts="완료 판정에 쓰인 val/test 결과가 같은 플랜트에서 나왔다 (결함 5)",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_completion_rejects_val_and_test_from_different_plants"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_completion_gates_pass_at_the_minimum_sample_boundary"
        ),
        positive_probe="val/test 같은 지문, 최소 표본 세션 1개·세그먼트 1개",
    ),
    # ---------------- 파인튜닝 완료 게이트 (audit_finetune_completion) ----------------
    GateDeclaration(
        gate_id="measured_finetune_checkpoint",
        owner=_READINESS,
        what_it_asserts="완료 판정 대상 checkpoint 가 measured 물리로 학습된 것이다",
        negative_fixture=f"{_NEG}::test_measured_finetune_checkpoint_gate_fails_on_surrogate_physics",
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_completion_gates_pass_at_the_minimum_sample_boundary"
        ),
        positive_probe="최소 표본 경계: 세션 1개 / 세그먼트 1개",
    ),
    GateDeclaration(
        gate_id="recorded_manifest_provenance",
        owner=_READINESS,
        what_it_asserts="완료 판정이 읽는 manifest 가 학습에 쓴 것과 같은 파일이다",
        negative_fixture=f"{_NEG}::test_recorded_manifest_provenance_gate_fails_without_manifest",
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_completion_gates_pass_at_the_minimum_sample_boundary"
        ),
        positive_probe="manifest SHA 일치, 최소 표본 1개에서 확인",
    ),
    GateDeclaration(
        gate_id="recorded_transfer_snapshot",
        owner=_READINESS,
        what_it_asserts="canonical fine-tune이 bootstrap transfer snapshot에 결속된 recorded 파일만 사용한다",
        negative_fixture=(
            f"{_NEG}::test_recorded_transfer_snapshot_gate_fails_without_transfer_snapshot"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_canonical_completion_verifies_selection_capability_marker_metrics_chain"
        ),
        positive_probe="bootstrap receipt와 transfer manifest의 64자리 SHA·inode snapshot이 일치",
    ),
    GateDeclaration(
        gate_id="recorded_selection_test_once_chain",
        owner=_READINESS,
        what_it_asserts="recorded-val 선택과 single-use test capability 체인이 완료 결과와 일치한다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::test_canonical_completion_verifies_selection_capability_marker_metrics_chain"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_canonical_completion_verifies_selection_capability_marker_metrics_chain"
        ),
        positive_probe="선택/발급/소비/완료 4개 marker의 SHA 체인 검증",
    ),
    GateDeclaration(
        gate_id="recorded_val_g4",
        owner=_READINESS,
        what_it_asserts="독립 recorded val 평가가 G4(최악 소스 계열 포함)를 통과했다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_g4_rejects_model_that_amplifies_one_source_family"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_completion_gates_pass_at_the_minimum_sample_boundary"
        ),
        positive_probe="최악 계열 −0.01 dB — 개선이라 말할 수 있는 최소값",
    ),
    GateDeclaration(
        gate_id="recorded_test_g4",
        owner=_READINESS,
        what_it_asserts="독립 recorded test 평가가 G4 를 통과했다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::test_g4_rejects_metrics_without_worst_source_fields"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_completion_gates_pass_at_the_minimum_sample_boundary"
        ),
        positive_probe="최악 옥타브 0.01 dB — 한계 바로 안쪽",
    ),
    # ---------------- 아티팩트 품질 게이트 (audit_official_path_model 내부) ----------------
    GateDeclaration(
        gate_id="official_path_sub_band_consistency",
        owner=_READINESS,
        what_it_asserts="필수 대역 안 **모든 부대역**의 반복 일관성이 하한을 넘는다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_weak_sub_band_is_rejected_even_when_the_total_passes"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_all_canonical_sub_bands_pass_at_the_required_boundary"
        ),
        positive_probe="canonical 4개 부대역이 official hard 하한 0.95에 정확히 붙어도 통과",
    ),
    GateDeclaration(
        gate_id="official_path_delay_spread",
        owner=_READINESS,
        what_it_asserts="P−S 상대 τ spread 를 **코드 상수**와 비교한다 (자기증명 차단)",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_artifact_cannot_declare_its_own_delay_jitter_allowance"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_official_path_delay_spread_passes_at_the_allowed_maximum"
        ),
        positive_probe="허용 최대 spread 3 샘플 정확히 (한 샘플 더 가면 FAIL)",
    ),
    GateDeclaration(
        gate_id="official_path_reanalysis_envelope",
        owner=_READINESS,
        what_it_asserts="재분석 아티팩트의 파라미터가 완화 방향이 아니다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::test_reanalysis_parameter_envelope_is_enforced"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_reanalysis_inside_the_envelope_is_accepted"
        ),
        positive_probe="봉투 안쪽 경계값(min_alignment_score 0.95)에서 통과",
    ),
    # ---------------- 측정·재분석 게이트 ----------------
    GateDeclaration(
        gate_id="measurement_relative_tau_outliers",
        owner=_PROBE,
        what_it_asserts="P−S 상대 τ 가 튀는 반복을 기각한다 (오염 과반에서도)",
        negative_fixture=(
            "tests/test_interleaved_probe.py::test_relative_tau_gate_rejects_the_measured_frame_slip"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_interleaved_probe.py::test_relative_tau_gate_survives_contamination_majority"
        ),
        positive_probe="실측 캡처 225546 의 오염 과반 상태에서도 정상 반복을 유지",
    ),
    GateDeclaration(
        gate_id="measurement_timebase_drift",
        owner=_PROBE,
        what_it_asserts="정상상태에 들지 못한 반복(국소 드리프트 이상치)을 기각한다",
        negative_fixture=(
            "tests/test_interleaved_probe.py::"
            "test_timebase_drift_flags_the_measured_warmup_transient"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_timebase_drift_accepts_the_steady_state_repeats_at_90_percent_of_the_tolerance"
        ),
        positive_probe="편차 1.8 = 허용 2.0 의 90%",
    ),
    GateDeclaration(
        gate_id="reanalysis_rejects_loosening",
        owner=_REANALYSE,
        what_it_asserts="게이트를 약화하는 재분석 인자를 거부한다 (강화는 허용)",
        negative_fixture="tests/test_reanalyse_paths.py::test_loosening_arguments_are_refused",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_reanalyse_paths.py::test_tightening_arguments_are_allowed"
        ),
        positive_probe="강화 방향 인자(0.99 / 0.02)는 그대로 받아들인다",
    ),
    GateDeclaration(
        gate_id="reanalysis_frame_slip",
        owner=_REANALYSE,
        what_it_asserts="주입된 프레임 슬립 반복을 정확히 기각하고 과반이면 실패 폐쇄한다",
        negative_fixture="tests/test_reanalyse_paths.py::test_majority_frame_slip_fails_closed",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_reanalyse_paths.py::test_clean_capture_round_trips_to_the_injected_plant"
        ),
        positive_probe="슬립 없는 캡처는 반복 정렬 점수 0.999 이상으로 왕복한다",
    ),
    GateDeclaration(
        gate_id="reanalysis_metadata_forgery",
        owner=_REANALYSE,
        what_it_asserts="metadata.json 과 NPZ 내부 사본이 다르면 거부한다",
        negative_fixture="tests/test_reanalyse_paths.py::test_metadata_forgery_is_detected",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_reanalysis_accepts_metadata_that_matches_the_npz_copy"
        ),
        positive_probe="metadata.json 과 NPZ 사본이 완전히 같은 정상 캡처 (xrun 0건)",
    ),
    # ---------------- 소스 풀 구성(prepare_noise_pool) 게이트 ----------------
    # ⚠ 2026-08-06 통합 검증: 이 게이트는 **선언 없이 만들어졌다.** 같은 변경이
    # "모든 게이트에 짝을 붙였다(72/72)"고 주장하면서 자기가 새로 만든 스크립트
    # 종료코드 게이트는 레지스트리에도 tests/ 에도 넣지 않았다. 스크립트 exit
    # 게이트가 대상이 아니라는 변명은 성립하지 않는다 — record_duct 3개,
    # reanalyse_paths 3개가 이미 같은 형태로 선언돼 있다.
    GateDeclaration(
        gate_id="noise_pool_declared_tags_exist",
        owner=_NOISE_POOL,
        what_it_asserts=(
            "source_mix_ratio 가 비율>0 으로 선언한 태그의 manifest 를 전부 만들지 "
            "못하면 종료코드 1 로 끝난다 (synth_dataset 의 조용한 합성원 폴백 차단)"
        ),
        negative_fixture=(
            "tests/test_prepare_noise_pool.py::"
            "test_missing_declared_tag_fails_the_build"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_prepare_noise_pool.py::"
            "test_every_declared_tag_present_builds_all_manifests"
        ),
        positive_probe="선언 태그 3종을 정확히 다 채운 최소 구성 (비율 0 인 태그 1종은 요구하지 않는다)",
    ),
    GateDeclaration(
        gate_id="noise_pool_recorded_holdout_required",
        owner=_NOISE_POOL,
        what_it_asserts=(
            "학습용 synthetic manifest를 쓰기 전에 recorded holdout 파일이 존재하고 "
            "최소 1개 클립을 포함하는지 확인한다"
        ),
        negative_fixture=(
            "tests/test_prepare_noise_pool.py::"
            "test_missing_holdout_fails_before_writing_any_manifest"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_prepare_noise_pool.py::"
            "test_every_declared_tag_present_builds_all_manifests"
        ),
        positive_probe="recorded holdout 1개 이상과 선언 태그 3종이 있는 최소 정상 구성",
    ),
    # ---------------- G4 판정 게이트 (eval/recorded.py) ----------------
    GateDeclaration(
        gate_id="g4_out_of_band_do_no_harm",
        owner="src/deep_anc/eval/recorded.py",
        what_it_asserts="신뢰 대역 밖 옥타브를 증폭하지 않는다 (절대목표 1)",
        negative_fixture="tests/test_recorded_eval.py::test_g4_rejects_out_of_band_amplifier",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_recorded_eval.py::test_metrics_markdown_and_npz_include_source_octave_and_worst10"
        ),
        positive_probe="옥타브 500/2000Hz 를 실제로 판정하는 정상 결과에서 PASS",
    ),
    GateDeclaration(
        gate_id="g4_statistical_power",
        owner="src/deep_anc/eval/recorded.py",
        what_it_asserts="표본이 부족하면 PASS 가 아니라 판정 불가를 반환한다 (D3)",
        negative_fixture=(
            "tests/test_recorded_eval.py::"
            "test_g4_is_inconclusive_when_a_family_has_too_few_groups"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_g4_statistical_power_passes_at_exactly_the_group_floor"
        ),
        positive_probe="계열당 그룹이 정확히 하한 4 일 때 판정 가능",
    ),
    GateDeclaration(
        gate_id="g4_cluster_bootstrap_ci",
        owner="src/deep_anc/eval/recorded.py",
        what_it_asserts="클러스터가 부족하면 CI 를 지어내지 않는다 (그룹 단위 재표집)",
        negative_fixture=(
            "tests/test_recorded_eval.py::"
            "test_cluster_bootstrap_ci_is_undefined_below_the_group_floor"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_recorded_eval.py::test_cluster_bootstrap_ci_is_wider_than_naive_segment_resampling"
        ),
        positive_probe="그룹 4개(하한) 에서 CI 가 정의되고 세그먼트 재표집보다 넓다",
    ),
    GateDeclaration(
        gate_id="g4_strict_trusted_subbands",
        owner="src/deep_anc/eval/recorded.py",
        what_it_asserts=(
            "150–1600Hz 평균이 좋아도 family별 1000–1600Hz를 포함한 네 strict "
            "부대역의 target(d=ERR) coverage·평균·최악10%·CI가 모두 통과해야 한다"
        ),
        negative_fixture=(
            "tests/test_recorded_eval.py::"
            "test_g4_rejects_aggregate_pass_when_upper_trusted_subband_amplifies"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_recorded_eval.py::"
            "test_metrics_markdown_and_npz_include_source_octave_and_worst10"
        ),
        positive_probe=(
            "150–300/300–600/600–1000/1000–1600Hz 각각에 FFT-bin 정렬 target과 "
            "family별 독립 group 4개를 둔 정상 결과"
        ),
    ),
    GateDeclaration(
        gate_id="metrics_plant_comparability",
        owner="src/deep_anc/eval/recorded.py",
        what_it_asserts="서로 다른 플랜트에서 나온 metrics 의 비교를 거부한다 (결함 5)",
        negative_fixture=(
            "tests/test_recorded_eval.py::test_metrics_comparison_rejects_different_plants"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_metrics_comparison_accepts_two_runs_from_the_same_plant"
        ),
        positive_probe=(
            "합성 fixture P 1602 / S 1462 / lead 116의 exact P/S SHA와 "
            "TrainingTimingContract 지문이 같을 때만 비교 허용; 현행 strict "
            "플랜트 수치 주장 아님"
        ),
    ),
    # ---------------- recorded QA 게이트 ----------------
    GateDeclaration(
        gate_id="recorded_qa_source_err_alignment",
        owner=_QA,
        what_it_asserts="source→ERR 결맞음과 지연 안정성이 살아 있다 (결함 2)",
        negative_fixture=(
            "tests/test_recorded_qa.py::test_qa_rejects_source_err_timebase_collapse"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_recorded_qa_delay_gates_pass_at_90_percent_of_the_robust_limits"
        ),
        positive_probe="robust-std 0.9 × 8.0 = 7.2 샘플까지 흔들어도 PASS",
    ),
    GateDeclaration(
        gate_id="recorded_qa_ref_err_control",
        owner=_QA,
        what_it_asserts="음향 대조군(REF→ERR)이 살아 있어 진단이 갈린다",
        negative_fixture=(
            "tests/test_recorded_qa.py::"
            "test_qa_rejects_a_dead_reference_mic_as_an_acoustic_problem"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_recorded_qa_gates_pass_at_the_minimum_length_and_full_coverage"
        ),
        positive_probe="정렬 대역 150–1600Hz 전 구간에서 REF→ERR 대조군 유지",
    ),
    GateDeclaration(
        gate_id="recorded_qa_delay_stability",
        owner=_QA,
        what_it_asserts="source→ERR 지연이 세션 안에서 떠다니지 않는다",
        negative_fixture=(
            "tests/test_recorded_qa.py::test_qa_rejects_a_drifting_source_err_delay"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_recorded_qa_delay_gates_pass_at_90_percent_of_the_robust_limits"
        ),
        positive_probe="p95−p5 0.9 × 48.0 = 43.2 샘플까지 정상으로 본다",
    ),
    # ---------------- 교차 도메인 불변식 (신설분) ----------------
    GateDeclaration(
        gate_id="invariant_corpus_disjoint",
        owner=_INVARIANTS,
        what_it_asserts="두 학습 스트림이 같은 원본 오디오를 쓰지 않는다 (D1)",
        negative_fixture=(
            "tests/test_invariants.py::test_corpus_disjoint_fails_on_the_music_leak"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_invariants.py::test_corpus_disjoint_passes_on_separate_corpora"
        ),
        positive_probe="서로 다른 코퍼스 3개에서 겹침 0건",
    ),
    GateDeclaration(
        gate_id="invariant_measured_delay_agreement",
        owner=_INVARIANTS,
        what_it_asserts="같은 지연을 두 방법으로 잰 값이 일치한다 (D2)",
        negative_fixture=(
            "tests/test_invariants.py::test_measured_delay_agreement_fails_on_the_d2_gap"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_invariants.py::test_measured_delay_agreement_passes_within_tolerance"
        ),
        positive_probe="두 방법의 차 18.0 샘플 = 허용치 안쪽 경계",
    ),
    GateDeclaration(
        gate_id="invariant_stream_delay_stability",
        owner=_INVARIANTS,
        what_it_asserts="재생→캡처 지연이 세션 내내 한 값이다",
        negative_fixture=(
            "tests/test_invariants.py::test_stream_delay_stability_fails_on_a_drifting_timebase"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_invariants.py::test_stream_delay_stability_passes_on_a_constant_delay"
        ),
        positive_probe="상수 지연 120 샘플, p95 기준 판정",
    ),
    GateDeclaration(
        gate_id="recorded_qa_group_split_leak",
        owner=_QA,
        what_it_asserts="같은 group_id 가 여러 split 에 걸치지 않는다",
        negative_fixture="tests/test_recorded_qa.py::test_group_split_leak_is_a_fatal_global_error",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_recorded_qa_gates_pass_at_the_minimum_length_and_full_coverage"
        ),
        positive_probe="group_id 12개가 split 을 넘지 않는 정상 manifest",
    ),
    GateDeclaration(
        gate_id="recorded_qa_audio_shape_and_length",
        owner=_QA,
        what_it_asserts="채널 수·샘플레이트·lead 를 고려한 최소 길이를 만족한다",
        negative_fixture=(
            "tests/test_recorded_qa.py::test_audio_shape_rate_and_lead_aware_minimum_length_fail"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_recorded_qa_gates_pass_at_the_minimum_length_and_full_coverage"
        ),
        positive_probe=(
            "합성 fixture TrainingTimingContract의 segment 48000 + lead 116 + 1 = "
            "48117 최소 길이를 정확히 만족; 현행 strict 수치 주장 아님"
        ),
    ),
    GateDeclaration(
        gate_id="recorded_qa_level_and_metadata",
        owner=_QA,
        what_it_asserts="RMS/클리핑/메타데이터 일치가 기준을 만족한다",
        negative_fixture="tests/test_recorded_qa.py::test_nonfinite_rms_clip_and_metadata_mismatch_fail",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_recorded_qa_level_and_clip_gates_pass_at_90_percent_of_their_limits"
        ),
        positive_probe="클립 비율 0.0045 = 한계 0.005 의 90%, RMS −72 dBFS",
    ),
    GateDeclaration(
        gate_id="recorded_qa_family_coverage",
        owner=_QA,
        what_it_asserts="모든 source_family 가 필수 split 을 덮는다",
        negative_fixture=(
            "tests/test_recorded_qa.py::"
            "test_family_must_cover_all_required_splits_unless_diagnostic_override"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_recorded_qa_gates_pass_at_the_minimum_length_and_full_coverage"
        ),
        positive_probe="계열 4종 × split 3종을 정확히 채운 최소 커버리지",
    ),
    # ---------------- 수집(record_duct) 게이트 — 재생 전/저장 전 판정 ----------------
    GateDeclaration(
        gate_id="recording_input_rail_preflight",
        owner=_RECORD,
        what_it_asserts="마이크 입력이 풀스케일에 붙어 있으면 재생 전에 멈춘다",
        negative_fixture=(
            "tests/test_record_duct_gates.py::"
            "test_input_rail_gate_rejects_the_measured_railing_microphones"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_record_duct_gates.py::test_input_rail_gate_passes_a_healthy_quiet_room"
        ),
        positive_probe="정상 정숙 실내 peak 0.006 수준에서 통과",
    ),
    GateDeclaration(
        gate_id="recording_timeline_fail_closed",
        owner=_RECORD,
        what_it_asserts="재정렬이 성공하지 못한 세션은 저장하지 않는다",
        negative_fixture=(
            "tests/test_record_duct_gates.py::"
            "test_timeline_gate_rejects_the_shipped_corpus_numbers"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_record_duct_gates.py::test_timeline_gate_accepts_a_recovered_session"
        ),
        positive_probe="실측 재정렬 결과 coh² 0.952 / 유효창 0.981 (하한 0.90 바로 위)",
    ),
    GateDeclaration(
        gate_id="recording_gate_tightening_only",
        owner=_RECORD,
        what_it_asserts="CLI 로 수집 게이트를 완화할 수 없다 (강화만 허용)",
        negative_fixture="tests/test_record_duct_gates.py::test_cli_refuses_to_loosen_the_gates",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_recording_gate_cli_accepts_the_default_and_tightened_values"
        ),
        positive_probe="기본값 0.90 그대로와 강화값 0.99 둘 다 거부되지 않는다",
    ),
    # ---------------- 시간축 재정렬 게이트 ----------------
    GateDeclaration(
        gate_id="timeline_warp_holdout_validation",
        owner=_TIMELINE,
        what_it_asserts="워프 검증은 추정에 쓰지 않은 채널(ERR)로만 한다",
        negative_fixture=(
            "tests/test_recorded_timeline.py::"
            "test_collapsed_timebase_is_not_silently_repaired"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_recorded_timeline.py::test_known_time_warp_is_recovered_and_coherence_is_restored"
        ),
        positive_probe="진폭 134 샘플의 알려진 워프를 넣고 coh² 0.9 이상으로 복구",
    ),
    GateDeclaration(
        gate_id="timeline_valid_window_ratio",
        owner=_TIMELINE,
        what_it_asserts="지연을 추정할 수 없었던 창을 유효창으로 세지 않는다",
        negative_fixture=(
            "tests/test_recorded_timeline.py::"
            "test_silent_source_is_detected_and_its_windows_are_rejected"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_timeline_valid_window_ratio_passes_just_above_the_floor"
        ),
        positive_probe="유효창 비율 0.9 하한 바로 위에서 추적을 인정",
    ),
    # ---------------- 실측 데이터셋 게이트 ----------------
    GateDeclaration(
        gate_id="recorded_dataset_require_aligned_source",
        owner=_RECORDED_DATASET,
        what_it_asserts="재정렬본을 요구하면 원본 source.wav 로 조용히 폴백하지 않는다",
        negative_fixture=(
            "tests/test_recorded_dataset_augment.py::test_require_aligned_source_fails_closed"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_recorded_dataset_augment.py::test_aligned_source_is_preferred_over_the_raw_playback_array"
        ),
        positive_probe="재정렬본이 있는 정상 세션 1개에서 그것을 읽는다 (32 샘플 지연)",
    ),
    GateDeclaration(
        gate_id="recorded_lead_single_derivation",
        owner=_RECORDED_DATASET,
        what_it_asserts="실측 lead 는 측정 지연에서만 유도되고 손으로 쓸 수 없다",
        negative_fixture=(
            "tests/test_recorded_dataset_augment.py::"
            "test_lead_cannot_be_hand_written_against_the_derivation"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_recorded_dataset_augment.py::test_timeline_lead_mode_derives_the_lead_from_the_session"
        ),
        positive_probe=(
            "합성 fixture의 세션 지연 1602에서 lead 116을 timeline과 "
            "TrainingTimingContract로 유도; 현행 strict 숫자는 NPZ가 단일 출처"
        ),
    ),
    GateDeclaration(
        gate_id="recorded_augment_plant_commutes",
        owner=_RECORDED_DATASET,
        what_it_asserts="증강은 LTI 플랜트와 교환 가능해야 한다 (양쪽에 같은 연산)",
        negative_fixture=(
            "tests/test_recorded_dataset_augment.py::"
            "test_mic_noise_goes_to_the_input_only_never_to_the_target"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_recorded_dataset_augment.py::test_augmentation_preserves_the_plant_relation_end_to_end"
        ),
        positive_probe="레벨·극성·EQ 를 전부 건 상태에서 플랜트 관계 유지 (허용 0.02)",
    ),
    # ---------------- 런타임 게이트 ----------------
    GateDeclaration(
        gate_id="runtime_digital_reference_lead",
        owner=_RUNTIME,
        what_it_asserts="런타임 lead 가 checkpoint lead 와 정확히 같다",
        negative_fixture=f"{_NEG}::test_runtime_lead_gate_fails_when_checkpoint_disagrees",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_runtime_lead_gate_accepts_the_measured_lead_115_exactly"
        ),
        positive_probe="현재 strict lead 115 정확 일치 (여유 0, 1 샘플 어긋나면 거부)",
    ),
    GateDeclaration(
        gate_id="runtime_strict_plant_contract",
        owner="src/deep_anc/realtime/plant_contract.py",
        what_it_asserts="digital-reference DL runtime이 strict P/S·raw provenance에서 유도한 lead와 일치한다",
        negative_fixture=(
            "tests/test_runtime_plant_contract.py::"
            "test_runtime_strict_plant_contract_rejects_the_legacy_109_sample_lead"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_runtime_plant_contract.py::"
            "test_runtime_strict_plant_contract_accepts_the_actual_115_sample_contract"
        ),
        positive_probe="same-capture P=1386/S=1245/handoff=256 → lead 115, raw/analysis/level SHA까지 대조",
    ),
    GateDeclaration(
        gate_id="runtime_engine_artifact_preflight",
        owner=_RUNTIME,
        what_it_asserts="engine 블록이 가리키는 모든 파일이 시작 전에 실제로 존재한다",
        negative_fixture=(
            "tests/test_realtime_start.py::"
            "test_engine_preflight_rejects_a_missing_artifact_even_when_unused"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_realtime_start.py::test_engine_preflight_accepts_the_shipped_runtime_configs"
        ),
        positive_probe="출하 runtime 설정 2종이 가리키는 파일이 전부 실존",
    ),
    GateDeclaration(
        gate_id="runtime_input_preflight",
        owner="src/deep_anc/audio_io.py",
        what_it_asserts="죽은/고정된 에러 마이크 채널로는 시작하지 않는다",
        negative_fixture=(
            "tests/test_realtime_start.py::test_runtime_input_preflight_rejects_stuck_error_channel"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_runtime_input_preflight_accepts_a_quiet_room_at_90_percent_of_its_limits"
        ),
        positive_probe="RMS −72 dBFS, 클립 0.0045 = 한계 0.005 의 90%",
    ),
    GateDeclaration(
        gate_id="runtime_fxlms_adaptation_fail_closed",
        owner="src/deep_anc/realtime/engines.py",
        what_it_asserts="FxLMS 적응은 명시적으로 켜기 전에는 꺼져 있다",
        negative_fixture="tests/test_realtime_start.py::test_fxlms_adaptation_gate_is_fail_closed",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_fxlms_adaptation_gate_accepts_the_one_fully_safe_combination"
        ),
        positive_probe="8축 안전 조건이 전부 만족될 때는 적응이 허용된다",
    ),
    # ---------------- 실시간 워치독 (2026-08-06 실시간 감사) ----------------
    # 이 7개는 safety.WatchdogId 열거와 1:1 이다. tests/test_realtime_safety.py 의
    # test_every_watchdog_is_declared_as_a_gate 가 그 대응을 강제하므로 워치독을
    # 추가하면서 선언을 빠뜨리면 테스트가 실패한다. "감시 중" 이라는 주장은
    # **그것을 발동시킨 시나리오**로만 증명된다 — S1 은 그 증명이 없어 죽어 있었다.
    GateDeclaration(
        gate_id="runtime_nonfinite_output",
        owner=_SAFETY,
        what_it_asserts="엔진이 NaN/Inf 를 내면 보고하고 ANC 를 끈다 (조용히 삼키지 않는다)",
        negative_fixture=f"{_RTSAFE}::test_nonfinite_output_watchdog_detects_nan_instead_of_swallowing_it",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_every_watchdog_stays_silent_across_the_operating_band_at_90_percent_of_its_limit"
        ),
        positive_probe="80–1600Hz 정상 출력 3초 × 위상 8종에서 NaN 오검출 0",
    ),
    GateDeclaration(
        gate_id="runtime_output_dc",
        owner=_SAFETY,
        what_it_asserts="상쇄 출력의 지속 DC 를 제거하고 ANC 를 끈다 (보이스코일 보호)",
        negative_fixture=f"{_RTSAFE}::test_output_dc_watchdog_detects_a_sustained_offset",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_every_watchdog_stays_silent_across_the_operating_band_at_90_percent_of_its_limit"
        ),
        positive_probe="80Hz(블록당 0.43주기) 포함 목표 대역 전체, 위상 8종",
    ),
    GateDeclaration(
        gate_id="runtime_output_saturation",
        owner=_SAFETY,
        what_it_asserts="tanh 리미터에 눌린 출력을 검출한다 (리미터와 다른 값을 잰다)",
        negative_fixture=f"{_RTSAFE}::test_saturation_watchdog_detects_output_the_old_clip_counter_declared_clean",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_every_watchdog_stays_silent_across_the_operating_band_at_90_percent_of_its_limit"
        ),
        positive_probe="출력 RMS 를 한계 0.12 의 90% 로 몰아도 포화 오검출 0",
    ),
    GateDeclaration(
        gate_id="runtime_output_rms",
        owner=_SAFETY,
        what_it_asserts="상쇄 출력 RMS 상한을 지속 초과하면 ANC 를 끈다",
        negative_fixture=f"{_RTSAFE}::test_output_rms_watchdog_detects_sustained_over_power",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_every_watchdog_stays_silent_across_the_operating_band_at_90_percent_of_its_limit"
        ),
        positive_probe="RMS = 0.9 × 0.12 = 한계의 90%",
    ),
    GateDeclaration(
        gate_id="runtime_deadline_miss_rate",
        owner=_SAFETY,
        what_it_asserts="최근 1초 미스율이 한계를 넘으면 ANC 를 끈다 (교대 미스 포함)",
        negative_fixture=f"{_RTSAFE}::test_deadline_watchdog_detects_the_alternating_miss_the_streak_never_caught",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_every_watchdog_stays_silent_across_the_operating_band_at_90_percent_of_its_limit"
        ),
        positive_probe="미스율 18% = 한계 0.2 의 90%",
    ),
    GateDeclaration(
        gate_id="runtime_divergence",
        owner=_SAFETY,
        what_it_asserts="발산을 검출하고, 베이스라인이 없어 판정 불가면 fail-closed 로 끈다",
        negative_fixture=f"{_RTSAFE}::test_divergence_watchdog_detects_the_missing_baseline_instead_of_going_quiet",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_every_watchdog_stays_silent_across_the_operating_band_at_90_percent_of_its_limit"
        ),
        positive_probe="발산비 3.6 = 한계 4.0 의 90%",
    ),
    GateDeclaration(
        gate_id="runtime_handoff_backlog",
        owner=_SAFETY,
        what_it_asserts="입력 백로그가 1 hop 을 넘어 오래된 입력을 버리면 ANC 를 끈다",
        negative_fixture=f"{_RTSAFE}::test_handoff_backlog_watchdog_detects_dropped_stale_input",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_every_watchdog_stays_silent_across_the_operating_band_at_90_percent_of_its_limit"
        ),
        positive_probe="백로그 드롭율 18% = 한계 0.2 의 90%",
    ),
    GateDeclaration(
        gate_id="runtime_pipeline_handoff_budget",
        owner=_SAFETY,
        what_it_asserts="입력/출력 백로그 예산이 대칭이고 실효 핸드오프가 학습 가정과 같다",
        negative_fixture=f"{_RTSAFE}::test_handoff_budget_rejects_the_input_output_asymmetry",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::test_handoff_budget_accepts_the_shipped_256_sample_hop"
        ),
        positive_probe="출하 duct.yaml handoff 256 에서 대칭 1 hop 정확히",
    ),
    # ---------------- 교차 도메인 불변식 (발생기 A') ----------------
    GateDeclaration(
        gate_id="invariant_relative_tau_constancy",
        owner=_INVARIANTS,
        what_it_asserts="같은 출력 스트림의 두 채널은 상대 지연이 상수다",
        negative_fixture=(
            "tests/test_invariants.py::test_relative_tau_constancy_fails_on_the_measured_frame_slip"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_invariants.py::test_relative_tau_constancy_passes_on_the_clean_subset"
        ),
        positive_probe="오염 전 11 반복, 허용 3.0 샘플 안",
    ),
    GateDeclaration(
        gate_id="invariant_stream_coherence",
        owner=_INVARIANTS,
        what_it_asserts="재생 신호와 캡처 신호의 대응(coh²)이 살아 있다",
        negative_fixture=(
            "tests/test_invariants.py::test_stream_coherence_fails_on_a_collapsed_timebase"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_invariants.py::test_stream_coherence_passes_on_a_delayed_copy"
        ),
        positive_probe="coh² 하한 0.60 에 대해 실측 정상 0.96 수준",
    ),
    GateDeclaration(
        gate_id="invariant_plant_fingerprint_match",
        owner=_INVARIANTS,
        what_it_asserts="비교되는 두 메트릭이 같은 플랜트에서 나왔다",
        negative_fixture=(
            "tests/test_invariants.py::test_plant_fingerprint_match_fails_across_the_20260804_plants"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_invariants.py::test_plant_fingerprint_match_passes_for_the_same_plant"
        ),
        positive_probe=(
            "합성 fixture P 1602 / S 1462의 exact SHA와 timing digest가 같은 두 "
            "지문; 현행 strict 플랜트 수치 주장 아님"
        ),
    ),
    GateDeclaration(
        gate_id="invariant_lead_agreement",
        owner=_INVARIANTS,
        what_it_asserts="설정 lead 가 측정 지연에서 유도되는 값과 같다",
        negative_fixture=(
            "tests/test_invariants.py::test_lead_agreement_fails_on_the_aaeef41_mismatch"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_invariants.py::test_lead_agreement_passes_on_the_measured_plant"
        ),
        positive_probe=(
            "합성 fixture P−S 140과 handoff에서 PlantDelays.lead()=116을 유도; "
            "현행 값은 strict NPZ와 handoff에서만 유도"
        ),
    ),
    # ---------------- 손실 (절대목표 1·2 를 손실 안에서 강제) ----------------
    GateDeclaration(
        gate_id="loss_do_no_harm_band_overlap",
        owner=_LOSS_CFG,
        what_it_asserts=(
            "대역 밖 단측 힌지가 개선을 요구하는 대역과 겹치지 않는다 — 겹치면 두 항이"
            " 서로 상쇄하고 지표로는 보이지 않는다"
        ),
        negative_fixture=(
            f"{_LOSS_TESTS}::test_do_no_harm_band_overlapping_the_trusted_band_is_rejected"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_anc_loss.py::test_do_no_harm_bands_are_derived_by_subtracting_the_trusted_band"
        ),
        positive_probe="신뢰대역 150–1600Hz를 뺀 여집합이 빈틈 없이 덮인다",
    ),
    GateDeclaration(
        gate_id="loss_do_no_harm_margin_matches_the_gate",
        owner=_LOSS_CFG,
        what_it_asserts=(
            "대역 밖 힌지 마진이 G4 옥타브 임계에서 유도된 상한을 넘지 않는다 — 넘으면"
            " 손실을 정확히 만족한 모델이 게이트를 FAIL 한다 (실측 8.5 dB 차이)"
        ),
        negative_fixture=(
            "tests/test_do_no_harm_contract.py::"
            "test_config_rejects_a_margin_looser_than_the_gate_allows"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_do_no_harm_contract.py::"
            "test_a_model_that_exactly_satisfies_the_hinge_passes_the_g4_gate"
        ),
        positive_probe="유도 마진 −18.27 dB 를 정확히 만족한 최악 신호가 전 옥타브에서 통과",
    ),
    GateDeclaration(
        gate_id="loss_do_no_harm_bands_align_to_octaves",
        owner=_LOSS_CFG,
        what_it_asserts=(
            "대역 밖 힌지 대역이 G4 옥타브 경계를 가로지르지 않는다 — 가로지르면 대역"
            " 비율을 만족한 채 한 옥타브에 에너지를 몰아넣을 수 있다 (실측 3.1 dB 손해)"
        ),
        negative_fixture=(
            "tests/test_do_no_harm_contract.py::test_config_rejects_bands_that_cross_an_octave_edge"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_do_no_harm_contract.py::test_energy_concentrated_in_one_octave_still_passes"
        ),
        positive_probe=(
            "옥타브 정렬 후 2000 Hz 옥타브에 에너지를 전부 몰아넣어도 최악 −1.00 dB "
            "(정렬 전에는 −12.63 dB 로 실패했다)"
        ),
    ),
    GateDeclaration(
        gate_id="plant_confidence_ceiling_is_recomputed",
        owner=_READINESS,
        what_it_asserts=(
            "선언된 설계 상한이 실측 P/S 아티팩트에서 다시 푼 값보다 낙관적이지 않다 —"
            " 설정에 적힌 숫자를 그대로 믿으면 물리적으로 불가능한 목표도 통과한다"
        ),
        negative_fixture=(
            "tests/test_design_ceiling.py::"
            "test_a_numerically_exploding_solution_is_not_called_stable"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_design_ceiling.py::test_shipped_declaration_matches_the_recomputation"
        ),
        positive_probe=(
            "출하 선언 2.15 dB가 현행 strict P/S 최악 옥타브 재계산 17.28 dB보다 "
            "보수적이다 (허용 오차 0.5 dB); current 재계산보다 낙관적인 선언은 거부된다"
        ),
    ),
    GateDeclaration(
        gate_id="recorded_high_band_alignment",
        owner=_QA,
        what_it_asserts=(
            "source→ERR 코히런스가 **고역(600–1600Hz)에서도** 하한을 만족한다 —"
            " 절대목표 1의 나머지 절반이고, 2026-08-06 이전에는 보는 게이트가 0개였다"
        ),
        negative_fixture=(
            "tests/test_recorded_qa.py::test_high_band_alignment_failure_is_rejected"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_gate_positive_fixtures.py::"
            "test_the_delay_gate_now_implies_the_coherence_gate"
        ),
        positive_probe=(
            "지터 상한 3.41 샘플의 94% 까지 민 세션에서 고역 coh² 가 하한 0.60 위에 남는다"
        ),
    ),
    GateDeclaration(
        gate_id="recorded_source_pool_agrees_with_sessions",
        owner=_READINESS,
        what_it_asserts=(
            "설정이 선언한 실측 소스풀과 세션이 실제로 재생한 풀이 같다 — 다르면 누수"
            " 게이트가 엉뚱한 클립끼리 비교해 PASS 하면서 누수를 통과시킨다"
        ),
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_a_pool_the_sessions_did_not_actually_play"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_recorded_source_pool.py::"
            "test_observed_pool_is_read_from_what_the_session_played"
        ),
        positive_probe="세션 2개가 전부 v2 를 재생했을 때 관측이 v2 1종만 보고한다",
    ),
    # ---------------- 합성 소스 분포 (절대목표 2 — 모든 소리를 본다) ----------------
    GateDeclaration(
        gate_id="synth_declared_source_manifests_exist",
        owner=_SYNTH_DATASET,
        what_it_asserts=(
            "source_mix_ratio 가 선언한 모든 태그의 manifest 가 구성 시점에 존재한다 —"
            " 없으면 학습기가 조용히 합성원으로 대체해 선언과 다른 분포로 돈다"
        ),
        negative_fixture=(
            "tests/test_synth_source_manifests.py::"
            "test_missing_declared_source_manifest_is_rejected"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_synth_source_manifests.py::"
            "test_all_declared_manifests_present_passes"
        ),
        positive_probe=(
            "선언 3태그(speech 0.15 · music 0.10 · esc50 0.05)의 manifest 가 전부 있으면 "
            "통과하고 풀을 건드리지 않는다 — 게이트가 꺼져서 통과하는 것이 아니다"
        ),
    ),
    GateDeclaration(
        gate_id="loss_config_schema",
        owner=_LOSS_CFG,
        what_it_asserts="loss 블록에 모르는 키가 없고 값 범위가 유효하다",
        negative_fixture=f"{_LOSS_TESTS}::test_unknown_loss_key_is_rejected",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_anc_loss.py::test_shipped_training_configs_pass_the_loss_schema"
        ),
        positive_probe="출하 학습 설정 3개가 전부 스키마를 통과 (미지 키 0건)",
    ),
    GateDeclaration(
        gate_id="loss_limiter_limit_single_source",
        owner=_LOSS,
        what_it_asserts="리미터 한계를 모델과 설정이 따로 정하지 않는다",
        negative_fixture=f"{_LOSS_TESTS}::test_limiter_limit_disagreement_is_rejected",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_anc_loss.py::test_saturation_penalty_replaces_the_structurally_dead_clip_term"
        ),
        positive_probe="리미터 한계 0.2, sat_margin 2.0 의 정상 구성에서 통과",
    ),
    GateDeclaration(
        gate_id="timing_single_source_of_plant_settle",
        owner=_TIMING,
        what_it_asserts="정착 구간은 PlantSettle.derive() 밖에서 만들 수 없다",
        negative_fixture=f"{_START_TESTS}::test_plant_settle_cannot_be_constructed_by_hand",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_loss_start_sample.py::test_plant_settle_is_derived_from_the_measured_plant"
        ),
        positive_probe=(
            "합성 fixture S 1462 + handoff 256 + FIR 2048에서 "
            "PlantSettle.derive()로만 유도; 현행 S 지연은 strict NPZ가 단일 출처"
        ),
    ),
    GateDeclaration(
        gate_id="evaluation_warmup_floor",
        owner=_EVAL_RECORDED,
        what_it_asserts=(
            "평가 warmup 이 플랜트 정착 구간 아래로 내려가지 않는다 — 그 구간은 e = d 라"
            " 상쇄량을 잴 수 없다"
        ),
        negative_fixture=(
            f"{_START_TESTS}::test_evaluation_warmup_cannot_drop_below_the_plant_settle_window"
        ),
        discoverable_id=False,
        positive_fixture=(
            "tests/test_loss_start_sample.py::test_warmup_floor_does_not_shrink_a_longer_requested_warmup"
        ),
        positive_probe=(
            "합성 fixture 요청 warmup 12000 > PlantSettle 유도 하한 3769이면 "
            "요청값 유지; 현행 strict 수치 주장 아님"
        ),
    ),
    GateDeclaration(
        gate_id="timing_single_source_of_lead",
        owner=_TIMING,
        what_it_asserts="lead 는 PlantDelays.lead() 밖에서 만들 수 없다",
        negative_fixture="tests/test_timing.py::test_lead_cannot_be_constructed_by_hand",
        discoverable_id=False,
        positive_fixture=(
            "tests/test_timing.py::test_lead_is_derived_from_measured_delays"
        ),
        positive_probe="실측 P−S 139~141 범위에서 lead 를 유도",
    ),
)


def declared_gate_ids() -> frozenset[str]:
    return frozenset(item.gate_id for item in GATES)


def gate(gate_id: str) -> GateDeclaration:
    for item in GATES:
        if item.gate_id == gate_id:
            return item
    raise KeyError(
        f"선언되지 않은 게이트: {gate_id!r} — 게이트를 만들었다면 "
        "그것을 FAIL 시키는 fixture 와 함께 gate_registry.GATES 에 선언하세요"
    )


def gates_for_owner(owner: str) -> tuple[GateDeclaration, ...]:
    return tuple(item for item in GATES if item.owner == owner)


class _AuditGateIdVisitor(ast.NodeVisitor):
    """audit call의 literal과 단순 문자열 바인딩을 scope별로 찾는다."""

    def __init__(self) -> None:
        self.found: set[str] = set()
        self._scopes: list[dict[str, str | None]] = []

    @staticmethod
    def _scope_bindings(body: list[ast.stmt]) -> dict[str, str | None]:
        class _BindingCollector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.bindings: dict[str, str | None] = {}

            def _bind(self, name: str, value: ast.expr | None) -> None:
                literal = (
                    value.value
                    if isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and _GATE_ID.fullmatch(value.value)
                    else None
                )
                previous = self.bindings.get(name, literal)
                self.bindings[name] = literal if previous == literal else None

            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._bind(target.id, node.value)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if isinstance(node.target, ast.Name):
                    self._bind(node.target.id, node.value)

            # 중첩 scope의 바인딩은 바깥 scope에 섞지 않는다.
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

        collector = _BindingCollector()
        for statement in body:
            collector.visit(statement)
        return collector.bindings

    def _visit_scope(self, body: list[ast.stmt]) -> None:
        self._scopes.append(self._scope_bindings(body))
        try:
            for statement in body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def visit_Module(self, node: ast.Module) -> None:
        self._visit_scope(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_scope(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_scope(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node.body)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        is_audit_call = (
            isinstance(function, ast.Attribute)
            and function.attr in {"pass_", "fail"}
            and isinstance(function.value, ast.Name)
            and function.value.id == "audit"
        )
        if is_audit_call and node.args:
            argument = node.args[0]
            gate_id: str | None = None
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and _GATE_ID.fullmatch(argument.value)
            ):
                gate_id = argument.value
            elif isinstance(argument, ast.Name):
                for scope in reversed(self._scopes):
                    if argument.id in scope:
                        gate_id = scope[argument.id]
                        break
            if gate_id is not None:
                self.found.add(gate_id)
        self.generic_visit(node)


def discover_audit_gate_ids(source_path: str | Path) -> frozenset[str]:
    """AST에서 ``audit.pass_``/``audit.fail`` 의 정적 gate id를 찾는다.

    문자열 literal뿐 아니라 같은 lexical scope의 단순 문자열 변수도 해석한다. f-string
    또는 런타임 매개변수로 만들어지는 id는 선언에서 ``discoverable_id=False``로 표시한다.
    """

    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    visitor = _AuditGateIdVisitor()
    visitor.visit(tree)
    return frozenset(visitor.found)
