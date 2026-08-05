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

게이트를 새로 만들 때
--------------------
1. 게이트를 구현한다.
2. **그것을 FAIL 시키는 테스트를 쓴다.**
3. 여기에 :class:`GateDeclaration` 을 추가한다.
2를 건너뛰면 3을 쓸 수 없고, 3을 건너뛰면 메타 테스트가 실패한다.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "GATES",
    "GateDeclaration",
    "declared_gate_ids",
    "discover_audit_gate_ids",
    "gate",
    "gates_for_owner",
]


_NODE_ID = re.compile(r"^tests/[A-Za-z0-9_]+\.py::[A-Za-z0-9_]+(\[[^\]]+\])?$")
_GATE_ID = re.compile(r"^[a-z][a-z0-9_]*$")


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

    positive_fixture: str | None = None
    """정상 입력이 통과하는 것을 보이는 노드 id (있으면). 오기각 방지용."""

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
        if self.positive_fixture is not None and not _NODE_ID.match(self.positive_fixture):
            raise ValueError(
                f"{self.gate_id}: positive_fixture 형식 오류: {self.positive_fixture!r}"
            )
        return self


_READINESS = "src/deep_anc/train/finetune_readiness.py"
_QA = "src/deep_anc/data/recorded_qa.py"
_PROBE = "src/deep_anc/dsp/interleaved_probe.py"
_INVARIANTS = "src/deep_anc/dsp/invariants.py"
_TIMING = "src/deep_anc/dsp/timing.py"
_REANALYSE = "scripts/data/reanalyse_paths_interleaved.py"
_RUNTIME = "src/deep_anc/realtime/run_realtime.py"
_SAFETY = "src/deep_anc/realtime/safety.py"
_RECORD = "scripts/data/record_duct.py"
_TIMELINE = "src/deep_anc/data/timeline.py"
_RECORDED_DATASET = "src/deep_anc/data/recorded_dataset.py"

_LOSS = "src/deep_anc/losses/anc_loss.py"
_LOSS_CFG = "src/deep_anc/losses/config.py"
_EVAL_RECORDED = "src/deep_anc/eval/recorded.py"

_NEG = "tests/test_gate_negative_fixtures.py"
_RTSAFE = "tests/test_realtime_safety.py"
_LOSS_TESTS = "tests/test_anc_loss.py"
_START_TESTS = "tests/test_loss_start_sample.py"


GATES: tuple[GateDeclaration, ...] = (
    # ---------------- 파인튜닝 진입 게이트 (audit_finetune_readiness) ----------------
    GateDeclaration(
        gate_id="config_fail_closed_flags",
        owner=_READINESS,
        what_it_asserts="필수 fail-closed 설정 3종이 켜져 있다",
        negative_fixture=f"{_NEG}::test_config_fail_closed_flags_gate_fails_when_a_flag_is_off",
        positive_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_passes_only_with_official_paths_completed_init_and_full_recorded_qa"
        ),
    ),
    GateDeclaration(
        gate_id="measured_primary_mode",
        owner=_READINESS,
        what_it_asserts="digital-ref 파인튜닝이 실측 P(z) 모드로 돈다",
        negative_fixture=f"{_NEG}::test_measured_primary_mode_gate_fails_on_surrogate",
    ),
    GateDeclaration(
        gate_id="recorded_mix_ratio",
        owner=_READINESS,
        what_it_asserts="실측 데이터 혼합비가 요구 범위 안이다",
        negative_fixture=f"{_NEG}::test_recorded_mix_ratio_gate_fails_when_recorded_share_is_too_small",
    ),
    GateDeclaration(
        gate_id="official_secondary_path",
        owner=_READINESS,
        what_it_asserts="S(z) 아티팩트가 official 품질 메타데이터와 일관성 기준을 만족한다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_timing_invalid_or_legacy_path_metadata"
        ),
    ),
    GateDeclaration(
        gate_id="official_primary_path",
        owner=_READINESS,
        what_it_asserts="P(z) 아티팩트가 official 품질 기준과 출력 채널 규약을 만족한다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_official_path_gate_rejects_wrong_channel_and_low_consistency"
        ),
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
            "tests/test_finetune_readiness.py::"
            "test_interleaved_pair_from_one_capture_passes_matched_conditions"
        ),
    ),
    GateDeclaration(
        gate_id="path_delay_and_lead",
        owner=_READINESS,
        what_it_asserts="설정 lead 가 측정 P/S 지연에서 유도되는 값과 정확히 같다",
        negative_fixture=f"{_NEG}::test_path_delay_and_lead_gate_fails_when_lead_is_hand_written",
    ),
    GateDeclaration(
        gate_id="completed_init_checkpoint",
        owner=_READINESS,
        what_it_asserts="init checkpoint 가 완주했고 lead·물리 모드가 허용 범위다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::test_init_lead_mismatch_is_rejected_by_default"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_init_lead_mismatch_within_declared_tolerance_passes"
        ),
    ),
    GateDeclaration(
        gate_id="recorded_dataset_qa",
        owner=_READINESS,
        what_it_asserts="실측 manifest 의 모든 세션이 QA 를 통과하고 커버리지가 충족된다",
        negative_fixture=f"{_NEG}::test_recorded_dataset_qa_gate_fails_on_a_broken_session",
    ),
    GateDeclaration(
        gate_id="readiness",
        owner=_READINESS,
        what_it_asserts="진입 게이트 전부가 통과했다 (집계)",
        negative_fixture=f"{_NEG}::test_readiness_aggregate_gate_fails_when_any_child_fails",
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
            "tests/test_finetune_readiness.py::"
            "test_readiness_passes_only_with_official_paths_completed_init_and_full_recorded_qa"
        ),
    ),
    GateDeclaration(
        gate_id="recorded_statistical_power",
        owner=_READINESS,
        what_it_asserts="val/test 의 계열당 그룹이 CI 를 정의할 수 있는 하한 이상이다 (D3)",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_underpowered_val_and_test_groups"
        ),
    ),
    GateDeclaration(
        gate_id="corpus_disjoint",
        owner=_READINESS,
        what_it_asserts="합성 학습 스트림과 실측이 같은 원본 오디오를 쓰지 않는다 (D1)",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_corpus_leak_between_synthetic_and_recorded"
        ),
    ),
    GateDeclaration(
        gate_id="measured_source_delay_agreement",
        owner=_READINESS,
        what_it_asserts="실측 세션의 source→ERR 지연이 P(z) 유도값과 일치한다 (D2)",
        negative_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_readiness_rejects_recorded_delay_disagreeing_with_the_primary_path"
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
            "tests/test_finetune_readiness.py::"
            "test_achievable_ceiling_matches_the_measured_plants"
        ),
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
            "tests/test_finetune_readiness.py::"
            "test_completion_accepts_val_and_test_from_the_same_plant"
        ),
    ),
    # ---------------- 파인튜닝 완료 게이트 (audit_finetune_completion) ----------------
    GateDeclaration(
        gate_id="measured_finetune_checkpoint",
        owner=_READINESS,
        what_it_asserts="완료 판정 대상 checkpoint 가 measured 물리로 학습된 것이다",
        negative_fixture=f"{_NEG}::test_measured_finetune_checkpoint_gate_fails_on_surrogate_physics",
    ),
    GateDeclaration(
        gate_id="recorded_manifest_provenance",
        owner=_READINESS,
        what_it_asserts="완료 판정이 읽는 manifest 가 학습에 쓴 것과 같은 파일이다",
        negative_fixture=f"{_NEG}::test_recorded_manifest_provenance_gate_fails_without_manifest",
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
    ),
    GateDeclaration(
        gate_id="recorded_test_g4",
        owner=_READINESS,
        what_it_asserts="독립 recorded test 평가가 G4 를 통과했다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::test_g4_rejects_metrics_without_worst_source_fields"
        ),
        discoverable_id=False,
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
        positive_fixture=(
            "tests/test_finetune_readiness.py::"
            "test_weak_sub_band_outside_the_required_band_is_not_judged"
        ),
        discoverable_id=False,
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
    ),
    GateDeclaration(
        gate_id="official_path_reanalysis_envelope",
        owner=_READINESS,
        what_it_asserts="재분석 아티팩트의 파라미터가 완화 방향이 아니다",
        negative_fixture=(
            "tests/test_finetune_readiness.py::test_reanalysis_parameter_envelope_is_enforced"
        ),
        positive_fixture=(
            "tests/test_finetune_readiness.py::test_reanalysis_inside_the_envelope_is_accepted"
        ),
        discoverable_id=False,
    ),
    # ---------------- 측정·재분석 게이트 ----------------
    GateDeclaration(
        gate_id="measurement_relative_tau_outliers",
        owner=_PROBE,
        what_it_asserts="P−S 상대 τ 가 튀는 반복을 기각한다 (오염 과반에서도)",
        negative_fixture=(
            "tests/test_interleaved_probe.py::test_relative_tau_gate_rejects_the_measured_frame_slip"
        ),
        positive_fixture=(
            "tests/test_interleaved_probe.py::test_relative_tau_gate_survives_contamination_majority"
        ),
        discoverable_id=False,
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
    ),
    GateDeclaration(
        gate_id="reanalysis_rejects_loosening",
        owner=_REANALYSE,
        what_it_asserts="게이트를 약화하는 재분석 인자를 거부한다 (강화는 허용)",
        negative_fixture="tests/test_reanalyse_paths.py::test_loosening_arguments_are_refused",
        positive_fixture="tests/test_reanalyse_paths.py::test_tightening_arguments_are_allowed",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="reanalysis_frame_slip",
        owner=_REANALYSE,
        what_it_asserts="주입된 프레임 슬립 반복을 정확히 기각하고 과반이면 실패 폐쇄한다",
        negative_fixture="tests/test_reanalyse_paths.py::test_majority_frame_slip_fails_closed",
        positive_fixture=(
            "tests/test_reanalyse_paths.py::test_clean_capture_round_trips_to_the_injected_plant"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="reanalysis_metadata_forgery",
        owner=_REANALYSE,
        what_it_asserts="metadata.json 과 NPZ 내부 사본이 다르면 거부한다",
        negative_fixture="tests/test_reanalyse_paths.py::test_metadata_forgery_is_detected",
        discoverable_id=False,
    ),
    # ---------------- G4 판정 게이트 (eval/recorded.py) ----------------
    GateDeclaration(
        gate_id="g4_out_of_band_do_no_harm",
        owner="src/deep_anc/eval/recorded.py",
        what_it_asserts="신뢰 대역 밖 옥타브를 증폭하지 않는다 (절대목표 1)",
        negative_fixture="tests/test_recorded_eval.py::test_g4_rejects_out_of_band_amplifier",
        positive_fixture=(
            "tests/test_recorded_eval.py::"
            "test_metrics_markdown_and_npz_include_source_octave_and_worst10"
        ),
        discoverable_id=False,
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
    ),
    GateDeclaration(
        gate_id="g4_cluster_bootstrap_ci",
        owner="src/deep_anc/eval/recorded.py",
        what_it_asserts="클러스터가 부족하면 CI 를 지어내지 않는다 (그룹 단위 재표집)",
        negative_fixture=(
            "tests/test_recorded_eval.py::"
            "test_cluster_bootstrap_ci_is_undefined_below_the_group_floor"
        ),
        positive_fixture=(
            "tests/test_recorded_eval.py::"
            "test_cluster_bootstrap_ci_is_wider_than_naive_segment_resampling"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="metrics_plant_comparability",
        owner="src/deep_anc/eval/recorded.py",
        what_it_asserts="서로 다른 플랜트에서 나온 metrics 의 비교를 거부한다 (결함 5)",
        negative_fixture=(
            "tests/test_recorded_eval.py::test_metrics_comparison_rejects_different_plants"
        ),
        discoverable_id=False,
    ),
    # ---------------- recorded QA 게이트 ----------------
    GateDeclaration(
        gate_id="recorded_qa_source_err_alignment",
        owner=_QA,
        what_it_asserts="source→ERR 결맞음과 지연 안정성이 살아 있다 (결함 2)",
        negative_fixture=(
            "tests/test_recorded_qa.py::test_qa_rejects_source_err_timebase_collapse"
        ),
        positive_fixture=(
            "tests/test_recorded_qa.py::"
            "test_qa_measures_source_err_alignment_on_a_healthy_session"
        ),
        discoverable_id=False,
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
    ),
    GateDeclaration(
        gate_id="recorded_qa_delay_stability",
        owner=_QA,
        what_it_asserts="source→ERR 지연이 세션 안에서 떠다니지 않는다",
        negative_fixture=(
            "tests/test_recorded_qa.py::test_qa_rejects_a_drifting_source_err_delay"
        ),
        discoverable_id=False,
    ),
    # ---------------- 교차 도메인 불변식 (신설분) ----------------
    GateDeclaration(
        gate_id="invariant_corpus_disjoint",
        owner=_INVARIANTS,
        what_it_asserts="두 학습 스트림이 같은 원본 오디오를 쓰지 않는다 (D1)",
        negative_fixture=(
            "tests/test_invariants.py::test_corpus_disjoint_fails_on_the_music_leak"
        ),
        positive_fixture=(
            "tests/test_invariants.py::test_corpus_disjoint_passes_on_separate_corpora"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="invariant_measured_delay_agreement",
        owner=_INVARIANTS,
        what_it_asserts="같은 지연을 두 방법으로 잰 값이 일치한다 (D2)",
        negative_fixture=(
            "tests/test_invariants.py::test_measured_delay_agreement_fails_on_the_d2_gap"
        ),
        positive_fixture=(
            "tests/test_invariants.py::test_measured_delay_agreement_passes_within_tolerance"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="invariant_stream_delay_stability",
        owner=_INVARIANTS,
        what_it_asserts="재생→캡처 지연이 세션 내내 한 값이다",
        negative_fixture=(
            "tests/test_invariants.py::test_stream_delay_stability_fails_on_a_drifting_timebase"
        ),
        positive_fixture=(
            "tests/test_invariants.py::test_stream_delay_stability_passes_on_a_constant_delay"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="recorded_qa_group_split_leak",
        owner=_QA,
        what_it_asserts="같은 group_id 가 여러 split 에 걸치지 않는다",
        negative_fixture="tests/test_recorded_qa.py::test_group_split_leak_is_a_fatal_global_error",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="recorded_qa_audio_shape_and_length",
        owner=_QA,
        what_it_asserts="채널 수·샘플레이트·lead 를 고려한 최소 길이를 만족한다",
        negative_fixture=(
            "tests/test_recorded_qa.py::test_audio_shape_rate_and_lead_aware_minimum_length_fail"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="recorded_qa_level_and_metadata",
        owner=_QA,
        what_it_asserts="RMS/클리핑/메타데이터 일치가 기준을 만족한다",
        negative_fixture="tests/test_recorded_qa.py::test_nonfinite_rms_clip_and_metadata_mismatch_fail",
        discoverable_id=False,
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
        positive_fixture=(
            "tests/test_record_duct_gates.py::test_input_rail_gate_passes_a_healthy_quiet_room"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="recording_timeline_fail_closed",
        owner=_RECORD,
        what_it_asserts="재정렬이 성공하지 못한 세션은 저장하지 않는다",
        negative_fixture=(
            "tests/test_record_duct_gates.py::"
            "test_timeline_gate_rejects_the_shipped_corpus_numbers"
        ),
        positive_fixture=(
            "tests/test_record_duct_gates.py::test_timeline_gate_accepts_a_recovered_session"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="recording_gate_tightening_only",
        owner=_RECORD,
        what_it_asserts="CLI 로 수집 게이트를 완화할 수 없다 (강화만 허용)",
        negative_fixture="tests/test_record_duct_gates.py::test_cli_refuses_to_loosen_the_gates",
        discoverable_id=False,
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
        positive_fixture=(
            "tests/test_recorded_timeline.py::"
            "test_known_time_warp_is_recovered_and_coherence_is_restored"
        ),
        discoverable_id=False,
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
    ),
    # ---------------- 실측 데이터셋 게이트 ----------------
    GateDeclaration(
        gate_id="recorded_dataset_require_aligned_source",
        owner=_RECORDED_DATASET,
        what_it_asserts="재정렬본을 요구하면 원본 source.wav 로 조용히 폴백하지 않는다",
        negative_fixture=(
            "tests/test_recorded_dataset_augment.py::test_require_aligned_source_fails_closed"
        ),
        positive_fixture=(
            "tests/test_recorded_dataset_augment.py::"
            "test_aligned_source_is_preferred_over_the_raw_playback_array"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="recorded_lead_single_derivation",
        owner=_RECORDED_DATASET,
        what_it_asserts="실측 lead 는 측정 지연에서만 유도되고 손으로 쓸 수 없다",
        negative_fixture=(
            "tests/test_recorded_dataset_augment.py::"
            "test_lead_cannot_be_hand_written_against_the_derivation"
        ),
        positive_fixture=(
            "tests/test_recorded_dataset_augment.py::"
            "test_timeline_lead_mode_derives_the_lead_from_the_session"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="recorded_augment_plant_commutes",
        owner=_RECORDED_DATASET,
        what_it_asserts="증강은 LTI 플랜트와 교환 가능해야 한다 (양쪽에 같은 연산)",
        negative_fixture=(
            "tests/test_recorded_dataset_augment.py::"
            "test_mic_noise_goes_to_the_input_only_never_to_the_target"
        ),
        positive_fixture=(
            "tests/test_recorded_dataset_augment.py::"
            "test_augmentation_preserves_the_plant_relation_end_to_end"
        ),
        discoverable_id=False,
    ),
    # ---------------- 런타임 게이트 ----------------
    GateDeclaration(
        gate_id="runtime_digital_reference_lead",
        owner=_RUNTIME,
        what_it_asserts="런타임 lead 가 checkpoint lead 와 정확히 같다",
        negative_fixture=f"{_NEG}::test_runtime_lead_gate_fails_when_checkpoint_disagrees",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_engine_artifact_preflight",
        owner=_RUNTIME,
        what_it_asserts="engine 블록이 가리키는 모든 파일이 시작 전에 실제로 존재한다",
        negative_fixture=(
            "tests/test_realtime_start.py::"
            "test_engine_preflight_rejects_a_missing_artifact_even_when_unused"
        ),
        positive_fixture=(
            "tests/test_realtime_start.py::"
            "test_engine_preflight_accepts_the_shipped_runtime_configs"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_input_preflight",
        owner="src/deep_anc/audio_io.py",
        what_it_asserts="죽은/고정된 에러 마이크 채널로는 시작하지 않는다",
        negative_fixture=(
            "tests/test_realtime_start.py::test_runtime_input_preflight_rejects_stuck_error_channel"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_fxlms_adaptation_fail_closed",
        owner="src/deep_anc/realtime/engines.py",
        what_it_asserts="FxLMS 적응은 명시적으로 켜기 전에는 꺼져 있다",
        negative_fixture="tests/test_realtime_start.py::test_fxlms_adaptation_gate_is_fail_closed",
        discoverable_id=False,
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
    ),
    GateDeclaration(
        gate_id="runtime_output_dc",
        owner=_SAFETY,
        what_it_asserts="상쇄 출력의 지속 DC 를 제거하고 ANC 를 끈다 (보이스코일 보호)",
        negative_fixture=f"{_RTSAFE}::test_output_dc_watchdog_detects_a_sustained_offset",
        positive_fixture=f"{_RTSAFE}::test_clean_anti_noise_trips_no_watchdog",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_output_saturation",
        owner=_SAFETY,
        what_it_asserts="tanh 리미터에 눌린 출력을 검출한다 (리미터와 다른 값을 잰다)",
        negative_fixture=f"{_RTSAFE}::test_saturation_watchdog_detects_output_the_old_clip_counter_declared_clean",
        positive_fixture=f"{_RTSAFE}::test_clean_anti_noise_trips_no_watchdog",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_output_rms",
        owner=_SAFETY,
        what_it_asserts="상쇄 출력 RMS 상한을 지속 초과하면 ANC 를 끈다",
        negative_fixture=f"{_RTSAFE}::test_output_rms_watchdog_detects_sustained_over_power",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_deadline_miss_rate",
        owner=_SAFETY,
        what_it_asserts="최근 1초 미스율이 한계를 넘으면 ANC 를 끈다 (교대 미스 포함)",
        negative_fixture=f"{_RTSAFE}::test_deadline_watchdog_detects_the_alternating_miss_the_streak_never_caught",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_divergence",
        owner=_SAFETY,
        what_it_asserts="발산을 검출하고, 베이스라인이 없어 판정 불가면 fail-closed 로 끈다",
        negative_fixture=f"{_RTSAFE}::test_divergence_watchdog_detects_the_missing_baseline_instead_of_going_quiet",
        positive_fixture=f"{_RTSAFE}::test_clean_anti_noise_trips_no_watchdog",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_handoff_backlog",
        owner=_SAFETY,
        what_it_asserts="입력 백로그가 1 hop 을 넘어 오래된 입력을 버리면 ANC 를 끈다",
        negative_fixture=f"{_RTSAFE}::test_handoff_backlog_watchdog_detects_dropped_stale_input",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="runtime_pipeline_handoff_budget",
        owner=_SAFETY,
        what_it_asserts="입력/출력 백로그 예산이 대칭이고 실효 핸드오프가 학습 가정과 같다",
        negative_fixture=f"{_RTSAFE}::test_handoff_budget_rejects_the_input_output_asymmetry",
        positive_fixture=f"{_RTSAFE}::test_handoff_budget_derives_one_hop_from_the_duct_config",
        discoverable_id=False,
    ),
    # ---------------- 교차 도메인 불변식 (발생기 A') ----------------
    GateDeclaration(
        gate_id="invariant_relative_tau_constancy",
        owner=_INVARIANTS,
        what_it_asserts="같은 출력 스트림의 두 채널은 상대 지연이 상수다",
        negative_fixture=(
            "tests/test_invariants.py::test_relative_tau_constancy_fails_on_the_measured_frame_slip"
        ),
        positive_fixture=(
            "tests/test_invariants.py::test_relative_tau_constancy_passes_on_the_clean_subset"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="invariant_stream_coherence",
        owner=_INVARIANTS,
        what_it_asserts="재생 신호와 캡처 신호의 대응(coh²)이 살아 있다",
        negative_fixture=(
            "tests/test_invariants.py::test_stream_coherence_fails_on_a_collapsed_timebase"
        ),
        positive_fixture=(
            "tests/test_invariants.py::test_stream_coherence_passes_on_a_delayed_copy"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="invariant_plant_fingerprint_match",
        owner=_INVARIANTS,
        what_it_asserts="비교되는 두 메트릭이 같은 플랜트에서 나왔다",
        negative_fixture=(
            "tests/test_invariants.py::test_plant_fingerprint_match_fails_across_the_20260804_plants"
        ),
        positive_fixture=(
            "tests/test_invariants.py::test_plant_fingerprint_match_passes_for_the_same_plant"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="invariant_lead_agreement",
        owner=_INVARIANTS,
        what_it_asserts="설정 lead 가 측정 지연에서 유도되는 값과 같다",
        negative_fixture=(
            "tests/test_invariants.py::test_lead_agreement_fails_on_the_aaeef41_mismatch"
        ),
        positive_fixture=(
            "tests/test_invariants.py::test_lead_agreement_passes_on_the_measured_plant"
        ),
        discoverable_id=False,
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
        positive_fixture=(
            f"{_LOSS_TESTS}::test_do_no_harm_bands_are_derived_by_subtracting_the_trusted_band"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="loss_config_schema",
        owner=_LOSS_CFG,
        what_it_asserts="loss 블록에 모르는 키가 없고 값 범위가 유효하다",
        negative_fixture=f"{_LOSS_TESTS}::test_unknown_loss_key_is_rejected",
        positive_fixture=f"{_LOSS_TESTS}::test_shipped_training_configs_pass_the_loss_schema",
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="loss_limiter_limit_single_source",
        owner=_LOSS,
        what_it_asserts="리미터 한계를 모델과 설정이 따로 정하지 않는다",
        negative_fixture=f"{_LOSS_TESTS}::test_limiter_limit_disagreement_is_rejected",
        positive_fixture=(
            f"{_LOSS_TESTS}::test_saturation_penalty_replaces_the_structurally_dead_clip_term"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="timing_single_source_of_plant_settle",
        owner=_TIMING,
        what_it_asserts="정착 구간은 PlantSettle.derive() 밖에서 만들 수 없다",
        negative_fixture=f"{_START_TESTS}::test_plant_settle_cannot_be_constructed_by_hand",
        positive_fixture=(
            f"{_START_TESTS}::test_plant_settle_is_derived_from_the_measured_plant"
        ),
        discoverable_id=False,
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
        positive_fixture=(
            f"{_START_TESTS}::test_warmup_floor_does_not_shrink_a_longer_requested_warmup"
        ),
        discoverable_id=False,
    ),
    GateDeclaration(
        gate_id="timing_single_source_of_lead",
        owner=_TIMING,
        what_it_asserts="lead 는 PlantDelays.lead() 밖에서 만들 수 없다",
        negative_fixture="tests/test_timing.py::test_lead_cannot_be_constructed_by_hand",
        positive_fixture="tests/test_timing.py::test_lead_is_derived_from_measured_delays",
        discoverable_id=False,
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


_AUDIT_CALL = re.compile(
    r"audit\.(?:pass_|fail)\(\s*(?:#[^\n]*\n\s*)?[\"']([a-z][a-z0-9_]*)[\"']"
)


def discover_audit_gate_ids(source_path: str | Path) -> frozenset[str]:
    """소스에서 ``audit.pass_``/``audit.fail`` 의 게이트 id 문자열을 긁어낸다.

    선언을 빠뜨린 게이트를 메타 테스트가 잡아내기 위한 것이다. f-string 으로 만들어지는
    id 는 여기서 발견되지 않으므로 선언에서 ``discoverable_id=False`` 로 표시한다.
    """

    text = Path(source_path).read_text(encoding="utf-8")
    return frozenset(_AUDIT_CALL.findall(text))
