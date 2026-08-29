"""canonical 파인튜닝 가드레일 문서와 코드 authority의 최소 결속."""

from pathlib import Path

from deep_anc.dsp.invariants import (
    REQUIRED_SOURCE_FAMILIES,
    REQUIRED_SOURCE_FAMILY_MIX_TAGS,
)
from deep_anc.train.finetune_readiness import _Audit, _audit_absolute_objective_scope


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = REPO_ROOT / "docs" / "16_canonical_finetune_guardrails.md"
BROADBAND_DOCUMENT = REPO_ROOT / "docs" / "18_broadband_anc_guardrails.md"
TRAINING_DOCUMENT = REPO_ROOT / "docs" / "05_training_elice.md"
RECORDED_GENERATION_DOCUMENT = REPO_ROOT / "docs" / "17_recorded_generation.md"

REQUIRED_GATE_AUTHORITIES = {
    "absolute_objective_scope": "src/deep_anc/train/finetune_readiness.py",
    "recorded_transfer_snapshot": "src/deep_anc/train/finetune_readiness.py",
    "official_secondary_path": "src/deep_anc/train/finetune_readiness.py",
    "official_primary_path": "src/deep_anc/train/finetune_readiness.py",
    "matched_path_measurement_conditions": "src/deep_anc/train/finetune_readiness.py",
    "path_delay_and_lead": "src/deep_anc/train/finetune_readiness.py",
    "recorded_dataset_qa": "src/deep_anc/train/finetune_readiness.py",
    "recorded_alignment_integrity": "src/deep_anc/train/finetune_readiness.py",
    "recorded_statistical_power": "src/deep_anc/train/finetune_readiness.py",
    "recorded_subband_coverage": "src/deep_anc/train/finetune_readiness.py",
    "corpus_disjoint": "src/deep_anc/train/finetune_readiness.py",
    "measured_source_delay_agreement": "src/deep_anc/train/finetune_readiness.py",
    "plant_confidence_ceiling": "src/deep_anc/train/finetune_readiness.py",
    "completed_init_checkpoint": "src/deep_anc/train/finetune_readiness.py",
    "g4_strict_trusted_subbands": "src/deep_anc/eval/recorded.py",
    "g4_out_of_band_do_no_harm": "src/deep_anc/eval/recorded.py",
    "g4_statistical_power": "src/deep_anc/eval/recorded.py",
    "g4_cluster_bootstrap_ci": "src/deep_anc/eval/recorded.py",
    "recorded_val_g4": "src/deep_anc/train/finetune_readiness.py",
    "recorded_test_g4": "src/deep_anc/train/finetune_readiness.py",
    "recorded_selection_test_once_chain": "src/deep_anc/train/evaluation_contract.py",
    "runtime_strict_plant_contract": "src/deep_anc/realtime/plant_contract.py",
    "runtime_engine_artifact_preflight": "src/deep_anc/realtime/run_realtime.py",
    "runtime_deadline_miss_rate": "src/deep_anc/realtime/safety.py",
    "runtime_handoff_backlog": "src/deep_anc/realtime/safety.py",
    "runtime_pipeline_handoff_budget": "src/deep_anc/realtime/safety.py",
}


def _document_text() -> str:
    assert DOCUMENT.is_file(), "canonical 파인튜닝 가드레일 문서가 없습니다"
    return DOCUMENT.read_text(encoding="utf-8")


def test_guardrail_document_names_every_required_gate_with_its_authority():
    text = _document_text()
    lines = text.splitlines()
    for gate_id, authority in REQUIRED_GATE_AUTHORITIES.items():
        matches = [line for line in lines if f"`{gate_id}`" in line]
        assert matches, f"문서에서 필수 gate ID가 빠졌습니다: {gate_id}"
        assert any(f"`{authority}`" in line for line in matches), (
            f"{gate_id}와 authority {authority}가 같은 표 행에 결속되지 않았습니다"
        )
        assert (REPO_ROOT / authority).is_file(), f"authority 파일이 없습니다: {authority}"


def test_guardrail_document_is_fail_closed_and_forbids_legacy_promotion():
    text = _document_text()
    for token in (
        "**PASS**",
        "**FAIL**",
        "**BLOCKED**",
        "INCONCLUSIVE",
        "threshold",
        "임계값",
        "legacy",
        "승격 금지",
        "Level-5",
        "one-shot",
        "17/17",
        "P99 <3.0 ms",
        "cross-public speech lineage",
        "dns_book",
        "dns_reader",
    ):
        assert token in text, f"가드레일 문서 필수 문구가 빠졌습니다: {token}"


def test_broadband_guardrail_cannot_be_reduced_to_stage1_or_dnh_only():
    text = BROADBAND_DOCUMENT.read_text(encoding="utf-8")
    for token in (
        "11,313.708",
        "matched FxLMS",
        "positive_attenuation",
        "speech/music/environment/machine",
        "Level-5",
        "최소 5개 ERR 위치",
        "clock valid repeat ≥8",
        "target-d energy-density ratio ≥0.25",
        "source는 전 구간 계속 ON",
        "Stage-1 PASS를 2/4/8 kHz 최종 성공",
    ):
        assert token in text, f"광대역 가드레일 필수 문구가 빠졌습니다: {token}"


def test_historical_highband_generation_cannot_be_promoted_to_broadband_v2():
    text = RECORDED_GENERATION_DOCUMENT.read_text(encoding="utf-8")
    for token in (
        "Stage-1의 600--1600 Hz",
        "2/4/8 kHz 광대역-v2 데이터가 아니며",
        "broadband_point_control_150_11314_v2",
        "증거로 승격할 수 없다",
        "broadband_coverage_receipt.py",
    ):
        assert token in text, (
            "역사적 highband-coverage-v1의 광대역 승격 차단 문구가 빠졌습니다: "
            f"{token}"
        )


def test_elice_training_doc_preserves_readiness_recovery_order_before_g0():
    """현재 blocker를 건너뛰고 G0를 먼저 실행하라는 stale 지시를 금지한다."""

    text = TRAINING_DOCUMENT.read_text(encoding="utf-8")
    bootstrap_section = text.split("bootstrap 종료 후", 1)[1].split("## 5.", 1)[0]
    ordered_tokens = (
        "14/17 PASS",
        "speech lineage",
        "15/17 PASS",
        "coverage",
        "16/17 PASS",
        "G0",
    )
    cursor = 0
    for token in ordered_tokens:
        position = bootstrap_section.find(token, cursor)
        assert position >= 0, (
            "readiness 복구 순서는 현재 14/17 → speech lineage 15/17 → "
            f"coverage 16/17 → G0여야 합니다: {token!r} 누락/역전"
        )
        cursor = position + len(token)


def test_absolute_objective_requires_all_four_families_and_environment_pool_mapping():
    assert REQUIRED_SOURCE_FAMILIES == (
        "speech",
        "music",
        "environment",
        "machine",
    )
    assert REQUIRED_SOURCE_FAMILY_MIX_TAGS == {
        "speech": ("speech",),
        "music": ("music",),
        "environment": ("demand", "esc50"),
        "machine": ("machine",),
    }

    base_mix = {
        "speech": 0.15,
        "music": 0.10,
        "demand": 0.08,
        "esc50": 0.05,
        "machine": 0.07,
    }
    readiness = {"required_path_band_hz": [150, 1600]}
    duct = {"acoustics": {"realistic_target_band_hz": [80, 1600]}}

    passed = _Audit("objective")
    _audit_absolute_objective_scope(passed, readiness, {"source_mix_ratio": base_mix}, duct)
    assert passed.report()["ok"]

    for family, removed_tags in {
        "speech": ("speech",),
        "music": ("music",),
        "environment": ("demand", "esc50"),
        "machine": ("machine",),
    }.items():
        attacked_mix = dict(base_mix)
        for tag in removed_tags:
            attacked_mix[tag] = 0.0
        failed = _Audit("objective")
        _audit_absolute_objective_scope(
            failed, readiness, {"source_mix_ratio": attacked_mix}, duct
        )
        report = failed.report()
        assert not report["ok"]
        assert family in report["checks"][0]["message"]
