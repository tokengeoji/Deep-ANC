"""지연·정렬·대역 부기의 단일 출처 검증 (발생기 A).

이 파일이 지키는 것은 "값이 맞는가"가 아니라 **"값이 한 곳에서만 나오는가"**다.
2026-08-05 결함 군집 분석에서 확인된 결함 18건 중 9건이 같은 물리량을 두 곳 이상에서
따로 유도한 데서 나왔다. 그래서 아래 테스트 중 절반은 **소스 스캔**이다 — 여섯 번째
복붙을 사람이 눈으로 잡는 대신 CI 가 잡는다.
"""

from __future__ import annotations

import re
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_anc.config import DEFAULT_HANDOFF_SAMPLES, REPO_ROOT, load_train_config
from deep_anc.dsp.timing import (
    BandPlan,
    FrequencyBand,
    Lead,
    PlantDelays,
    PlantFingerprint,
    TrainingTimingContract,
    handoff_samples_from_config,
    intersect_frequency_bands,
    target_band_from_config,
)


# 2026-08-05 플랜트 복구 확정값 (캡처 20260804_225546_f7b0fecd).
MEASURED_PRIMARY_DELAY = 1602
MEASURED_SECONDARY_DELAY = 1462
MEASURED_HANDOFF = 256
MEASURED_LEAD = 116
MEASURED_RELATIVE_DELAY = 140


def _load_bound_canonical(overrides: list[str] | None = None) -> dict:
    def _bind(data: dict, *, repo_root):
        data.update(
            transfer_manifest="data/manifests/elice_transfer_manifest.json",
            transfer_manifest_sha256="b" * 64,
            recorded_transfer_aggregate_sha256="c" * 64,
        )

    values = list(overrides or []) + [
        f"data.bootstrap_receipt_sha256={'a' * 64}",
        f"campaign_prerequisite_sha256={'d' * 64}",
    ]
    with patch(
        "deep_anc.data.transfer_contract.bind_recorded_transfer_config", _bind
    ):
        return load_train_config(
            REPO_ROOT / "configs/train_pretrain_tiny.yaml", values
        )


def _duct_cfg(*, handoff: int = MEASURED_HANDOFF, target=(80.0, 800.0)) -> dict:
    return {
        "secondary_path": {"handoff_extra_samples": handoff},
        "acoustics": {"realistic_target_band_hz": list(target)},
    }


# ----------------------------------------------------------------------------------
# lead — 생성 경로가 하나뿐인가
# ----------------------------------------------------------------------------------
def test_lead_is_derived_from_measured_delays():
    delays = PlantDelays(
        primary_delay_samples=MEASURED_PRIMARY_DELAY,
        secondary_delay_samples=MEASURED_SECONDARY_DELAY,
        handoff_samples=MEASURED_HANDOFF,
        sample_rate=48_000,
    )
    lead = delays.lead()

    assert int(lead) == MEASURED_LEAD
    assert lead.raw_samples == MEASURED_LEAD
    assert not lead.is_clamped
    # P−S 는 이 측정의 유일한 물리 불변량이다 (캡처 9건에서 139~141).
    assert delays.relative_delay_samples == MEASURED_RELATIVE_DELAY
    assert delays.secondary_total_delay_samples == MEASURED_SECONDARY_DELAY + MEASURED_HANDOFF


def test_lead_cannot_be_constructed_by_hand():
    """lead 를 손으로 쓰는 순간 그것이 **두 번째 유도**가 된다 — 구조적으로 막는다.

    실제 사고(커밋 aaeef41): trainer 와 게이트가 각자 lead 를 유도해 109 와 113 으로
    갈라진 채 양쪽 다 자기 기준으로는 "통과"했다.
    """

    with pytest.raises(TypeError, match="PlantDelays.lead"):
        Lead(
            samples=113,
            raw_samples=113,
            secondary_delay_samples=1465,
            handoff_samples=256,
            primary_delay_samples=1608,
        )


def test_negative_lead_is_clamped_but_the_raw_value_survives():
    """handoff 를 줄이면 lead 가 음수가 된다 — 0 으로 자르되 사실을 숨기지 않는다.

    handoff 128 이면 자연 lead 는 0 이 아니라 **−15** 다(P−S=143 시절 실측). "예측
    요구가 사라진다"는 주장은 틀렸고, 사라지는 게 아니라 15 샘플 생긴다.
    """

    delays = PlantDelays(
        primary_delay_samples=1602,
        secondary_delay_samples=1462,
        handoff_samples=0,
        sample_rate=48_000,
    )
    lead = delays.lead()

    assert int(lead) == 0
    assert lead.raw_samples == -MEASURED_RELATIVE_DELAY
    assert lead.is_clamped


def test_training_timing_contract_is_self_contained_and_plant_derived():
    fir = [0.0, -0.2, 0.9, 0.1]
    delays = PlantDelays(
        primary_delay_samples=MEASURED_PRIMARY_DELAY,
        secondary_delay_samples=MEASURED_SECONDARY_DELAY,
        handoff_samples=MEASURED_HANDOFF,
        sample_rate=48_000,
    )
    contract = TrainingTimingContract.derive(
        primary_fir=fir,
        plant_delays=delays,
    )

    assert contract.primary_zeros_before_fir_samples == MEASURED_PRIMARY_DELAY
    assert contract.primary_fir_peak_offset_samples == 2
    assert contract.secondary_delay_samples == MEASURED_SECONDARY_DELAY
    assert contract.handoff_samples == MEASURED_HANDOFF
    assert contract.raw_digital_reference_lead_samples == MEASURED_LEAD
    assert contract.digital_reference_lead_samples == MEASURED_LEAD
    assert contract.synthetic_total_advance_samples == MEASURED_PRIMARY_DELAY + 2 + MEASURED_LEAD

    forged = contract.model_dump()
    forged["digital_reference_lead_samples"] += 1
    with pytest.raises(Exception, match="clamped lead"):
        TrainingTimingContract.model_validate(forged)


def test_compact_digital_config_injects_lead_from_timing_contract_only():
    raw = (REPO_ROOT / "configs/data_sim.yaml").read_text(encoding="utf-8")
    assert not re.search(r"^digital_reference_lead_samples\s*:", raw, re.MULTILINE)

    cfg = _load_bound_canonical()
    contract = TrainingTimingContract.from_data_config(cfg["data"])
    assert cfg["data"]["digital_reference_lead_samples"] == int(
        contract.digital_reference_lead_samples
    )
    with pytest.raises(ValueError, match="PlantDelays.lead"):
        _load_bound_canonical(["data.digital_reference_lead_samples=999"])


@pytest.mark.parametrize(
    "mode_overrides, expected",
    [
        (["data.digital_primary_path_mode=rir_surrogate", "data.digital_reference_lead_samples=7"], 7),
        # canonical REF-only payload은 digital-reference 전용이다. acoustic
        # diagnostic은 그 payload를 묵시적으로 재사용할 수 없고, top-level
        # input-contract 경로를 명시적으로 해제해야만 한다.
        (
            [
                "data_model_input_contract_config=null",
                "data.reference_mode=acoustic",
                "data.digital_reference_lead_samples=0",
            ],
            0,
        ),
    ],
)
def test_noncompact_timing_modes_keep_their_explicit_legacy_lead(
    mode_overrides, expected
):
    cfg = load_train_config(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        [
            "experiment_role=diagnostic_overfit",
            "init_eligible=false",
            "contract_run_dir=false",
            *mode_overrides,
        ],
    )
    assert cfg["data"]["digital_reference_lead_samples"] == expected
    assert "training_timing_contract" not in cfg["data"]


def test_acoustic_diagnostic_requires_explicit_ref_only_input_opt_out():
    """acoustic 진단이 canonical output-clock input 계약을 묵시적으로 약화하지 않는다."""

    common = [
        "experiment_role=diagnostic_overfit",
        "init_eligible=false",
        "contract_run_dir=false",
        "data.reference_mode=acoustic",
        "data.digital_reference_lead_samples=0",
    ]
    with pytest.raises(ValueError, match="reference_mode=digital 전용"):
        load_train_config(REPO_ROOT / "configs/train_pretrain_tiny.yaml", common)

    cfg = load_train_config(
        REPO_ROOT / "configs/train_pretrain_tiny.yaml",
        ["data_model_input_contract_config=null", *common],
    )
    assert cfg["data_model_input_contract_config"] is None
    assert "model_input_contract" not in cfg["data"]
    assert "model_input_contract_sha256" not in cfg


def test_compare_fxlms_is_explicit_legacy_only_and_never_reads_manual_delay():
    script = REPO_ROOT / "scripts/eval/compare_fxlms.py"
    source = script.read_text(encoding="utf-8")
    assert '.get("d_noise_delay_samples")' not in source
    spec = importlib.util.spec_from_file_location("_compare_fxlms_timing", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="legacy geometry timing 진단 전용"):
        module.validate_legacy_diagnostic_checkpoint(
            {"experiment_role": "diagnostic"},
            allow_legacy_diagnostic_timing=False,
        )
    with pytest.raises(ValueError, match="TrainingTimingContract/canonical"):
        module.validate_legacy_diagnostic_checkpoint(
            {"experiment_role": "canonical_pretrain", "data": {}},
            allow_legacy_diagnostic_timing=True,
        )
    module.validate_legacy_diagnostic_checkpoint(
        {"experiment_role": "legacy_diagnostic", "data": {}},
        allow_legacy_diagnostic_timing=True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"primary_delay_samples": -1},
        {"secondary_delay_samples": -1},
        {"handoff_samples": -1},
        {"sample_rate": 0},
    ],
)
def test_physically_impossible_delays_are_refused_at_construction(kwargs):
    base = dict(
        primary_delay_samples=1602,
        secondary_delay_samples=1462,
        handoff_samples=256,
        sample_rate=48_000,
    )
    base.update(kwargs)
    with pytest.raises(Exception):
        PlantDelays(**base)


def test_plant_delays_are_frozen():
    delays = PlantDelays(
        primary_delay_samples=1602,
        secondary_delay_samples=1462,
        handoff_samples=256,
        sample_rate=48_000,
    )
    with pytest.raises(Exception):
        delays.primary_delay_samples = 1  # type: ignore[misc]


# ----------------------------------------------------------------------------------
# handoff — 기본값이 한 곳에서만 나오는가
# ----------------------------------------------------------------------------------
def test_handoff_comes_from_the_config_and_falls_back_to_one_constant():
    assert handoff_samples_from_config(_duct_cfg(handoff=128)) == 128
    assert handoff_samples_from_config({}) == DEFAULT_HANDOFF_SAMPLES
    with pytest.raises(ValueError, match="0 이상"):
        handoff_samples_from_config(_duct_cfg(handoff=-1))


# ----------------------------------------------------------------------------------
# 대역 — 손실 대역과 보고 대역이 타입 수준에서 갈라져 있는가
# ----------------------------------------------------------------------------------
def test_band_plan_separates_the_optimise_band_from_the_measure_band():
    plan = BandPlan.resolve(
        plant_trusted_band_hz=(150.0, 1600.0),
        duct_cfg=_duct_cfg(target=(80.0, 800.0)),
        sample_rate=48_000,
    )

    # 손실 대역은 보수적으로 (기존 동작과 동일: S 신뢰 ∩ 목표).
    assert plan.optimize.as_tuple() == (150.0, 800.0)
    # 보고 대역은 넓다 — 넓지 않으면 절대목표 1(고역도 제거)을 **검증할 수 없다**.
    assert plan.measure.as_tuple() == (150.0, 1600.0)
    assert plan.measure.covers(plan.optimize)


def test_band_plan_matches_the_expression_it_replaced():
    """다섯 파일에 복붙돼 있던 그 한 줄과 결과가 같아야 한다 (동작 불변)."""

    duct = _duct_cfg(target=(80.0, 800.0))
    legacy = intersect_frequency_bands((150.0, 1600.0), (80.0, 800.0), 48_000 / 2.0)
    plan = BandPlan.resolve(
        plant_trusted_band_hz=(150.0, 1600.0), duct_cfg=duct, sample_rate=48_000
    )
    assert plan.optimize.as_tuple() == legacy


def test_band_plan_follows_the_duct_config_target_band():
    """duct.yaml 한 곳만 바꾸면 손실 대역이 따라온다."""

    narrow = BandPlan.resolve(
        plant_trusted_band_hz=(150.0, 1600.0),
        duct_cfg=_duct_cfg(target=(80.0, 800.0)),
        sample_rate=48_000,
    )
    wide = BandPlan.resolve(
        plant_trusted_band_hz=(150.0, 1600.0),
        duct_cfg=_duct_cfg(target=(80.0, 1600.0)),
        sample_rate=48_000,
    )
    assert narrow.optimize.as_tuple() == (150.0, 800.0)
    assert wide.optimize.as_tuple() == (150.0, 1600.0)
    # 보고 대역은 목표 대역에 좌우되지 않는다 — 플랜트 신뢰대역이 정한다.
    assert narrow.measure.as_tuple() == wide.measure.as_tuple() == (150.0, 1600.0)


def test_target_band_from_config_uses_one_default():
    from deep_anc.dsp.timing import DEFAULT_TARGET_BAND_HZ

    assert target_band_from_config({}).as_tuple() == DEFAULT_TARGET_BAND_HZ


@pytest.mark.parametrize(
    ("first", "second", "match"),
    [
        ((600, 150), (80, 800), "잘못된"),
        ((-1, 600), (80, 800), "잘못된"),
        ((150, 600), (800, 1600), "교집이 비어"),
        ((150, float("nan")), (80, 800), "유한한"),
        ((150, 600, 900), (80, 800), "형식"),
    ],
)
def test_intersect_frequency_bands_fails_fast(first, second, match):
    with pytest.raises(ValueError, match=match):
        intersect_frequency_bands(first, second, 24_000.0)


def test_frequency_band_rejects_inverted_and_infinite_bands():
    with pytest.raises(Exception):
        FrequencyBand(lo_hz=600.0, hi_hz=150.0)
    with pytest.raises(Exception):
        FrequencyBand(lo_hz=float("inf"), hi_hz=1.0)


# ----------------------------------------------------------------------------------
# 플랜트 지문
# ----------------------------------------------------------------------------------
def _fingerprint(**overrides) -> PlantFingerprint:
    delays = PlantDelays(
        primary_delay_samples=int(overrides.pop("primary", MEASURED_PRIMARY_DELAY)),
        secondary_delay_samples=int(overrides.pop("secondary", MEASURED_SECONDARY_DELAY)),
        handoff_samples=MEASURED_HANDOFF,
        sample_rate=48_000,
    )
    bands = BandPlan.resolve(
        plant_trusted_band_hz=(150.0, 1600.0), duct_cfg=_duct_cfg(), sample_rate=48_000
    )
    return PlantFingerprint.build(
        delays=delays,
        lead=delays.lead(),
        physics_status=str(overrides.pop("physics_status", "measured_primary_path")),
        bands=bands,
        **overrides,
    )


def test_fingerprint_refuses_a_lead_that_does_not_match_the_delays():
    delays = PlantDelays(
        primary_delay_samples=1602,
        secondary_delay_samples=1462,
        handoff_samples=256,
        sample_rate=48_000,
    )
    other = PlantDelays(
        primary_delay_samples=1608,
        secondary_delay_samples=1465,
        handoff_samples=256,
        sample_rate=48_000,
    )
    bands = BandPlan.resolve(
        plant_trusted_band_hz=(150.0, 1600.0), duct_cfg=_duct_cfg(), sample_rate=48_000
    )
    with pytest.raises(ValueError, match="지연 부기와 맞지 않습니다"):
        PlantFingerprint.build(
            delays=delays,
            lead=other.lead(),
            physics_status="measured_primary_path",
            bands=bands,
        )


def test_fingerprints_of_the_two_20260804_plants_differ():
    """전 = S 1342 / lead 109 / surrogate, 후 = 1465 / 113 / measured — 다른 물리다."""

    before = _fingerprint(
        primary=1489, secondary=1342, physics_status="secondary_surrogate_representation_pretrain"
    )
    after = _fingerprint(primary=1608, secondary=1465, physics_status="measured_primary_path")

    diffs = before.differences(after)
    assert diffs
    assert before.digest() != after.digest()
    with pytest.raises(ValueError, match="서로 다른 플랜트"):
        before.assert_same(after)


def test_identical_plants_compare_equal():
    assert _fingerprint().differences(_fingerprint()) == []
    _fingerprint().assert_same(_fingerprint())


# ----------------------------------------------------------------------------------
# 소스 스캔 — 재유도가 되살아나면 실패한다
# ----------------------------------------------------------------------------------
_SINGLE_SOURCE = REPO_ROOT / "src/deep_anc/dsp/timing.py"


def _python_sources() -> list[Path]:
    roots = [REPO_ROOT / "src", REPO_ROOT / "scripts"]
    files: list[Path] = []
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts and path != _SINGLE_SOURCE
        )
    return files


def _hits(pattern: re.Pattern[str], *, allow: set[str] = frozenset()) -> list[str]:
    out: list[str] = []
    for path in _python_sources():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in allow:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or pattern.search(line) is None:
                continue
            out.append(f"{rel}:{number}: {stripped}")
    return out


def test_no_module_re_derives_the_handoff_default():
    """``.get("handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES)`` 는 7벌 있었다."""

    pattern = re.compile(r"handoff_extra_samples[\"']\s*,\s*DEFAULT_HANDOFF_SAMPLES")
    assert _hits(pattern) == []


def test_no_module_hardcodes_the_handoff_constant():
    """``handoff = 256`` 이라는 사본이 측정 스크립트 두 곳에 있었다."""

    pattern = re.compile(r"^\s*handoff\s*=\s*\d+\s*$")
    assert _hits(pattern) == []


def test_no_module_re_derives_the_lead_relation():
    """``S + handoff − P`` 를 손으로 다시 계산하는 코드가 없어야 한다.

    문서 그림 스크립트는 역관계(P = S + handoff − lead)를 **그리기 위해** 쓰므로 예외다.
    """

    pattern = re.compile(r"=\s*(?:max\(\s*0\s*,\s*)?[\w\[\]\"'.]*s_?_?delay[\w\[\]\"'.]*\s*\+")
    allow = {
        "scripts/docs/render_architecture_figures.py",
        "scripts/docs/render_readme_figures.py",
    }
    assert _hits(pattern, allow=allow) == []


def test_no_module_re_derives_the_trusted_band_intersection():
    """``intersect(sp.trusted_band_hz(), realistic_target_band_hz, ...)`` 는 5벌 있었다."""

    pattern = re.compile(
        r"intersect_frequency_bands\(\s*[^)]*trusted_band_hz\(\)", re.DOTALL
    )
    hits = [
        str(path.relative_to(REPO_ROOT))
        for path in _python_sources()
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert hits == []


def test_intersect_frequency_bands_is_defined_exactly_once():
    pattern = re.compile(r"^def intersect_frequency_bands\(")
    assert _hits(pattern) == []


def test_the_consumers_import_the_single_source():
    """대역·지연을 쓰는 모듈이 timing 모듈을 실제로 참조하는가."""

    expected = {
        "src/deep_anc/train/trainer.py",
        "src/deep_anc/eval/recorded.py",
        "src/deep_anc/eval/metrics.py",
        "src/deep_anc/losses/anc_loss.py",
        "src/deep_anc/train/finetune_readiness.py",
        "src/deep_anc/realtime/run_realtime.py",
        "scripts/eval/evaluate_offline.py",
        "scripts/demo/evaluate_session.py",
        "scripts/demo/render_anc_demo.py",
        # measure_paths_interleaved는 P/S 독립 delay artifact의 producer이고
        # lead를 소비/발행하지 않는다. TrainingTimingContract는 resolved config와
        # reanalysis/readiness/eval 소비 경계에서만 만들어진다.
        "scripts/data/reanalyse_paths_interleaved.py",
    }
    missing = {
        rel
        for rel in expected
        if "dsp.timing" not in (REPO_ROOT / rel).read_text(encoding="utf-8")
        and "dsp.invariants" not in (REPO_ROOT / rel).read_text(encoding="utf-8")
    }
    assert missing == set()
