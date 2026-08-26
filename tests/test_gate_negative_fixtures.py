"""짝이 없던 게이트의 **negative fixture** — 각 게이트를 실제로 FAIL 시킨다 (발생기 B).

왜 이 파일이 있는가
------------------
2026-08-05 결함 군집 B: 게이트 9개가 전부 PASS 인데 플랜트가 형상 기준 50% 틀려 있었고,
recorded QA 는 80/80 PASS 인데 학습 데이터의 시간축이 붕괴해 있었다. 공통점은
**"좋아 보이는 데이터에서 한 번 통과시켜 보고 끝"** 이었다는 것이다. 나쁜 데이터에서
FAIL 하는 것을 확인한 게이트가 하나도 없었다.

여기 있는 테스트는 전부 "게이트가 통과한다"가 아니라 **"게이트가 거부한다"** 를 단언한다.
``deep_anc.ops.gate_registry`` 의 선언이 이 파일의 노드 id 를 가리키고,
``tests/test_gate_registry.py`` 가 그 대응을 강제한다.
"""

from __future__ import annotations


import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.realtime.run_realtime import validate_digital_reference_lead
from deep_anc.train.finetune_readiness import (
    audit_finetune_completion,
    audit_finetune_readiness,
)

from test_finetune_readiness import _checkpoint, _completion_setup, _g4_metrics, _ready_config


def _check(report: dict, gate_id: str) -> dict:
    for item in report["checks"]:
        if item["id"] == gate_id:
            return item
    raise AssertionError(f"게이트 {gate_id!r} 가 리포트에 없습니다: {report}")


def _readiness_gate(cfg: dict, gate_id: str) -> dict:
    report = audit_finetune_readiness(cfg)
    check = _check(report, gate_id)
    assert not report["ok"], f"{gate_id} 가 실패했는데 전체가 통과했습니다"
    return check


# ----------------------------------------------------------------------------------
# 진입 게이트
# ----------------------------------------------------------------------------------
def test_config_fail_closed_flags_gate_fails_when_a_flag_is_off(tmp_path):
    """fail-closed 플래그 하나만 꺼도 진입이 막혀야 한다.

    이 게이트는 "실측 P(z) 를 요구한다"는 **의도 자체**를 지킨다. 플래그를 끄면
    surrogate 로 파인튜닝하고도 measured 라고 주장할 수 있다.
    """

    cfg = _ready_config(tmp_path)
    cfg["require_measured_primary_path"] = False

    check = _readiness_gate(cfg, "config_fail_closed_flags")
    assert not check["ok"]
    assert check["details"]["missing"] == ["require_measured_primary_path"]


def test_measured_primary_mode_gate_fails_on_surrogate(tmp_path):
    """surrogate P(z) 로는 실측 파인튜닝에 들어갈 수 없다.

    surrogate 모드는 F_P = F_S 라 최적해가 1탭 항등필터이고 상한이 −92 dB 다 — 과제가
    자명하다. 그 상태로 measured 라고 주장하면 성능 해석이 통째로 무너진다.
    """

    cfg = _ready_config(tmp_path)
    cfg["data"]["digital_primary_path_mode"] = "secondary_surrogate"

    check = _readiness_gate(cfg, "measured_primary_mode")
    assert not check["ok"]
    assert check["details"]["digital_primary_path_mode"] == "secondary_surrogate"


def test_recorded_mix_ratio_gate_fails_when_recorded_share_is_too_small(tmp_path):
    """승인된 혼합비와 다르면 막는다 — 실측 비중이 조용히 줄면 결과 해석이 달라진다."""

    cfg = _ready_config(tmp_path)
    cfg["recorded_ratio"] = 0.1

    check = _readiness_gate(cfg, "recorded_mix_ratio")
    assert not check["ok"]
    assert check["details"]["recorded_ratio"] == pytest.approx(0.1)


def test_recorded_transfer_snapshot_gate_fails_without_transfer_snapshot(tmp_path):
    """canonical fine-tune은 Elice transfer snapshot 없이는 시작할 수 없다."""

    cfg = _ready_config(tmp_path)
    cfg["experiment_role"] = "canonical_finetune"
    cfg["data"]["bootstrap_receipt"] = str(tmp_path / "missing_bootstrap_receipt.json")
    check = _readiness_gate(cfg, "recorded_transfer_snapshot")
    assert not check["ok"]
    assert "receipt" in check["message"] or "transfer" in check["message"]


def test_path_delay_and_lead_gate_fails_when_lead_is_hand_written(tmp_path):
    """설정 lead 가 측정 P/S 지연에서 유도되는 값과 다르면 막는다.

    커밋 aaeef41 의 사고를 그대로 재현한다: 누군가 lead 를 손으로 적었고, trainer 와
    게이트가 각자 유도해 109 와 113 으로 갈라진 채 양쪽 다 "통과" 였다.
    이제 lead 는 ``PlantDelays.lead()`` 한 곳에서만 나오고, 손으로 적힌 값은
    **유도값과 대조되어** 거부된다.
    """

    cfg = _ready_config(tmp_path)
    derived = cfg["data"]["digital_reference_lead_samples"]
    cfg["data"]["digital_reference_lead_samples"] = derived + 4

    check = _readiness_gate(cfg, "path_delay_and_lead")
    assert not check["ok"]
    assert check["details"]["configured_lead"] == derived + 4
    assert check["details"]["expected_lead"] == derived


def test_path_delay_and_lead_gate_follows_the_duct_config(tmp_path):
    """duct.yaml 의 지연을 바꾸면 게이트가 **새 값**을 요구한다 (단일 출처 전파).

    한 곳만 바꿔도 전부 따라오는지 확인한다 — handoff 를 늘리면 요구 lead 가 같은
    양만큼 늘어야 한다. 예전에는 lead 유도가 5곳에 흩어져 있어 한 곳만 따라왔다.
    """

    cfg = _ready_config(tmp_path)
    base = audit_finetune_readiness(cfg)
    assert _check(base, "path_delay_and_lead")["ok"], base

    cfg["duct"]["secondary_path"]["handoff_extra_samples"] += 3
    moved = _check(audit_finetune_readiness(cfg), "path_delay_and_lead")
    assert not moved["ok"]
    assert (
        moved["details"]["expected_lead"]
        == _check(base, "path_delay_and_lead")["details"]["digital_reference_lead_samples"] + 3
    )


def test_recorded_dataset_qa_gate_fails_on_a_broken_session(tmp_path):
    """세션 하나만 깨져도 진입이 막혀야 한다 — QA 는 최악값 게이트다."""

    cfg = _ready_config(tmp_path)
    manifest = Path(cfg["recorded_manifest"])
    first = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    broken = Path(first["path"]) / "mics.wav"
    sf.write(broken, np.zeros((8, 2), dtype=np.float32), int(first["sample_rate"]), subtype="FLOAT")

    check = _readiness_gate(cfg, "recorded_dataset_qa")
    assert not check["ok"]


def test_readiness_aggregate_gate_fails_when_any_child_fails(tmp_path):
    """집계 게이트는 자식 하나만 실패해도 실패해야 한다 (실패 폐쇄).

    ``require_finetune_readiness`` 가 이 값을 읽어 학습 진입을 막는다. 집계가
    자식과 어긋나면 게이트 전체가 장식이 된다.
    """

    from deep_anc.train.finetune_readiness import require_finetune_readiness

    cfg = _ready_config(tmp_path)
    cfg["recorded_ratio"] = 0.1

    report = audit_finetune_readiness(cfg)
    assert not report["ok"]
    assert not _check(report, "recorded_mix_ratio")["ok"]
    with pytest.raises(RuntimeError):
        require_finetune_readiness(cfg)


# ----------------------------------------------------------------------------------
# 완료 게이트
# ----------------------------------------------------------------------------------
def test_measured_finetune_checkpoint_gate_fails_on_surrogate_physics(tmp_path):
    """surrogate 로 학습한 checkpoint 는 완료 후보가 될 수 없다."""

    cfg, best, manifest, run = _completion_setup(tmp_path)
    surrogate_cfg = {
        **cfg,
        "physics_status": "secondary_surrogate_representation_pretrain",
        "digital_reference_lead_samples": cfg["data"]["digital_reference_lead_samples"],
    }
    _checkpoint(best, cfg=surrogate_cfg, step=4)
    _checkpoint(best.parent / "last.pt", cfg=surrogate_cfg, step=6)
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    _g4_metrics(test_metrics, split="test", checkpoint=best, manifest=manifest)

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )
    check = _check(report, "measured_finetune_checkpoint")
    assert not report["ok"]
    assert not check["ok"]
    assert "physics_status" in check["message"]


def test_recorded_manifest_provenance_gate_fails_without_manifest(tmp_path):
    """manifest 지문을 계산할 수 없으면 G4 판정 자체가 무의미하다 — 실패 폐쇄."""

    cfg, best, manifest, run = _completion_setup(tmp_path)
    val_metrics = run / "eval_recorded_val" / "metrics.npz"
    test_metrics = run / "eval_recorded_test" / "metrics.npz"
    _g4_metrics(val_metrics, split="val", checkpoint=best, manifest=manifest)
    _g4_metrics(test_metrics, split="test", checkpoint=best, manifest=manifest)
    manifest.unlink()

    report = audit_finetune_completion(
        cfg, checkpoint=best, val_metrics=val_metrics, test_metrics=test_metrics
    )
    assert not report["ok"]
    assert not _check(report, "recorded_manifest_provenance")["ok"]
    # provenance 가 없으면 G4 는 판정 불가로 함께 막혀야 한다.
    assert not _check(report, "recorded_val_g4")["ok"]


# ----------------------------------------------------------------------------------
# 런타임 게이트
# ----------------------------------------------------------------------------------
def test_runtime_lead_gate_fails_when_checkpoint_disagrees():
    """배포 lead 와 checkpoint lead 가 다르면 시작하지 않는다.

    실측: 결정론적 lead 오차 δ=16 샘플이면 150–600Hz 평균 잔차가 −1.95 dB 이고
    600Hz 에서는 **+1.40 dB 증폭**이다. 조용한 성능 손실이 아니라 부호가 바뀐다.
    """

    with pytest.raises(ValueError, match="lead 불일치"):
        validate_digital_reference_lead("digital", 116, 109)

    # 정상: 같으면 통과하고 정규화된 값을 돌려준다.
    assert validate_digital_reference_lead("digital", 116, 116) == 116
