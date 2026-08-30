"""``scripts/data/prepare_noise_pool.py`` 의 빌드 게이트 짝.

⚠ 2026-08-06 통합 검증에서 추가됐다. 이 게이트(선언한 태그를 못 만들면 종료코드 1)는
게이트 레지스트리 정비와 **같은 변경 안에서 선언 없이 만들어졌다** — ``grep -rn
prepare_noise_pool tests/`` 0건, ``gates_for_owner(...)`` 빈 튜플이었다. 즉 "모든
게이트는 짝 없이 존재할 수 없다"가 선언된 게이트에 대해서만 참이었고, 새 게이트를
선언 없이 만드는 발생기는 그 변경 안에서 다시 돌았다.

여기서는 양방향을 다 본다: 선언 태그가 없으면 실패하고(negative), 전부 있으면
성공한다(positive). 후자가 없으면 "항상 실패하는 게이트" 와 구별되지 않는다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/prepare_noise_pool.py"
FS = 48_000


def _load_script():
    spec = importlib.util.spec_from_file_location("_prepare_noise_pool_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_clips(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for index in range(count):
        data = rng.standard_normal(FS).astype(np.float32) * 0.05
        sf.write(directory / f"clip_{index:03d}.wav", data, FS)


def _build_tree(tmp_path: Path, present_tags: dict[str, float], ratios: dict[str, float]):
    """가짜 REPO_ROOT 를 만든다. 태그별 원본 유무를 시험이 직접 정한다."""

    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/data_sim.yaml").write_text(
        yaml.safe_dump({"source_mix_ratio": ratios}), encoding="utf-8"
    )
    for tag in present_tags:
        # data/raw/<계열>/<tag>/ — 스크립트가 깊이를 가정하지 않는지도 함께 본다.
        _write_clips(tmp_path / "data/raw" / f"{tag}_family" / tag, 6)
    return tmp_path


def _run(module, root: Path, monkeypatch) -> int:
    monkeypatch.setattr(module, "REPO_ROOT", root)
    monkeypatch.setattr(
        sys, "argv", ["prepare_noise_pool.py", "--out", "data/manifests"]
    )
    return module.main()


def test_missing_declared_tag_fails_the_build(tmp_path, monkeypatch, capsys):
    """비율 > 0 인데 원본이 없는 태그가 있으면 **종료코드 1**.

    이것이 없으면 ``synth_dataset`` 이 그 태그를 로그 없이 합성원으로 폴백하고,
    학습은 선언한 ``source_mix_ratio`` 와 다른 데이터로 돈다. 출하 상태가 실제로
    이렇다 — ``dns_fullband``/``demand``/``machine`` 원본이 유실됐다.
    """

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 0.4, "speech": 0.3},
        ratios={"esc50": 0.4, "speech": 0.3, "machine": 0.3, "synthetic": 0.1},
    )

    assert _run(module, root, monkeypatch) == 1

    err = capsys.readouterr().err
    assert "machine" in err and "[실패]" in err
    # 원인을 진단으로 남겨야 한다 — 조용한 폴백이 이 게이트가 막는 것이다.
    assert "폴백" in err


def test_every_declared_tag_present_builds_all_manifests(tmp_path, monkeypatch, capsys):
    """**positive 짝** — 선언 태그를 정확히 다 채운 최소 구성에서 통과한다.

    경계까지 몰아본다: 비율 0 인 태그(``demand``)는 원본이 없어도 요구하지 않아야
    한다(``required_tags()`` 가 비율 > 0 만 센다). 그 태그까지 요구하면 게이트가
    "항상 실패" 로 굳고, 그러면 다음 사람이 게이트째로 끈다.
    """

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 0.4, "speech": 0.3, "music": 0.3},
        ratios={
            "esc50": 0.4,
            "speech": 0.3,
            "music": 0.3,
            "demand": 0.0,  # 선언은 남아 있으나 지금은 쓰지 않는다
            "synthetic": 0.1,
        },
    )

    assert _run(module, root, monkeypatch) == 0

    out = capsys.readouterr().out
    assert "완료: manifest 3개" in out
    for tag in ("esc50", "speech", "music"):
        assert (root / "data/manifests" / f"{tag}.jsonl").is_file()
    # 비율 0 인 태그는 만들지도, 요구하지도 않는다.
    assert not (root / "data/manifests/demand.jsonl").exists()


def test_the_gate_is_declared_in_the_registry():
    """스크립트 exit 게이트도 레지스트리에 있어야 한다 — 이 파일이 생긴 이유다."""

    from deep_anc.ops.gate_registry import gates_for_owner

    declared = gates_for_owner("scripts/data/prepare_noise_pool.py")
    assert [gate.gate_id for gate in declared] == ["noise_pool_declared_tags_exist"]
    gate = declared[0]
    assert gate.negative_fixture.startswith("tests/test_prepare_noise_pool.py::")
    assert gate.positive_fixture.startswith("tests/test_prepare_noise_pool.py::")


def test_the_shipped_config_still_declares_tags_whose_sources_are_missing():
    """출하 상태를 못 박는다 — 이 게이트는 지금 **실패하는 것이 정답**이다.

    ``dns_fullband``/``demand``/``machine`` 원본은 실제로 유실됐다. 있는 척하지
    않는다. 사람이 (원본 재수집) 또는 (혼합비에서 태그 제거) 중 하나를 선언해야
    하고, 그때 이 테스트가 먼저 깨져서 결정이 눈에 띈다.
    """

    from deep_anc.config import REPO_ROOT

    module = _load_script()
    pools = module.declared_pools(REPO_ROOT / "configs/data_sim.yaml")
    required = set(module.PoolPlan(pools=pools, roots=("data/raw",)).required_tags())

    found = module.discover_tag_dirs(REPO_ROOT / "data/raw", frozenset(required))
    if not (REPO_ROOT / "data/raw").is_dir():
        pytest.skip("data/raw 가 없는 트리입니다 (.gitignore 대상)")
    assert required - set(found), (
        "선언 태그의 원본이 전부 생겼습니다 — prepare_noise_pool 이 이제 통과합니다. "
        "이 테스트를 지우고 HANDOFF 의 '원본 유실' 항목을 닫으세요"
    )
