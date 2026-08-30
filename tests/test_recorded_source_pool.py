"""누수 게이트가 **설정이 아니라 세션이 실제로 재생한 것**을 보는지 강제한다.

왜 이 파일이 있는가
------------------
2026-08-06 감사가 재현한 fail-open. ``readiness.recorded_source_pool_csv`` 는 v1
(``data/source_pool/sources.csv``)을 가리키고 있었는데, v1 은 machine 이 **8 그룹**뿐이라
분할 하한(계열별 9 = val 4 + test 4 + train 1)을 만족할 수 없다. 즉 재녹음은 v2 로 해야
한다. 그런데 이 키를 안 고치고 v2 로 녹음하면, 누수 게이트가 **v1 클립끼리 비교해 PASS
하면서 v2 누수를 100% 통과**시킨다.

원인은 발생기 A 다 — "실측이 어떤 오디오를 썼는가" 라는 하나의 물리량을 설정과 세션이
따로 들고 있고 아무도 대조하지 않았다. 세션의 ``program.file`` 이 물리적 사실이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_anc.train.finetune_readiness import observed_source_pools


def _session(root: Path, name: str, played: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "session.json").write_text(
        json.dumps(
            {
                "session_id": name,
                "source_family": "machine",
                "group_id": "machine-fan",
                "program": {"type": "file", "file": played},
            }
        ),
        encoding="utf-8",
    )


# ------------------------------------------------------------------- 관측이 사실을 말하는가
def test_observed_pool_is_read_from_what_the_session_played(tmp_path: Path) -> None:
    _session(tmp_path, "s1", "data/source_pool_v2/machine/machine_000.wav")
    _session(tmp_path, "s2", "data/source_pool_v2/speech/speech_003.wav")
    assert observed_source_pools(tmp_path) == {"data/source_pool_v2/sources.csv": 2}


def test_two_pools_in_one_recording_are_both_reported(tmp_path: Path) -> None:
    """풀이 섞이면 **양쪽이 다 보여야** 한다. 하나만 골라 쓰면 나머지 클립이 샌다.

    섞임 자체는 정상이다 — 복구 47세션(v1)과 신규 33세션(v2)을 합치는 것이 스피커
    시간을 93.3분에서 38.5분으로 줄이는 정상 경로다. 위험한 것은 게이트가 한쪽만
    보는 것이고, 그래서 관측은 **본 것을 전부** 보고해야 한다.
    """

    _session(tmp_path, "old", "data/source_pool/environment/environment_000.wav")
    _session(tmp_path, "new", "data/source_pool_v2/environment/environment_000.wav")
    assert observed_source_pools(tmp_path) == {
        "data/source_pool/sources.csv": 1,
        "data/source_pool_v2/sources.csv": 1,
    }


def test_missing_or_broken_sessions_are_skipped_not_guessed(tmp_path: Path) -> None:
    (tmp_path / "no_meta").mkdir()
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "session.json").write_text("{ not json", encoding="utf-8")
    _session(tmp_path, "good", "data/source_pool_v2/music/music_001.wav")
    assert observed_source_pools(tmp_path) == {"data/source_pool_v2/sources.csv": 1}


def test_empty_root_reports_nothing_rather_than_a_default(tmp_path: Path) -> None:
    """세션이 없으면 **모른다**고 해야 한다. 기본값을 지어내면 그것이 두 번째 유도다."""

    assert observed_source_pools(tmp_path) == {}
    assert observed_source_pools(tmp_path / "does_not_exist") == {}


# ------------------------------------------------------- 이 저장소의 실제 풀이 하한을 만족하는가
def _pool_groups(csv_path: Path) -> dict[str, set[str]]:
    import csv

    out: dict[str, set[str]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["source_family"], set()).add(row["group_id"])
    return out


def test_v1_pool_cannot_satisfy_the_split_floor_but_v2_can() -> None:
    """왜 기본값을 v2 로 바꿨는지를 **숫자로** 고정한다.

    누가 편의로 v1 을 되돌리면 여기서 실패한다. 하한 9 = val 4 + test 4 + train 1 이고,
    이것은 ``eval.recorded.MIN_GROUPS_PER_FAMILY`` 에서 유도된다.
    """

    from deep_anc.config import REPO_ROOT
    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY

    floor = 2 * MIN_GROUPS_PER_FAMILY + 1
    v1 = REPO_ROOT / "data" / "source_pool" / "sources.csv"
    v2 = REPO_ROOT / "data" / "source_pool_v2" / "sources.csv"
    if not v1.is_file() or not v2.is_file():
        pytest.skip("소스풀 CSV 가 없는 환경")

    v1_groups = {f: len(g) for f, g in _pool_groups(v1).items()}
    v2_groups = {f: len(g) for f, g in _pool_groups(v2).items()}
    assert min(v1_groups.values()) < floor, (
        f"v1 이 하한을 만족합니다({v1_groups}) — 그렇다면 기본값을 v2 로 바꾼 근거가 "
        "사라졌으니 이 테스트와 기본값을 함께 재검토하세요"
    )
    assert v1_groups["machine"] == 8, f"v1 machine 그룹 수가 바뀌었습니다: {v1_groups}"
    assert min(v2_groups.values()) >= floor, f"v2 가 하한 {floor} 을 못 채웁니다: {v2_groups}"


def test_shipped_defaults_point_at_v2() -> None:
    """세 곳의 기본값이 다시 갈라지지 않게 한다 (발생기 A)."""

    import yaml

    from deep_anc.config import REPO_ROOT

    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "train_finetune.yaml").read_text(encoding="utf-8")
    )
    declared = cfg["readiness"]["recorded_source_pool_csv"]
    declared = [declared] if isinstance(declared, str) else list(declared)
    # v2 는 반드시 있어야 한다 — 신규 녹음은 v2 로만 받는다(v1 은 machine 8 그룹).
    assert "data/source_pool_v2/sources.csv" in declared, declared
    # v1 도 선언돼 있어야 복구 47세션을 섞어 쓸 수 있다 (93.3분 → 38.5분).
    assert "data/source_pool/sources.csv" in declared, declared

    batch = (REPO_ROOT / "scripts" / "data" / "record_session_batch.py").read_text(
        encoding="utf-8"
    )
    assert '"data/source_pool_v2/sources.csv"' in batch, (
        "record_session_batch 의 --sources 기본값이 v2 가 아닙니다 — 그 풀로 녹음하면 "
        "make_recorded_manifest 가 EXIT=2 로 거부합니다"
    )
