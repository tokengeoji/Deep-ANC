"""녹음 소스 풀의 **그룹 정의**를 검사한다.

왜 이 파일이 있는가
------------------
2026-08-06 실측: `group_id` 를 ESC-50 **카테고리**로 잡았더니 machine 이 8 그룹뿐이었고,
그 8 그룹으로는 `min_groups_per_family_per_split=4` (val 4 + test 4 + train 1 = 9)를
만족할 수 없어 manifest 생성이 실패한다. **세션을 아무리 늘려도 그룹은 늘지 않는다** —
같은 그룹 안의 세션은 독립이 아니기 때문이다.

그룹 수는 재녹음 분량을 직접 정하는 값이라(스피커 연결 시간 = 하드웨어 수명) 회귀가
생기면 조용히 통과시키면 안 된다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_ESC50_ROOT = REPO_ROOT / "data/raw/noise/esc50"
_META = _ESC50_ROOT / "ESC-50-master" / "meta" / "esc50.csv"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_recording_sources", REPO_ROOT / "scripts/data/build_recording_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_meta(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """(filename, category, src_file) 로 최소 ESC-50 메타를 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["filename,fold,target,category,esc10,src_file,take"]
    for filename, category, src_file in rows:
        lines.append(f"{filename},1,0,{category},False,{src_file},A")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_groups_split_a_category_by_source_recording():
    """한 카테고리 안의 서로 다른 원본 녹음이 서로 다른 그룹이 되어야 한다.

    `SRC_FILES_PER_GROUP` 개씩 묶이므로 src_file 12 개 → 그룹 3 개다.
    """
    module = _load_builder()
    per_group = int(module.SRC_FILES_PER_GROUP)
    rows = [
        (f"1-{index:05d}-A-0.wav", "engine", f"{index:05d}")
        for index in range(per_group * 3)
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_meta(root / "ESC-50-master" / "meta" / "esc50.csv", rows)
        buckets = module.collect_esc50(root)

    groups = {group for _, group in buckets["machine"]}
    assert len(groups) == 3, groups
    assert all(group.startswith("engine-") for group in groups)
    # 파일은 하나도 잃지 않는다.
    assert len(buckets["machine"]) == len(rows)


def test_a_group_never_spans_two_categories():
    """세션이 여러 카테고리를 섞으면 계열별 최악값(절대목표 2)을 귀속시킬 수 없다."""
    module = _load_builder()
    rows = [(f"1-{i:05d}-A-0.wav", "engine", f"{i:05d}") for i in range(2)]
    rows += [(f"1-{i:05d}-A-1.wav", "chainsaw", f"{i:05d}") for i in range(2)]
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_meta(root / "ESC-50-master" / "meta" / "esc50.csv", rows)
        buckets = module.collect_esc50(root)

    by_group: dict[str, set[str]] = {}
    for path, group in buckets["machine"]:
        # 파일명 마지막 필드로 카테고리를 되짚는다 (fixture 규약).
        by_group.setdefault(group, set()).add(path.name.rsplit("-", 1)[-1])
    for group, suffixes in by_group.items():
        assert len(suffixes) == 1, f"{group} 이 카테고리를 가로질렀다: {suffixes}"


def test_missing_src_file_column_falls_back_to_category():
    """구형 메타데이터에서도 죽지 않는다 — 그룹이 거칠어질 뿐이다."""
    module = _load_builder()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        meta = root / "ESC-50-master" / "meta" / "esc50.csv"
        meta.parent.mkdir(parents=True)
        meta.write_text(
            "filename,fold,target,category,esc10,take\n"
            "1-1-A-0.wav,1,0,engine,False,A\n"
            "1-2-A-0.wav,1,0,engine,False,A\n",
            encoding="utf-8",
        )
        buckets = module.collect_esc50(root)

    assert len(buckets["machine"]) == 2
    assert {group for _, group in buckets["machine"]} == {"engine-00"}


@pytest.mark.skipif(not _META.exists(), reason="ESC-50 원본이 없는 트리")
def test_real_esc50_yields_enough_groups_for_the_split_floor():
    """실제 ESC-50 에서 계열당 그룹이 G4 하한 + train 최소 1 을 넘어야 한다.

    이 수가 재녹음 분량을 정한다. 회귀하면 스피커를 더 오래 켜야 하고, 그것이
    사용자가 명시한 하드웨어 수명 제약을 직접 건드린다.
    """
    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY

    module = _load_builder()
    buckets = module.collect_esc50(_ESC50_ROOT)
    floor = 2 * int(MIN_GROUPS_PER_FAMILY) + 1
    for family in ("machine", "environment"):
        groups = {group for _, group in buckets[family]}
        assert len(groups) >= floor, (
            f"{family} 그룹이 {len(groups)} 개로 하한 {floor} 미만이다 — "
            "이 상태로는 manifest 생성이 실패한다"
        )
    # 회귀 감시용 실측 기준선 (2026-08-06): machine 55 / environment 126.
    assert len({g for _, g in buckets["machine"]}) >= 40
    assert len({g for _, g in buckets["environment"]}) >= 100


def test_already_recorded_clips_are_excluded_to_keep_groups_disjoint():
    """일부 세션만 다시 녹음할 때 같은 클립이 옛 그룹과 새 그룹에 함께 들어가면 안 된다.

    실측 근거(2026-08-06): 재정렬로 47세션이 살아남아 33세션만 다시 받으면 되는데,
    새 소스를 세분 그룹으로 다시 만들면 같은 오디오가 train 과 test 에 함께 놓인다.
    group 단위 split 이 그 자리에서 무의미해진다.
    """
    module = _load_builder()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "sources.csv"
        csv_path.write_text(
            "source_family,session_index,group_id,path,seconds,sample_rate_hz,"
            "clip_count,crest_factor_db,rms_at_unit_peak,clips\n"
            'machine,0,machine-engine,p.wav,70.0,48000,2,9.7,0.3,'
            '"[""1-00001-A-0.wav"", ""1-00002-A-0.wav""]"\n',
            encoding="utf-8",
        )
        used = module.clips_used_by_sessions([csv_path])

    assert used == {"1-00001-A-0.wav", "1-00002-A-0.wav"}


def test_unreadable_clip_provenance_stops_the_build_instead_of_pretending():
    """읽지 못한 목록을 빈 집합으로 흘려보내면 '누수를 막는 척'이 된다 — 멈춰야 한다.

    이것이 이 저장소에서 반복된 fail-open 의 모양이다: 검사가 돌기는 하는데
    입력이 비어 있어서 아무것도 걸러내지 못하고, 로그만 보면 통과한 것처럼 보인다.
    """
    module = _load_builder()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "sources.csv"
        csv_path.write_text(
            "source_family,clips\nmachine,not-a-python-literal\n", encoding="utf-8"
        )
        assert module.clips_used_by_sessions([csv_path]) == set()

        # 짝이 되는 방어: 빈 집합이면 소스를 만들지 않고 EXIT=2 로 멈춘다.
        exit_code = module.main(
            [
                "--out", str(Path(tmp) / "pool"),
                "--sessions-per-family", "1",
                "--seconds", "1.0",
                "--families", "machine",
                "--keep-disjoint-from", str(csv_path),
            ]
        )
    assert exit_code == 2
