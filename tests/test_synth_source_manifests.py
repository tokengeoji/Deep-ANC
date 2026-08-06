"""선언한 소스 태그가 조용히 사라지는 것을 막는다 (절대목표 2).

왜 이 파일이 있는가
------------------
2026-08-06 감사: ``source_mix_ratio`` 가 선언한 태그 중 manifest 가 없는 것이 5개였고
합계 선언 비중이 **0.70** 이었다(speech 0.15 · music 0.10 · machine 0.07 · demand 0.08 ·
dns_fullband 0.30). 그런데 학습은 멈추지 않았다 — ``SynthANCDataset._pool()`` 이 한 줄
출력하고 태그를 버린 뒤 합성원으로 대체했기 때문이다. 그 코드는 **DataLoader 워커
안에서** 돌아서 출력이 아무도 안 보는 로그로 간다.

그 상태로 100k step 을 돌리면 **음성과 음악을 한 번도 보지 못한 모델**이 나온다. 절대목표
2는 "소음·음성·음악을 모두, 최악값 기준으로 제거" 이므로 이것은 목표 위반이고, 60시간을
태운 뒤에야 알게 된다.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from deep_anc.data.synth_dataset import assert_declared_sources_exist


def _manifest(tmp_path: Path, tag: str) -> str:
    path = tmp_path / f"{tag}.jsonl"
    path.write_text(
        json.dumps(
            {
                "path": str(tmp_path / f"{tag}.wav"),
                "duration_s": 5.0,
                "sample_rate": 48000,
                "channels": 1,
                "tag": tag,
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


# --------------------------------------------------------------------------- 음성 대조
def test_missing_declared_source_manifest_is_rejected(tmp_path: Path) -> None:
    pools = {
        "speech": [_manifest(tmp_path, "speech")],
        "dns_fullband": [str(tmp_path / "dns_fullband.jsonl")],  # 없음
        "machine": [str(tmp_path / "machine.jsonl")],  # 없음
    }
    mix = {"synthetic": 0.55, "speech": 0.15, "dns_fullband": 0.23, "machine": 0.07}
    with pytest.raises(FileNotFoundError) as excinfo:
        assert_declared_sources_exist(pools, mix)
    message = str(excinfo.value)
    # 무엇이 없는지와 **얼마나 잃는지**가 메시지에 있어야 한다. 태그 이름만 있으면
    # 사람이 0.30 을 잃는 것과 0.03 을 잃는 것을 구별하지 못한다.
    assert "dns_fullband" in message and "machine" in message
    assert "speech" not in message.split("둘 중 하나")[0]
    assert "0.30" in message, f"합계 선언 비중이 메시지에 없습니다:\n{message}"


def test_the_declared_weight_of_every_missing_tag_is_reported(tmp_path: Path) -> None:
    pools = {"music": [str(tmp_path / "music.jsonl")]}
    with pytest.raises(FileNotFoundError, match=r"0\.10"):
        assert_declared_sources_exist(pools, {"music": 0.10})


# --------------------------------------------------------------------------- 양성 대조
def test_all_declared_manifests_present_passes(tmp_path: Path) -> None:
    """게이트가 **꺼져서** 통과하는 것이 아님을 증명한다."""

    pools = {tag: [_manifest(tmp_path, tag)] for tag in ("speech", "music", "esc50")}
    assert_declared_sources_exist(pools, {"speech": 0.15, "music": 0.10, "esc50": 0.05})
    assert set(pools) == {"speech", "music", "esc50"}, "통과 경로가 풀을 건드렸습니다"


def test_synthetic_only_configuration_passes(tmp_path: Path) -> None:
    assert_declared_sources_exist({}, {"synthetic": 1.0})


# ------------------------------------------------------------------- 탈출구는 시끄럽다
def test_diagnostic_escape_warns_and_drops_the_tag(tmp_path: Path) -> None:
    """``allow_missing`` 은 통과시키되 **경고를 남기고** 풀에서 태그를 제거한다.

    조용히 통과하면 예전 상태로 정확히 되돌아간다. 탈출구가 있어도 된다 — 다만
    그것을 쓴 흔적이 남아야 한다.
    """

    pools = {"dns_fullband": [str(tmp_path / "dns.jsonl")]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_declared_sources_exist(pools, {"dns_fullband": 0.30}, allow_missing=True)
    assert any("진단 모드" in str(w.message) for w in caught), "탈출구가 조용했습니다"
    assert pools == {}, "탈출구가 없는 태그를 풀에 남겼습니다"


# --------------------------------------------------- 출하 설정이 실제로 이 게이트를 통과하는가
def test_shipped_mix_ratio_is_checked_against_real_manifests() -> None:
    """지금 저장소 상태를 그대로 판정한다 — 통과하든 실패하든 **사실을 말한다**.

    이 테스트는 "manifest 를 만들어라" 를 강제하지 않는다. 유실 코퍼스를 확보할지
    혼합비를 다시 선언할지는 사람이 정할 문제다. 여기서 강제하는 것은 **그 상태가
    조용히 지나가지 않는다** 는 것뿐이다.
    """

    import yaml

    from deep_anc.config import REPO_ROOT, _resolve_path

    data_cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "data_sim.yaml").read_text(encoding="utf-8")
    )
    mix = data_cfg["source_mix_ratio"]
    manifest_dir = _resolve_path(data_cfg.get("noise_manifest_dir", "data/manifests"))
    pools = {
        tag: [str(manifest_dir / f"{tag}.jsonl")]
        for tag, ratio in mix.items()
        if tag != "synthetic" and float(ratio) > 0.0
    }
    missing = [tag for tag, paths in pools.items() if not Path(paths[0]).is_file()]
    if missing:
        with pytest.raises(FileNotFoundError):
            assert_declared_sources_exist(dict(pools), mix)
    else:
        assert_declared_sources_exist(dict(pools), mix)
