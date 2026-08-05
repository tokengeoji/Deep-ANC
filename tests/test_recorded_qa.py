"""실측 세션 스트리밍 QA 게이트 회귀 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter

from deep_anc.data.manifest import MANIFEST_PATH_BASE, read_manifest, write_manifest
from deep_anc.data.recorded_qa import (
    RecordedQASettings,
    render_recorded_qa_markdown,
    validate_recorded_sessions,
)
from scripts.data.validate_recorded_sessions import main as qa_main


FS = 48_000


def _settings(*, minimum_segment: int = 512, lead: int = 7) -> RecordedQASettings:
    return RecordedQASettings(
        sample_rate=FS,
        segment_samples=minimum_segment,
        digital_reference_lead_samples=lead,
        block_frames=127,
        required_splits=("train", "val", "test"),
    )


ERR_DELAY_SAMPLES = 120
REF_DELAY_SAMPLES = 60

SESSION_FRAMES = 3 * FS
"""픽스처 세션의 기본 길이(3 초).

옛 기본값은 1200 샘플(25 ms)이었다. 그 길이로는 지연 궤적 창이 하나도 서지 않으므로
정렬 게이트가 "측정했다"고 말할 수 없다 — 그런데도 전부 PASS 였다. 즉 픽스처 자체가
게이트를 우회시키고 있었고, 그것이 이 저장소가 반복해 온 실패 방식(반증된 적 없는
게이트)이다. 실기 세션은 70 초이고 QA 는 앞 30 초를 본다. 3 초는 단위 테스트가
감당하는 범위에서 창이 40개 이상 서는 최소 길이다."""


def _band_noise(frames: int, seed: int) -> np.ndarray:
    """제어 대역을 덮는 광대역 잡음. 톤이 아니라 잡음이어야 하는 이유가 있다.

    순음은 상호상관 최대점이 주기마다 반복돼 지연이 다중값이 된다. 실제 프로그램
    소재(음성/음악/환경음)는 광대역이므로 픽스처도 광대역이어야 게이트가 실기와
    같은 조건에서 검증된다.
    """

    rng = np.random.default_rng(seed)
    b, a = butter(4, [100.0 / (FS / 2), 2000.0 / (FS / 2)], btype="band")
    filtered = lfilter(b, a, rng.standard_normal(frames + 4096))[4096:]
    peak = float(np.max(np.abs(filtered))) or 1.0
    return (0.25 * filtered / peak).astype(np.float64)


def _signals(
    frames: int, *, source_channels: int = 1, seed: int = 20260805
) -> tuple[np.ndarray, np.ndarray]:
    """**하나의 소스에서 지연·감쇠로 파생된** 3채널 세션.

    왜 이 픽스처를 바꿨는가 (이 교체 자체가 결함 2의 증거다)
    ------------------------------------------------------
    이전 픽스처는 ``mics ch0 = 310Hz 사인``, ``ch1 = 430Hz 사인``, ``source = 310Hz
    사인`` 이었다. ch0 와 ch1 은 **음향적으로 아무 관계가 없는 두 신호**인데 QA 는
    이것을 통과시켰다 — 즉 그 시절 QA 는 채널 사이의 관계를 아예 보지 않았다는 것이
    픽스처로 증명돼 있었고, 아무도 그렇게 읽지 않았다.

    실기에서 ERR·REF 두 마이크는 같은 음장을 서로 다른 지점에서 듣는다. 따라서 둘은
    지연만 다른 강한 상관 관계여야 한다(실측 coh² 0.959~0.991). 픽스처도 그래야 한다.
    """

    source = _band_noise(frames, seed)
    detector = np.random.default_rng(seed + 1)
    err = 0.8 * np.roll(source, ERR_DELAY_SAMPLES) + 0.01 * detector.standard_normal(frames)
    ref = 0.9 * np.roll(source, REF_DELAY_SAMPLES) + 0.01 * detector.standard_normal(frames)
    mics = np.stack([err, ref], axis=1).astype(np.float32)
    source_out = source.astype(np.float32)
    if source_channels == 2:
        source_out = np.stack([source_out, source_out], axis=1)
    return mics, source_out


def _timebase_collapsed_source(frames: int, *, seed: int = 20260805) -> np.ndarray:
    """``source.wav`` 만 시간축이 깨진 세션 — 2026-08-04 사고의 재현.

    ERR/REF 두 마이크는 손대지 않는다. 실측에서 무너진 것은 음향이 아니라 재생과
    캡처의 시간 대응이었기 때문이다: ``coh²(REF→ERR) = 0.959~0.991`` 로 멀쩡한데
    ``coh²(source→ERR) = 0.021~0.126`` 이었다. 이 픽스처가 그 조합을 그대로 만든다.
    """

    source = _band_noise(frames, seed)
    rng = np.random.default_rng(seed + 99)
    block = max(64, frames // 8)
    broken = np.zeros(frames, dtype=np.float64)
    for start in range(0, frames, block):
        stop = min(frames, start + block)
        jump = int(rng.integers(-4000, 4000))
        broken[start:stop] = np.roll(source, jump)[start:stop]
    return broken.astype(np.float32)


def _write_session(
    session: Path,
    *,
    session_id: str,
    group_id: str,
    family: str = "speech",
    frames: int = SESSION_FRAMES,
    mics: np.ndarray | None = None,
    source: np.ndarray | None = None,
    mics_sr: int = FS,
    source_sr: int = FS,
    metadata: dict | None = None,
) -> dict:
    session.mkdir(parents=True)
    default_mics, default_source = _signals(frames)
    mics = default_mics if mics is None else mics
    source = default_source if source is None else source
    sf.write(session / "mics.wav", mics, mics_sr, subtype="FLOAT")
    sf.write(session / "source.wav", source, source_sr, subtype="FLOAT")
    if metadata is None:
        metadata = {
            "session_id": session_id,
            "group_id": group_id,
            "source_family": family,
            "sample_rate": FS,
            "seconds": frames / FS,
            "program": {"type": "tone"},
        }
    (session / "session.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "path": str(session),
        "duration_s": frames / FS,
        "sample_rate": FS,
        "channels": 2,
        "tag": "recorded",
        "session_id": session_id,
        "group_id": group_id,
        "source_family": family,
    }


def test_recorded_qa_consumes_resolved_manifest_paths_and_streams_all_blocks(tmp_path):
    manifest = tmp_path / "data" / "manifests" / "recorded.jsonl"
    sessions_root = tmp_path / "data" / "recorded"
    entries = []
    for split, suffix in (("train", "a"), ("val", "b"), ("test", "c")):
        session_id = f"session-{suffix}"
        entry = _write_session(
            sessions_root / session_id,
            session_id=session_id,
            group_id=f"speaker-{suffix}",
        )
        entry.update(
            {
                "path": f"../recorded/{session_id}",
                "path_base": MANIFEST_PATH_BASE,
                "split": split,
            }
        )
        entries.append(entry)
    write_manifest(entries, manifest)

    resolved = read_manifest(manifest)
    report = validate_recorded_sessions(
        resolved, _settings(), manifest_path=str(manifest)
    )

    assert report["ok"]
    assert report["summary"]["sessions"] == 3
    assert report["summary"]["groups"] == 3
    assert all(Path(entry["path"]).is_absolute() for entry in resolved)
    assert all(
        session["audio"]["mics"]["blocks_read"] > 1
        and session["audio"]["source"]["blocks_read"] > 1
        for session in report["sessions"]
    )
    json.dumps(report, ensure_ascii=False)
    markdown = render_recorded_qa_markdown(report)
    assert "판정: **PASS**" in markdown
    assert "Source-family 커버리지" in markdown


def test_group_split_leak_is_a_fatal_global_error(tmp_path):
    first = _write_session(
        tmp_path / "session-a",
        session_id="session-a",
        group_id="same-speaker",
    )
    second = _write_session(
        tmp_path / "session-b",
        session_id="session-b",
        group_id="same-speaker",
    )
    first["split"] = "train"
    second["split"] = "test"

    report = validate_recorded_sessions(
        [first, second],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=256,
            required_splits=("train", "test"),
        ),
    )

    assert not report["ok"]
    assert any("여러 split" in message for message in report["errors"])


def test_family_must_cover_all_required_splits_unless_diagnostic_override(tmp_path):
    entries = []
    for split, family in (("train", "speech"), ("val", "speech"), ("test", "music")):
        session_id = f"{split}-{family}"
        entry = _write_session(
            tmp_path / session_id,
            session_id=session_id,
            group_id=f"group-{session_id}",
            family=family,
        )
        entry["split"] = split
        entries.append(entry)

    strict = validate_recorded_sessions(entries, _settings())
    diagnostic = validate_recorded_sessions(
        entries,
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=127,
            required_splits=("train", "val", "test"),
            allow_incomplete_family_coverage=True,
        ),
    )

    assert not strict["ok"]
    assert any("source_family=" in message for message in strict["errors"])
    assert diagnostic["ok"]
    assert diagnostic["warnings"]


def test_acoustic_reference_does_not_require_or_read_source_wav(tmp_path):
    entries = []
    for split in ("train", "val", "test"):
        session_id = f"acoustic-{split}"
        entry = _write_session(
            tmp_path / session_id,
            session_id=session_id,
            group_id=f"room-{split}",
            family="machine",
        )
        (tmp_path / session_id / "source.wav").unlink()
        entry["split"] = split
        entries.append(entry)

    report = validate_recorded_sessions(
        entries,
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=0,
            reference_mode="acoustic",
            block_frames=127,
            required_splits=("train", "val", "test"),
        ),
    )

    assert report["ok"]
    assert all("source" not in session["audio"] for session in report["sessions"])


def test_audio_shape_rate_and_lead_aware_minimum_length_fail(tmp_path):
    frames = 518  # segment 512 + lead 7 + 1보다 2샘플 부족
    mono_mics = np.full(frames, 0.05, dtype=np.float32)
    _, stereo_source = _signals(frames, source_channels=2)
    entry = _write_session(
        tmp_path / "bad-shape",
        session_id="bad-shape",
        group_id="group-shape",
        frames=frames,
        mics=mono_mics,
        source=stereo_source,
        source_sr=44_100,
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=128,
            required_splits=("train",),
        ),
    )
    errors = "\n".join(report["sessions"][0]["errors"])

    assert not report["ok"]
    assert "mics.wav는 정확히 2채널" in errors
    assert "source.wav는 mono" in errors
    assert "source.wav sample rate" in errors
    assert "최소길이 미달" in errors


def test_nonfinite_rms_clip_and_metadata_mismatch_fail(tmp_path):
    frames = 900
    mics, source = _signals(frames)
    mics[:, 1] = 0.0
    source[:] = 1.0
    source[0] = np.nan
    entry = _write_session(
        tmp_path / "bad-values",
        session_id="bad-values",
        group_id="group-values",
        family="music",
        frames=frames,
        mics=mics,
        source=source,
        metadata={
            "session_id": "different-session",
            "group_id": "different-group",
            "source_family": "speech",
            "sample_rate": 16_000,
            "seconds": 9.0,
            "program": {"type": "file"},
        },
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=128,
            required_splits=("train",),
        ),
    )
    errors = "\n".join(report["sessions"][0]["errors"])

    assert not report["ok"]
    assert "비유한 샘플" in errors
    assert "clip ratio" in errors
    assert "ch1 RMS" in errors
    assert "source_family" in errors
    assert "group_id" in errors
    assert "session_id" in errors
    assert "sample_rate" in errors
    assert "seconds" in errors


def test_missing_required_session_files_fail(tmp_path):
    session = tmp_path / "missing-files"
    session.mkdir()
    entry = {
        "path": str(session),
        "duration_s": 1.0,
        "sample_rate": FS,
        "channels": 2,
        "tag": "recorded",
        "session_id": "missing-files",
        "group_id": "group-missing",
        "source_family": "environment",
        "split": "train",
    }

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            required_splits=("train",),
        ),
    )
    errors = "\n".join(report["sessions"][0]["errors"])

    assert not report["ok"]
    assert "mics.wav" in errors
    assert "source.wav" in errors
    assert "session.json" in errors


def test_cli_writes_failure_reports_and_returns_nonzero_for_leaky_manifest(tmp_path):
    manifest = tmp_path / "leaky.jsonl"
    raw_entries = [
        {
            "path": "session-a",
            "split": "train",
            "group_id": "same",
            "source_family": "speech",
            "session_id": "session-a",
        },
        {
            "path": "session-b",
            "split": "test",
            "group_id": "same",
            "source_family": "speech",
            "session_id": "session-b",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in raw_entries), encoding="utf-8"
    )
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "sample_rate": FS,
                "segment_seconds": 0.02,
                "reference_mode": "digital",
                "digital_reference_lead_samples": 7,
            }
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "qa.md"
    out_json = tmp_path / "qa.json"

    code = qa_main(
        [
            "--manifest",
            str(manifest),
            "--data-config",
            str(data_config),
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ]
    )

    assert code == 1
    assert out_md.is_file() and out_json.is_file()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert not report["ok"]
    assert any("여러 split" in message for message in report["errors"])
    assert "판정: **FAIL**" in out_md.read_text(encoding="utf-8")


# ======================================================================================
# 채널 간 관계 게이트 (결함 2) — negative test 가 짝으로 붙는다
#
# 이 게이트가 없던 시절 실측 80 세션이 **전부** PASS 였다. 게이트가 작동하는지 아는
# 유일한 방법은 실제로 일어난 오염을 주입해 FAIL 하는 것을 보는 것이다.
# ======================================================================================
def test_qa_measures_source_err_alignment_on_a_healthy_session(tmp_path):
    """정상 세션에서 정렬 지표가 **실제로 측정되고 기록**되는지 본다.

    PASS 만으로는 부족하다 — 검사를 건너뛰고 PASS 한 것과 구분되지 않기 때문이다.
    그래서 측정값의 존재와 크기를 함께 단언한다.
    """

    entries = []
    for split in ("train", "val", "test"):
        entry = _write_session(
            tmp_path / f"aligned-{split}",
            session_id=f"aligned-{split}",
            group_id=f"group-{split}",
        )
        entry["split"] = split
        entries.append(entry)

    report = validate_recorded_sessions(entries, _settings())

    assert report["ok"]
    for session in report["sessions"]:
        alignment = session["alignment"]
        assert alignment["ok"]
        # 파생 관계가 살아 있으면 두 코히런스가 모두 높다.
        assert alignment["source_err_coherence"] > 0.9
        assert alignment["ref_err_coherence"] > 0.9
        # 지연이 상수이므로 흔들림이 거의 없어야 한다.
        assert alignment["source_err_delay_robust_std_samples"] < 8.0
        assert alignment["source_err_delay_p95_p5_samples"] < 48.0
        assert abs(alignment["source_err_delay_median_samples"] - ERR_DELAY_SAMPLES) <= 2
        # "측정했는가"까지 기록으로 남아야 한다 — 창이 서지 않은 채 통과하면
        # 검사를 건너뛴 것과 구분되지 않는다.
        assert alignment["delay_valid_windows"] >= 8
        assert alignment["delay_valid_window_ratio"] >= 0.5
        assert (
            alignment["delay_estimator"]
            == "deep_anc.data.timeline.measure_delay_trajectory"
        )


def test_qa_rejects_source_err_timebase_collapse(tmp_path):
    """2026-08-04 사고의 재현 — source 시간축만 깨진 세션이 FAIL 해야 한다.

    ERR/REF 마이크는 건드리지 않는다. 그래서 이 테스트는 게이트가
    "소리가 나쁘다"가 아니라 **"재생과 캡처의 시간 대응이 없다"** 를 잡는지 확인한다.
    실측 대응: coh²(source→ERR) 0.021~0.126 / coh²(REF→ERR) 0.959~0.991.
    """

    frames = 24_000
    mics, _ = _signals(frames)
    entry = _write_session(
        tmp_path / "collapsed",
        session_id="collapsed",
        group_id="group-collapsed",
        frames=frames,
        mics=mics,
        source=_timebase_collapsed_source(frames),
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=4096,
            required_splits=("train",),
        ),
    )

    session = report["sessions"][0]
    errors = "\n".join(session["errors"])
    assert not report["ok"]
    assert "결맞음" in errors
    # 진단까지 나와야 한다: 음향 대조군이 살아 있으므로 원인은 녹음 소프트웨어다.
    assert "녹음 소프트웨어 타임베이스 문제" in errors
    assert session["alignment"]["source_err_coherence"] < 0.6
    assert session["alignment"]["ref_err_coherence"] > 0.9


def test_qa_rejects_a_drifting_source_err_delay(tmp_path):
    """코히런스는 살아 있는데 **지연이 떠다니는** 세션도 잡아야 한다.

    지연이 천천히 움직이면 긴 창 코히런스는 떨어지지만 짧은 창에서는 살아 있다.
    코히런스 하나만 보면 "조금 나쁜 세션"으로 보이므로, 지연 안정성을 독립 축으로
    검사한다 — 이 테스트가 그 축이 실제로 작동하는지 증명한다.
    """

    frames = 240_000  # 5초
    source = _band_noise(frames, 4242)
    rng = np.random.default_rng(11)
    # 실측 붕괴 세션과 같은 모양의 **연속** 드리프트를 만든다: 헌팅 주기 4.5초,
    # 진폭 150 샘플. 계단형으로 만들면 추정기가 추적을 놓쳐 "판정 불가" 로 떨어지고,
    # 그러면 이 테스트는 로버스트 통계 축을 전혀 검증하지 못한다.
    times = np.arange(frames, dtype=np.float64)
    lag = float(ERR_DELAY_SAMPLES) + 150.0 * np.sin(2.0 * np.pi * times / FS / 4.5)
    positions = np.clip(times - lag, 0.0, frames - 2.0)
    base = positions.astype(np.int64)
    frac = positions - base
    err = 0.8 * ((1.0 - frac) * source[base] + frac * source[base + 1])
    err += 0.01 * rng.standard_normal(frames)
    ref = 0.9 * np.roll(source, REF_DELAY_SAMPLES) + 0.01 * rng.standard_normal(frames)
    mics = np.stack([err, ref], axis=1).astype(np.float32)

    entry = _write_session(
        tmp_path / "drifting",
        session_id="drifting",
        group_id="group-drifting",
        frames=frames,
        mics=mics,
        source=source.astype(np.float32),
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=8192,
            required_splits=("train",),
        ),
    )

    session = report["sessions"][0]
    errors = "\n".join(session["errors"])
    assert not report["ok"]
    assert "떠다닙니다" in errors
    alignment = session["alignment"]
    # 추적은 됐는데(유효창 비율 정상) 값이 움직인다 — 로버스트 축이 잡은 것이다.
    assert alignment["delay_valid_window_ratio"] >= 0.5
    assert alignment["source_err_delay_robust_std_samples"] > 8.0
    assert alignment["source_err_delay_p95_p5_samples"] > 48.0


def test_qa_rejects_a_dead_reference_mic_as_an_acoustic_problem(tmp_path):
    """REF→ERR 이 죽으면 **다른 진단**이 나와야 한다 — 음향/배치 문제.

    두 코히런스를 함께 보는 설계의 값어치가 여기에 있다. 하나만 봤다면 원인을
    구분하지 못하고 매번 재녹음을 지시하게 된다.
    """

    frames = 24_000
    mics, source = _signals(frames)
    rng = np.random.default_rng(5)
    mics[:, 1] = (0.05 * rng.standard_normal(frames)).astype(np.float32)  # 무관한 잡음
    entry = _write_session(
        tmp_path / "dead-ref",
        session_id="dead-ref",
        group_id="group-dead-ref",
        frames=frames,
        mics=mics,
        source=source,
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=4096,
            required_splits=("train",),
        ),
    )

    session = report["sessions"][0]
    errors = "\n".join(session["errors"])
    assert not report["ok"]
    assert "마이크/배치 문제" in errors
    # source→ERR 은 멀쩡하므로 타임베이스 진단이 나오면 안 된다.
    assert "녹음 소프트웨어 타임베이스 문제" not in errors


def test_the_old_unrelated_sine_fixture_would_now_fail(tmp_path):
    """**게이트의 부재를 증명하던 옛 픽스처가 이제는 거부되는지** 확인한다.

    교체 전 픽스처는 ``mics ch0 = 310Hz``, ``ch1 = 430Hz``, ``source = 310Hz`` 였다.
    ch0 와 ch1 은 서로 아무 관계도 없는데 QA 80/80 이 PASS 였다. 그 조합을 그대로
    되살려 지금은 FAIL 하는 것을 못 박아 둔다 — 이 테스트가 깨지면 누군가 채널 간
    관계 게이트를 되돌린 것이다.
    """

    frames = 24_000
    t = np.arange(frames, dtype=np.float64) / FS
    legacy_mics = np.stack(
        [0.08 * np.sin(2 * np.pi * 310 * t), 0.04 * np.sin(2 * np.pi * 430 * t)],
        axis=1,
    ).astype(np.float32)
    legacy_source = (0.05 * np.sin(2 * np.pi * 310 * t)).astype(np.float32)

    entry = _write_session(
        tmp_path / "legacy-sine",
        session_id="legacy-sine",
        group_id="group-legacy",
        frames=frames,
        mics=legacy_mics,
        source=legacy_source,
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=4096,
            required_splits=("train",),
        ),
    )

    assert not report["ok"]
    assert any(
        "마이크/배치 문제" in message for message in report["sessions"][0]["errors"]
    )


# ======================================================================================
# 오기각 회귀 방어 (반증 #14 / #18)
#
# 게이트가 "발동하는가" 만 확인하면 절반이다. 이 저장소에서 실제로 사고를 낸 쪽은
# **정상 데이터에서 발동한 쪽**이었다: 제대로 재정렬된 47 세션 중 22 개(47%)가
# QA 지연 게이트에서 떨어졌고, 원인은 데이터가 아니라 추정기였다.
# ======================================================================================
def _legacy_wideband_argmax(
    playback: np.ndarray, capture: np.ndarray, *, window: int = FS
) -> np.ndarray:
    """**옛 추정기의 재현** — 대역제한도 PHAT 도 없는 1초창 광대역 상관의 |argmax|.

    운영 코드에서는 지웠다. 테스트에만 남긴 이유는 "이 픽스처가 옛 게이트를 실제로
    발동시켰는가"를 단언하기 위해서다. 그 단언이 없으면 오기각 회귀 테스트가
    "아무 신호나 통과한다"와 구별되지 않는다.
    """

    from scipy.signal import correlate

    n = int(min(playback.size, capture.size))
    lags = np.arange(-window + 1, window)
    mask = np.abs(lags) <= min(8_000, window - 1)
    taus = []
    for start in range(0, n - window + 1, window):
        left = playback[start : start + window]
        right = capture[start : start + window]
        corr = correlate(right - right.mean(), left - left.mean(), mode="full")
        taus.append(int(lags[mask][int(np.argmax(np.abs(corr[mask])))]))
    return np.asarray(taus, dtype=np.float64)


def test_qa_does_not_reject_a_realigned_session_over_reverberant_outlier_windows(
    tmp_path,
):
    """실측 101813 의 구조를 합성으로 재현한다 — 이 세션은 **통과해야 한다.**

    실측 창별 τ(옛 광대역 argmax 추정기): 30 창 중 27 창이 125~150 인데 3 개가
    479 / 487 / 6311 이었다. 그 3 개가 std 를 1106.55, range 를 6186 으로 만들어
    coh² 0.926 짜리 정상 세션을 FAIL 시켰다. 같은 신호의 단일 출처 추정값은
    robust-std 1.84 다.

    여기서는 상수 지연 120 의 직접파에, 세 구간에서만 직접파보다 강한 늦은 반사
    (6311 샘플)를 얹어 같은 함정을 만든다. 덕트 잔향·다중경로가 있는 실제 음향
    경로에서만 나타나는 실패 모드라 픽스처에도 그 구조가 있어야 한다.
    """

    frames = 10 * FS  # 실측처럼 오염 창이 소수(3/40)가 되도록 길게 잡는다
    source = _band_noise(frames, 101813)
    rng = np.random.default_rng(1813)
    err = 0.8 * np.roll(source, ERR_DELAY_SAMPLES)
    late = np.roll(source, 6_311)
    slot = FS // 4
    for index in (5, 6, 29):  # 0.25초 구간 3개에만 직접파보다 강한 늦은 반사
        err[index * slot : (index + 1) * slot] += (
            2.0 * late[index * slot : (index + 1) * slot]
        )
    err += 0.01 * rng.standard_normal(frames)
    ref = 0.9 * np.roll(source, REF_DELAY_SAMPLES) + 0.01 * rng.standard_normal(frames)
    mics = np.stack([err, ref], axis=1).astype(np.float32)

    entry = _write_session(
        tmp_path / "reverberant",
        session_id="reverberant",
        group_id="group-reverberant",
        frames=frames,
        mics=mics,
        source=source.astype(np.float32),
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=8192,
            required_splits=("train",),
        ),
    )

    # (1) 이 픽스처가 실제로 옛 게이트를 발동시키는 신호인지부터 확인한다.
    #     이 단언이 없으면 "아무 신호나 통과한다"와 구별되지 않는다.
    legacy = _legacy_wideband_argmax(source, err)
    assert float(np.std(legacy)) > 64.0  # 옛 임계
    assert float(legacy.max() - legacy.min()) > 256.0  # 옛 임계

    # (2) 단일 출처 추정기 + 로버스트 통계로는 통과해야 한다.
    session = report["sessions"][0]
    assert report["ok"], session["errors"]
    alignment = session["alignment"]
    assert alignment["ok"]
    assert abs(alignment["source_err_delay_median_samples"] - ERR_DELAY_SAMPLES) <= 2
    assert alignment["source_err_delay_robust_std_samples"] < 8.0
    # 원시 통계는 진단으로 남되 판정에는 쓰이지 않는다는 것도 함께 못 박는다.
    assert "source_err_delay_raw_std_samples" in alignment
    assert "source_err_delay_ptp_samples" in alignment


def test_qa_and_timeline_report_the_same_delay_numbers(tmp_path):
    """같은 파일쌍에 대해 QA 와 timeline 이 **같은 숫자**를 내는가 (발생기 A 대조).

    반증 #14/#18 이 반증한 상태가 정확히 이것이 깨진 상태였다:
    ``timeline.estimate_lag_track`` robust-std 1.84 vs
    ``invariants.measure_stream_delay_trajectory`` std 1106.55 — **600배**.
    두 값이 갈리면 "수집은 통과했는데 QA 는 떨어진다"가 되고, 아무도 어느 쪽이
    맞는지 모른다. 실측 47 세션에서도 이 두 값의 차이는 현재 0.0 이다.
    """

    from deep_anc.data.timeline import measure_delay_trajectory

    frames = SESSION_FRAMES
    mics, source = _signals(frames)
    entry = _write_session(
        tmp_path / "agree",
        session_id="agree",
        group_id="group-agree",
        frames=frames,
        mics=mics,
        source=source,
    )
    entry["split"] = "train"

    report = validate_recorded_sessions(
        [entry],
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            block_frames=8192,
            required_splits=("train",),
        ),
    )
    alignment = report["sessions"][0]["alignment"]

    direct = measure_delay_trajectory(
        np.asarray(source, dtype=np.float64),
        np.asarray(mics[:, 0], dtype=np.float64),
        sample_rate=FS,
    )
    assert alignment["source_err_delay_median_samples"] == direct.median_samples
    assert alignment["source_err_delay_robust_std_samples"] == direct.robust_std_samples
    assert alignment["source_err_delay_p95_p5_samples"] == direct.p95_p5_samples


# ======================================================================================
# 임계 선언 — 폐기 키는 조용히 무시되지 않고, 게이트는 강화 방향으로만 움직인다
# ======================================================================================
def _data_cfg() -> dict:
    return {
        "sample_rate": FS,
        "segment_seconds": 0.02,
        "reference_mode": "digital",
        "digital_reference_lead_samples": 7,
    }


def test_legacy_delay_thresholds_are_migrated_and_reported_not_silently_ignored():
    """출하 config 의 옛 키(64 / 256)가 들어와도 **더 엄격한 쪽**이 적용된다.

    64/256 은 광대역 argmax 추정기의 잡음 바닥(18 샘플)에 맞춰진 값이라, 대역제한
    PHAT 로 바꾼 지금 그대로 쓰면 오염군(robust-std 최소 25.2)을 통째로 통과시킨다.
    그렇다고 조용히 무시하면 설정 파일만 읽은 사람이 게이트를 오해한다 — 그래서
    무엇이 무시됐는지 리포트에 남긴다.
    """

    from deep_anc.data.recorded_qa import settings_from_data_config

    settings = settings_from_data_config(
        _data_cfg(),
        alignment_overrides={
            "max_source_err_delay_std_samples": 64.0,
            "max_source_err_delay_range_samples": 256.0,
        },
    )

    assert settings.max_source_err_delay_robust_std_samples == 8.0
    assert settings.max_source_err_delay_p95_p5_samples == 48.0
    notes = "\n".join(settings.deprecated_threshold_notes)
    assert "max_source_err_delay_std_samples" in notes
    assert "max_source_err_delay_range_samples" in notes

    report = validate_recorded_sessions([], settings)
    assert any("폐기됐습니다" in warning for warning in report["warnings"])
    assert report["settings"]["max_source_err_delay_robust_std_samples"] == 8.0
    assert report["settings"]["deprecated_threshold_notes"]


def test_a_stricter_legacy_threshold_is_honoured():
    """더 엄격한 선언은 그대로 살아야 한다 — 무조건 기본값으로 덮으면 그것도 죽은 설정이다."""

    from deep_anc.data.recorded_qa import settings_from_data_config

    settings = settings_from_data_config(
        _data_cfg(),
        alignment_overrides={"max_source_err_delay_std_samples": 4.0},
    )
    assert settings.max_source_err_delay_robust_std_samples == 4.0


def test_delay_thresholds_cannot_be_loosened_past_the_measured_valley():
    """게이트를 통과시키려고 임계를 키우는 것은 **생성 시점에** 막힌다."""

    import pytest

    with pytest.raises(ValueError, match="실측 근거 상한"):
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            max_source_err_delay_robust_std_samples=64.0,
        )
    with pytest.raises(ValueError, match="실측 근거 상한"):
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            max_source_err_delay_p95_p5_samples=256.0,
        )
    with pytest.raises(ValueError, match="실측 근거 하한"):
        RecordedQASettings(
            sample_rate=FS,
            segment_samples=512,
            digital_reference_lead_samples=7,
            min_source_err_delay_window_ratio=0.05,
        )


def test_qa_no_longer_owns_a_second_delay_window_declaration():
    """지연 궤적의 창 길이는 timeline 이 단독 소유한다.

    QA 가 자기 창 길이를 들고 있으면 ``session.json`` 의 ``timeline.aligned_lag_*``
    와 QA 의 지연 통계가 다시 갈라진다 — 같은 물리량 두 벌이 되는 것이다.
    """

    from deep_anc.data.recorded_qa import settings_from_data_config

    assert not hasattr(
        RecordedQASettings(
            sample_rate=FS, segment_samples=512, digital_reference_lead_samples=7
        ),
        "alignment_window_seconds",
    )
    settings = settings_from_data_config(
        _data_cfg(), alignment_overrides={"alignment_window_seconds": 1.0}
    )
    assert any(
        "alignment_window_seconds" in note for note in settings.deprecated_threshold_notes
    )
