"""오디오 장치를 여는 **모든** 코드가 실기 진입 규약을 지키는지 강제한다.

왜 이 테스트가 있는가
--------------------
2026-08-06 하루에 같은 결함을 **세 번** 재생산했다. 새 계측 도구를 만들 때마다
저장소의 기존 규약을 쓰지 않고 각자 다시 만들었다:

1. 채널 분리 검사기 → 두 번째 지연 추정기를 만들었다. ``timeline`` 과 600배 갈릴
   뻔했고, 실제로 그 갈림이 재정렬된 세션의 47%를 오기각시킨 전례가 있다.
2. 레벨 미터 프로브 → 밴드 노이즈로 만들어 인터리브 멀티톤과 크레스트가 6dB 어긋났다.
   눈금이 통째로 틀렸다.
3. 레벨 미터 입력 → ``dtype="float32"`` 로 열어 PortAudio 변환 규약을 벗어났고,
   레일 게이트가 없어 **죽은 마이크의 잡음을 레벨로 표시**했다. 사용자가 볼륨을
   완전히 꺼도 눈금이 안 움직였다.

셋 다 발생기 A(같은 물리량을 두 곳에서 따로 유도)이고, **셋 다 코드가 아니라 사람이
잡았다.** 주석으로 "규약을 지켜라" 라고 적는 방식은 이미 실패했다 — 그래서 여기서
기계적으로 검사한다.

이 테스트가 막는 것
------------------
* 승인 목록에 없는 파일이 ``sounddevice`` 를 쓰는 것
* 소리를 내는 진입점이 재생 전 전제조건 검사를 건너뛰는 것
* 입력을 int32 가 아닌 dtype 으로 여는 것

새 도구를 만들 때 이 테스트가 실패하면, **테스트를 고치지 말고 도구를 규약에 맞춰라.**
승인 목록에 추가하려면 사유를 함께 적어야 한다.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "scripts")

# ── 승인 목록 ────────────────────────────────────────────────────────────────
# key = 저장소 상대 경로, value = (소리를 내는가, 전제조건이 배선됐는가, 사유)
#
# **래칫 규약**: `배선됨=False` 는 이 규약보다 먼저 있던 부채다. 그 집합은
# `test_unwired_backlog_only_shrinks` 가 고정한다 — 줄어들 수는 있어도 늘어날 수 없다.
# 새로 만드는 진입점은 **반드시** 배선돼야 한다. 목록에 False 로 추가하는 것은 금지다.
AUDIO_ENTRY_POINTS: dict[str, tuple[bool, bool, str]] = {
    # 규약 자체를 정의하는 곳
    "src/deep_anc/audio_io.py": (False, True, "규약의 단일 출처. 장치 해석과 게이트를 제공한다"),
    # 실기 런타임
    "src/deep_anc/realtime/run_realtime.py": (True, True, "ANC 런타임. 명시적 사용자·볼륨·장치 게이트"),
    "src/deep_anc/baselines/fxlms_core.py": (False, True, "FxLMS 오프라인 기준선 유틸리티"),
    # 측정·수집
    "scripts/data/record_duct.py": (True, True, "실측 세션 수집. 레일 게이트 + 저장 시점 정렬 게이트"),
    "scripts/data/measure_paths_interleaved.py": (True, True, "P/S 동시 측정"),
    "scripts/data/calibrate_wideband.py": (True, True, "채널별 ESS 측정"),
    "scripts/data/set_amp_level.py": (True, True, "앰프 레벨 교정 미터"),
    # 벤치·진단 (실기 필요, 학습 산출물에 직접 들어가지 않음)
    "scripts/bench/measure_io_latency.py": (True, True, "I/O 왕복 지연 측정"),
    "scripts/bench/measure_io_jitter.py": (True, True, "콜백 지터 측정"),
    "scripts/bench/measure_clock_drift.py": (True, True, "클록 드리프트 측정"),
    "scripts/bench/measure_channel_paths.py": (True, True, "채널별 경로 측정"),
    "scripts/bench/measure_duct_transfer_map.py": (True, True, "덕트 전달맵 측정"),
    "scripts/bench/playback_duct_probe.py": (True, True, "재생 프로브"),
    "scripts/bench/sweep_probe_level.py": (True, True, "프로브 레벨 스윕"),
    "scripts/demo/evaluate_fxlms_direct.py": (True, True, "FxLMS 실기 평가"),
}

# 재생 전 전제조건을 만족시키는 것으로 인정하는 호출.
PRECONDITION_CALLS = frozenset(
    {
        "assert_measurement_preconditions",
        "assert_live_pcm_clock_preconditions",
        "input_rail_gate",  # 레일 게이트를 직접 부르는 진입점
    }
)

UNWIRED_BACKLOG = frozenset(
    k for k, (plays, wired, _) in AUDIO_ENTRY_POINTS.items() if plays and not wired
)
"""이 규약보다 먼저 있던 진입점들. **늘릴 수 없다.**"""


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def _uses_sounddevice(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "import sounddevice" in text


def _called_names(path: Path) -> set[str]:
    """파일 안에서 호출되는 이름 전부 (속성 접근 포함)."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:  # pragma: no cover - 컴파일 테스트가 따로 잡는다
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_every_sounddevice_user_is_declared():
    """승인 목록에 없는 파일이 오디오 장치를 열 수 없다.

    새 계측 도구가 규약을 모른 채 만들어지는 것을 여기서 막는다.
    """

    found = {
        str(p.relative_to(REPO_ROOT)) for p in _python_files() if _uses_sounddevice(p)
    }
    undeclared = sorted(found - set(AUDIO_ENTRY_POINTS))
    assert not undeclared, (
        "sounddevice 를 쓰는데 승인 목록에 없는 파일입니다:\n  "
        + "\n  ".join(undeclared)
        + "\n\n이 테스트를 고치지 말고 도구를 규약에 맞추세요:\n"
        "  - 입력은 audio_io.MEASUREMENT_DTYPE[0] (int32) 로 열고 pcm_int32_to_float32 로 변환\n"
        "  - 소리를 내기 전에 audio_io.assert_measurement_preconditions() 호출\n"
        "  - 지연/코히런스/대역은 기존 단일 출처를 호출 (dsp.timing / data.timeline / dsp.invariants)\n"
        "그 뒤 AUDIO_ENTRY_POINTS 에 사유와 함께 등록하세요."
    )


def test_declared_entry_points_still_exist():
    """지워진 파일이 목록에 남아 있으면 목록이 거짓말을 시작한다."""

    missing = sorted(k for k in AUDIO_ENTRY_POINTS if not (REPO_ROOT / k).exists())
    assert not missing, f"승인 목록에 있으나 파일이 없습니다: {missing}"


@pytest.mark.parametrize(
    "relative",
    sorted(k for k, (plays, wired, _) in AUDIO_ENTRY_POINTS.items() if plays and wired),
)
def test_playing_entry_points_check_preconditions(relative: str):
    """소리를 내는 진입점은 재생 전 전제조건 검사를 반드시 호출한다.

    이 검사가 없으면 죽은 마이크로 스피커를 울리게 된다 — 스피커 연결 시간만 버리고
    (사용자 명시 제약: 스피커 구동 하드웨어 수명), 얻은 숫자는 음향과 무관하다.
    """

    path = REPO_ROOT / relative
    names = _called_names(path)
    assert names & PRECONDITION_CALLS, (
        f"{relative} 이 소리를 내는데 재생 전 전제조건 검사를 부르지 않습니다.\n"
        f"인정되는 호출: {sorted(PRECONDITION_CALLS)}\n"
        "audio_io.assert_measurement_preconditions(sd, hardware) 를 재생 **전에** 부르세요."
    )


def test_rail_gate_has_exactly_one_definition():
    """레일 게이트가 두 곳에서 정의되면 임계가 갈린다 (발생기 A).

    원래 이 함수는 scripts/data/record_duct.py 안에 있었고, set_amp_level.py 가
    sys.path 를 조작해 스크립트에서 import 해야 했다. 그 구조가 "새 도구는 그냥
    안 쓴다" 로 이어졌다.
    """

    definitions = [
        str(p.relative_to(REPO_ROOT))
        for p in _python_files()
        if "def input_rail_gate(" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert definitions == ["src/deep_anc/audio_io.py"], (
        f"input_rail_gate 정의가 {definitions} 에 있습니다. "
        "단일 출처는 src/deep_anc/audio_io.py 이고 나머지는 import 해야 합니다."
    )


def test_measurement_dtype_is_the_single_source():
    """입력 dtype 규약이 상수 하나에서 나오는지 확인한다.

    실측: dtype="float32" 로 연 미터가 같은 신호를 pcm_int32_to_float32 대비
    수십 dB 다르게 읽었다. 그 눈금으로 사용자가 볼륨 노브를 헛돌렸다.
    """

    from deep_anc.audio_io import MEASUREMENT_DTYPE

    assert MEASUREMENT_DTYPE == ("int32", "int16")

    offenders: list[str] = []
    for relative, (plays, wired, _) in AUDIO_ENTRY_POINTS.items():
        if not (plays and wired):
            continue  # 부채는 test_unwired_backlog_only_shrinks 가 고정한다
        text = (REPO_ROOT / relative).read_text(encoding="utf-8", errors="ignore")
        if 'dtype="float32"' in text or "dtype='float32'" in text:
            offenders.append(relative)
    assert not offenders, (
        "실기 진입점이 입력을 float32 로 엽니다: " + ", ".join(offenders) + "\n"
        "PortAudio 변환은 장치마다 풀스케일이 달라진다 — int32 로 받아 "
        "pcm_int32_to_float32 로 변환하세요."
    )


def test_unwired_backlog_only_shrinks():
    """규약 이전 부채 집합을 고정한다 — 줄어들 수는 있어도 **늘어날 수 없다**.

    이것이 래칫이다. 새 진입점을 `배선됨=False` 로 등록하려 하면 여기서 막힌다.
    부채를 갚으면(전제조건을 배선하면) 이 목록에서 지우고 `True` 로 바꿔라.
    """

    known = set()
    added = sorted(UNWIRED_BACKLOG - known)
    assert not added, (
        "전제조건이 배선되지 않은 진입점이 새로 생겼습니다: " + ", ".join(added) + "\n"
        "새 진입점은 audio_io.assert_measurement_preconditions() 를 반드시 부르세요. "
        "이 목록에 추가하는 것은 허용되지 않습니다."
    )
    # 갚은 부채는 known 에서도 지워야 목록이 거짓말을 안 한다.
    stale = sorted(known - UNWIRED_BACKLOG)
    assert not stale, (
        f"배선을 마친 항목이 known 에 남아 있습니다: {stale} — 이 목록에서 지우세요."
    )


def test_precondition_rejects_a_dead_microphone():
    """무신호 마이크를 **재생 전에** 막는다 (하한).

    2026-08-06 실측: 마이크가 정확히 0(peak 0.000000, -240 dBFS)을 내보내는데
    레일 게이트만 있던 공용 함수가 "정상" 으로 통과시켰다. 상한만 보면 반쪽이다 —
    죽은 센서로 얻은 숫자는 틀린 게 아니라 무의미하고, 그걸 모르면 사람이
    하드웨어를 헛만진다(실제로 그렇게 볼륨 노브를 헛돌렸다).
    """

    import numpy as np

    from deep_anc.audio_io import MIN_PROBE_DBFS, assert_measurement_preconditions

    fs = 48000
    hardware = {"sample_rate": fs, "input": {"card": "APE", "pcm": 1}}

    class _SilentDevice:
        """항상 디지털 무음을 돌려주는 가짜 장치."""

        def rec(self, frames, **_kwargs):
            return np.zeros((int(frames), 2), dtype=np.int32)

        def wait(self):
            return None

    import deep_anc.audio_io as audio_io

    original = audio_io.resolve_alsa_portaudio_device
    audio_io.resolve_alsa_portaudio_device = lambda *a, **k: 0
    try:
        with pytest.raises(RuntimeError, match="무신호"):
            assert_measurement_preconditions(_SilentDevice(), hardware, seconds=1.0)
    finally:
        audio_io.resolve_alsa_portaudio_device = original

    assert MIN_PROBE_DBFS == -80.0


def test_precondition_rejects_a_railed_microphone():
    """풀스케일에 붙은 마이크를 **재생 전에** 막는다 (상한). 짝이 되는 음성 대조다."""

    import numpy as np

    from deep_anc.audio_io import assert_measurement_preconditions

    fs = 48000
    hardware = {"sample_rate": fs, "input": {"card": "APE", "pcm": 1}}

    class _RailedDevice:
        def rec(self, frames, **_kwargs):
            return np.full((int(frames), 2), 2**31 - 1, dtype=np.int32)

        def wait(self):
            return None

    import deep_anc.audio_io as audio_io

    original = audio_io.resolve_alsa_portaudio_device
    audio_io.resolve_alsa_portaudio_device = lambda *a, **k: 0
    try:
        with pytest.raises(RuntimeError, match="풀스케일"):
            assert_measurement_preconditions(_RailedDevice(), hardware, seconds=1.0)
    finally:
        audio_io.resolve_alsa_portaudio_device = original


def test_precondition_accepts_a_healthy_quiet_room():
    """정상 신호는 통과해야 한다 — 게이트가 꺼져서 통과하는 게 아님을 짝으로 확인한다."""

    import numpy as np

    from deep_anc.audio_io import assert_measurement_preconditions

    fs = 48000
    hardware = {"sample_rate": fs, "input": {"card": "APE", "pcm": 1}}

    class _HealthyDevice:
        def rec(self, frames, **_kwargs):
            rng = np.random.default_rng(0)
            # 실측 정상 세션 수준: RMS 약 0.001 (-60 dBFS), 레일 0
            x = rng.standard_normal((int(frames), 2)) * 0.001
            return (x * (2**31)).astype(np.int32)

        def wait(self):
            return None

    import deep_anc.audio_io as audio_io

    original = audio_io.resolve_alsa_portaudio_device
    audio_io.resolve_alsa_portaudio_device = lambda *a, **k: 0
    try:
        ratios = assert_measurement_preconditions(_HealthyDevice(), hardware, seconds=1.0)
    finally:
        audio_io.resolve_alsa_portaudio_device = original
    assert max(ratios) == 0.0


def test_precondition_checks_both_pcm_endpoints_before_recording(monkeypatch):
    """별도 USB 출력 PCM 점유도 입력 probe 전에 fail-closed여야 한다."""

    import deep_anc.audio_io as audio_io

    hardware = {
        "sample_rate": 48_000,
        "input": {"card": "APE", "pcm": 1},
        "output": {"card": "Audio", "pcm": 0},
    }

    class _MustNotRecord:
        def rec(self, *_args, **_kwargs):
            raise AssertionError("PCM 무점유 확인 전에 rec를 열면 안 됩니다")

    monkeypatch.setattr(
        audio_io,
        "assert_measurement_pcm_unoccupied",
        lambda _hardware: (_ for _ in ()).throw(RuntimeError("output PCM busy")),
    )

    with pytest.raises(RuntimeError, match="output PCM busy"):
        audio_io.assert_measurement_preconditions(_MustNotRecord(), hardware)
