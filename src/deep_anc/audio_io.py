"""오디오 장치 해석·PCM 변환 유틸.

anc_project/fxlms_core.py 에서 실기 검증된 함수를 그대로 이식했다 (출처 명기).
Jetson AGX Orin: 입력 hw:APE,1 (S32_LE), 출력 AB13X USB (S16_LE).
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

_INT32_SCALE = np.float32(1.0 / 2147483648.0)
_INT16_MAX = np.float32(32767.0)


def alsa_card_index(card_id: str) -> int:
    """ALSA 짧은 카드 ID('APE', 'Audio')를 카드 번호로 해석."""
    cards_path = Path("/proc/asound/cards")
    if not cards_path.exists():
        raise RuntimeError("/proc/asound/cards 를 읽을 수 없습니다")

    wanted = card_id.strip()
    pattern = re.compile(r"^\s*(\d+)\s+\[([^\]]+)\]")
    for line in cards_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match and match.group(2).strip() == wanted:
            return int(match.group(1))

    raise RuntimeError(f"ALSA 카드 ID '{wanted}' 를 찾지 못했습니다")


def format_sounddevice_devices() -> str:
    """PortAudio/sounddevice 장치 목록을 표로 반환."""
    import sounddevice as sd

    lines = []
    for index, device in enumerate(sd.query_devices()):
        lines.append(
            f"{index:3d}: in={int(device['max_input_channels']):2d}, "
            f"out={int(device['max_output_channels']):2d}, "
            f"rate={float(device['default_samplerate']):8.1f} | {device['name']}"
        )
    return "\n".join(lines)


def resolve_alsa_portaudio_device(
    card_id: str,
    pcm_device: int,
    direction: str,
    required_channels: int,
    override_index: int | None = None,
) -> int:
    """ALSA 카드 ID/디바이스 번호를 sounddevice 장치 인덱스로 매핑."""
    import sounddevice as sd

    direction = direction.lower().strip()
    if direction not in {"input", "output"}:
        raise ValueError("direction 은 'input' 또는 'output'")

    devices = sd.query_devices()
    capability_key = "max_input_channels" if direction == "input" else "max_output_channels"

    if override_index is not None:
        if not 0 <= override_index < len(devices):
            raise RuntimeError(f"잘못된 sounddevice 인덱스: {override_index}")
        if int(devices[override_index][capability_key]) < required_channels:
            raise RuntimeError(
                f"장치 {override_index} 는 {direction} {required_channels}채널을 지원하지 않습니다"
            )
        return int(override_index)

    card_number = alsa_card_index(card_id)
    token = f"hw:{card_number},{int(pcm_device)}"
    matches: list[int] = []

    for index, device in enumerate(devices):
        name = str(device["name"])
        if token in name and int(device[capability_key]) >= required_channels:
            matches.append(index)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        for index in matches:
            if f"({token})" in str(devices[index]["name"]):
                return index
        return matches[0]

    raise RuntimeError(
        f"ALSA {card_id}, device {pcm_device} 를 PortAudio {direction} 장치로 매핑하지 "
        f"못했습니다.\n장치 목록:\n{format_sounddevice_devices()}"
    )


def pcm_int32_to_float32(samples: np.ndarray) -> np.ndarray:
    """S32_LE PCM → 정규화 float32 (채널 유지)."""
    return np.asarray(samples, dtype=np.int32).astype(np.float32) * _INT32_SCALE


def float32_to_pcm_int16(samples: np.ndarray) -> np.ndarray:
    """정규화 float32 → 클리핑 후 S16_LE PCM."""
    values = np.asarray(samples, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.rint(np.clip(values, -1.0, 1.0) * _INT16_MAX).astype(np.int16)


def rms_dbfs(samples: np.ndarray, floor_db: float = -200.0) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return floor_db
    power = float(np.mean(values * values))
    if not np.isfinite(power) or power <= 0.0:
        return floor_db
    return max(floor_db, 10.0 * float(np.log10(power)))


def analyze_int32_input_probe(
    samples: np.ndarray,
    *,
    min_rms_dbfs: float = -80.0,
    max_clip_ratio: float = 0.005,
    min_unique_codes: int = 8,
) -> dict:
    """짧은 S32_LE 입력이 실제 마이크 신호인지 채널별로 판정한다.

    ``-1``/``0`` 같은 고정 I2S 라인은 float 변환 뒤 작은 0처럼 보여 RMS만으로
    놓칠 수 있다. 원시 코드 다양성·클리핑·RMS를 함께 검사한다.
    """
    raw = np.asarray(samples)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError(f"입력 probe는 [frames, channels]여야 합니다: {raw.shape}")
    if raw.shape[1] < 2:
        raise ValueError(f"ERR/REF 2채널 입력이 필요합니다: {raw.shape[1]}채널")
    raw = raw.astype(np.int32, copy=False)
    normalized = pcm_int32_to_float32(raw)
    channels = []
    for channel in range(raw.shape[1]):
        values = raw[:, channel]
        signal = normalized[:, channel]
        unique_codes = int(np.unique(values).size)
        clip_ratio = float(np.mean(np.abs(signal.astype(np.float64)) >= 0.99))
        rms = rms_dbfs(signal)
        stuck = unique_codes < int(min_unique_codes)
        valid = (
            not stuck
            and rms >= float(min_rms_dbfs)
            and clip_ratio <= float(max_clip_ratio)
        )
        channels.append(
            {
                "channel": channel,
                "rms_dbfs": float(rms),
                "peak": float(np.max(np.abs(signal))),
                "clip_ratio": clip_ratio,
                "unique_codes": unique_codes,
                "raw_min": int(np.min(values)),
                "raw_max": int(np.max(values)),
                "stuck": stuck,
                "valid": valid,
            }
        )
    return {"frames": int(raw.shape[0]), "channels": channels}


# I2S 입력은 스트림을 연 직후 큰 기동 트랜지언트를 낸다 — 실측으로 0.0-0.5초 구간이
# RMS -36.3 dBFS / peak 0.062 이고, 0.3초쯤부터 정상 바닥(-67.4 dBFS / peak 0.002)으로
# 내려온다. 이 구간을 버리지 않으면 **이 함수의 목적 자체가 무너진다**: 트랜지언트만으로
# RMS 가 -42 dBFS 로 나오므로, 신호가 전혀 없는 죽은 마이크도 min_rms_dbfs(-80) 게이트를
# 통과한다. 실제로 저장소의 "잡음 바닥 -42 ~ -46 dBFS" 기록은 전부 이 트랜지언트였다
# (2초 창 예측 -42.31 vs 로그 -42.34/-42.40/-42.49, 5초 창 예측 -46.27 vs 문서 -46.33).
DEFAULT_PROBE_SETTLE_SECONDS = 1.0


def capture_input_probe(
    audio_cfg: dict,
    *,
    seconds: float = 2.0,
    settle_seconds: float = DEFAULT_PROBE_SETTLE_SECONDS,
    min_rms_dbfs: float = -80.0,
    max_clip_ratio: float = 0.005,
) -> dict:
    """출력 장치를 열지 않고 설정된 APE 입력만 캡처해 분석한다.

    ``settle_seconds`` 만큼 더 캡처해서 앞부분을 버린다. 판정에 쓰는 것은 그 뒤 구간이다.
    """
    import sounddevice as sd

    fs = int(audio_cfg["sample_rate"])
    if seconds <= 0.0:
        raise ValueError("probe seconds는 양수여야 합니다")
    settle = max(0.0, float(settle_seconds))
    input_cfg = audio_cfg["input"]
    device = resolve_alsa_portaudio_device(
        input_cfg["card"], input_cfg["pcm"], "input", 2
    )
    settle_frames = int(round(settle * fs))
    raw = sd.rec(
        settle_frames + int(round(float(seconds) * fs)),
        samplerate=fs,
        channels=2,
        dtype="int32",
        device=device,
    )
    sd.wait()
    report = analyze_int32_input_probe(
        raw[settle_frames:],
        min_rms_dbfs=min_rms_dbfs,
        max_clip_ratio=max_clip_ratio,
    )
    report.update(
        {
            "device": int(device),
            "sample_rate": fs,
            "settle_seconds": settle,
            "analyzed_seconds": float(seconds),
        }
    )
    return report


# ---------------------------------------------------------------------------
# 캡처 클록 교란 방지
# ---------------------------------------------------------------------------
def check_capture_clock_undisturbed(card_id: str) -> tuple[bool, list[str]]:
    """캡처 카드의 클록을 흔들 수 있는 것이 붙어 있는지 본다.

    왜 필요한가
    ----------
    Tegra APE 는 **PLL_A 하나를 I2S1~I2S6 전체가 공유**한다
    (부팅 DTS ``/sound`` 노드가 ``pll_a``/``plla_out0`` 를 직접 쥐고, 모든 I2S 가
    ``assigned-clock-parents = <plla_out0>``). 머신 드라이버는 스트림 레이트에 맞춰
    PLL_A 를 재설정한다.

    그런데 PulseAudio 가 같은 카드에 **44100 Hz** 프로파일을 들고 있다
    (실측: ``alsa_output.platform-sound.analog-stereo``, ``device.string="front:1"``).
    데스크톱 알림음 하나로 그 sink 가 깨어나면 PLL_A 가 44.1k 계열로 재조정되고,
    **그 순간 돌고 있던 우리 48kHz 캡처의 BCLK 가 세션 중간에 이동한다.**

    이건 XRUN 이 아니다 — 어떤 기존 게이트도 못 잡는다. 녹음은 정상 종료되고
    파일도 멀쩡해 보이는데 시간축만 틀어진다. 2026-08-04 에 80세션을 날린 것과
    같은 종류의 조용한 실패다.

    반환: ``(안전한가, 사유 목록)``. 사유가 있으면 호출부가 **실패-폐쇄**해야 한다.
    """

    import subprocess

    reasons: list[str] = []

    # (a) 다른 프로세스가 이 카드의 PCM 을 열고 있는가
    try:
        index = alsa_card_index(card_id)
    except Exception:  # pragma: no cover - 카드가 없으면 다른 곳에서 이미 실패한다
        return True, []
    for status in sorted(Path(f"/proc/asound/card{index}").glob("pcm*/sub*/status")):
        try:
            first = status.read_text(encoding="utf-8").splitlines()[0].strip()
        except OSError:
            continue
        if first and first != "closed":
            reasons.append(f"{status.parent.parent.name} 가 열려 있습니다 ({first})")

    # (b) PulseAudio 가 같은 카드를 다른 레이트로 들고 있는가
    try:
        out = subprocess.run(
            ["pactl", "list", "sinks"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return not reasons, reasons  # pactl 이 없으면 (a) 만으로 판단한다

    block: list[str] = []
    for line in out.splitlines() + ["Sink #"]:
        if line.startswith("Sink #") and block:
            text = "\n".join(block)
            if f'device.string = "front:{index}"' in text or f'alsa.card = "{index}"' in text:
                state = next(
                    (l.split(":", 1)[1].strip() for l in block if "State:" in l), "?"
                )
                spec = next(
                    (l.split(":", 1)[1].strip() for l in block if "Sample Specification" in l),
                    "?",
                )
                if state.upper() == "RUNNING":
                    reasons.append(
                        f"PulseAudio 가 같은 카드를 재생 중입니다 ({spec}) — "
                        "PLL_A 가 재조정되어 캡처 BCLK 가 세션 중에 이동합니다"
                    )
            block = []
        block.append(line)

    return not reasons, reasons


def assert_capture_clock_undisturbed(card_id: str) -> None:
    """:func:`check_capture_clock_undisturbed` 가 걸리면 안내와 함께 예외를 던진다."""

    ok, reasons = check_capture_clock_undisturbed(card_id)
    if ok:
        return
    raise RuntimeError(
        "캡처 클록이 교란될 수 있습니다:\n  - "
        + "\n  - ".join(reasons)
        + "\n\n해결: 측정 전에 데스크톱 오디오가 이 카드를 놓게 하세요.\n"
        "  pactl set-card-profile alsa_card.platform-sound off   # 완전히 놓는다\n"
        "  pactl set-card-profile alsa_card.platform-sound output:analog-stereo+input:analog-stereo   # 되돌리기"
    )


# ---------------------------------------------------------------------------
# 실기 진입 규약 — 오디오 장치를 여는 모든 코드가 지켜야 하는 것
# ---------------------------------------------------------------------------
#
# 왜 규약이 필요한가 (2026-08-06, 하루에 세 번 같은 결함을 재생산했다)
# ------------------------------------------------------------------
# 새 계측 도구를 만들 때마다 저장소의 기존 규약을 쓰지 않고 각자 다시 만들었다:
#
#   1. 채널 분리 검사기 → 두 번째 지연 추정기를 만들었다 (timeline 과 600배 갈릴 뻔)
#   2. 레벨 미터 프로브 → 밴드 노이즈로 만들어 멀티톤과 크레스트가 6dB 어긋났다
#   3. 레벨 미터 입력  → dtype="float32" 로 열어 PortAudio 변환 규약을 벗어났고,
#                        레일 게이트가 없어 죽은 마이크의 잡음을 레벨로 표시했다
#
# 셋 다 "같은 물리량을 두 곳에서 따로 유도한다"(발생기 A)이고, 셋 다 사용자 관측이
# 잡았다 — 코드가 아니라 사람이 잡았다는 뜻이다. 그래서 규약을 문서가 아니라
# **테스트로 강제**한다: tests/test_audio_entry_contract.py 가 sounddevice 를 쓰는
# 모든 파일을 열거하고 아래 규약을 지키는지 검사한다.

MEASUREMENT_DTYPE = ("int32", "int16")
"""실기 스트림의 dtype 규약 — 입력 int32 / 출력 int16.

PortAudio 에 float 변환을 맡기면 풀스케일 규약이 장치마다 달라진다. 실측:
``dtype="float32"`` 로 연 미터가 같은 신호를 ``pcm_int32_to_float32`` 대비 수십 dB
다르게 읽었다. 입력은 **항상** int32 로 받아 :func:`pcm_int32_to_float32` 로 변환하고,
출력은 int16 으로 :func:`float32_to_pcm_int16` 을 거친다.
"""

MAX_PROBE_CLIP_RATIO = 0.005
"""마이크 자가진단 **상한**. QA 의 ``max_clip_ratio`` 와 같은 자리에서, 재생 **전에** 본다."""

MIN_PROBE_DBFS = -80.0
"""마이크 자가진단 **하한**. 이보다 조용하면 무신호로 본다.

상한만 보면 반쪽이다. 2026-08-06 실측: 마이크가 **정확히 0**(peak 0.000000,
-240 dBFS)을 내보내는데 레일 게이트만으로는 "정상" 으로 통과했다. 죽은 센서로 얻은
숫자는 틀린 게 아니라 무의미하고, 그걸 모르면 사람이 하드웨어를 헛만진다.
``record_duct.py`` 는 원래 이 하한을 갖고 있었다(``--ref-check-dbfs`` 기본 -80).
공용 함수로 옮기면서 하한을 빠뜨린 것이 이 결함이다.
"""


def input_rail_gate(
    probe_float: np.ndarray, *, max_clip_ratio: float = MAX_PROBE_CLIP_RATIO
) -> tuple[bool, list[float]]:
    """gate: ``recording_input_rail_preflight`` — 마이크가 풀스케일에 붙어 있는가.

    자가진단에 **상한이 없었다.** 하한만 보면 "죽은 마이크"는 잡지만 "레일에 붙은
    마이크"는 통과한다 — 오히려 아주 살아 있어 보인다. 2026-08-05/06 실측: 입력단
    접촉이 빠지면 두 채널이 6~23 Hz 초저역으로 int32 풀스케일을 때린다
    (레일 비율 0.047~0.111, RMS 0.34~0.45). 정상 세션은 peak 0.006 / RMS 0.001 이다.

    이 상태를 못 막으면 계측이 **음향과 무관한 숫자**를 낸다 — 2026-08-06 에
    레벨 미터가 그 잡음을 -21 dBFS 로 표시해 사용자가 볼륨 노브를 헛돌렸다.

    반환 ``(ok, clip_ratio_per_channel)``. 재생 **전에** 판정하는 것이 요점이다 —
    스피커를 울린 뒤 QA 에서 걸러 봐야 스피커 연결 시간만 버린다.
    """

    data = np.asarray(probe_float, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    ratios = [float(np.mean(np.abs(data[:, ch]) >= 0.999)) for ch in range(data.shape[1])]
    return bool(max(ratios, default=0.0) <= float(max_clip_ratio)), ratios


def assert_measurement_preconditions(sd, hardware: dict, *, seconds: float = 1.5) -> list[float]:
    """스피커를 울리기 **전에** 실기 전제조건을 전부 확인한다. 하나라도 깨지면 예외.

    (a) 캡처 클록이 교란되지 않는가 — PulseAudio 가 같은 APE 카드를 다른 레이트로
        잡고 있으면 PLL_A 재조정으로 BCLK 가 세션 중에 이동한다.
    (b) 마이크가 풀스케일 레일에 붙어 있지 않은가.

    반환: 채널별 레일 비율(진단용). 이 함수를 부르지 않고 소리를 내는 진입점은
    ``tests/test_audio_entry_contract.py`` 가 거부한다.
    """

    card = hardware["input"]["card"]
    assert_capture_clock_undisturbed(card)

    device = resolve_alsa_portaudio_device(card, hardware["input"]["pcm"], "input", 2)
    probe = sd.rec(
        int(seconds * int(hardware["sample_rate"])),
        samplerate=int(hardware["sample_rate"]),
        channels=2,
        dtype=MEASUREMENT_DTYPE[0],
        device=device,
    )
    sd.wait()
    settle = int(0.5 * int(hardware["sample_rate"]))
    probe_f = pcm_int32_to_float32(probe[settle:])
    ok, ratios = input_rail_gate(probe_f)
    if not ok:
        raise RuntimeError(
            f"마이크 입력이 풀스케일에 붙어 있습니다 (레일 비율 "
            f"{ratios[0]:.4f}/{ratios[1]:.4f} > {MAX_PROBE_CLIP_RATIO}). "
            "이 상태의 계측은 음향과 무관한 숫자를 냅니다 — 볼륨을 돌려도 안 움직입니다.\n"
            "입력단 전원·배선(J30 핀 접촉)을 확인한 뒤 다시 실행하세요."
        )

    # 하한 — 상한만 보면 반쪽이다. 무신호 마이크도 "레일 아님" 으로 통과한다.
    levels = [rms_dbfs(probe_f[:, ch]) for ch in range(probe_f.shape[1])]
    dead = [ch for ch, db in enumerate(levels) if db < MIN_PROBE_DBFS]
    if dead:
        detail = " / ".join(f"ch{ch} {levels[ch]:.1f} dBFS" for ch in range(len(levels)))
        raise RuntimeError(
            f"마이크가 무신호입니다 ({detail} < {MIN_PROBE_DBFS:.0f} dBFS). "
            f"채널 {dead} 에 데이터가 오지 않습니다.\n"
            "이 상태로 계측하면 숫자가 음향과 무관해집니다. "
            "입력단 전원·배선(J30 핀 접촉)과 I2S 클록을 확인한 뒤 다시 실행하세요."
        )
    return ratios
