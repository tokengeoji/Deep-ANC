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
