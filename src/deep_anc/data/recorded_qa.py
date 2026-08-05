"""실측 ANC 세션의 스트리밍 무결성·메타데이터 QA.

이 모듈은 오디오 파일을 읽기만 하며 재생 장치나 ``sounddevice``를 열지 않는다.
긴 녹음도 ``block_frames`` 단위로 순회해 전체 파형을 메모리에 올리지 않는다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from ..dsp.invariants import (
    MAX_STREAM_DELAY_P95_P5_SAMPLES,
    MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
    MIN_STREAM_DELAY_VALID_WINDOW_RATIO,
    MIN_STREAM_DELAY_VALID_WINDOWS,
)
from .manifest import VALID_SPLITS, validate_group_id, validate_source_family


@dataclass(frozen=True)
class RecordedQASettings:
    """실측 세션 QA 판정값."""

    sample_rate: int
    segment_samples: int
    digital_reference_lead_samples: int
    reference_mode: str = "digital"
    block_frames: int = 262_144
    clip_threshold: float = 0.99
    max_clip_ratio: float = 0.005
    min_mic_rms_dbfs: float = -80.0
    min_source_rms_dbfs: float = -80.0
    required_splits: tuple[str, ...] = VALID_SPLITS
    allow_incomplete_family_coverage: bool = False

    # ---- 채널 간 관계 게이트 (결함 2) -------------------------------------------
    # 2026-08-04: 실측 80 세션이 **전부** QA 를 통과했는데 학습 데이터의 시간축이
    # 붕괴해 있었다. 통과한 이유는 QA 가 무엇을 봤는가가 아니라 **안 봤는가**다 —
    # RMS·클리핑·길이·샘플레이트·메타데이터 일치. 전부 파일 **하나하나**의 통계다.
    # 학습이 실제로 배워야 하는 것은 채널 **사이**의 관계(source→ERR)인데 그것을
    # 한 번도 보지 않았다. 아래 값들이 그 관계를 판정에 넣는다.
    check_alignment: bool = True
    alignment_band_hz: tuple[float, float] = (150.0, 600.0)
    alignment_nperseg: int = 8192
    alignment_max_seconds: float = 30.0
    """정렬 검사에 읽어 들일 최대 길이(초). 스트리밍 QA 의 메모리 규약을 깨지 않기
    위한 상한이다. 30 초 = 채널당 5.7 MB 이고, 실측 세션 70 초 중 30 초면 창 30개라
    지연 궤적 통계가 충분히 안정된다."""

    min_source_err_coherence: float = 0.60
    """실측 정상 0.96~0.99 / 붕괴 0.02~0.13 사이의 넓은 골짜기에서 고른 값."""

    min_ref_err_coherence: float = 0.60
    """음향 대조군. 이 값이 살아 있는데 source→ERR 만 죽으면 원인은 음향이 아니라
    녹음 소프트웨어의 타임베이스다 — 진단이 자동으로 갈린다."""

    max_source_err_delay_robust_std_samples: float = MAX_STREAM_DELAY_ROBUST_STD_SAMPLES
    """지연 궤적 산포(1.4826×MAD)의 상한. **원시 std 가 아니다.**

    2026-08-06 반증 #14/#18: 원시 std/range 로 판정하던 옛 게이트는 제대로 재정렬된
    47 세션 중 22개(47%)를 오기각했다. 창 30개 중 27개가 125~150 인데 이상치 3개가
    std 를 1106 으로 만든 것이 원인이고, 그 이상치는 광대역 argmax 추정기가 만들었다.
    추정기를 대역제한 PHAT 단일 출처로 바꾸고 판정량을 로버스트 통계로 바꿨다.
    임계 근거(전수 실측)는 :data:`deep_anc.dsp.invariants.MAX_STREAM_DELAY_ROBUST_STD_SAMPLES`.
    """

    max_source_err_delay_p95_p5_samples: float = MAX_STREAM_DELAY_P95_P5_SAMPLES
    """지연 궤적 변동폭(p95−p5)의 상한. min/max range 가 아니다 — 이유는 위와 같다."""

    min_source_err_delay_windows: int = MIN_STREAM_DELAY_VALID_WINDOWS
    """유효창이 이보다 적으면 "안정" 이 아니라 **판정 불가(FAIL)** 다."""

    min_source_err_delay_window_ratio: float = MIN_STREAM_DELAY_VALID_WINDOW_RATIO
    """추적을 놓친 창의 비율이 이보다 크면 FAIL. 로버스트 통계의 fail-open 구멍을 막는다."""

    deprecated_threshold_notes: tuple[str, ...] = ()
    """폐기된 임계 키가 들어왔을 때 남기는 안내. 조용히 무시하지 않는다 —
    "죽은 설정이 다음 사람을 속인다"가 이 저장소에서 반복된 실패 방식이다."""

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate는 양수여야 합니다")
        if self.segment_samples <= 0:
            raise ValueError("segment_samples는 양수여야 합니다")
        if self.digital_reference_lead_samples < 0:
            raise ValueError("digital_reference_lead_samples는 0 이상이어야 합니다")
        if self.reference_mode not in {"digital", "acoustic"}:
            raise ValueError(f"지원하지 않는 reference_mode: {self.reference_mode!r}")
        if self.reference_mode != "digital" and self.digital_reference_lead_samples:
            raise ValueError("acoustic reference QA에는 digital lead를 적용할 수 없습니다")
        if self.block_frames <= 0:
            raise ValueError("block_frames는 양수여야 합니다")
        if not 0.0 < self.clip_threshold <= 1.0:
            raise ValueError("clip_threshold는 0보다 크고 1 이하여야 합니다")
        if not 0.0 <= self.max_clip_ratio <= 1.0:
            raise ValueError("max_clip_ratio는 0 이상 1 이하여야 합니다")
        for value, name in (
            (self.min_mic_rms_dbfs, "min_mic_rms_dbfs"),
            (self.min_source_rms_dbfs, "min_source_rms_dbfs"),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name}는 유한값이어야 합니다")
        invalid = [split for split in self.required_splits if split not in VALID_SPLITS]
        if invalid:
            raise ValueError(f"지원하지 않는 required split: {invalid}")
        # 정렬 게이트 값도 생성 시점에 검증한다. 물리적으로 불가능한 임계로 게이트를
        # 무력화하는 것(예: 음수 코히런스 하한)은 값이 만들어지기 전에 막아야 한다.
        lo, hi = (float(value) for value in self.alignment_band_hz)
        if not (0.0 < lo < hi <= float(self.sample_rate) / 2.0):
            raise ValueError(
                f"alignment_band_hz가 유효하지 않습니다: {self.alignment_band_hz!r} "
                f"(0 < lo < hi ≤ {self.sample_rate / 2:.0f}Hz)"
            )
        if self.alignment_nperseg < 64 or self.alignment_nperseg & (self.alignment_nperseg - 1):
            raise ValueError(
                f"alignment_nperseg는 64 이상의 2의 거듭제곱이어야 합니다: "
                f"{self.alignment_nperseg}"
            )
        if not 0.0 < float(self.alignment_max_seconds):
            raise ValueError(
                f"alignment_max_seconds는 양수여야 합니다: {self.alignment_max_seconds!r}"
            )
        for value, name in (
            (self.min_source_err_coherence, "min_source_err_coherence"),
            (self.min_ref_err_coherence, "min_ref_err_coherence"),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name}는 (0, 1] 이어야 합니다: {value!r}")
        for value, name, ceiling in (
            (
                self.max_source_err_delay_robust_std_samples,
                "max_source_err_delay_robust_std_samples",
                MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
            ),
            (
                self.max_source_err_delay_p95_p5_samples,
                "max_source_err_delay_p95_p5_samples",
                MAX_STREAM_DELAY_P95_P5_SAMPLES,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name}는 유한한 양수여야 합니다: {value!r}")
            # 게이트는 강화 방향으로만 조정할 수 있다. 통과시키려고 임계를 키우는 것을
            # 설정 파일 한 줄로 할 수 있으면 그건 게이트가 아니다.
            if float(value) > float(ceiling):
                raise ValueError(
                    f"{name}({value!r})는 실측 근거 상한 {ceiling} 보다 클 수 없습니다 — "
                    "게이트는 강화 방향으로만 조정합니다 (정상군 최대와 오염군 최소 "
                    "사이의 골짜기에서 고른 값입니다)"
                )
        if int(self.min_source_err_delay_windows) < 1:
            raise ValueError(
                f"min_source_err_delay_windows는 1 이상이어야 합니다: "
                f"{self.min_source_err_delay_windows!r}"
            )
        ratio = float(self.min_source_err_delay_window_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(
                f"min_source_err_delay_window_ratio는 (0, 1] 이어야 합니다: {ratio!r}"
            )
        if ratio < MIN_STREAM_DELAY_VALID_WINDOW_RATIO:
            raise ValueError(
                f"min_source_err_delay_window_ratio({ratio!r})는 실측 근거 하한 "
                f"{MIN_STREAM_DELAY_VALID_WINDOW_RATIO} 보다 작을 수 없습니다 — "
                "게이트는 강화 방향으로만 조정합니다"
            )

    @property
    def effective_lead_samples(self) -> int:
        return (
            self.digital_reference_lead_samples
            if self.reference_mode == "digital"
            else 0
        )

    @property
    def minimum_frames(self) -> int:
        # RecordedANCDataset은 start 상한을 만들기 위해 segment+lead보다 최소
        # 1샘플 더 긴 세션을 요구한다.
        return self.segment_samples + self.effective_lead_samples + 1


_ALIGNMENT_OVERRIDE_KEYS = frozenset(
    {
        "check_alignment",
        "alignment_band_hz",
        "alignment_nperseg",
        "alignment_max_seconds",
        "min_source_err_coherence",
        "min_ref_err_coherence",
        "max_source_err_delay_robust_std_samples",
        "max_source_err_delay_p95_p5_samples",
        "min_source_err_delay_windows",
        "min_source_err_delay_window_ratio",
    }
)


_DEPRECATED_ALIGNMENT_KEYS: dict[str, tuple[str, float]] = {
    "max_source_err_delay_std_samples": (
        "max_source_err_delay_robust_std_samples",
        MAX_STREAM_DELAY_ROBUST_STD_SAMPLES,
    ),
    "max_source_err_delay_range_samples": (
        "max_source_err_delay_p95_p5_samples",
        MAX_STREAM_DELAY_P95_P5_SAMPLES,
    ),
}
"""옛 임계 키 → 새 키. **값은 옮겨 오지 않는다.**

옛 64/256 은 광대역 argmax 추정기의 잡음 바닥(18 샘플)에 맞춰진 값이라, 대역제한
PHAT 단일 출처로 바꾼 지금 그대로 쓰면 오염군(robust-std 최소 25.2)을 통째로
통과시킨다. 그래서 선언값과 실측 근거 상한 중 **작은 쪽**을 쓰고, 무엇이 무시됐는지를
``deprecated_threshold_notes`` 로 남긴다 — 조용히 무시하면 다음 사람이 설정 파일을
읽고 게이트가 64 라고 믿는다."""


_DEPRECATED_ALIGNMENT_DROPPED: dict[str, str] = {
    "alignment_window_seconds": (
        "지연 궤적의 창 길이는 이제 deep_anc.data.timeline.TimelineSettings 가 "
        "단독으로 소유합니다 (window 0.25s / hop 0.0625s). QA 가 별도 창을 선언하면 "
        "session.json 의 timeline.aligned_lag_* 와 QA 의 지연 통계가 다시 갈라집니다"
    ),
}


def settings_from_data_config(
    data_cfg: dict,
    *,
    block_frames: int = 262_144,
    clip_threshold: float = 0.99,
    max_clip_ratio: float = 0.005,
    min_mic_rms_dbfs: float = -80.0,
    min_source_rms_dbfs: float = -80.0,
    required_splits: Iterable[str] = VALID_SPLITS,
    allow_incomplete_family_coverage: bool = False,
    alignment_overrides: dict | None = None,
) -> RecordedQASettings:
    """학습 데이터 설정과 동일한 세그먼트/lead 최소 길이를 해석한다.

    ``alignment_overrides`` 는 ``readiness`` 블록이 선언한 정렬 임계를 받는 통로다.
    QA 와 게이트가 **같은 임계**로 판정해야 한다 — 두 곳이 각자 기본값을 들고 있으면
    "QA 는 통과했는데 게이트는 실패" 같은 해석 불가능한 상태가 생긴다.
    """

    sample_rate = int(data_cfg["sample_rate"])
    raw_segment = int(round(float(data_cfg["segment_seconds"]) * sample_rate))
    segment_samples = max(256, (raw_segment // 256) * 256)
    overrides = dict(alignment_overrides or {})
    notes: list[str] = []
    for legacy, (replacement, ceiling) in _DEPRECATED_ALIGNMENT_KEYS.items():
        if legacy not in overrides:
            continue
        declared = float(overrides.pop(legacy))
        effective = min(declared, float(ceiling))
        notes.append(
            f"{legacy}={declared:g} 는 폐기됐습니다 (광대역 argmax 추정기 기준 값). "
            f"{replacement} 를 쓰고, 실측 근거 상한 {ceiling:g} 과 비교해 더 엄격한 쪽인 "
            f"{effective:g} 을 적용합니다"
        )
        overrides.setdefault(replacement, effective)
    for dropped, reason in _DEPRECATED_ALIGNMENT_DROPPED.items():
        if dropped in overrides:
            overrides.pop(dropped)
            notes.append(f"{dropped} 는 더 이상 쓰이지 않습니다 — {reason}")
    unknown = sorted(set(overrides).difference(_ALIGNMENT_OVERRIDE_KEYS))
    if unknown:
        raise ValueError(f"알 수 없는 정렬 설정 키: {unknown}")
    return RecordedQASettings(
        **overrides,
        deprecated_threshold_notes=tuple(notes),
        sample_rate=sample_rate,
        segment_samples=segment_samples,
        digital_reference_lead_samples=int(
            data_cfg.get("digital_reference_lead_samples", 0)
        ),
        reference_mode=str(data_cfg.get("reference_mode", "digital")),
        block_frames=int(block_frames),
        clip_threshold=float(clip_threshold),
        max_clip_ratio=float(max_clip_ratio),
        min_mic_rms_dbfs=float(min_mic_rms_dbfs),
        min_source_rms_dbfs=float(min_source_rms_dbfs),
        required_splits=tuple(required_splits),
        allow_incomplete_family_coverage=bool(allow_incomplete_family_coverage),
    )


def _identifier(field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}는 비어 있지 않은 문자열이어야 합니다")
    if value != value.strip() or len(value) > 128:
        raise ValueError(f"{field} 형식이 올바르지 않습니다: {value!r}")
    if any(ch in value for ch in ("/", "\\", "\0", "\n", "\r")):
        raise ValueError(f"{field}에 경로 구분자/제어문자를 사용할 수 없습니다")
    return value


def _dbfs_from_sum_squares(sum_squares: np.ndarray, frames: int) -> list[float]:
    rms = np.sqrt(sum_squares / max(1, frames))
    return [20.0 * math.log10(max(float(value), 1.0e-12)) for value in rms]


def _stream_audio(path: Path, settings: RecordedQASettings) -> dict:
    """오디오 하나를 블록 단위로 전수 검사한다."""

    with sf.SoundFile(str(path), mode="r") as audio:
        channels = int(audio.channels)
        declared_frames = int(audio.frames)
        sum_squares = np.zeros(channels, dtype=np.float64)
        clipped = np.zeros(channels, dtype=np.int64)
        nonfinite = np.zeros(channels, dtype=np.int64)
        peak = np.zeros(channels, dtype=np.float64)
        frames = 0
        blocks_read = 0

        while True:
            block = audio.read(
                frames=settings.block_frames,
                dtype="float32",
                always_2d=True,
            )
            if block.shape[0] == 0:
                break
            blocks_read += 1
            frames += int(block.shape[0])
            finite = np.isfinite(block)
            nonfinite += np.count_nonzero(~finite, axis=0)
            safe = np.where(finite, block, 0.0).astype(np.float64, copy=False)
            sum_squares += np.einsum("ij,ij->j", safe, safe)
            magnitude = np.abs(safe)
            clipped += np.count_nonzero(
                magnitude >= settings.clip_threshold, axis=0
            )
            peak = np.maximum(peak, np.max(magnitude, axis=0))

        return {
            "path": str(path),
            "sample_rate": int(audio.samplerate),
            "channels": channels,
            "frames": frames,
            "declared_frames": declared_frames,
            "duration_s": frames / float(audio.samplerate),
            "format": str(audio.format),
            "subtype": str(audio.subtype),
            "blocks_read": blocks_read,
            "rms_dbfs": _dbfs_from_sum_squares(sum_squares, frames),
            "peak": [float(value) for value in peak],
            "clip_ratio": [
                float(value) / max(1, frames) for value in clipped
            ],
            "nonfinite_samples": [int(value) for value in nonfinite],
        }


def _read_session_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"session.json을 읽을 수 없습니다: {exc}"
    if not isinstance(value, dict):
        return None, "session.json 최상위 값은 JSON 객체여야 합니다"
    return value, None


def _append_error(result: dict, message: str) -> None:
    result["errors"].append(message)


def _validate_manifest_metadata(entry: dict, session_path: Path, result: dict) -> None:
    for key in ("split", "source_family", "group_id", "session_id"):
        if key not in entry:
            _append_error(result, f"manifest 필수 필드 누락: {key}")

    split = entry.get("split")
    if split not in VALID_SPLITS:
        _append_error(result, f"잘못된 split: {split!r}")

    try:
        result["source_family"] = validate_source_family(entry.get("source_family"))
    except ValueError as exc:
        _append_error(result, str(exc))
    try:
        result["group_id"] = validate_group_id(entry.get("group_id"))
    except ValueError as exc:
        _append_error(result, str(exc))
    try:
        result["session_id"] = _identifier("session_id", entry.get("session_id"))
    except ValueError as exc:
        _append_error(result, str(exc))

    session_id = result.get("session_id")
    if session_id is not None and session_id != session_path.name:
        _append_error(
            result,
            f"manifest session_id({session_id!r})와 디렉터리명({session_path.name!r}) 불일치",
        )
    if entry.get("tag") not in (None, "recorded"):
        _append_error(result, f"recorded manifest의 tag가 아닙니다: {entry.get('tag')!r}")


def _validate_session_metadata(
    metadata: dict,
    entry: dict,
    result: dict,
    settings: RecordedQASettings,
) -> None:
    for key, validator in (
        ("source_family", validate_source_family),
        ("group_id", validate_group_id),
    ):
        if key not in metadata:
            _append_error(result, f"session.json 필수 필드 누락: {key}")
            continue
        try:
            value = validator(metadata[key])
        except ValueError as exc:
            _append_error(result, f"session.json: {exc}")
            continue
        if entry.get(key) != value:
            _append_error(
                result,
                f"session.json {key}({value!r})와 manifest({entry.get(key)!r}) 불일치",
            )

    # 현재 수집 포맷은 session_id를 디렉터리/manifest가 소유한다. 향후 JSON에도
    # 기록되면 그 값까지 엄격히 교차검증한다.
    if "session_id" in metadata:
        try:
            metadata_session_id = _identifier("session_id", metadata["session_id"])
        except ValueError as exc:
            _append_error(result, f"session.json: {exc}")
        else:
            if metadata_session_id != entry.get("session_id"):
                _append_error(
                    result,
                    "session.json session_id와 manifest session_id가 다릅니다: "
                    f"{metadata_session_id!r} != {entry.get('session_id')!r}",
                )

    if "sample_rate" in metadata:
        try:
            metadata_sr = int(metadata["sample_rate"])
        except (TypeError, ValueError):
            _append_error(result, "session.json sample_rate가 정수가 아닙니다")
        else:
            if metadata_sr != settings.sample_rate:
                _append_error(
                    result,
                    f"session.json sample_rate {metadata_sr} != {settings.sample_rate}",
                )


CHANNEL_ERR_MIC = 0
CHANNEL_REF_MIC = 1
"""``session.json`` 채널 규약: ``err_mic: 0, ref_mic: 1``. 두 곳에서 따로 0/1 을
쓰지 않도록 상수로 올린다 — 채널을 뒤바꿔 읽으면 정렬 게이트가 대조군과 피검사군을
맞바꾸고, 붕괴한 세션이 통과한다."""


def _read_alignment_excerpt(
    path: Path, settings: RecordedQASettings
) -> tuple[np.ndarray, int]:
    """정렬 검사용 앞부분 발췌를 읽는다. 길이는 ``alignment_max_seconds`` 로 묶인다."""

    max_frames = int(round(float(settings.alignment_max_seconds) * settings.sample_rate))
    with sf.SoundFile(str(path)) as handle:
        data = handle.read(frames=max_frames, dtype="float64", always_2d=True)
    return np.asarray(data, dtype=np.float64), int(data.shape[0])


def _validate_alignment(
    session_path: Path,
    result: dict,
    settings: RecordedQASettings,
) -> None:
    """학습이 배워야 하는 **관계 자체**를 검사한다 (결함 2).

    파일별 통계(RMS/clip/길이)는 채널 사이의 시간 관계에 대해 아무것도 말하지 않는다.
    실측 80 세션이 전부 통과한 이유가 정확히 그것이다.

    세 값을 함께 본다 — 셋을 따로 보면 진단이 서지 않는다.

    ============================  ==============  ==============  ================
    지표                          붕괴 세션 실측  음향 대조군      임계
    ============================  ==============  ==============  ================
    coh²(source→ERR)              0.021~0.126     —               ≥ 0.60
    coh²(REF→ERR)                 —               0.959~0.991     ≥ 0.60
    source→ERR τ std (1초창)      1019~2216       17.7~20.1       ≤ 64
    source→ERR τ range            8869~13532      106~215         ≤ 256
    ============================  ==============  ==============  ================

    REF→ERR 이 살아 있는데 source→ERR 만 죽었다는 조합이 **진단 그 자체**다: 마이크도
    스피커도 배치도 멀쩡하고, 재생(USB)과 캡처(I²S)가 서로 다른 클록 도메인인데
    콜백이 인덱스로만 정렬했다는 뜻이다. 그래서 대조군을 함께 잰다.

    검사는 :mod:`deep_anc.dsp.invariants` 의 공용 검사기를 부른다. 여기서 코히런스를
    다시 구현하면 그것이 두 번째 정의가 되고, 언젠가 측정·QA·게이트가 서로 다른 답을
    내놓는다 — 이 저장소에서 이미 여러 번 일어난 일이다.
    """

    from ..dsp.invariants import check_stream_coherence, check_stream_delay_stability

    mics_path = session_path / "mics.wav"
    # **학습이 실제로 읽는 파일**을 잰다. RecordedANCDataset 은 source_aligned.wav 가
    # 있으면 그것을 x_ref 로 쓰므로, QA 가 source.wav 만 보면 "QA 가 잰 신호"와
    # "학습이 쓴 신호"가 갈린다 — 같은 물리량을 두 곳에서 따로 보는 발생기 A 그 자체다.
    # source.wav 는 원본 provenance 로 남아 있고 재정렬 전에는 영원히 붕괴 상태이므로,
    # 재정렬본이 있어도 source.wav 를 재면 복구된 세션까지 전부 FAIL 이 된다.
    aligned_path = session_path / "source_aligned.wav"
    source_path = aligned_path if aligned_path.is_file() else session_path / "source.wav"
    result.setdefault("alignment_reference", source_path.name)
    try:
        mics_data, _ = _read_alignment_excerpt(mics_path, settings)
        source_data, _ = _read_alignment_excerpt(source_path, settings)
    except (OSError, RuntimeError, ValueError) as exc:
        _append_error(result, f"정렬 검사용 오디오를 읽지 못했습니다: {exc}")
        return
    if mics_data.shape[1] < 2 or source_data.shape[1] < 1:
        return  # 채널 수 오류는 _validate_audio 가 이미 보고했다
    frames = int(min(mics_data.shape[0], source_data.shape[0]))
    if frames < 2:
        _append_error(result, "정렬 검사를 할 만큼의 오디오가 없습니다")
        return

    err = mics_data[:frames, CHANNEL_ERR_MIC]
    ref = mics_data[:frames, CHANNEL_REF_MIC]
    src = source_data[:frames, 0]
    if not (
        np.all(np.isfinite(err)) and np.all(np.isfinite(ref)) and np.all(np.isfinite(src))
    ):
        # NaN/Inf 는 코히런스를 조용히 NaN 으로 만들고 `NaN < 임계` 는 False 라
        # **통과처럼 보인다**. 비유한 샘플 자체는 _validate_audio 가 이미 오류로
        # 보고했으므로 여기서는 판정을 흉내내지 않고 측정하지 않았음을 남긴다.
        result["alignment"] = {"ok": False, "skipped": "비유한 샘플이 있어 측정 불가"}
        return
    band = (float(settings.alignment_band_hz[0]), float(settings.alignment_band_hz[1]))

    alignment: dict[str, Any] = {
        "band_hz": [band[0], band[1]],
        "nperseg": int(settings.alignment_nperseg),
        "analysed_seconds": frames / float(settings.sample_rate),
        "reference_file": source_path.name,
    }
    try:
        coherence_check = check_stream_coherence(
            src,
            err,
            sample_rate=settings.sample_rate,
            band_hz=band,
            min_coherence=float(settings.min_source_err_coherence),
            nperseg=int(settings.alignment_nperseg),
            control=ref,
            name="source_err_coherence",
        )
        delay_check = check_stream_delay_stability(
            src,
            err,
            sample_rate=settings.sample_rate,
            max_robust_std_samples=float(settings.max_source_err_delay_robust_std_samples),
            max_p95_p5_samples=float(settings.max_source_err_delay_p95_p5_samples),
            min_valid_windows=int(settings.min_source_err_delay_windows),
            min_valid_window_ratio=float(settings.min_source_err_delay_window_ratio),
            name="source_err_delay_stability",
        )
    except (ValueError, RuntimeError) as exc:
        _append_error(result, f"정렬 검사 실패: {exc}")
        return

    control_coherence = coherence_check.measured.get("control_coherence")
    alignment.update(
        {
            "source_err_coherence": float(coherence_check.measured["coherence"]),
            "ref_err_coherence": (
                float(control_coherence) if control_coherence is not None else float("nan")
            ),
            # 부기 이름이 곧 물리량이다. robust_std/p95_p5 는 판정량이고,
            # raw_std/ptp 는 진단이다 — 이름이 섞이면 다음 사람이 다시 생 std 로
            # 게이트를 만든다.
            "source_err_delay_median_samples": float(
                delay_check.measured["median_samples"]
            ),
            "source_err_delay_robust_std_samples": float(
                delay_check.measured["robust_std_samples"]
            ),
            "source_err_delay_p95_p5_samples": float(
                delay_check.measured["p95_p5_samples"]
            ),
            "source_err_delay_raw_std_samples": float(
                delay_check.measured["raw_std_samples"]
            ),
            "source_err_delay_ptp_samples": float(delay_check.measured["ptp_samples"]),
            "delay_windows": int(delay_check.measured["windows"]),
            "delay_valid_windows": int(delay_check.measured["valid_windows"]),
            "delay_valid_window_ratio": float(
                delay_check.measured["valid_window_ratio"]
            ),
            "delay_track_band_hz": list(delay_check.measured["band_hz"]),
            "delay_window_samples": int(delay_check.measured["window_samples"]),
            "delay_estimator": "deep_anc.data.timeline.measure_delay_trajectory",
            "ok": bool(coherence_check.ok and delay_check.ok),
        }
    )
    result["alignment"] = alignment

    if not coherence_check.ok:
        _append_error(result, coherence_check.detail)
    if control_coherence is not None and float(control_coherence) < float(
        settings.min_ref_err_coherence
    ):
        _append_error(
            result,
            f"REF→ERR {band[0]:.0f}-{band[1]:.0f}Hz 결맞음 {float(control_coherence):.3f} < "
            f"{float(settings.min_ref_err_coherence):.2f} — 마이크/배치 문제입니다 "
            "(소프트웨어 타임베이스가 아니라 음향 쪽)",
        )
    if not delay_check.ok:
        _append_error(result, delay_check.detail)


def _validate_audio(
    entry: dict,
    metadata: dict | None,
    mics: dict,
    source: dict | None,
    result: dict,
    settings: RecordedQASettings,
) -> None:
    if mics["channels"] != 2:
        _append_error(result, f"mics.wav는 정확히 2채널이어야 합니다: {mics['channels']}")
    audio_items = [("mics.wav", mics)]
    if source is not None:
        if source["channels"] != 1:
            _append_error(result, f"source.wav는 mono여야 합니다: {source['channels']}채널")
        audio_items.append(("source.wav", source))
    for label, stats in audio_items:
        if stats["sample_rate"] != settings.sample_rate:
            _append_error(
                result,
                f"{label} sample rate {stats['sample_rate']} != {settings.sample_rate}",
            )
        if stats["frames"] != stats["declared_frames"]:
            _append_error(
                result,
                f"{label} 헤더 frames({stats['declared_frames']})와 읽은 frames"
                f"({stats['frames']}) 불일치",
            )
        if sum(stats["nonfinite_samples"]) > 0:
            _append_error(
                result,
                f"{label}에 비유한 샘플 {sum(stats['nonfinite_samples'])}개",
            )
        for channel, ratio in enumerate(stats["clip_ratio"]):
            if ratio > settings.max_clip_ratio:
                _append_error(
                    result,
                    f"{label} ch{channel} clip ratio {ratio:.3%} > "
                    f"{settings.max_clip_ratio:.3%}",
                )

    if source is not None and mics["frames"] != source["frames"]:
        _append_error(
            result,
            f"mics/source 길이 불일치: {mics['frames']} != {source['frames']}",
        )
    shortest = min(mics["frames"], source["frames"]) if source is not None else mics["frames"]
    if shortest < settings.minimum_frames:
        lead_label = "digital lead" if settings.reference_mode == "digital" else "acoustic"
        _append_error(
            result,
            f"학습 세그먼트+{lead_label} 최소길이 미달: "
            f"{shortest} < {settings.minimum_frames}",
        )

    for channel, rms in enumerate(mics["rms_dbfs"]):
        if rms < settings.min_mic_rms_dbfs:
            _append_error(
                result,
                f"mics.wav ch{channel} RMS {rms:.1f}dBFS < "
                f"{settings.min_mic_rms_dbfs:.1f}dBFS",
            )

    if source is not None:
        family = str(entry.get("source_family", "")).lower()
        program = metadata.get("program", {}) if isinstance(metadata, dict) else {}
        program_type = (
            str(program.get("type", "")).lower() if isinstance(program, dict) else ""
        )
        intentional_silence = family == "silence" or program_type == "silence"
        source_rms = source["rms_dbfs"][0] if source["rms_dbfs"] else -240.0
        if not intentional_silence and source_rms < settings.min_source_rms_dbfs:
            _append_error(
                result,
                f"source.wav RMS {source_rms:.1f}dBFS < "
                f"{settings.min_source_rms_dbfs:.1f}dBFS",
            )

    manifest_sr = entry.get("sample_rate")
    if manifest_sr is None:
        _append_error(result, "manifest 필수 필드 누락: sample_rate")
    else:
        try:
            manifest_sr_value = int(manifest_sr)
        except (TypeError, ValueError):
            _append_error(result, f"manifest sample_rate가 정수가 아닙니다: {manifest_sr!r}")
        else:
            if manifest_sr_value != mics["sample_rate"]:
                _append_error(
                    result,
                    f"manifest sample_rate {manifest_sr_value} != mics {mics['sample_rate']}",
                )

    manifest_duration = entry.get("duration_s")
    if manifest_duration is None:
        _append_error(result, "manifest 필수 필드 누락: duration_s")
    else:
        try:
            duration_value = float(manifest_duration)
        except (TypeError, ValueError):
            _append_error(result, f"manifest duration_s가 숫자가 아닙니다: {manifest_duration!r}")
        else:
            if not math.isfinite(duration_value) or abs(duration_value - mics["duration_s"]) > (
                1.0 / settings.sample_rate
            ):
                _append_error(
                    result,
                    f"manifest duration_s {duration_value!r} != mics {mics['duration_s']:.6f}",
                )

    if "channels" in entry:
        try:
            manifest_channels = int(entry["channels"])
        except (TypeError, ValueError):
            _append_error(result, "manifest channels가 정수가 아닙니다")
        else:
            if manifest_channels != mics["channels"]:
                _append_error(
                    result,
                    f"manifest channels {manifest_channels} != mics {mics['channels']}",
                )

    if metadata is not None and "seconds" in metadata:
        try:
            metadata_seconds = float(metadata["seconds"])
        except (TypeError, ValueError):
            _append_error(result, "session.json seconds가 숫자가 아닙니다")
        else:
            if not math.isfinite(metadata_seconds) or abs(
                metadata_seconds - mics["duration_s"]
            ) > (1.0 / settings.sample_rate):
                _append_error(
                    result,
                    f"session.json seconds {metadata_seconds!r} != mics "
                    f"{mics['duration_s']:.6f}",
                )


def _validate_one_session(entry: dict, settings: RecordedQASettings) -> dict:
    session_path = Path(str(entry.get("path", "")))
    result: dict[str, Any] = {
        "path": str(session_path),
        "split": entry.get("split"),
        "session_id": entry.get("session_id"),
        "group_id": entry.get("group_id"),
        "source_family": entry.get("source_family"),
        "errors": [],
        "warnings": [],
        "audio": {},
        # 정렬 결과는 **항상** 자리를 갖는다. 키가 없으면 소비자가 `.get("alignment", {})`
        # 로 조용히 넘어가고, "검사하지 않았다"와 "검사해서 통과했다"가 구분되지 않는다.
        "alignment": {},
    }
    _validate_manifest_metadata(entry, session_path, result)

    if not session_path.is_dir():
        _append_error(result, f"세션 디렉터리가 없습니다: {session_path}")
        result["ok"] = False
        return result

    paths = {
        "mics": session_path / "mics.wav",
        "source": session_path / "source.wav",
        "metadata": session_path / "session.json",
    }
    required_files = {"mics", "metadata"}
    if settings.reference_mode == "digital":
        required_files.add("source")
    for label in required_files:
        path = paths[label]
        if not path.is_file():
            _append_error(result, f"필수 파일 누락: {path.name}")

    metadata: dict | None = None
    if paths["metadata"].is_file():
        metadata, metadata_error = _read_session_json(paths["metadata"])
        if metadata_error is not None:
            _append_error(result, metadata_error)
        elif metadata is not None:
            _validate_session_metadata(metadata, entry, result, settings)

    stats: dict[str, dict] = {}
    audio_keys = ("mics", "source") if settings.reference_mode == "digital" else ("mics",)
    for key in audio_keys:
        path = paths[key]
        if not path.is_file():
            continue
        try:
            stats[key] = _stream_audio(path, settings)
        except (OSError, RuntimeError, ValueError) as exc:
            _append_error(result, f"{path.name} 스트리밍 읽기 실패: {exc}")

    result["audio"] = stats
    source_ready = settings.reference_mode != "digital" or "source" in stats
    if "mics" in stats and source_ready:
        _validate_audio(
            entry, metadata, stats["mics"], stats.get("source"), result, settings
        )
        # 채널 간 관계 게이트. digital reference 일 때만 source.wav 가 존재하며,
        # 그때가 바로 "학습이 source→ERR 관계를 배운다"고 주장하는 경우다.
        if (
            settings.check_alignment
            and settings.reference_mode == "digital"
            and "source" in stats
            and stats["mics"]["channels"] == 2
            and stats["source"]["channels"] == 1
        ):
            _validate_alignment(session_path, result, settings)
        result["duration_s"] = float(stats["mics"]["duration_s"])
    else:
        result["duration_s"] = 0.0
    result["ok"] = not result["errors"]
    return result


def _coverage_summary(results: list[dict]) -> tuple[dict, dict]:
    split_summary: dict[str, dict] = {}
    family_summary: dict[str, dict] = {}
    for result in results:
        split = str(result.get("split"))
        family = str(result.get("source_family"))
        group = str(result.get("group_id"))
        duration = float(result.get("duration_s", 0.0))

        split_item = split_summary.setdefault(
            split,
            {"sessions": 0, "valid_sessions": 0, "duration_s": 0.0, "groups": set(), "families": {}},
        )
        split_item["sessions"] += 1
        split_item["valid_sessions"] += int(bool(result.get("ok")))
        split_item["duration_s"] += duration
        split_item["groups"].add(group)
        split_item["families"][family] = split_item["families"].get(family, 0) + 1

        family_item = family_summary.setdefault(
            family,
            {"sessions": 0, "duration_s": 0.0, "groups": set(), "splits": {}},
        )
        family_item["sessions"] += 1
        family_item["duration_s"] += duration
        family_item["groups"].add(group)
        family_item["splits"][split] = family_item["splits"].get(split, 0) + 1

    for mapping in (split_summary, family_summary):
        for item in mapping.values():
            item["groups"] = len(item["groups"])
    return split_summary, family_summary


def validate_recorded_sessions(
    entries: list[dict], settings: RecordedQASettings, *, manifest_path: str = ""
) -> dict:
    """``read_manifest``가 반환한 세션 경로를 소비해 QA 리포트를 만든다."""

    results = [_validate_one_session(dict(entry), settings) for entry in entries]
    global_errors: list[str] = []
    global_warnings: list[str] = []

    if not entries:
        global_errors.append("manifest에 실측 세션이 없습니다")
    # 폐기된 임계 키를 조용히 무시하지 않는다 — 설정 파일만 읽은 사람이 게이트를
    # 오해하는 것이 이 저장소에서 반복된 실패 방식이다.
    global_warnings.extend(settings.deprecated_threshold_notes)

    session_ids: dict[str, list[str]] = {}
    paths: dict[str, list[str]] = {}
    group_splits: dict[str, set[str]] = {}
    group_families: dict[str, set[str]] = {}
    for result in results:
        session_id = str(result.get("session_id"))
        path = str(result.get("path"))
        group = str(result.get("group_id"))
        split = str(result.get("split"))
        family = str(result.get("source_family"))
        session_ids.setdefault(session_id, []).append(path)
        paths.setdefault(path, []).append(session_id)
        group_splits.setdefault(group, set()).add(split)
        group_families.setdefault(group, set()).add(family)

    for session_id, session_paths in session_ids.items():
        if len(session_paths) > 1:
            global_errors.append(
                f"중복 session_id={session_id!r}: {', '.join(session_paths)}"
            )
    for path, identifiers in paths.items():
        if len(identifiers) > 1:
            global_errors.append(f"중복 세션 경로={path!r}: {identifiers}")
    for group, splits in group_splits.items():
        if len(splits) > 1:
            global_errors.append(
                f"치명: group_id={group!r}가 여러 split에 걸쳐 있습니다: {sorted(splits)}"
            )
    for group, families in group_families.items():
        if len(families) > 1:
            global_errors.append(
                f"group_id={group!r}의 source_family가 일관되지 않습니다: {sorted(families)}"
            )

    split_summary, family_summary = _coverage_summary(results)
    for split in settings.required_splits:
        if split not in split_summary or split_summary[split]["sessions"] == 0:
            global_errors.append(f"필수 split에 세션이 없습니다: {split}")

    for family, item in sorted(family_summary.items()):
        missing = [split for split in settings.required_splits if split not in item["splits"]]
        if missing:
            message = f"source_family={family!r}가 다음 split에 없습니다: {missing}"
            if settings.allow_incomplete_family_coverage:
                global_warnings.append(message)
            else:
                global_errors.append(message)

    total_duration = sum(float(result.get("duration_s", 0.0)) for result in results)
    report = {
        "ok": not global_errors and all(result.get("ok") for result in results),
        "manifest": manifest_path,
        "settings": {
            "sample_rate": settings.sample_rate,
            "segment_samples": settings.segment_samples,
            "digital_reference_lead_samples": settings.effective_lead_samples,
            "minimum_frames": settings.minimum_frames,
            "block_frames": settings.block_frames,
            "clip_threshold": settings.clip_threshold,
            "max_clip_ratio": settings.max_clip_ratio,
            "min_mic_rms_dbfs": settings.min_mic_rms_dbfs,
            "min_source_rms_dbfs": settings.min_source_rms_dbfs,
            "required_splits": list(settings.required_splits),
            "allow_incomplete_family_coverage": settings.allow_incomplete_family_coverage,
            "check_alignment": settings.check_alignment,
            "alignment_band_hz": list(settings.alignment_band_hz),
            "min_source_err_coherence": settings.min_source_err_coherence,
            "min_ref_err_coherence": settings.min_ref_err_coherence,
            "max_source_err_delay_robust_std_samples": (
                settings.max_source_err_delay_robust_std_samples
            ),
            "max_source_err_delay_p95_p5_samples": (
                settings.max_source_err_delay_p95_p5_samples
            ),
            "min_source_err_delay_windows": settings.min_source_err_delay_windows,
            "min_source_err_delay_window_ratio": (
                settings.min_source_err_delay_window_ratio
            ),
            "delay_estimator": "deep_anc.data.timeline.measure_delay_trajectory",
            "deprecated_threshold_notes": list(settings.deprecated_threshold_notes),
        },
        "summary": {
            "sessions": len(results),
            "valid_sessions": sum(int(bool(result.get("ok"))) for result in results),
            "invalid_sessions": sum(int(not bool(result.get("ok"))) for result in results),
            "groups": len(group_splits),
            "families": len(family_summary),
            "duration_s": total_duration,
            "splits": split_summary,
            "source_families": family_summary,
        },
        "errors": global_errors,
        "warnings": global_warnings,
        "sessions": results,
    }
    return report


def failure_report(message: str, *, manifest_path: str = "") -> dict:
    """manifest/config 로드 자체가 실패했을 때도 저장 가능한 리포트."""

    return {
        "ok": False,
        "manifest": manifest_path,
        "settings": {},
        "summary": {
            "sessions": 0,
            "valid_sessions": 0,
            "invalid_sessions": 0,
            "groups": 0,
            "families": 0,
            "duration_s": 0.0,
            "splits": {},
            "source_families": {},
        },
        "errors": [message],
        "warnings": [],
        "sessions": [],
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_recorded_qa_markdown(report: dict) -> str:
    """JSON 리포트와 같은 내용을 사람이 읽는 Markdown으로 렌더링한다."""

    summary = report.get("summary", {})
    settings = report.get("settings", {})
    lines = [
        "# 실측 ANC 세션 QA 리포트",
        "",
        f"- 판정: **{'PASS' if report.get('ok') else 'FAIL'}**",
        f"- manifest: `{_markdown_cell(report.get('manifest', ''))}`",
        f"- 세션: {summary.get('valid_sessions', 0)}/{summary.get('sessions', 0)} 유효",
        f"- 분량: {float(summary.get('duration_s', 0.0)) / 60.0:.2f}분",
    ]
    if settings:
        lines += [
            f"- 최소 길이: {settings.get('minimum_frames')} samples "
            f"(segment {settings.get('segment_samples')} + lead "
            f"{settings.get('digital_reference_lead_samples')} + 1)",
            f"- 판정값: mic/source RMS ≥ {settings.get('min_mic_rms_dbfs')}/"
            f"{settings.get('min_source_rms_dbfs')}dBFS, clip ≤ "
            f"{100.0 * float(settings.get('max_clip_ratio', 0.0)):.3f}%",
        ]

    lines += [
        "",
        "## Split 커버리지",
        "",
        "| split | sessions | valid | groups | duration(min) | source families |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in VALID_SPLITS:
        item = summary.get("splits", {}).get(split, {})
        families = ", ".join(
            f"{name}:{count}" for name, count in sorted(item.get("families", {}).items())
        ) or "-"
        lines.append(
            f"| {split} | {item.get('sessions', 0)} | {item.get('valid_sessions', 0)} | "
            f"{item.get('groups', 0)} | {float(item.get('duration_s', 0.0)) / 60.0:.2f} | "
            f"{_markdown_cell(families)} |"
        )

    lines += [
        "",
        "## Source-family 커버리지",
        "",
        "| family | sessions | groups | duration(min) | train / val / test |",
        "|---|---:|---:|---:|---|",
    ]
    for family, item in sorted(summary.get("source_families", {}).items()):
        split_counts = item.get("splits", {})
        lines.append(
            f"| {_markdown_cell(family)} | {item.get('sessions', 0)} | {item.get('groups', 0)} | "
            f"{float(item.get('duration_s', 0.0)) / 60.0:.2f} | "
            f"{split_counts.get('train', 0)} / {split_counts.get('val', 0)} / "
            f"{split_counts.get('test', 0)} |"
        )

    lines += [
        "",
        "## 세션 검사",
        "",
        "| session | split | family | group | duration(s) | mic RMS ch0/ch1 | source RMS | "
        "max clip | blocks | 판정 |",
        "|---|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for result in report.get("sessions", []):
        audio = result.get("audio", {})
        mics = audio.get("mics", {})
        source = audio.get("source", {})
        mic_rms = "/".join(f"{value:.1f}" for value in mics.get("rms_dbfs", [])) or "-"
        source_rms_values = source.get("rms_dbfs", [])
        source_rms = f"{source_rms_values[0]:.1f}" if source_rms_values else "-"
        clip_values = list(mics.get("clip_ratio", [])) + list(source.get("clip_ratio", []))
        max_clip = 100.0 * max(clip_values, default=0.0)
        blocks = int(mics.get("blocks_read", 0)) + int(source.get("blocks_read", 0))
        verdict = "PASS" if result.get("ok") else "FAIL: " + "; ".join(result.get("errors", []))
        lines.append(
            f"| {_markdown_cell(result.get('session_id'))} | {_markdown_cell(result.get('split'))} | "
            f"{_markdown_cell(result.get('source_family'))} | {_markdown_cell(result.get('group_id'))} | "
            f"{float(result.get('duration_s', 0.0)):.2f} | {mic_rms} | {source_rms} | "
            f"{max_clip:.3f}% | {blocks} | {_markdown_cell(verdict)} |"
        )

    if report.get("errors"):
        lines += ["", "## 치명 오류", ""]
        lines.extend(f"- {_markdown_cell(message)}" for message in report["errors"])
    if report.get("warnings"):
        lines += ["", "## 경고", ""]
        lines.extend(f"- {_markdown_cell(message)}" for message in report["warnings"])

    lines += [
        "",
        "> 이 검사는 파일을 블록 단위로 읽기만 하며 오디오 출력 장치를 열지 않는다.",
        "",
    ]
    return "\n".join(lines)
