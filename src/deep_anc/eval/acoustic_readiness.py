"""오디오·추론 엔진을 열지 않는 acoustic-reference 설정/인과성 진단.

이 보고서는 선언된 기하와 측정 S(z)의 필요조건을 확인한다. 실제 입력 품질,
런타임 지터, 안정성, 비선형, 공간 감쇠 또는 checkpoint 배포 자격을 인증하지 않는다.
digital P(z), 소음 출력 gain, digital playback lead를 예측 여유로 사용하지 않는다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from deep_anc.config import REPO_ROOT, load_runtime_config


# 기존 파인튜닝 경로 측정 게이트와 같은 반복 일관성 기준.
MIN_PATH_CONSISTENCY = 0.9


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다")
    return number


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    number = _number(value, name)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{name}: {minimum} 이상의 정수가 필요합니다")
    return int(number)


def _band(value: Any, name: str, sample_rate: int) -> list[float]:
    values = np.asarray(value).reshape(-1)
    if values.size != 2:
        raise ValueError(f"{name}: [하한, 상한] 두 값이 필요합니다")
    low, high = (_number(v, name) for v in values)
    if not 0 <= low < high <= sample_rate / 2:
        raise ValueError(f"{name}: 0 ≤ 하한 < 상한 ≤ Nyquist이어야 합니다")
    return [low, high]


def _scalar(data: Any, key: str) -> Any:
    if key not in data:
        raise ValueError(f"S(z) NPZ에 필수 메타데이터 {key}가 없습니다")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"S(z) {key}: 스칼라가 필요합니다")
    return value.item()


def _read_secondary_metadata(path: str | Path) -> dict:
    """기존 secondary_path 로더는 torch를 import하므로 여기서는 NPZ만 읽는다.

    delay/sample_rate의 legacy 기본값과 excitation→trusted 폴백은 사용하지 않는다.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = p if p.exists() else REPO_ROOT / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"2차경로 모델이 없습니다: {p}")
    loaded = np.load(p, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"S(z)는 NPZ 파일이어야 합니다: {p}")
    with loaded as data:
        if "fir" not in data:
            raise ValueError("S(z) NPZ에 fir이 없습니다")
        fir = np.asarray(data["fir"])
        if (
            fir.size == 0 or not np.issubdtype(fir.dtype, np.number)
            or np.iscomplexobj(fir) or not np.all(np.isfinite(fir))
            or not np.any(fir != 0)
        ):
            raise ValueError("S(z) FIR은 유한하고 0이 아닌 실수 배열이어야 합니다")
        sample_rate = _integer(_scalar(data, "sample_rate"), "S(z) sample_rate", 1)
        delay = _integer(_scalar(data, "delay_samples"), "S(z) delay_samples")
        consistency_band = (
            _band(data["consistency_band_hz"], "S(z) consistency_band_hz", sample_rate)
            if "consistency_band_hz" in data else None
        )
        excitation_band = (
            _band(data["excitation_band_hz"], "S(z) excitation_band_hz", sample_rate)
            if "excitation_band_hz" in data else None
        )
        consistency = (
            _number(_scalar(data, "coherence_median"), "S(z) coherence_median")
            if "coherence_median" in data else None
        )
        if consistency is not None and not 0 <= consistency <= 1 + 1e-9:
            raise ValueError("S(z) coherence_median은 0~1이어야 합니다")
        block = (
            _integer(_scalar(data, "calibration_block_size"), "S(z) calibration_block_size", 1)
            if "calibration_block_size" in data else None
        )
        latency = (
            str(_scalar(data, "calibration_latency"))
            if "calibration_latency" in data else None
        )
        channel = str(_scalar(data, "output_channel")) if "output_channel" in data else None
        if channel is not None and channel != "cancel":
            raise ValueError("S(z) output_channel은 cancel이어야 합니다")
    return {
        "path": str(p), "sample_rate": sample_rate, "delay_samples": delay,
        "fir_length": int(fir.size), "consistency_band_hz": consistency_band,
        "excitation_band_hz": excitation_band, "repeat_consistency": consistency,
        "calibration_block_size": block, "calibration_latency": latency,
    }


def build_acoustic_readiness_report(
    cfg: dict,
    *,
    require_broadband: bool = False,
    required_band_hz: tuple[float, float] | list[float] | None = None,
) -> dict:
    """load_runtime_config 결과를 진단한다. 잘못된 설정/파일은 예외로 실패한다.

    ``passed``는 요청한 선택 게이트만의 결과다. 기본 보고 모드의 True는 acoustic
    broadband 상쇄 성공이나 스피커 실행 허가를 뜻하지 않는다. 필수 메타가 없으면
    지연을 추정하지 않으며, 신뢰대역 메타가 없으면 대역을 미검증으로 표시한다.
    """
    try:
        if cfg["reference"] != "mic":
            raise ValueError("acoustic 진단에는 runtime.reference=mic이 필요합니다")
        if _integer(cfg["digital_reference_lead_samples"], "digital_reference_lead_samples") != 0:
            raise ValueError("acoustic reference의 digital_reference_lead_samples는 0이어야 합니다")
        audio = cfg["hardware"]["audio"]
        sample_rate = _integer(audio["sample_rate"], "audio.sample_rate", 1)
        block = _integer(audio["block_size"], "audio.block_size", 1)
        hop = _integer(cfg["hop"], "runtime.hop", 1)
        if hop != block:
            raise ValueError("현재 런타임은 hop == audio.block_size를 요구합니다")
        duct = cfg["duct"]
        handoff = _integer(duct["secondary_path"]["handoff_extra_samples"], "handoff_extra_samples")
        if handoff != hop:
            raise ValueError("현재 3-스레드 런타임의 handoff_extra_samples는 1 hop이어야 합니다")
        secondary = _read_secondary_metadata(duct["secondary_path"]["npz"])
        reference_x = _number(duct["positions_m"]["reference_mic"], "reference_mic")
        error_x = _number(duct["positions_m"]["error_mic"], "error_mic")
        sound_speed = _number(duct["duct"]["speed_of_sound_mps"], "speed_of_sound_mps")
        if sound_speed <= 0:
            raise ValueError("speed_of_sound_mps는 양수여야 합니다")
    except KeyError as exc:
        raise ValueError(f"필수 설정이 없습니다: {exc.args[0]}") from exc
    if secondary["sample_rate"] != sample_rate:
        raise ValueError("오디오와 S(z)의 sample_rate가 다릅니다")
    if secondary["calibration_block_size"] not in (None, block):
        raise ValueError("오디오 block_size가 S(z) 측정 조건과 다릅니다")
    if secondary["calibration_latency"] is not None:
        if audio.get("latency") != secondary["calibration_latency"]:
            raise ValueError("오디오 latency가 S(z) 측정 조건과 다릅니다")

    # 덕트의 +X 방향 입사라는 명시된 기하 가정. 역순 배치에 abs를 적용하지 않는다.
    preview = (error_x - reference_x) / sound_speed * sample_rate
    control_delay = secondary["delay_samples"] + handoff
    prediction = control_delay - preview
    arrival_met = prediction <= 0
    band = secondary["consistency_band_hz"]
    consistency = secondary["repeat_consistency"]
    quality_met = consistency is not None and consistency >= MIN_PATH_CONSISTENCY
    above_1khz = bool(band is not None and band[1] > 1000 and quality_met)
    requested = _band(required_band_hz, "required_band_hz", sample_rate) if required_band_hz is not None else None
    band_met = bool(
        requested is not None and band is not None and quality_met
        and band[0] <= requested[0] and band[1] >= requested[1]
    )
    warnings: list[str] = []
    if not arrival_met:
        warnings.append("제어음 도착이 늦어 예측 불가능한 광대역 신호의 인과성 필요조건을 만족하지 않습니다. 주기음 사용 자체를 금지하는 판정은 아닙니다.")
    if not above_1khz:
        warnings.append("S(z)의 1 kHz 초과 대역은 반복 일관성 기준으로 검증되지 않았습니다.")
    if band is None:
        warnings.append("consistency_band_hz가 없어 신뢰대역을 알 수 없습니다. 가진대역으로 대체하지 않습니다.")
    if not quality_met:
        warnings.append("S(z) 반복 일관성이 없거나 기존 경로 기준 0.9에 미달합니다.")
    if secondary["calibration_block_size"] is None or secondary["calibration_latency"] is None:
        warnings.append("S(z) 측정 block/latency 메타가 부족해 현재 I/O 설정과의 일치를 확인하지 못했습니다.")
    failures = []
    if require_broadband and not arrival_met:
        failures.append("broadband_arrival_condition")
    if requested is not None and not band_met:
        failures.append("required_band")
    return {
        "config_valid": True, "passed": not failures, "reference": "mic",
        "digital_reference_lead_samples": 0, "sample_rate": sample_rate,
        "secondary_path": secondary,
        "causality": {
            "geometry_assumption": "+X 방향 평면파 입사, REF/ERR 위치는 설정값이며 실측 선행시간이 아님",
            "reference_preview_samples": preview,
            "reference_preview_ms": 1000 * preview / sample_rate,
            "handoff_samples": handoff,
            "control_delay_samples": control_delay,
            "control_delay_ms": 1000 * control_delay / sample_rate,
            "required_prediction_samples": prediction,
            "required_prediction_ms": 1000 * prediction / sample_rate,
            "broadband_arrival_condition_met": arrival_met,
            "interpretation": "S 순수지연 + 1 hop − REF→ERR 기하선행. 필요 예측이 0 이하인 것은 단순 도착 필요조건 통과이며, 감쇠·안정성·quiet zone 성공을 보장하지 않습니다. 추론이 다음 콜백 전에 끝난다는 가정이며 추론시간을 중복 합산하지 않습니다.",
        },
        "band_validation": {
            "minimum_repeat_consistency": MIN_PATH_CONSISTENCY,
            "validated_band_extends_above_1khz": above_1khz,
            "required_band_hz": requested,
            "required_band_covered": band_met if requested is not None else None,
            "interpretation": "반복 정렬 후의 대역 일관성 메타이며 실시간 위상 안정성과 비선형 응답은 별도 검증이 필요합니다.",
        },
        "requirements": {"broadband": bool(require_broadband), "band_hz": requested},
        "failed_requirements": failures, "warnings": warnings,
        "scope": "설정/NPZ만 읽는 무출력 진단. 오디오 장치, torch, 모델, P(z)는 로드하지 않습니다.",
    }


def check_acoustic_readiness(
    config_path: str | Path,
    overrides: list[str] | None = None,
    *,
    require_broadband: bool = False,
    required_band_hz: tuple[float, float] | list[float] | None = None,
) -> dict:
    """설정 파일을 로드하여 JSON 직렬화 가능한 보고서를 반환한다."""
    return build_acoustic_readiness_report(
        load_runtime_config(config_path, overrides),
        require_broadband=require_broadband, required_band_hz=required_band_hz,
    )
