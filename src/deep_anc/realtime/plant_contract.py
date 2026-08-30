"""실시간 digital-reference ANC의 실제 덕트 플랜트 계약.

학습 체크포인트와 ONNX sidecar가 서로 같은 lead를 적고 있다는 사실만으로는
실제 덕트와 시간축이 맞는지 알 수 없다. 이 모듈은 오디오 장치를 열기 **전** strict
P/S 원시 증거와 현재 runtime 설정을 함께 읽어, runtime이 실제 플랜트의
``PlantDelays.lead()``를 쓰는지 검증한다.

이 검사는 digital-reference DL 경로에만 적용한다. FxLMS와 acoustic-reference는
별도의 적응/인과성 규약이 있으므로 여기서 억지로 같은 계약을 강제하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..config import _resolve_path
from ..dsp.secondary_path import SecondaryPathData, load_secondary_path
from ..dsp.timing import PlantDelays, TrainingTimingContract


STRICT_RUNTIME_BAND_HZ = (150.0, 1600.0)
"""현재 실배포 전 검증에 요구하는 P/S 공통 신뢰 대역.

고주파를 아직 측정하지 않았다는 사실을 숨기지 않기 위해, 이 상수는 2 kHz 이상을
"통과"시키지 않는다. 150–1600 Hz만 strict plant가 실제로 입증한 범위다.
"""


class RuntimePlantContractError(ValueError):
    """실제 P/S·raw provenance와 runtime 사이의 모순."""


@dataclass(frozen=True)
class RuntimePlantContract:
    """오디오 시작 직전에 확인한 strict P/S 플랜트 지문."""

    timing: TrainingTimingContract
    capture_id: str
    primary_path_sha256: str
    secondary_path_sha256: str
    raw_measurement_sha256: str
    analysis_sha256: str
    measurement_level_evidence_sha256: str


def runtime_requires_strict_plant_contract(cfg: dict[str, Any]) -> bool:
    """실제 P/S lead 검증이 필요한 runtime인지 반환한다."""

    return (
        str(cfg.get("reference", "digital")) == "digital"
        and str(cfg.get("controller", "dl")) == "dl"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(data: Any, key: str, *, label: str) -> Any:
    if key not in data:
        raise RuntimePlantContractError(f"{label}에 필수 metadata {key!r}가 없습니다")
    values = np.asarray(data[key])
    if values.size != 1:
        raise RuntimePlantContractError(
            f"{label} metadata {key!r}는 scalar여야 합니다: shape={values.shape}"
        )
    value = values.reshape(-1)[0]
    return value.item() if hasattr(value, "item") else value


def _vector(data: Any, key: str, *, label: str) -> np.ndarray:
    if key not in data:
        raise RuntimePlantContractError(f"{label}에 필수 metadata {key!r}가 없습니다")
    values = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise RuntimePlantContractError(
            f"{label} metadata {key!r}는 유한한 2원 대역이어야 합니다"
        )
    return values


def _metadata(path: Path, *, label: str) -> dict[str, Any]:
    """strict interleaved P/S에 필요한 작은 metadata만 안전하게 읽는다."""

    scalar_keys = (
        "capture_id",
        "sample_rate",
        "calibration_block_size",
        "calibration_latency",
        "output_channel",
        "output_pcm_provenance",
        "error_mic_channel",
        "reference_mic_channel",
        "noise_output_channel",
        "cancel_output_channel",
        "xrun_count",
        "repeats",
        "amplitude",
        "source_raw_npz_path",
        "source_raw_npz_sha256",
        "source_analysis_npz_path",
        "source_analysis_npz_sha256",
        "delay_semantics",
        "operator_confirmed_routing_and_geometry",
        "operator_confirmed_user_present",
        "operator_confirmed_volume_minimum",
    )
    try:
        with np.load(path, allow_pickle=False) as data:
            out = {key: _scalar(data, key, label=label) for key in scalar_keys}
            out["consistency_band_hz"] = _vector(
                data, "consistency_band_hz", label=label
            )
    except OSError as exc:
        raise RuntimePlantContractError(f"{label}를 읽을 수 없습니다: {path}: {exc}") from exc
    return out


def _as_sha256(value: Any, *, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimePlantContractError(f"{label} SHA-256 형식이 잘못됐습니다: {value!r}")
    return digest


def _verify_referenced_file(
    *, path_value: Any, declared_sha256: Any, label: str
) -> tuple[Path, str]:
    path = _resolve_path(str(path_value)).resolve()
    if not path.is_file():
        raise RuntimePlantContractError(
            f"{label} raw provenance 파일이 없습니다: {path}"
        )
    declared = _as_sha256(declared_sha256, label=label)
    actual = _sha256_file(path)
    if actual != declared:
        raise RuntimePlantContractError(
            f"{label} raw provenance SHA가 다릅니다: declared={declared}, actual={actual}"
        )
    return path, actual


def _require_true(value: Any, *, label: str) -> None:
    if not isinstance(value, (bool, np.bool_)) or not bool(value):
        raise RuntimePlantContractError(f"{label} operator confirmation이 PASS가 아닙니다")


def _validate_measurement_metadata(
    metadata: dict[str, Any],
    *,
    label: str,
    expected_output_name: str,
    expected_output_channel: int,
    sample_rate: int,
    block_size: int,
    latency: str,
    error_channel: int,
    reference_channel: int,
) -> None:
    if str(metadata["capture_id"]).strip() == "":
        raise RuntimePlantContractError(f"{label} capture_id가 비어 있습니다")
    if int(metadata["sample_rate"]) != int(sample_rate):
        raise RuntimePlantContractError(
            f"{label} sample_rate={metadata['sample_rate']} != runtime={sample_rate}"
        )
    if int(metadata["calibration_block_size"]) != int(block_size):
        raise RuntimePlantContractError(
            f"{label} calibration block={metadata['calibration_block_size']} != runtime={block_size}"
        )
    if str(metadata["calibration_latency"]) != str(latency):
        raise RuntimePlantContractError(
            f"{label} calibration latency={metadata['calibration_latency']!r} != runtime={latency!r}"
        )
    if str(metadata["output_channel"]) != expected_output_name:
        raise RuntimePlantContractError(
            f"{label} output_channel={metadata['output_channel']!r}; "
            f"expected={expected_output_name!r}"
        )
    actual_channel_key = (
        "noise_output_channel" if expected_output_name == "noise" else "cancel_output_channel"
    )
    if int(metadata[actual_channel_key]) != int(expected_output_channel):
        raise RuntimePlantContractError(
            f"{label} {actual_channel_key}={metadata[actual_channel_key]} != runtime={expected_output_channel}"
        )
    if int(metadata["error_mic_channel"]) != int(error_channel):
        raise RuntimePlantContractError(
            f"{label} error_mic_channel={metadata['error_mic_channel']} != runtime={error_channel}"
        )
    if int(metadata["reference_mic_channel"]) != int(reference_channel):
        raise RuntimePlantContractError(
            f"{label} reference_mic_channel={metadata['reference_mic_channel']} != runtime={reference_channel}"
        )
    if str(metadata["output_pcm_provenance"]) != "observed_submitted_int16":
        raise RuntimePlantContractError(
            f"{label} 출력 PCM provenance가 observed_submitted_int16이 아닙니다"
        )
    if str(metadata["delay_semantics"]) != "effective_zeros_before_compact_fir":
        raise RuntimePlantContractError(
            f"{label} delay semantics가 현재 compact-P/S 규약과 다릅니다"
        )
    if int(metadata["xrun_count"]) != 0:
        raise RuntimePlantContractError(f"{label} 측정 xrun_count가 0이 아닙니다")
    if int(metadata["repeats"]) < 8:
        raise RuntimePlantContractError(f"{label} kept repeats가 8보다 작습니다")
    band = np.asarray(metadata["consistency_band_hz"], dtype=np.float64)
    if band[0] > STRICT_RUNTIME_BAND_HZ[0] or band[1] < STRICT_RUNTIME_BAND_HZ[1]:
        raise RuntimePlantContractError(
            f"{label} consistency band={tuple(band)}가 required "
            f"{STRICT_RUNTIME_BAND_HZ}를 덮지 않습니다"
        )
    for key in (
        "operator_confirmed_routing_and_geometry",
        "operator_confirmed_user_present",
        "operator_confirmed_volume_minimum",
    ):
        _require_true(metadata[key], label=f"{label} {key}")


def _level_evidence(
    *,
    path: Path,
    primary: dict[str, Any],
    secondary: dict[str, Any],
    sample_rate: int,
) -> str:
    if not path.is_file():
        raise RuntimePlantContractError(f"measurement level evidence가 없습니다: {path}")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePlantContractError(
            f"measurement level evidence를 읽을 수 없습니다: {path}: {exc}"
        ) from exc
    if not isinstance(evidence, dict) or evidence.get("passed") is not True:
        raise RuntimePlantContractError("measurement level evidence가 PASS가 아닙니다")
    if int(evidence.get("sample_rate", -1)) != int(sample_rate):
        raise RuntimePlantContractError("measurement level evidence sample_rate가 runtime과 다릅니다")
    probe = float(evidence.get("probe_amplitude", float("nan")))
    if not math.isfinite(probe) or not math.isclose(probe, 0.003, abs_tol=1e-12):
        raise RuntimePlantContractError(
            f"measurement level evidence probe_amplitude={probe!r}; strict 규약 0.003이 아닙니다"
        )
    for label, metadata in (("P", primary), ("S", secondary)):
        amplitude = float(metadata["amplitude"])
        if not math.isfinite(amplitude) or not math.isclose(amplitude, probe, abs_tol=1e-12):
            raise RuntimePlantContractError(
                f"{label} probe amplitude={amplitude!r}가 level evidence={probe!r}와 다릅니다"
            )
    raw = evidence.get("interleaved_raw")
    if not isinstance(raw, dict):
        raise RuntimePlantContractError("measurement level evidence interleaved_raw가 없습니다")
    evidence_raw_sha = _as_sha256(raw.get("sha256"), label="level evidence raw")
    p_raw_sha = _as_sha256(primary["source_raw_npz_sha256"], label="P raw")
    s_raw_sha = _as_sha256(secondary["source_raw_npz_sha256"], label="S raw")
    if evidence_raw_sha != p_raw_sha or p_raw_sha != s_raw_sha:
        raise RuntimePlantContractError(
            "measurement level evidence와 P/S가 같은 interleaved raw capture를 가리키지 않습니다"
        )
    return _sha256_file(path)


def validate_runtime_plant_contract(cfg: dict[str, Any]) -> RuntimePlantContract | None:
    """실제 strict P/S에서 runtime lead를 유도하고 일치 여부를 확인한다.

    호출자는 이 함수를 ``sounddevice`` import, engine 생성, 입력 probe보다 앞서 호출해야
    한다. 실패 시 어떤 오디오 장치도 열려서는 안 된다.
    """

    if not runtime_requires_strict_plant_contract(cfg):
        return None

    hardware = cfg.get("hardware") or {}
    audio = hardware.get("audio") or {}
    channels = hardware.get("channels") or {}
    duct = cfg.get("duct") or {}
    secondary_cfg = duct.get("secondary_path") or {}
    digital_cfg = duct.get("digital_reference") or {}

    sample_rate = int(audio.get("sample_rate", 0))
    block_size = int(audio.get("block_size", 0))
    hop = int(cfg.get("hop", block_size))
    latency = str(audio.get("latency", ""))
    if sample_rate <= 0 or block_size <= 0 or hop != block_size:
        raise RuntimePlantContractError(
            "runtime strict plant 검증에는 양의 sample_rate/block_size와 hop==block_size가 필요합니다"
        )
    secondary_value = secondary_cfg.get("npz")
    primary_value = digital_cfg.get("primary_path_npz")
    if not secondary_value or not primary_value:
        raise RuntimePlantContractError("strict runtime에는 duct.yaml의 P/S NPZ가 모두 필요합니다")
    primary_path = _resolve_path(primary_value).resolve()
    secondary_path = _resolve_path(secondary_value).resolve()
    if not primary_path.is_file() or not secondary_path.is_file():
        raise RuntimePlantContractError("strict runtime P/S NPZ 파일이 없습니다")

    primary_meta = _metadata(primary_path, label="P(z)")
    secondary_meta = _metadata(secondary_path, label="S(z)")
    _validate_measurement_metadata(
        primary_meta,
        label="P(z)",
        expected_output_name="noise",
        expected_output_channel=int(channels.get("noise_out", -1)),
        sample_rate=sample_rate,
        block_size=block_size,
        latency=latency,
        error_channel=int(channels.get("error_mic", -1)),
        reference_channel=int(channels.get("reference_mic", -1)),
    )
    _validate_measurement_metadata(
        secondary_meta,
        label="S(z)",
        expected_output_name="cancel",
        expected_output_channel=int(channels.get("cancel_out", -1)),
        sample_rate=sample_rate,
        block_size=block_size,
        latency=latency,
        error_channel=int(channels.get("error_mic", -1)),
        reference_channel=int(channels.get("reference_mic", -1)),
    )
    if str(primary_meta["capture_id"]) != str(secondary_meta["capture_id"]):
        raise RuntimePlantContractError("P/S capture_id가 달라 같은 플랜트가 아닙니다")

    _, p_raw_sha = _verify_referenced_file(
        path_value=primary_meta["source_raw_npz_path"],
        declared_sha256=primary_meta["source_raw_npz_sha256"],
        label="P(z)",
    )
    _, s_raw_sha = _verify_referenced_file(
        path_value=secondary_meta["source_raw_npz_path"],
        declared_sha256=secondary_meta["source_raw_npz_sha256"],
        label="S(z)",
    )
    _, p_analysis_sha = _verify_referenced_file(
        path_value=primary_meta["source_analysis_npz_path"],
        declared_sha256=primary_meta["source_analysis_npz_sha256"],
        label="P(z) analysis",
    )
    _, s_analysis_sha = _verify_referenced_file(
        path_value=secondary_meta["source_analysis_npz_path"],
        declared_sha256=secondary_meta["source_analysis_npz_sha256"],
        label="S(z) analysis",
    )
    if p_raw_sha != s_raw_sha or p_analysis_sha != s_analysis_sha:
        raise RuntimePlantContractError("P/S raw 또는 analysis provenance가 서로 다릅니다")

    strict_cfg = duct.get("strict_measurement") or {}
    level_value = strict_cfg.get("measurement_level_evidence")
    if not level_value:
        raise RuntimePlantContractError("duct.strict_measurement.measurement_level_evidence가 없습니다")
    level_sha = _level_evidence(
        path=_resolve_path(level_value).resolve(),
        primary=primary_meta,
        secondary=secondary_meta,
        sample_rate=sample_rate,
    )

    primary: SecondaryPathData = load_secondary_path(primary_path)
    secondary: SecondaryPathData = load_secondary_path(secondary_path)
    if int(primary.sample_rate) != sample_rate or int(secondary.sample_rate) != sample_rate:
        raise RuntimePlantContractError("P/S NPZ sample rate가 runtime과 다릅니다")
    delays = PlantDelays.from_config(
        duct_cfg=duct,
        primary_delay_samples=int(primary.delay_samples),
        secondary_delay_samples=int(secondary.delay_samples),
        sample_rate=sample_rate,
    )
    timing = TrainingTimingContract.derive(primary_fir=primary.fir, plant_delays=delays)
    configured_lead = int(cfg.get("digital_reference_lead_samples", -1))
    expected_lead = int(timing.digital_reference_lead_samples)
    if configured_lead != expected_lead:
        raise RuntimePlantContractError(
            "runtime digital-reference lead가 strict P/S와 다릅니다: "
            f"runtime={configured_lead}, derived={expected_lead} "
            f"(P={primary.delay_samples}, S={secondary.delay_samples}, "
            f"handoff={delays.handoff_samples}). legacy artifact의 숫자를 실제 plant에 "
            "억지로 맞추지 말고 canonical 115-sample checkpoint/ONNX를 사용하세요."
        )
    return RuntimePlantContract(
        timing=timing,
        capture_id=str(primary_meta["capture_id"]),
        primary_path_sha256=_sha256_file(primary_path),
        secondary_path_sha256=_sha256_file(secondary_path),
        raw_measurement_sha256=p_raw_sha,
        analysis_sha256=p_analysis_sha,
        measurement_level_evidence_sha256=level_sha,
    )
