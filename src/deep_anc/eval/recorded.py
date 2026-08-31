"""독립 recorded val/test의 결정론적 오프라인 평가.

실측 세션은 ANC OFF로 녹음되므로 error mic 신호가 ``d``이다. 모델 출력 ``y``에
체크포인트가 보존한 S(z)와 런타임 handoff 지연을 적용해 ``e = d + S*y``를
계산한다. 이 모듈은 오디오 장치를 열거나 소리를 출력하지 않는다.
"""

from __future__ import annotations

import json
import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import soundfile as sf
import torch

from ..config import REPO_ROOT
from ..data.manifest import read_manifest, read_manifest_bytes
from ..data.primary_path import resolve_digital_primary_path
from ..data.synth_dataset import _delay_np
from ..dsp.invariants import check_lead_agreement, check_plant_fingerprint_match
from ..dsp.secondary_path import (
    DifferentiableSecondaryPath,
    SecondaryPathData,
    load_secondary_path,
)
# 대역 밖 예산의 단일 출처 — 손실 힌지가 이 임계에서 유도된다 (발생기 A).
from ..dsp.do_no_harm import (
    MAX_OUT_OF_BAND_AMPLIFICATION_DB,
    OCTAVE_BAND_CENTERS_HZ,
)

# 지연·lead·대역 부기의 단일 출처 (발생기 A).
from ..dsp.timing import (
    BandPlan,
    FrequencyBand,
    PlantDelays,
    PlantFingerprint,
    TrainingTimingContract,
    handoff_samples_from_config,
)
from ..models import build_model
from ..model_input import (
    RefOnlyModelInputContract,
    apply_stage1_ref_only_numpy,
    resolve_stage1_model_input_contract,
    validate_stage1_ref_only_tensor,
)
from ..train.trainer import validate_training_physics
from ..train.evaluation_contract import snapshot_regular_file
from ..train.experiment_contract import validate_embedded_experiment_contract
from .metrics import (
    band_power,
    band_nmse_db,
    intersect_frequency_bands,
    nmse_db,
    octave_band_attenuation,
)
from .recorded_sampling import (
    CANONICAL_EDGE_TRIM_SECONDS,
    CANONICAL_MAX_SEGMENTS_PER_SESSION,
    CANONICAL_SEGMENT_SECONDS,
    RECORDED_SAMPLING_CONTRACT_SCHEMA,
    canonical_feedback_delay_samples,
    canonical_warmup_samples,
    deterministic_segment_starts,
    effective_segment_samples,
)
from .trusted_subbands import (
    MIN_GROUPS_PER_FAMILY,
    MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO,
    STRICT_TRUSTED_BAND_HZ,
    STRICT_TRUSTED_SUBBAND_SCHEMA,
    STRICT_TRUSTED_SUBBANDS_HZ,
    cluster_bootstrap_ci as _strict_cluster_bootstrap_ci,
    source_energy_covered,
    strict_subband_includes_upper_edge,
    strict_trusted_subbands_for,
)


@dataclass(frozen=True)
class RecordedEvalContext:
    """체크포인트에서 복원된 모델·플랜트·물리 메타데이터."""

    model: torch.nn.Module
    plant: DifferentiableSecondaryPath
    cfg: dict
    device: torch.device
    sample_rate: int
    trusted_band_hz: tuple[float, float]
    physics_status: str
    reference_mode: str
    digital_reference_lead_samples: int
    expected_digital_reference_lead_samples: int
    primary_delay_samples: int | None
    secondary_path: SecondaryPathData
    secondary_handoff_samples: int
    checkpoint_sha256: str = ""
    model_input_contract: RefOnlyModelInputContract | None = None
    model_input_contract_sha256: str = ""


@dataclass(frozen=True)
class RecordedSegment:
    """한 세션에서 잘라낸 고정 평가 구간."""

    x: np.ndarray  # [2, T]
    d: np.ndarray  # [T]
    session_id: str
    group_id: str
    source_family: str
    start_sample: int
    recorded_lead_samples: int = 0
    recorded_delay_samples: float = -1.0
    timing_contract_sha256: str = ""
    source_timeline: str = "legacy"
    model_input_contract_sha256: str = ""


def _config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_if_file(value: str | Path) -> str:
    """평가 provenance용 파일 지문. 없는 테스트 placeholder는 빈 값으로 남긴다."""

    path = _config_path(value)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve_checkpoint_model_input_contract(
    cfg: dict,
) -> RefOnlyModelInputContract | None:
    """체크포인트의 resolved 입력 payload와 최상위 digest를 함께 검증한다.

    구형 diagnostic checkpoint는 입력 계약 자체가 없으므로 과거 2채널 동작을
    보존한다. 반면 canonical fine-tune은 REF-only payload와 그 digest 중 하나라도
    빠지면 공식 recorded 평가를 열지 않는다.
    """

    data_cfg = cfg.get("data")
    contract = resolve_stage1_model_input_contract(
        data_cfg if isinstance(data_cfg, dict) else None
    )
    declared_sha = cfg.get("model_input_contract_sha256")
    canonical = str(cfg.get("experiment_role", "")) == "canonical_finetune"
    if contract is None:
        if declared_sha is not None:
            raise ValueError(
                "checkpoint model_input_contract_sha256가 있지만 "
                "data.model_input_contract가 없습니다"
            )
        if canonical:
            raise ValueError(
                "canonical fine-tune checkpoint에 data.model_input_contract가 없습니다"
            )
        # legacy/diagnostic checkpoint의 기존 ERR-context 동작을 그대로 보존한다.
        return None
    digest = contract.digest()
    if declared_sha != digest:
        raise ValueError(
            "checkpoint model_input_contract_sha256가 resolved "
            "data.model_input_contract digest와 다릅니다"
        )
    return contract


def validate_resolved_checkpoint(
    state: dict, *, allow_surrogate: bool = False
) -> tuple[dict, int, str, str]:
    """resolved checkpoint와 물리 상태/lead alias를 검증한다.

    반환은 ``(cfg, lead, physics_status, reference_mode)``이다. 기본 모드에서는
    measured P(z)로 학습한 체크포인트만 허용한다. ``allow_surrogate``는 실제
    성능 판정이 아닌 진단을 명시적으로 요청할 때만 사용한다.
    """

    if not isinstance(state, dict) or not isinstance(state.get("cfg"), dict):
        raise ValueError("checkpoint에 resolved cfg가 없습니다")
    cfg = state["cfg"]
    if str(cfg.get("experiment_role", "")) == "canonical_finetune":
        validate_embedded_experiment_contract(cfg)
    if not isinstance(state.get("model"), dict):
        raise ValueError("checkpoint에 model state_dict가 없습니다")
    for key in ("model", "data", "duct"):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f"checkpoint resolved cfg에 {key!r} 설정이 없습니다")
    if "physics_status" not in cfg:
        raise ValueError("checkpoint resolved cfg에 physics_status가 없습니다")
    if "trusted_band_hz" not in cfg:
        raise ValueError("checkpoint resolved cfg에 trusted_band_hz가 없습니다")
    if "digital_reference_lead_samples" not in cfg:
        raise ValueError(
            "checkpoint resolved cfg에 digital_reference_lead_samples alias가 없습니다"
        )

    data_cfg = cfg["data"]
    _resolve_checkpoint_model_input_contract(cfg)
    reference_mode = str(data_cfg.get("reference_mode", "digital"))
    if reference_mode not in {"digital", "acoustic"}:
        raise ValueError(
            f"지원하지 않는 checkpoint reference_mode: {reference_mode!r}"
        )
    nested_lead = int(data_cfg.get("digital_reference_lead_samples", 0))
    alias_lead = int(cfg["digital_reference_lead_samples"])
    if nested_lead < 0 or alias_lead < 0:
        raise ValueError("digital-reference lead는 0 이상이어야 합니다")
    if nested_lead != alias_lead:
        raise ValueError(
            "checkpoint digital-reference lead alias 불일치: "
            f"cfg={alias_lead}, data={nested_lead}"
        )
    if reference_mode != "digital" and nested_lead != 0:
        raise ValueError(
            "digital_reference_lead_samples는 reference_mode=digital에서만 "
            "사용할 수 있습니다"
        )

    saved_status = str(cfg["physics_status"])
    computed_status = validate_training_physics(cfg)
    if saved_status != computed_status:
        raise ValueError(
            "checkpoint physics_status와 resolved 설정이 다릅니다: "
            f"saved={saved_status}, resolved={computed_status}"
        )
    if not allow_surrogate and saved_status != "measured_primary_path":
        raise ValueError(
            "recorded 성능 평가는 measured_primary_path checkpoint만 허용합니다: "
            f"{saved_status}. 진단 목적이면 --allow-surrogate를 명시하세요."
        )
    if saved_status == "measured_primary_path":
        primary_path = cfg["duct"].get("digital_reference", {}).get(
            "primary_path_npz"
        )
        if not primary_path:
            raise ValueError(
                "measured_primary_path checkpoint에 primary_path_npz가 없습니다"
            )
    return cfg, nested_lead, saved_status, reference_mode


def load_recorded_eval_context(
    checkpoint: str | Path,
    *,
    allow_surrogate: bool = False,
    device: str | torch.device | None = None,
    checkpoint_bytes: bytes | None = None,
    checkpoint_sha256: str | None = None,
) -> RecordedEvalContext:
    """체크포인트의 resolved cfg만 사용해 모델과 공칭 S(z)를 복원한다."""

    checkpoint = Path(checkpoint)
    if checkpoint_bytes is None:
        snapshot = snapshot_regular_file(checkpoint)
        checkpoint_bytes = snapshot.content
        checkpoint_sha256 = snapshot.sha256
    else:
        actual_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
        if checkpoint_sha256 is not None and checkpoint_sha256 != actual_sha:
            raise ValueError("checkpoint byte snapshot SHA가 다릅니다")
        checkpoint_sha256 = actual_sha
    state = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
    )
    cfg, lead, physics_status, reference_mode = validate_resolved_checkpoint(
        state, allow_surrogate=allow_surrogate
    )
    data_cfg = cfg["data"]
    model_input_contract = _resolve_checkpoint_model_input_contract(cfg)
    model_input_contract_sha = (
        model_input_contract.digest() if model_input_contract is not None else ""
    )
    duct_cfg = cfg["duct"]
    fs = int(data_cfg["sample_rate"])
    if fs <= 0:
        raise ValueError(f"잘못된 checkpoint sample_rate: {fs}")

    secondary_cfg = duct_cfg.get("secondary_path", {})
    secondary_path_value = secondary_cfg.get("npz")
    if not secondary_path_value:
        raise ValueError("checkpoint duct.secondary_path.npz가 없습니다")
    secondary = load_secondary_path(_config_path(secondary_path_value))
    if int(secondary.sample_rate) != fs:
        raise ValueError(
            f"S(z) sample_rate={secondary.sample_rate}Hz != checkpoint={fs}Hz"
        )
    handoff = handoff_samples_from_config(duct_cfg)

    primary_delay: int | None = None
    expected_lead = 0
    if reference_mode == "digital":
        primary_data, primary_delay = resolve_digital_primary_path(
            data_cfg, duct_cfg, fs, secondary
        )
        # lead 는 여기서 다시 유도하지 않는다 — PlantDelays 가 유일한 발원지이고
        # 게이트(finetune_readiness)도 **같은 함수**를 호출한다. 두 곳이 각자 유도해
        # 109 와 113 으로 갈라졌던 것이 커밋 aaeef41 의 사고다.
        delays = PlantDelays.from_config(
            duct_cfg=duct_cfg,
            secondary_delay_samples=int(secondary.delay_samples),
            primary_delay_samples=int(primary_delay),
            sample_rate=fs,
        )
        if physics_status == "measured_primary_path":
            if primary_data is None:  # pragma: no cover - measured resolver 방어
                raise ValueError("measured checkpoint에 compact P(z)가 없습니다")
            saved_timing = TrainingTimingContract.from_data_config(data_cfg)
            actual_timing = TrainingTimingContract.derive(
                primary_fir=primary_data.fir,
                plant_delays=delays,
            )
            if saved_timing != actual_timing:
                raise ValueError(
                    "checkpoint training_timing_contract가 resolved P/S와 다릅니다"
                )
            expected_lead = int(actual_timing.digital_reference_lead_samples)
            if int(lead) != int(saved_timing.digital_reference_lead_samples):
                raise ValueError(
                    "checkpoint digital-reference lead가 TrainingTimingContract와 "
                    f"다릅니다: checkpoint={lead}, expected={expected_lead}"
                )
        else:
            # 명시 surrogate 진단만 legacy alias를 PlantDelays와 대조한다.
            expected_lead = int(delays.lead().samples)
            result = check_lead_agreement(int(lead), delays)
            if not result.ok:
                raise ValueError(
                    "checkpoint digital-reference lead가 P/S 지연과 다릅니다: "
                    f"checkpoint={lead}, expected={expected_lead} "
                    f"(S={secondary.delay_samples}+handoff {handoff}, P={primary_delay})"
                )

    band_plan = BandPlan.resolve(
        plant_trusted_band_hz=secondary.trusted_band_hz(),
        duct_cfg=duct_cfg,
        sample_rate=fs,
    )
    computed_trusted = band_plan.optimize.as_tuple()
    saved_trusted_raw = cfg["trusted_band_hz"]
    saved_trusted = intersect_frequency_bands(
        saved_trusted_raw, saved_trusted_raw, fs / 2.0
    )
    if not np.allclose(saved_trusted, computed_trusted, rtol=0.0, atol=1e-6):
        raise ValueError(
            "checkpoint trusted_band_hz가 현재 resolved P/S 설정과 다릅니다: "
            f"saved={saved_trusted}, expected={computed_trusted}"
        )

    if device is None or str(device) == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device를 요청했지만 CUDA를 사용할 수 없습니다")

    model = build_model(cfg["model"])
    model.load_state_dict(state["model"])
    model.eval().to(resolved_device)
    plant = DifferentiableSecondaryPath(
        secondary, handoff_extra_samples=handoff
    ).eval().to(resolved_device)
    return RecordedEvalContext(
        model=model,
        plant=plant,
        cfg=cfg,
        device=resolved_device,
        sample_rate=fs,
        trusted_band_hz=computed_trusted,
        physics_status=physics_status,
        reference_mode=reference_mode,
        digital_reference_lead_samples=lead,
        expected_digital_reference_lead_samples=expected_lead,
        primary_delay_samples=primary_delay,
        secondary_path=secondary,
        secondary_handoff_samples=handoff,
        checkpoint_sha256=str(checkpoint_sha256),
        model_input_contract=model_input_contract,
        model_input_contract_sha256=model_input_contract_sha,
    )


def load_and_audit_recorded_manifest(
    manifest_path: str | Path, split: str, *, manifest_bytes: bytes | None = None
) -> list[dict]:
    """전체 manifest의 path/session/group split 누수를 검사하고 split을 반환."""

    if split not in {"val", "test"}:
        raise ValueError("독립 recorded 평가는 split=val 또는 test만 허용합니다")
    entries = (
        read_manifest(manifest_path)
        if manifest_bytes is None
        else read_manifest_bytes(manifest_bytes, manifest_path=manifest_path)
    )
    if not entries:
        raise ValueError(f"recorded manifest가 비어 있습니다: {manifest_path}")

    seen_paths: dict[str, str] = {}
    seen_sessions: dict[str, tuple[str, str]] = {}
    group_splits: dict[str, str] = {}
    group_families: dict[str, str] = {}
    for index, entry in enumerate(entries):
        for key in ("path", "split", "session_id", "group_id", "source_family"):
            if key not in entry or not str(entry[key]).strip():
                raise ValueError(f"manifest entry #{index}에 {key!r}가 없습니다")
        entry_split = str(entry["split"])
        if entry_split not in {"train", "val", "test"}:
            raise ValueError(
                f"manifest entry #{index}의 split이 잘못되었습니다: {entry_split}"
            )
        resolved_path = str(Path(entry["path"]).expanduser().resolve())
        previous_path_split = seen_paths.setdefault(resolved_path, entry_split)
        if previous_path_split != entry_split:
            raise ValueError(
                f"같은 session path가 여러 split에 있습니다: {resolved_path} "
                f"({previous_path_split}, {entry_split})"
            )
        if previous_path_split == entry_split and any(
            str(Path(previous["path"]).expanduser().resolve()) == resolved_path
            for previous in entries[:index]
        ):
            raise ValueError(f"manifest에 중복 session path가 있습니다: {resolved_path}")

        session_id = str(entry["session_id"])
        previous_session = seen_sessions.setdefault(
            session_id, (entry_split, resolved_path)
        )
        if previous_session != (entry_split, resolved_path):
            raise ValueError(
                f"session_id={session_id!r}가 중복되거나 split을 넘나듭니다"
            )

        group_id = str(entry["group_id"])
        source_family = str(entry["source_family"])
        previous_group_split = group_splits.setdefault(group_id, entry_split)
        if previous_group_split != entry_split:
            raise ValueError(
                f"group_id={group_id!r}가 여러 split에 있습니다: "
                f"{previous_group_split}, {entry_split}"
            )
        previous_family = group_families.setdefault(group_id, source_family)
        if previous_family != source_family:
            raise ValueError(
                f"group_id={group_id!r}가 여러 source_family에 속합니다: "
                f"{previous_family}, {source_family}"
            )

    selected = [dict(entry) for entry in entries if entry["split"] == split]
    if not selected:
        raise ValueError(f"recorded manifest에 '{split}' split 세션이 없습니다")
    return sorted(
        selected,
        key=lambda entry: (
            str(entry["group_id"]),
            str(entry["session_id"]),
            str(entry["path"]),
        ),
    )


def resolve_feedback_delay(data_cfg: dict, requested: int | None = None) -> int:
    """학습 범위 안의 결정론적 feedback 지연을 선택한다(기본=중앙값)."""

    raw = data_cfg.get("closed_loop", {}).get(
        "feedback_delay_samples", [0, 0]
    )
    if isinstance(raw, (int, float)):
        lo = hi = int(raw)
    else:
        if not isinstance(raw, Sequence) or len(raw) != 2:
            raise ValueError(
                "closed_loop.feedback_delay_samples는 [lo, hi]여야 합니다"
            )
        lo, hi = int(raw[0]), int(raw[1])
    if lo < 0 or hi < lo:
        raise ValueError(f"잘못된 feedback delay 범위: {raw}")
    value = int(round((lo + hi) / 2.0)) if requested is None else int(requested)
    if not lo <= value <= hi:
        raise ValueError(
            f"feedback delay {value}가 checkpoint 학습 범위 [{lo}, {hi}] 밖입니다"
        )
    return value


def _plant_settle_samples(context: "RecordedEvalContext") -> int:
    """평가 컨텍스트의 플랜트 정착 구간 (단일 출처 위임)."""

    from ..dsp.timing import PlantSettle

    return PlantSettle.derive(
        secondary_delay_samples=int(context.secondary_path.delay_samples),
        handoff_samples=int(context.secondary_handoff_samples),
        fir_taps=int(context.secondary_path.fir.size),
        sample_rate=int(context.sample_rate),
    ).samples


def resolve_warmup_samples(
    data_cfg: dict,
    sample_rate: int,
    requested_seconds: float | None = None,
    min_samples: int = 0,
) -> int:
    """평가 지표에서 제외할 플랜트 적용 후 warmup 길이를 반환.

    ``min_samples`` 는 S(z) 총지연 + FIR 정착이다(단일 출처: ``dsp.timing.PlantSettle``).
    이 구간은 y 가 무엇이든 ``e = d`` 라 상쇄량을 잴 수 없다. 학습이 버리는 구간
    (``trainer.loss_start_sample``)과 **같은 양을 가리켜야** 두 숫자를 비교할 수 있다 —
    예전에는 학습이 0, 평가가 ``closed_loop.warmup_seconds``(12000) 을 버렸고 아무도
    두 숫자가 다른 것을 몰랐다 (발생기 A).

    하한을 두는 방향은 게이트를 **강화**한다: warmup_seconds 가 나중에 0 으로 내려가도
    구조적으로 상쇄 불가능한 구간(실측 recorded 하한 worst −4.8 dB)이 지표에 섞이지
    않는다. 현재 값(12000 > 3769)에서는 동작이 변하지 않는다.
    """

    seconds = (
        float(data_cfg.get("closed_loop", {}).get("warmup_seconds", 0.25))
        if requested_seconds is None
        else float(requested_seconds)
    )
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("warmup_seconds는 유한한 0 이상 값이어야 합니다")
    floor = int(min_samples)
    if floor < 0:
        raise ValueError("min_samples는 0 이상이어야 합니다")
    return max(floor, int(round(seconds * int(sample_rate))))


def _read_session_metadata(session_dir: Path) -> dict:
    path = session_dir / "session.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: session metadata를 읽을 수 없습니다: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: 최상위는 JSON 객체여야 합니다")
    return value


def _load_session_audio(
    entry: dict,
    sample_rate: int,
    reference_mode: str,
    *,
    allow_legacy_source_timeline: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict, str]:
    session_dir = Path(entry["path"])
    if not session_dir.is_dir():
        raise FileNotFoundError(f"recorded session 디렉터리가 없습니다: {session_dir}")

    metadata = _read_session_metadata(session_dir)
    # canonical regrouped manifest의 group_id는 leakage-safe lineage component이고,
    # immutable session.json의 group_id는 녹음 당시 source-pool group이다. 이 둘을
    # 그대로 같다고 요구하면 정상 canonical 82세션을 모두 거부한다. 반대로 session
    # metadata를 무시하면 manifest가 다른 원본 녹음으로 바뀐 것을 놓친다. 따라서
    # session.json은 source_pool_group_id(있을 때), bootstrap/CI의 독립 group은 manifest
    # group_id(component)라는 서로 다른 의미를 각각 검증한다.
    if "group_id" in metadata:
        source_pool_group = entry.get("source_pool_group_id")
        expected_group = (
            str(source_pool_group).strip()
            if source_pool_group is not None
            else str(entry["group_id"])
        )
        if not expected_group or str(metadata["group_id"]) != expected_group:
            raise ValueError(
                f"{session_dir}: manifest source-pool group={expected_group!r}와 "
                f"session.json group_id={metadata['group_id']!r}가 다릅니다"
            )
    if "source_family" in metadata and str(metadata["source_family"]) != str(
        entry["source_family"]
    ):
        raise ValueError(
            f"{session_dir}: manifest source_family={entry['source_family']!r}와 "
            f"session.json={metadata['source_family']!r}가 다릅니다"
        )
    if "sample_rate" in metadata and int(metadata["sample_rate"]) != sample_rate:
        raise ValueError(
            f"{session_dir}: session.json sample_rate={metadata['sample_rate']} "
            f"!= checkpoint={sample_rate}"
        )

    mics_path = session_dir / "mics.wav"
    if not mics_path.exists():
        raise FileNotFoundError(f"mics.wav가 없습니다: {mics_path}")
    mics, mic_rate = sf.read(mics_path, dtype="float32", always_2d=True)
    if int(mic_rate) != sample_rate:
        raise ValueError(
            f"{mics_path}: sample_rate={mic_rate} != checkpoint={sample_rate}"
        )
    if mics.shape[1] < 2:
        raise ValueError(f"{mics_path}: err/ref 2채널이 필요합니다")
    if not np.all(np.isfinite(mics[:, :2])):
        raise ValueError(f"{mics_path}: NaN/Inf가 있습니다")

    source: np.ndarray | None = None
    source_timeline = "acoustic_ref"
    if reference_mode == "digital":
        aligned_path = session_dir / "source_aligned.wav"
        source_path = (
            aligned_path
            if aligned_path.is_file()
            else session_dir / "source.wav"
        )
        if not aligned_path.is_file() and not allow_legacy_source_timeline:
            raise FileNotFoundError(
                "공식 digital measured 평가는 ADC 시간축 source_aligned.wav만 "
                f"허용합니다: {aligned_path}"
            )
        if not source_path.exists():
            raise FileNotFoundError(
                f"digital-reference 평가에 source가 필요합니다: {source_path}"
            )
        source_audio, source_rate = sf.read(
            source_path, dtype="float32", always_2d=True
        )
        if int(source_rate) != sample_rate:
            raise ValueError(
                f"{source_path}: sample_rate={source_rate} != checkpoint={sample_rate}"
            )
        source = source_audio[:, 0]
        if not np.all(np.isfinite(source)):
            raise ValueError(f"{source_path}: NaN/Inf가 있습니다")
        source_timeline = source_path.name

    return mics[:, 0], mics[:, 1], source, metadata, source_timeline


def iter_recorded_segments(
    entries: Sequence[dict],
    data_cfg: dict,
    *,
    model_hop: int,
    max_segments_per_session: int = 8,
    segment_seconds: float | None = None,
    feedback_delay_samples: int | None = None,
    edge_trim_seconds: float = 0.25,
    allow_legacy_source_timeline: bool = False,
) -> Iterator[RecordedSegment]:
    """manifest 세션을 유한하고 결정론적인 비중첩 segment로 변환한다."""

    fs = int(data_cfg["sample_rate"])
    hop = int(model_hop)
    if hop <= 0:
        raise ValueError("model hop은 양수여야 합니다")
    seconds = (
        float(data_cfg["segment_seconds"])
        if segment_seconds is None
        else float(segment_seconds)
    )
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("segment_seconds는 유한한 양수여야 합니다")
    raw_segment = int(round(seconds * fs))
    segment = (raw_segment // hop) * hop
    if segment < hop:
        raise ValueError(
            f"segment_seconds={seconds}가 model hop {hop}샘플보다 짧습니다"
        )
    edge_trim_seconds = float(edge_trim_seconds)
    if not math.isfinite(edge_trim_seconds) or edge_trim_seconds < 0.0:
        raise ValueError("edge_trim_seconds는 유한한 0 이상 값이어야 합니다")
    edge_trim = int(round(edge_trim_seconds * fs))

    reference_mode = str(data_cfg.get("reference_mode", "digital"))
    model_input_contract = resolve_stage1_model_input_contract(data_cfg)
    model_input_contract_sha = (
        model_input_contract.digest() if model_input_contract is not None else ""
    )
    constant_lead = int(data_cfg.get("digital_reference_lead_samples", 0))
    timing_contract: TrainingTimingContract | None = None
    timing_contract_sha = ""
    if reference_mode == "digital" and not allow_legacy_source_timeline:
        if str(data_cfg.get("recorded_lead_mode", "")) != "timeline":
            raise ValueError(
                "공식 digital measured 평가는 recorded_lead_mode=timeline이어야 합니다"
            )
        timing_contract = TrainingTimingContract.from_data_config(data_cfg)
        if int(timing_contract.sample_rate) != fs:
            raise ValueError("training_timing_contract sample_rate가 평가 설정과 다릅니다")
        if int(timing_contract.digital_reference_lead_samples) != constant_lead:
            raise ValueError(
                "training_timing_contract digital lead가 checkpoint data와 다릅니다"
            )
        timing_contract_sha = timing_contract.digest()
    feedback_delay = resolve_feedback_delay(data_cfg, feedback_delay_samples)

    for entry in entries:
        err, ref, source, metadata, source_timeline = _load_session_audio(
            entry,
            fs,
            reference_mode,
            allow_legacy_source_timeline=allow_legacy_source_timeline,
        )
        lead = constant_lead
        recorded_delay = -1.0
        if reference_mode == "digital":
            assert source is not None
            if timing_contract is not None:
                timeline = metadata.get("timeline")
                if not isinstance(timeline, dict) or (
                    "aligned_lag_median_samples" not in timeline
                ):
                    raise ValueError(
                        f"{entry['path']}: session.json timeline."
                        "aligned_lag_median_samples가 없습니다"
                    )
                recorded_delay = float(timeline["aligned_lag_median_samples"])
                lead = timing_contract.recorded_lead_samples(recorded_delay)
            usable = min(err.size, source.size - lead)
        else:
            usable = min(err.size, ref.size)
        starts = deterministic_segment_starts(
            usable,
            segment,
            max_segments_per_session,
            edge_trim_samples=edge_trim,
        )
        if not starts:
            raise ValueError(
                f"{entry['path']}: lead={lead}, edge trim={edge_trim} 적용 후 "
                f"{segment}샘플 segment가 없습니다"
            )

        delayed_err = _delay_np(err, feedback_delay)
        for start in starts:
            stop = start + segment
            d = np.ascontiguousarray(err[start:stop], dtype=np.float32)
            if reference_mode == "digital":
                assert source is not None
                x_ref = source[start + lead : stop + lead]
            else:
                x_ref = ref[start:stop]
            err_input = delayed_err[start:stop]
            if x_ref.size != segment or err_input.size != segment or d.size != segment:
                raise RuntimeError(f"{entry['path']}: segment 길이 계산 오류")
            x_ref, err_input = apply_stage1_ref_only_numpy(
                x_ref, err_input, model_input_contract
            )
            x = np.ascontiguousarray(
                np.stack([x_ref, err_input]), dtype=np.float32
            )
            yield RecordedSegment(
                x=x,
                d=d,
                session_id=str(entry["session_id"]),
                group_id=str(entry["group_id"]),
                source_family=str(entry["source_family"]),
                start_sample=int(start),
                recorded_lead_samples=int(lead),
                recorded_delay_samples=float(recorded_delay),
                timing_contract_sha256=timing_contract_sha,
                source_timeline=source_timeline,
                model_input_contract_sha256=model_input_contract_sha,
            )


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("평가 지표가 비었거나 NaN/Inf를 포함합니다")
    worst_count = max(1, int(math.ceil(array.size * 0.1)))
    worst = np.sort(array)[-worst_count:]
    return {
        "mean_db": float(np.mean(array)),
        "median_db": float(np.median(array)),
        "worst10_mean_db": float(np.mean(worst)),
        "worst10_threshold_db": float(np.percentile(array, 90.0)),
        "worst_db": float(np.max(array)),
    }


def evaluate_recorded_segments(
    model: torch.nn.Module,
    plant: DifferentiableSecondaryPath,
    segments: Iterable[RecordedSegment],
    *,
    sample_rate: int,
    trusted_band_hz: tuple[float, float],
    octave_bands_hz: Sequence[float],
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    warmup_samples: int = 0,
    model_input_contract: RefOnlyModelInputContract | None = None,
) -> dict:
    """segment iterable을 배치 평가하고 작은 메트릭 배열만 메모리에 보존한다."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size는 양수여야 합니다")
    warmup_samples = int(warmup_samples)
    if warmup_samples < 0:
        raise ValueError("warmup_samples는 0 이상이어야 합니다")
    device = torch.device(device)
    model.eval()
    plant.eval()

    # ``octave_rows``의 집계값만 남기면 나중에 scalar/summary를 바꿔도 실제
    # segment가 무엇을 보였는지 재감사할 방법이 없다. 입력 순서와 무관하게 항상
    # 오름차순 중심주파수 열을 쓰는 raw matrix를 함께 보존한다. 비정규 octave
    # 집합도 이 함수에서는 진단용으로 계산할 수 있지만, persisted canonical G4
    # validator는 125/250/500/1000/1600/2000/4000/8000Hz의 정확한 8-band schema만
    # 허용한다.
    requested_octaves = tuple(sorted(float(center) for center in octave_bands_hz))
    if not requested_octaves or any(
        not math.isfinite(center) or center <= 0.0 for center in requested_octaves
    ):
        raise ValueError("octave_bands_hz는 비어 있지 않은 유한 양수여야 합니다")
    if len(set(requested_octaves)) != len(requested_octaves):
        raise ValueError("octave_bands_hz 중심주파수는 중복될 수 없습니다")

    metadata: list[RecordedSegment] = []
    fullband_values: list[float] = []
    trusted_values: list[float] = []
    octave_values: dict[float, list[float]] = {
        center: [] for center in requested_octaves
    }
    octave_trusted: dict[float, bool] = {}
    per_segment_octave_values: list[list[float]] = []
    # 공식 150–1600 Hz가 아니면 이 평가는 진단용이다. 빈 목록을 "부대역 전부
    # 통과"로 읽지 않도록 schema/flag는 write_recorded_metrics에서 fail-closed한다.
    strict_subbands = strict_trusted_subbands_for(trusted_band_hz)
    per_segment_subband_values: list[list[float]] = []
    per_segment_subband_coverage: list[list[bool]] = []
    per_segment_subband_density: list[list[float]] = []
    model_input_contract_sha = (
        model_input_contract.digest() if model_input_contract is not None else ""
    )

    def evaluate_batch(batch: list[RecordedSegment]) -> None:
        x_np = np.stack([segment.x for segment in batch])
        d_np = np.stack([segment.d for segment in batch])
        x = torch.from_numpy(x_np).to(device)
        validate_stage1_ref_only_tensor(
            x, model_input_contract, label="recorded evaluation input"
        )
        d = torch.from_numpy(d_np[:, None, :]).to(device)
        with torch.no_grad():
            y = model(x)
            # 측정 FIR에 극성이 포함되어 있으므로 추가 부호 반전 금지.
            e = d + plant(y.float(), {"jitter": 0})
        # 반드시 전체 S(z)를 적용한 뒤 warmup을 자른다. y/d를 먼저 자르면
        # delay+FIR 상태가 사라져 segment 시작의 상쇄량을 잘못 계산한다.
        if warmup_samples >= d.shape[-1]:
            raise ValueError(
                f"warmup_samples={warmup_samples}가 segment 길이 "
                f"{d.shape[-1]} 이상입니다"
            )
        d_metric_np = d[:, 0, warmup_samples:].float().cpu().numpy()
        e_metric_np = e[:, 0, warmup_samples:].float().cpu().numpy()

        for index, segment in enumerate(batch):
            d_item = d_metric_np[index]
            e_item = e_metric_np[index]
            fullband_values.append(nmse_db(d_item, e_item))
            trusted_values.append(
                band_nmse_db(
                    d_item, e_item, sample_rate, trusted_band_hz
                )
            )
            band_rows = octave_band_attenuation(
                d_item,
                e_item,
                sample_rate,
                list(requested_octaves),
                trusted_band_hz,
            )
            by_center = {float(row["center_hz"]): row for row in band_rows}
            if set(by_center) != set(requested_octaves):
                raise RuntimeError("octave attenuation 결과 중심주파수가 요청과 다릅니다")
            for row in band_rows:
                center = float(row["center_hz"])
                octave_values.setdefault(center, []).append(
                    float(row["attenuation_db"])
                )
                octave_trusted[center] = bool(row["trusted"])
            per_segment_octave_values.append(
                [float(by_center[center]["attenuation_db"]) for center in requested_octaves]
            )
            if strict_subbands is not None:
                trusted_power = band_power(d_item, sample_rate, trusted_band_hz)
                subband_values: list[float] = []
                subband_coverage: list[bool] = []
                subband_density: list[float] = []
                for subband in strict_subbands:
                    include_upper = strict_subband_includes_upper_edge(subband)
                    subband_power = band_power(
                        d_item,
                        sample_rate,
                        subband,
                        include_upper=include_upper,
                    )
                    covered, density = source_energy_covered(
                        subband_power,
                        trusted_power,
                        subband,
                        min_density_ratio=MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO,
                    )
                    subband_values.append(
                        band_nmse_db(
                            d_item,
                            e_item,
                            sample_rate,
                            subband,
                            include_upper=include_upper,
                        )
                    )
                    subband_coverage.append(bool(covered))
                    subband_density.append(float(density))
                per_segment_subband_values.append(subband_values)
                per_segment_subband_coverage.append(subband_coverage)
                per_segment_subband_density.append(subband_density)
            metadata.append(segment)

    pending: list[RecordedSegment] = []
    expected_samples: int | None = None
    for segment in segments:
        if segment.x.ndim != 2 or segment.x.shape[0] != 2:
            raise ValueError(
                f"recorded segment x shape은 [2,T]여야 합니다: {segment.x.shape}"
            )
        if segment.d.ndim != 1 or segment.x.shape[-1] != segment.d.size:
            raise ValueError(
                f"recorded segment x/d 길이가 다릅니다: {segment.x.shape}, {segment.d.shape}"
            )
        if not np.all(np.isfinite(segment.x)) or not np.all(np.isfinite(segment.d)):
            raise ValueError("recorded segment에 NaN/Inf가 있습니다")
        if model_input_contract is not None:
            if segment.model_input_contract_sha256 != model_input_contract_sha:
                raise ValueError(
                    "recorded segment model-input contract SHA가 평가 계약과 다릅니다"
                )
            if np.count_nonzero(segment.x[1]) != 0:
                raise ValueError("recorded segment ERR feature는 exact zero여야 합니다")
        if expected_samples is None:
            expected_samples = int(segment.d.size)
        elif int(segment.d.size) != expected_samples:
            raise ValueError(
                "모든 recorded segment 길이는 같아야 합니다: "
                f"{expected_samples} != {segment.d.size}"
            )
        pending.append(segment)
        if len(pending) == batch_size:
            evaluate_batch(pending)
            pending = []
    if pending:
        evaluate_batch(pending)
    if not metadata:
        raise ValueError("평가할 recorded segment가 없습니다")

    trusted_array = np.asarray(trusted_values, dtype=np.float64)
    fullband_array = np.asarray(fullband_values, dtype=np.float64)
    gap_array = trusted_array - fullband_array
    families = sorted({segment.source_family for segment in metadata})
    source_rows: list[dict] = []
    for family in families:
        indices = np.asarray(
            [
                index
                for index, segment in enumerate(metadata)
                if segment.source_family == family
            ],
            dtype=np.int64,
        )
        trusted_stats = _distribution(trusted_array[indices])
        fullband_stats = _distribution(fullband_array[indices])
        source_rows.append(
            {
                "source_family": family,
                "n_segments": int(indices.size),
                "n_sessions": len({metadata[index].session_id for index in indices}),
                "n_groups": len({metadata[index].group_id for index in indices}),
                "trusted": trusted_stats,
                "fullband": fullband_stats,
                "gap_mean_db": float(np.mean(gap_array[indices])),
            }
        )

    octave_rows: list[dict] = []
    for center in requested_octaves:
        values = np.asarray(octave_values[center], dtype=np.float64)
        if values.size == 0:
            continue
        worst_count = max(1, int(math.ceil(values.size * 0.1)))
        worst = np.sort(values)[:worst_count]  # 감쇠는 작은 값이 나쁨
        octave_rows.append(
            {
                "center_hz": center,
                "attenuation_mean_db": float(np.mean(values)),
                "attenuation_median_db": float(np.median(values)),
                "attenuation_worst10_mean_db": float(np.mean(worst)),
                "trusted": bool(octave_trusted.get(center, False)),
            }
        )

    if strict_subbands is None:
        strict_subband_hz = np.empty((0, 2), dtype=np.float64)
        strict_subband_values = np.empty((len(metadata), 0), dtype=np.float64)
        strict_subband_coverage = np.empty((len(metadata), 0), dtype=np.bool_)
        strict_subband_density = np.empty((len(metadata), 0), dtype=np.float64)
        strict_subband_rows: list[dict] = []
        source_strict_subband_rows: list[dict] = []
    else:
        strict_subband_hz = np.asarray(strict_subbands, dtype=np.float64)
        strict_subband_values = np.asarray(per_segment_subband_values, dtype=np.float64)
        strict_subband_coverage = np.asarray(
            per_segment_subband_coverage, dtype=np.bool_
        )
        strict_subband_density = np.asarray(
            per_segment_subband_density, dtype=np.float64
        )
        if strict_subband_values.shape != (len(metadata), len(strict_subbands)):
            raise RuntimeError("strict trusted subband 결과 shape이 segment 수와 다릅니다")

        strict_subband_rows = []
        for band_index, subband in enumerate(strict_subbands):
            valid = strict_subband_coverage[:, band_index]
            values = strict_subband_values[valid, band_index]
            strict_subband_rows.append(
                {
                    "band_hz": tuple(float(value) for value in subband),
                    "n_segments": int(values.size),
                    "coverage_fraction": float(np.mean(valid)),
                    "source_energy_density_ratio_mean": float(
                        np.mean(strict_subband_density[:, band_index])
                    ),
                    "nmse": _distribution(values) if values.size else None,
                }
            )

        source_strict_subband_rows = []
        for family in families:
            family_indices = np.asarray(
                [
                    index
                    for index, segment in enumerate(metadata)
                    if segment.source_family == family
                ],
                dtype=np.int64,
            )
            for band_index, subband in enumerate(strict_subbands):
                valid = strict_subband_coverage[family_indices, band_index]
                values = strict_subband_values[family_indices, band_index][valid]
                groups = np.asarray(
                    [metadata[index].group_id for index in family_indices], dtype=np.str_
                )[valid]
                source_strict_subband_rows.append(
                    {
                        "source_family": family,
                        "band_hz": tuple(float(value) for value in subband),
                        "n_total_segments": int(family_indices.size),
                        "n_segments": int(values.size),
                        "n_groups": int(np.unique(groups).size),
                        "coverage_fraction": float(np.mean(valid)),
                        "source_energy_density_ratio_mean": float(
                            np.mean(strict_subband_density[family_indices, band_index])
                        ),
                        "nmse": _distribution(values) if values.size else None,
                    }
                )

    return {
        "n_segments": len(metadata),
        "n_sessions": len({segment.session_id for segment in metadata}),
        "n_groups": len({segment.group_id for segment in metadata}),
        "segment_samples": int(metadata[0].d.size),
        "metric_samples_per_segment": int(metadata[0].d.size - warmup_samples),
        "trusted": _distribution(trusted_array),
        "fullband": _distribution(fullband_array),
        "gap_mean_db": float(np.mean(gap_array)),
        "warmup_samples": warmup_samples,
        "model_input_contract_sha256": model_input_contract_sha,
        "per_segment_trusted_db": trusted_array,
        "per_segment_fullband_db": fullband_array,
        "per_segment_gap_db": gap_array,
        "segment_session_id": np.asarray(
            [segment.session_id for segment in metadata], dtype=np.str_
        ),
        "segment_group_id": np.asarray(
            [segment.group_id for segment in metadata], dtype=np.str_
        ),
        "segment_source_family": np.asarray(
            [segment.source_family for segment in metadata], dtype=np.str_
        ),
        "segment_start_sample": np.asarray(
            [segment.start_sample for segment in metadata], dtype=np.int64
        ),
        "segment_recorded_lead_samples": np.asarray(
            [segment.recorded_lead_samples for segment in metadata], dtype=np.int64
        ),
        "segment_recorded_delay_samples": np.asarray(
            [segment.recorded_delay_samples for segment in metadata], dtype=np.float64
        ),
        "segment_timing_contract_sha256": np.asarray(
            [segment.timing_contract_sha256 for segment in metadata], dtype=np.str_
        ),
        "segment_source_timeline": np.asarray(
            [segment.source_timeline for segment in metadata], dtype=np.str_
        ),
        "source_rows": source_rows,
        "octave_rows": octave_rows,
        "octave_center_hz": np.asarray(requested_octaves, dtype=np.float64),
        "per_segment_octave_attenuation_db": np.asarray(
            per_segment_octave_values, dtype=np.float64
        ),
        "strict_trusted_subband_schema": (
            STRICT_TRUSTED_SUBBAND_SCHEMA if strict_subbands is not None else ""
        ),
        "strict_trusted_subband_min_source_energy_density_ratio": (
            float(MIN_SUBBAND_SOURCE_ENERGY_DENSITY_RATIO)
            if strict_subbands is not None
            else float("nan")
        ),
        "strict_trusted_subband_hz": strict_subband_hz,
        "per_segment_trusted_subband_nmse_db": strict_subband_values,
        "per_segment_trusted_subband_coverage": strict_subband_coverage,
        "per_segment_trusted_subband_source_energy_density_ratio": strict_subband_density,
        "strict_trusted_subband_rows": strict_subband_rows,
        "source_strict_trusted_subband_rows": source_strict_subband_rows,
    }


# ``MAX_OUT_OF_BAND_AMPLIFICATION_DB`` 는 여기에 있었다. 손실 힌지 마진과 이 임계가
# 서로를 모른 채 각자 적혀 있었고, 실측 결과 **힌지를 정확히 만족한 모델이 게이트를
# 8.5 dB 차이로 FAIL** 했다. 이제 정의는 ``dsp/do_no_harm.py`` 한 곳이고 손실 마진은
# 거기에서 유도된다. 이 이름은 기존 참조(테스트·스크립트)를 위해 위에서 import 된다.

# ``MIN_GROUPS_PER_FAMILY``는 trusted_subbands의 단일 출처다. 같은 그룹 안의
# 세그먼트는 독립이 아니므로 CI/부대역 coverage는 세그먼트 수가 아니라 독립 group
# 수로 판정한다.

G4_PASS = "PASS"
G4_FAIL = "FAIL"
G4_INCONCLUSIVE = "INCONCLUSIVE"
"""G4 는 2값이 아니라 **3값** 판정이다.

"개선을 보이지 못했다"와 "악화를 보였다"는 다른 사실이고, 둘을 같은 FAIL 로 뭉치면
원인 진단이 불가능해진다. 더 중요한 것은 반대 방향이다 — 표본이 부족해 아무 말도 할
수 없는 상태를 PASS 로 흘려보내면 게이트가 있는 것이 없는 것보다 나쁘다. 실측
파인튜닝 val trusted −0.07 dB 는 cluster bootstrap CI [−0.456, +0.481] 로 0 과 구별
불가였는데 점추정만 보고 "개선"으로 읽혔다.

``INCONCLUSIVE`` 는 결코 ``g4_pass=True`` 가 되지 않는다.
"""


def cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 20260805,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """**그룹 단위**로 재표집한 평균의 신뢰구간. 반환은 ``(lo, hi, n_groups)``.

    세그먼트 단위 부트스트랩은 같은 음원에서 잘라낸 조각들을 독립 표본으로 착각해
    CI 를 실제보다 몇 배 좁게 만든다. 클러스터(=그룹)를 통째로 뽑아야 "다른 음원을
    가져왔다면 어땠을까"라는 질문에 답이 된다 — G4 가 실제로 묻는 질문이 그것이다.

    그룹 수가 :data:`MIN_GROUPS_PER_FAMILY` 미만이면 ``(nan, nan, n)`` 을 돌려준다.
    클러스터가 1개면 CI 가 수학적으로 정의되지 않는데, 그때 좁은 CI 를 지어내는 것이
    가장 위험한 실패다.
    """

    return _strict_cluster_bootstrap_ci(
        values,
        groups,
        min_groups=MIN_GROUPS_PER_FAMILY,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
    )


def _strict_trusted_subband_g4(result: dict) -> dict:
    """공식 네 trusted 부대역의 family별 G4 증거를 raw segment에서 재계산한다.

    전체 150–1600 Hz 평균은 한 부대역의 증폭을 다른 부대역의 큰 감쇠로 숨길 수 있다.
    이 함수는 사람이 읽는 ``source_*_rows``를 믿지 않고, result가 보존한 per-segment
    NMSE/coverage/group 배열에서 family×subband mean, worst10, CI를 다시 만든다.
    """

    source_rows = result.get("source_rows") or []
    families = [str(row["source_family"]) for row in source_rows]
    bands = np.asarray(STRICT_TRUSTED_SUBBANDS_HZ, dtype=np.float64)
    shape = (len(families), len(STRICT_TRUSTED_SUBBANDS_HZ))

    def empty(reason: str) -> dict:
        return {
            "schema_pass": False,
            "reason": reason,
            "families": families,
            "bands_hz": bands,
            "n_segments": np.zeros(shape, dtype=np.int64),
            "n_groups": np.zeros(shape, dtype=np.int64),
            "coverage_fraction": np.zeros(shape, dtype=np.float64),
            "density_ratio_mean": np.full(shape, np.nan, dtype=np.float64),
            "mean_db": np.full(shape, np.nan, dtype=np.float64),
            "worst10_db": np.full(shape, np.nan, dtype=np.float64),
            "ci_lo_db": np.full(shape, np.nan, dtype=np.float64),
            "ci_hi_db": np.full(shape, np.nan, dtype=np.float64),
            "coverage_pass": np.zeros(shape, dtype=np.bool_),
            "power_pass": np.zeros(shape, dtype=np.bool_),
            "mean_pass": np.zeros(shape, dtype=np.bool_),
            "worst10_pass": np.zeros(shape, dtype=np.bool_),
            "ci_pass": np.zeros(shape, dtype=np.bool_),
            "source_pass": np.zeros(shape, dtype=np.bool_),
            "g4_coverage_pass": False,
            "g4_power_pass": False,
            "g4_mean_pass": False,
            "g4_worst10_pass": False,
            "g4_ci_pass": False,
            "g4_pass": False,
            "upper_pass": False,
            "failed_mean_rows": [],
            "failed_worst10_rows": [],
        }

    if str(result.get("strict_trusted_subband_schema", "")) != (
        STRICT_TRUSTED_SUBBAND_SCHEMA
    ):
        return empty(
            "공식 G4는 strict trusted-band schema "
            f"{STRICT_TRUSTED_SUBBAND_SCHEMA!r}가 필요합니다"
        )
    stored_bands = np.asarray(result.get("strict_trusted_subband_hz"), dtype=np.float64)
    if stored_bands.shape != bands.shape or not np.array_equal(stored_bands, bands):
        return empty("strict trusted subband 경계가 canonical 150–1600Hz 분할과 다릅니다")

    try:
        values = np.asarray(
            result["per_segment_trusted_subband_nmse_db"], dtype=np.float64
        )
        coverage = np.asarray(
            result["per_segment_trusted_subband_coverage"], dtype=np.bool_
        )
        density = np.asarray(
            result["per_segment_trusted_subband_source_energy_density_ratio"],
            dtype=np.float64,
        )
        segment_family = np.asarray(result["segment_source_family"]).astype(str).reshape(-1)
        segment_group = np.asarray(result["segment_group_id"]).astype(str).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        return empty(f"strict trusted subband raw 배열이 없습니다: {exc}")
    expected_shape = (segment_family.size, len(STRICT_TRUSTED_SUBBANDS_HZ))
    if (
        values.shape != expected_shape
        or coverage.shape != expected_shape
        or density.shape != expected_shape
        or segment_group.size != segment_family.size
    ):
        return empty("strict trusted subband raw 배열 shape이 segment metadata와 다릅니다")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(density)):
        return empty("strict trusted subband raw NMSE/target-energy에 NaN/Inf가 있습니다")

    n_segments = np.zeros(shape, dtype=np.int64)
    n_groups = np.zeros(shape, dtype=np.int64)
    coverage_fraction = np.zeros(shape, dtype=np.float64)
    density_ratio_mean = np.full(shape, np.nan, dtype=np.float64)
    mean_db = np.full(shape, np.nan, dtype=np.float64)
    worst10_db = np.full(shape, np.nan, dtype=np.float64)
    ci_lo_db = np.full(shape, np.nan, dtype=np.float64)
    ci_hi_db = np.full(shape, np.nan, dtype=np.float64)
    coverage_pass = np.zeros(shape, dtype=np.bool_)
    power_pass = np.zeros(shape, dtype=np.bool_)
    mean_pass = np.zeros(shape, dtype=np.bool_)
    worst10_pass = np.zeros(shape, dtype=np.bool_)
    ci_pass = np.zeros(shape, dtype=np.bool_)
    source_pass = np.zeros(shape, dtype=np.bool_)
    failed_mean_rows: list[tuple[str, tuple[float, float], float]] = []
    failed_worst10_rows: list[tuple[str, tuple[float, float], float]] = []

    for family_index, family in enumerate(families):
        family_mask = segment_family == family
        if not np.any(family_mask):
            return empty(f"source family {family!r}의 segment metadata가 없습니다")
        for band_index, band in enumerate(STRICT_TRUSTED_SUBBANDS_HZ):
            valid = coverage[family_mask, band_index]
            selected_values = values[family_mask, band_index][valid]
            selected_groups = segment_group[family_mask][valid]
            n_segments[family_index, band_index] = int(selected_values.size)
            n_groups[family_index, band_index] = int(np.unique(selected_groups).size)
            coverage_fraction[family_index, band_index] = float(np.mean(valid))
            density_ratio_mean[family_index, band_index] = float(
                np.mean(density[family_mask, band_index])
            )
            has_coverage = bool(selected_values.size > 0)
            coverage_pass[family_index, band_index] = has_coverage
            has_power = bool(
                has_coverage
                and n_groups[family_index, band_index] >= MIN_GROUPS_PER_FAMILY
            )
            power_pass[family_index, band_index] = has_power
            if has_coverage:
                stats = _distribution(selected_values)
                mean_db[family_index, band_index] = float(stats["mean_db"])
                worst10_db[family_index, band_index] = float(
                    stats["worst10_mean_db"]
                )
                mean_pass[family_index, band_index] = bool(
                    mean_db[family_index, band_index] < 0.0
                )
                worst10_pass[family_index, band_index] = bool(
                    worst10_db[family_index, band_index] < 0.0
                )
                if not mean_pass[family_index, band_index]:
                    failed_mean_rows.append(
                        (family, tuple(band), float(mean_db[family_index, band_index]))
                    )
                if not worst10_pass[family_index, band_index]:
                    failed_worst10_rows.append(
                        (
                            family,
                            tuple(band),
                            float(worst10_db[family_index, band_index]),
                        )
                    )
                lo, hi, _ = cluster_bootstrap_ci(selected_values, selected_groups)
                ci_lo_db[family_index, band_index] = lo
                ci_hi_db[family_index, band_index] = hi
                ci_pass[family_index, band_index] = bool(
                    has_power and math.isfinite(hi) and hi < 0.0
                )
            source_pass[family_index, band_index] = bool(
                coverage_pass[family_index, band_index]
                and power_pass[family_index, band_index]
                and mean_pass[family_index, band_index]
                and worst10_pass[family_index, band_index]
                and ci_pass[family_index, band_index]
            )

    g4_coverage_pass = bool(coverage_pass.size and np.all(coverage_pass))
    g4_power_pass = bool(power_pass.size and np.all(power_pass))
    g4_mean_pass = bool(mean_pass.size and np.all(mean_pass))
    g4_worst10_pass = bool(worst10_pass.size and np.all(worst10_pass))
    g4_ci_pass = bool(ci_pass.size and np.all(ci_pass))
    g4_pass = bool(source_pass.size and np.all(source_pass))
    # 마지막 열은 single source인 1000–1600 Hz다. 별도 flag를 남겨 평균이 이 대역의
    # 실패를 감추는지 completion audit과 사람이 즉시 확인할 수 있게 한다.
    upper_pass = bool(source_pass.shape[1] and np.all(source_pass[:, -1]))
    return {
        "schema_pass": True,
        "reason": "",
        "families": families,
        "bands_hz": bands,
        "n_segments": n_segments,
        "n_groups": n_groups,
        "coverage_fraction": coverage_fraction,
        "density_ratio_mean": density_ratio_mean,
        "mean_db": mean_db,
        "worst10_db": worst10_db,
        "ci_lo_db": ci_lo_db,
        "ci_hi_db": ci_hi_db,
        "coverage_pass": coverage_pass,
        "power_pass": power_pass,
        "mean_pass": mean_pass,
        "worst10_pass": worst10_pass,
        "ci_pass": ci_pass,
        "source_pass": source_pass,
        "g4_coverage_pass": g4_coverage_pass,
        "g4_power_pass": g4_power_pass,
        "g4_mean_pass": g4_mean_pass,
        "g4_worst10_pass": g4_worst10_pass,
        "g4_ci_pass": g4_ci_pass,
        "g4_pass": g4_pass,
        "upper_pass": upper_pass,
        "failed_mean_rows": failed_mean_rows,
        "failed_worst10_rows": failed_worst10_rows,
    }


def _plant_fingerprint_for(context: "RecordedEvalContext") -> PlantFingerprint:
    """평가 결과에 박아 넣을 플랜트 지문. **결함 5의 구조적 차단이다.**

    2026-08-04 사고: 파인튜닝 전 기준선은 S 지연 1342 / lead 109 / surrogate 물리,
    후는 1465 / 113 / measured 였다. **서로 다른 물리**인데 "1.30 dB 개선"이라고
    적혔고 그것을 막는 장치가 아무 데도 없었다. metrics 가 지문을 들고 다니지 않으면
    이 사고는 구조적으로 반복된다.
    """

    duct_cfg = context.cfg.get("duct", {}) or {}
    secondary_value = str((duct_cfg.get("secondary_path", {}) or {}).get("npz", ""))
    primary_value = str(
        (duct_cfg.get("digital_reference", {}) or {}).get("primary_path_npz", "")
    )
    delays = PlantDelays(
        # P 지연이 없는 실행(acoustic reference 등)은 0 으로 둔다. 그 사실 자체가
        # 지문에 남아 measured 실행과 절대 같아 보이지 않는다.
        primary_delay_samples=max(0, int(context.primary_delay_samples or 0)),
        secondary_delay_samples=int(context.secondary_path.delay_samples),
        handoff_samples=int(context.secondary_handoff_samples),
        sample_rate=int(context.sample_rate),
    )
    bands = BandPlan(
        plant_trusted=FrequencyBand.parse(context.trusted_band_hz, name="S 신뢰"),
        target=FrequencyBand.parse(context.trusted_band_hz, name="목표"),
        optimize=FrequencyBand.parse(context.trusted_band_hz, name="손실"),
        measure=FrequencyBand.parse(context.trusted_band_hz, name="보고"),
        nyquist_hz=float(context.sample_rate) / 2.0,
    )
    return PlantFingerprint.build(
        delays=delays,
        lead=delays.lead(),
        physics_status=str(context.physics_status),
        bands=bands,
        secondary_sha256=_sha256_if_file(secondary_value) or None,
        primary_sha256=_sha256_if_file(primary_value) or None,
        configured_lead_samples=int(context.digital_reference_lead_samples),
    )


def plant_fingerprint_from_metrics(data) -> PlantFingerprint:
    """``metrics.npz`` 에 저장된 지문을 되살린다 (비교 전용).

    저장 형식이 아니라 **타입**으로 되살리는 이유: 필드를 손으로 골라 비교하면
    언젠가 한 필드를 빠뜨리고, 그 빠뜨린 필드가 하필 달랐던 것이 사고가 된다.
    """

    if "plant_fingerprint_json" not in getattr(data, "files", []):
        raise ValueError(
            "metrics.npz 에 plant_fingerprint_json 이 없습니다 — 플랜트 지문을 남기지 "
            "않던 구버전 평가기의 산출물이라 비교 가능성을 판정할 수 없습니다. "
            "evaluate_recorded.py 로 재평가하세요."
        )
    payload = json.loads(str(np.asarray(data["plant_fingerprint_json"]).reshape(-1)[0]))
    return PlantFingerprint(**payload)


def assert_comparable_metrics(before, after, *, context: str = "전후 비교") -> None:
    """두 ``metrics.npz`` 가 **같은 플랜트**에서 나왔을 때만 비교를 허용한다.

    ``np.load`` 로 연 두 아카이브를 그대로 넘겨라. 다르면 :class:`ValueError` 다 —
    "개선"을 적기 전에 멈추는 것이 이 함수의 목적이다.
    """

    result = check_plant_fingerprint_match(
        plant_fingerprint_from_metrics(before),
        plant_fingerprint_from_metrics(after),
    )
    if not result.ok:
        raise ValueError(f"[{context}] {result.detail}")


def write_recorded_metrics(
    result: dict,
    out_dir: str | Path,
    *,
    checkpoint: str | Path,
    manifest: str | Path,
    split: str,
    context: RecordedEvalContext,
    feedback_delay_samples: int,
    allow_surrogate: bool,
    model_hop: int,
    max_segments_per_session: int,
    segment_seconds: float,
    canonical_sampling: bool,
    edge_trim_samples: int,
    warmup_samples: int,
    checkpoint_sha256: str | None = None,
    manifest_sha256: str | None = None,
    experiment_contract_sha256: str | None = None,
    selection_sha256: str = "",
    test_capability_sha256: str = "",
    test_consumed_marker_sha256: str = "",
    exclusive: bool = False,
) -> tuple[Path, Path]:
    """사람용 Markdown과 기계용 NPZ를 원자적으로 생성한다."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=not exclusive)
    markdown_path = out_dir / "metrics.md"
    npz_path = out_dir / "metrics.npz"
    trusted = result["trusted"]
    fullband = result["fullband"]
    resolved_model_hop = int(model_hop)
    resolved_max_segments = int(max_segments_per_session)
    resolved_segment_seconds = float(segment_seconds)
    resolved_edge_trim = int(edge_trim_samples)
    expected_segment_samples = effective_segment_samples(
        sample_rate=int(context.sample_rate),
        model_hop=resolved_model_hop,
        segment_seconds=resolved_segment_seconds,
    )
    if int(result["segment_samples"]) != expected_segment_samples:
        raise ValueError(
            "평가 결과 segment_samples가 sampling 요청/hop 유도값과 다릅니다"
        )
    canonical_sampling = bool(canonical_sampling)
    context_model_input_sha = str(context.model_input_contract_sha256 or "")
    expected_context_model_input_sha = (
        context.model_input_contract.digest()
        if context.model_input_contract is not None
        else ""
    )
    if context_model_input_sha != expected_context_model_input_sha:
        raise ValueError(
            "recorded evaluation context의 model-input contract SHA가 payload와 다릅니다"
        )
    if str(result.get("model_input_contract_sha256", "")) != (
        context_model_input_sha
    ):
        raise ValueError(
            "recorded evaluation result의 model-input contract SHA가 checkpoint와 다릅니다"
        )
    plant_settle_samples = _plant_settle_samples(context)
    if canonical_sampling:
        canonical_edge_trim = int(
            round(CANONICAL_EDGE_TRIM_SECONDS * int(context.sample_rate))
        )
        if (
            resolved_max_segments != CANONICAL_MAX_SEGMENTS_PER_SESSION
            or resolved_segment_seconds != CANONICAL_SEGMENT_SECONDS
            or resolved_edge_trim != canonical_edge_trim
        ):
            raise ValueError(
                "canonical recorded sampling은 max=64, segment=1.5초, "
                "edge trim=0.25초와 정확히 같아야 합니다"
            )
        saved_loss_start = context.cfg.get("loss_start_sample")
        if (
            isinstance(saved_loss_start, bool)
            or not isinstance(saved_loss_start, int)
            or int(saved_loss_start) != plant_settle_samples
        ):
            raise ValueError(
                "canonical recorded sampling의 checkpoint loss_start_sample이 "
                "실제 S(z) PlantSettle과 정확히 같아야 합니다"
            )
        expected_feedback = canonical_feedback_delay_samples(context.cfg["data"])
        if int(feedback_delay_samples) != expected_feedback:
            raise ValueError(
                "canonical recorded sampling feedback delay가 checkpoint 기본 중앙값과 "
                "다릅니다"
            )
        expected_warmup = canonical_warmup_samples(
            context.cfg["data"],
            sample_rate=int(context.sample_rate),
            plant_settle_samples=plant_settle_samples,
        )
        if int(warmup_samples) != expected_warmup:
            raise ValueError(
                "canonical recorded sampling warmup이 checkpoint 기본값/PlantSettle과 "
                "다릅니다"
            )
    if int(result.get("warmup_samples", warmup_samples)) != int(warmup_samples):
        raise ValueError("평가 결과와 보고서 warmup_samples가 다릅니다")
    if int(result.get("metric_samples_per_segment", -1)) != (
        int(result["segment_samples"]) - int(warmup_samples)
    ):
        raise ValueError("평가 결과 metric_samples_per_segment가 segment-warmup과 다릅니다")
    trusted_pass = bool(trusted["mean_db"] < 0.0)
    fullband_pass = bool(fullband["mean_db"] <= 0.0)

    # 절대 목표 2번(모든 소리 제거)은 **평균이 아니라 최악값** 문제다. 전 세그먼트 평균만
    # 보면 machine -8 dB 와 speech +6 dB 가 섞여 평균 -1.75 dB 로 통과한다 — 즉 대화를
    # 6 dB 증폭하는 모델이 G4 PASS 로 배포 후보가 된다. per-family 통계는 이미 계산해
    # npz 에 넣고 있었지만(source_rows) 판정에는 한 번도 쓰이지 않았다.
    #
    # 두 축을 함께 본다:
    #   - 어떤 source_family 도 평균이 증폭이면 안 된다 (mean_db < 0)
    #   - 그 family 안의 최악 10% 구간도 증폭이면 안 된다 (worst10_mean_db < 0)
    # 후자가 없으면 "평균은 좋은데 특정 구간에서만 시끄러운" 모델을 걸러내지 못한다.
    source_rows = result.get("source_rows") or []
    worst_source_trusted_db = (
        max(float(row["trusted"]["mean_db"]) for row in source_rows)
        if source_rows else float("nan")
    )
    worst_source_trusted_worst10_db = (
        max(float(row["trusted"]["worst10_mean_db"]) for row in source_rows)
        if source_rows else float("nan")
    )
    worst_source_family = (
        max(source_rows, key=lambda row: float(row["trusted"]["mean_db"]))["source_family"]
        if source_rows else ""
    )
    # source_rows 가 비어 있으면 기능 2 를 측정하지 못한 것이다 — 통과시키지 않는다.
    source_pass = bool(
        source_rows
        and worst_source_trusted_db < 0.0
        and worst_source_trusted_worst10_db < 0.0
    )

    # ---- (a) 대역 밖 do-no-harm (절대목표 1) ------------------------------------
    octave_rows = result.get("octave_rows") or []
    # do-no-harm은 이름 그대로 **trusted 최적화 대역 밖**의 증폭을 막는 게이트다.
    # trusted 150–1600 Hz 안의 작은 감쇠량을 이 게이트에 섞으면, 해당 대역의
    # NMSE/strict-subband 게이트와 같은 사실을 중복 판정하고 정작 2/4/8 kHz
    # 증폭의 의미가 흐려진다. ``trusted`` 표시는 plant-derived band에서 왔다.
    out_of_band_octave_rows = [
        row for row in octave_rows if not bool(row.get("trusted", False))
    ]
    worst_octave = (
        min(
            out_of_band_octave_rows,
            key=lambda row: float(row["attenuation_worst10_mean_db"]),
        )
        if out_of_band_octave_rows
        else None
    )
    amplified = [
        row
        for row in out_of_band_octave_rows
        if float(row["attenuation_worst10_mean_db"]) <= -MAX_OUT_OF_BAND_AMPLIFICATION_DB
    ]
    # 대역 밖 옥타브 행이 아예 없으면 "해치지 않았다"를 **측정하지 못한** 것이다.
    # 측정하지 못한 것을 통과로 세지 않는다.
    do_no_harm_pass = bool(out_of_band_octave_rows) and not amplified

    # ---- (b) 통계적 검정력 (D3) ---------------------------------------------------
    underpowered = [
        (str(row["source_family"]), int(row["n_groups"]))
        for row in source_rows
        if int(row["n_groups"]) < MIN_GROUPS_PER_FAMILY
    ]
    power_pass = bool(source_rows) and not underpowered

    # ---- (c) 계열별 cluster bootstrap CI (D3) -------------------------------------
    # 점추정으로 "최악 계열"을 고르는 것은 계열 간 폭(0.92 dB)이 그룹 SE(1.03 dB)보다
    # 작을 때 동전 던지기다. 개선을 주장하려면 CI **상단**이 0 아래여야 한다.
    per_segment_trusted = np.asarray(result["per_segment_trusted_db"], dtype=np.float64)
    segment_family = np.asarray(result["segment_source_family"])
    segment_group = np.asarray(result["segment_group_id"])
    source_ci: list[tuple[str, float, float, int]] = []
    for row in source_rows:
        family = str(row["source_family"])
        mask = segment_family == family
        lo, hi, n_groups = cluster_bootstrap_ci(
            per_segment_trusted[mask], segment_group[mask]
        )
        source_ci.append((family, lo, hi, n_groups))
    ci_defined = bool(source_ci) and all(
        math.isfinite(hi) for _, _, hi, _ in source_ci
    )
    ci_pass = ci_defined and all(hi < 0.0 for _, _, hi, _ in source_ci)

    # ---- (d) strict trusted 부대역 (기능 1의 150–1600 Hz 내부) -------------------
    # 전체 trusted 평균만으로는 150–1000 Hz의 이득이 1000–1600 Hz의 증폭을 덮는다.
    # official G4는 네 부대역 모두에서 family별 target(d=ERR) coverage, 평균, 최악 10%,
    # 독립 group bootstrap CI를 요구한다. strict schema가 아니면 진단 결과일 뿐이다.
    strict_subband = _strict_trusted_subband_g4(result)

    # ---- 3값 판정 ------------------------------------------------------------------
    # 순서가 중요하다: **증명된 악화(FAIL)** 가 **판정 불가(INCONCLUSIVE)** 보다 먼저다.
    # 표본이 부족하더라도 이미 해를 끼친 것이 보인다면 그것은 결론이 난 사실이다.
    hard_failures: list[str] = []
    if not trusted_pass:
        hard_failures.append(f"trusted 평균 {trusted['mean_db']:+.2f} dB ≥ 0")
    if not fullband_pass:
        hard_failures.append(f"fullband 평균 {fullband['mean_db']:+.2f} dB > 0")
    if not source_pass:
        hard_failures.append(
            f"최악 계열 {worst_source_family or 'n/a'} {worst_source_trusted_db:+.2f} dB"
        )
    if not do_no_harm_pass:
        if not octave_rows:
            hard_failures.append("옥타브 감쇠를 측정하지 못했습니다")
        else:
            hard_failures.append(
                "대역 밖 증폭: "
                + ", ".join(
                    f"{row['center_hz']:.0f}Hz {row['attenuation_worst10_mean_db']:+.2f} dB"
                    for row in amplified
                )
            )
    if strict_subband["schema_pass"]:
        for family, band, value in strict_subband["failed_mean_rows"]:
            hard_failures.append(
                f"trusted {band[0]:.0f}–{band[1]:.0f}Hz {family} 평균 "
                f"{value:+.2f} dB ≥ 0"
            )
        for family, band, value in strict_subband["failed_worst10_rows"]:
            hard_failures.append(
                f"trusted {band[0]:.0f}–{band[1]:.0f}Hz {family} 최악 10% "
                f"{value:+.2f} dB ≥ 0"
            )

    inconclusive_reasons: list[str] = []
    if not power_pass:
        inconclusive_reasons.append(
            "계열당 그룹 부족 (최소 "
            f"{MIN_GROUPS_PER_FAMILY}): "
            + ", ".join(f"{family}={count}" for family, count in underpowered)
            + " — 그룹이 1개면 오차 추정 자체가 불가능합니다"
        )
    elif not ci_pass:
        inconclusive_reasons.append(
            "계열별 cluster bootstrap CI 상단이 0 아래가 아닙니다: "
            + ", ".join(
                f"{family} [{lo:+.2f}, {hi:+.2f}]" for family, lo, hi, _ in source_ci
            )
            + " — 점추정이 음수라도 0 과 구별되지 않으면 개선을 주장할 수 없습니다"
        )
    if not strict_subband["schema_pass"]:
        inconclusive_reasons.append(
            f"strict trusted 150–1600Hz 부대역 증거가 없습니다: {strict_subband['reason']}"
        )
    elif not strict_subband["g4_coverage_pass"]:
        incomplete = [
            f"{family} {band[0]:.0f}–{band[1]:.0f}Hz"
            for family_index, family in enumerate(strict_subband["families"])
            for band_index, band in enumerate(STRICT_TRUSTED_SUBBANDS_HZ)
            if not strict_subband["coverage_pass"][family_index, band_index]
        ]
        inconclusive_reasons.append(
            "trusted 부대역에 실제 target 에너지가 없어 평가할 수 없습니다: "
            + ", ".join(incomplete)
        )
    elif not strict_subband["g4_power_pass"]:
        weak = [
            f"{family} {band[0]:.0f}–{band[1]:.0f}Hz="
            f"{int(strict_subband['n_groups'][family_index, band_index])} groups"
            for family_index, family in enumerate(strict_subband["families"])
            for band_index, band in enumerate(STRICT_TRUSTED_SUBBANDS_HZ)
            if not strict_subband["power_pass"][family_index, band_index]
        ]
        inconclusive_reasons.append(
            "trusted 부대역 independent group coverage 부족 (최소 "
            f"{MIN_GROUPS_PER_FAMILY}): " + ", ".join(weak)
        )
    elif not strict_subband["g4_ci_pass"]:
        weak_ci = [
            f"{family} {band[0]:.0f}–{band[1]:.0f}Hz "
            f"[{strict_subband['ci_lo_db'][family_index, band_index]:+.2f}, "
            f"{strict_subband['ci_hi_db'][family_index, band_index]:+.2f}]"
            for family_index, family in enumerate(strict_subband["families"])
            for band_index, band in enumerate(STRICT_TRUSTED_SUBBANDS_HZ)
            if not strict_subband["ci_pass"][family_index, band_index]
        ]
        inconclusive_reasons.append(
            "trusted 부대역 family cluster bootstrap CI 상단이 0 아래가 아닙니다: "
            + ", ".join(weak_ci)
        )

    if hard_failures:
        verdict = G4_FAIL
        verdict_reason = "; ".join(hard_failures)
    elif inconclusive_reasons:
        verdict = G4_INCONCLUSIVE
        verdict_reason = "; ".join(inconclusive_reasons)
    else:
        verdict = G4_PASS
        verdict_reason = "모든 조건을 통계적 근거와 함께 만족했습니다"
    # INCONCLUSIVE 는 결코 통과가 아니다.
    g4_pass = verdict == G4_PASS

    fingerprint = _plant_fingerprint_for(context)
    checkpoint_sha256 = checkpoint_sha256 or context.checkpoint_sha256
    manifest_sha256 = manifest_sha256 or _sha256_if_file(manifest)
    experiment_contract_sha256 = experiment_contract_sha256 or str(
        context.cfg.get("experiment_contract_sha256", "")
    )

    lines = [
        f"# Recorded {split} 오프라인 평가",
        "",
    ]
    if context.physics_status != "measured_primary_path":
        lines += [
            "> [!WARNING]",
            "> `--allow-surrogate` 진단 결과입니다. 실측 덕트 성능으로 해석하면 안 됩니다.",
            "",
        ]
    lines += [
        f"- 체크포인트: `{Path(checkpoint)}`",
        f"- 체크포인트 SHA-256: `{checkpoint_sha256 or 'unavailable'}`",
        f"- Manifest: `{Path(manifest)}` (`{split}`)",
        f"- Manifest SHA-256: `{manifest_sha256 or 'unavailable'}`",
        f"- Model input contract SHA-256: "
        f"`{context_model_input_sha or 'legacy-diagnostic-unbound'}`",
        f"- 물리 상태: `{context.physics_status}`",
        f"- 세션/그룹/세그먼트: {result['n_sessions']}/{result['n_groups']}/{result['n_segments']}",
        f"- 세그먼트 길이: {result['segment_samples']} samples; 지표 구간: "
        f"{result['metric_samples_per_segment']} samples",
        f"- Trusted 대역: {context.trusted_band_hz[0]:.0f}–{context.trusted_band_hz[1]:.0f} Hz",
        f"- Digital lead: {context.digital_reference_lead_samples} samples",
        f"- S(z) 지연: {context.secondary_path.delay_samples} + handoff "
        f"{context.secondary_handoff_samples} samples",
        f"- Feedback 입력 지연: {feedback_delay_samples} samples",
        f"- 세션 양끝 제외: {edge_trim_samples} samples "
        f"({edge_trim_samples / context.sample_rate:.3f} s/edge)",
        f"- 지표 warmup 제외: {warmup_samples} samples "
        f"({warmup_samples / context.sample_rate:.3f} s, S(z) 적용 후 절단)"
        + (
            f" — 플랜트 정착 하한 {_plant_settle_samples(context)} samples 적용"
            if warmup_samples <= _plant_settle_samples(context)
            else f" (플랜트 정착 하한 {_plant_settle_samples(context)} samples 보다 김)"
        ),
        "",
        "## 전체 결과",
        "",
        "NMSE는 낮을수록 좋습니다. 최악 10%는 세그먼트 NMSE가 큰 상위 10%의 평균입니다.",
        "",
        "| 지표 | 평균 | 중앙값 | 최악 10% 평균 | 최악 |",
        "|---|---:|---:|---:|---:|",
        f"| Trusted NMSE | {trusted['mean_db']:+.2f} dB | "
        f"{trusted['median_db']:+.2f} dB | {trusted['worst10_mean_db']:+.2f} dB | "
        f"{trusted['worst_db']:+.2f} dB |",
        f"| Fullband NMSE | {fullband['mean_db']:+.2f} dB | "
        f"{fullband['median_db']:+.2f} dB | {fullband['worst10_mean_db']:+.2f} dB | "
        f"{fullband['worst_db']:+.2f} dB |",
        "",
        f"Trusted−fullband 평균 간극: **{result['gap_mean_db']:+.2f} dB**",
        "",
        "## G4 독립 recorded 판정",
        "",
        "| 조건 | 기준 | 결과 | 판정 |",
        "|---|---:|---:|---|",
        f"| Trusted 평균 NMSE | < 0 dB | {trusted['mean_db']:+.2f} dB | "
        f"{'PASS' if trusted_pass else 'FAIL'} |",
        f"| Fullband 평균 NMSE | ≤ 0 dB | {fullband['mean_db']:+.2f} dB | "
        f"{'PASS' if fullband_pass else 'FAIL'} |",
        f"| **최악 source family 평균** (기능 2) | < 0 dB | "
        f"{worst_source_trusted_db:+.2f} dB (`{worst_source_family or 'n/a'}`) | "
        f"{'PASS' if source_rows and worst_source_trusted_db < 0.0 else 'FAIL'} |",
        f"| **최악 source family 최악 10%** (기능 2) | < 0 dB | "
        f"{worst_source_trusted_worst10_db:+.2f} dB | "
        f"{'PASS' if source_rows and worst_source_trusted_worst10_db < 0.0 else 'FAIL'} |",
        f"| **대역 밖 do-no-harm** (기능 1) | > −{MAX_OUT_OF_BAND_AMPLIFICATION_DB:.1f} dB | "
        + (
            f"{worst_octave['center_hz']:.0f} Hz "
            f"{worst_octave['attenuation_worst10_mean_db']:+.2f} dB"
            if worst_octave
            else "측정 없음"
        )
        + f" | {'PASS' if do_no_harm_pass else 'FAIL'} |",
        f"| **계열당 그룹 수** (통계적 검정력) | ≥ {MIN_GROUPS_PER_FAMILY} | "
        + (
            ", ".join(f"{family}={count}" for family, count in underpowered)
            if underpowered
            else f"최소 {min((int(row['n_groups']) for row in source_rows), default=0)}"
        )
        + f" | {'PASS' if power_pass else '판정 불가'} |",
        f"| **계열별 CI 상단** (그룹 부트스트랩) | < 0 dB | "
        + (
            ", ".join(f"{family} {hi:+.2f}" for family, _, hi, _ in source_ci)
            if ci_defined
            else "정의 불가"
        )
        + f" | {'PASS' if ci_pass else '판정 불가'} |",
        f"| **strict trusted 부대역 schema** | `{STRICT_TRUSTED_SUBBAND_SCHEMA}` | "
        + (
            "150–300 / 300–600 / 600–1000 / 1000–1600 Hz"
            if strict_subband["schema_pass"]
            else strict_subband["reason"]
        )
        + f" | {'PASS' if strict_subband['schema_pass'] else '판정 불가'} |",
        f"| **모든 family×trusted 부대역 coverage** | target(d=ERR) energy + ≥ {MIN_GROUPS_PER_FAMILY} groups | "
        f"coverage={strict_subband['g4_coverage_pass']}, groups={strict_subband['g4_power_pass']} | "
        f"{'PASS' if strict_subband['g4_coverage_pass'] and strict_subband['g4_power_pass'] else '판정 불가'} |",
        f"| **모든 family×trusted 부대역 평균/최악 10%** | < 0 dB | "
        f"mean={strict_subband['g4_mean_pass']}, worst10={strict_subband['g4_worst10_pass']} | "
        f"{'PASS' if strict_subband['g4_mean_pass'] and strict_subband['g4_worst10_pass'] else 'FAIL'} |",
        f"| **1000–1600 Hz upper trusted 부대역** | 모든 family PASS | "
        f"{strict_subband['upper_pass']} | {'PASS' if strict_subband['upper_pass'] else 'FAIL/판정 불가'} |",
        f"| **trusted 부대역 CI 상단** | < 0 dB | {strict_subband['g4_ci_pass']} | "
        f"{'PASS' if strict_subband['g4_ci_pass'] else '판정 불가'} |",
        "",
        f"**G4 종합: {verdict}** — {verdict_reason}",
        "",
        "> 기능 2(모든 소리 제거)는 **평균이 아니라 최악값** 문제다. 여섯 소스 중 다섯이",
        "> −20 dB 이고 하나가 +6 dB 이면 평균은 좋아 보이지만, 그 하나가 들리는 순간",
        "> quiet zone 은 실패한 것이다. 그래서 평균 두 줄만으로는 G4 를 통과시키지 않는다.",
        ">",
        "> 기능 1(저·고역 모두 제거)의 게이트는 **대역 밖 do-no-harm** 이다. fullband 평균",
        "> NMSE 는 `d` 에 에너지가 없는 대역의 증폭을 원리적으로 잡지 못한다 — 실측에서",
        "> fullband +5.95 dB 와 8 kHz −21.56 dB 가 같은 실행에 공존했다.",
        ">",
        "> **판정 불가(INCONCLUSIVE)는 PASS 가 아니다.** 계열당 그룹이 1–2개면 cluster",
        "> bootstrap 의 클러스터 수가 CI 를 정의하지 못한다. 그때 좁은 CI 를 지어내는 것이",
        "> 가장 위험한 실패이므로, 판정할 수 없다는 사실을 그대로 남긴다.",
        "",
        "## 계열별 그룹 부트스트랩 신뢰구간",
        "",
        "| Source family | 그룹 | Trusted 평균 | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for (family, lo, hi, n_groups), row in zip(source_ci, source_rows):
        interval = (
            f"[{lo:+.2f}, {hi:+.2f}]"
            if math.isfinite(hi)
            else f"정의 불가 (그룹 {n_groups} < {MIN_GROUPS_PER_FAMILY})"
        )
        lines.append(
            f"| {family} | {n_groups} | {row['trusted']['mean_db']:+.2f} dB | {interval} |"
        )
    lines += [
        "",
        "## Source family별 결과",
        "",
        "| Source family | 세션 | 그룹 | 세그먼트 | Trusted 평균 | Trusted 최악 10% | Fullband 평균 | Fullband 최악 10% | 간극 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["source_rows"]:
        lines.append(
            f"| {row['source_family']} | {row['n_sessions']} | {row['n_groups']} | "
            f"{row['n_segments']} | {row['trusted']['mean_db']:+.2f} | "
            f"{row['trusted']['worst10_mean_db']:+.2f} | "
            f"{row['fullband']['mean_db']:+.2f} | "
            f"{row['fullband']['worst10_mean_db']:+.2f} | "
            f"{row['gap_mean_db']:+.2f} |"
        )
    lines += [
        "",
        "## Strict trusted 150–1600 Hz 부대역 결과",
        "",
        "공식 G4는 각 family가 네 부대역 모두에 실제 target(d=ERR) 에너지를 가져야 한다. "
        "평균/최악 10%는 NMSE이므로 음수일수록 좋고, CI 상단도 0 dB 아래여야 한다.",
        "",
    ]
    if not strict_subband["schema_pass"]:
        lines += [
            f"> [!WARNING] {strict_subband['reason']}",
            "> 이 결과는 진단용이며 canonical G4 PASS로 사용할 수 없습니다.",
        ]
    else:
        lines += [
            "| Source family | 부대역 | coverage segments/groups | density | 평균 NMSE | 최악 10% | 95% CI | 판정 |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for family_index, family in enumerate(strict_subband["families"]):
            for band_index, band in enumerate(STRICT_TRUSTED_SUBBANDS_HZ):
                lo = strict_subband["ci_lo_db"][family_index, band_index]
                hi = strict_subband["ci_hi_db"][family_index, band_index]
                interval = (
                    f"[{lo:+.2f}, {hi:+.2f}]"
                    if math.isfinite(hi)
                    else "정의 불가"
                )
                lines.append(
                    f"| {family} | {band[0]:.0f}–{band[1]:.0f} Hz | "
                    f"{int(strict_subband['n_segments'][family_index, band_index])}/"
                    f"{int(strict_subband['n_groups'][family_index, band_index])} "
                    f"({100.0 * strict_subband['coverage_fraction'][family_index, band_index]:.0f}%) | "
                    f"{strict_subband['density_ratio_mean'][family_index, band_index]:.2f} | "
                    f"{strict_subband['mean_db'][family_index, band_index]:+.2f} | "
                    f"{strict_subband['worst10_db'][family_index, band_index]:+.2f} | "
                    f"{interval} | "
                    f"{'PASS' if strict_subband['source_pass'][family_index, band_index] else 'FAIL/INCONCLUSIVE'} |"
                )
    lines += [
        "",
        "## 옥타브 밴드 감쇠",
        "",
        "감쇠는 높을수록 좋으며, 최악 10%는 감쇠가 작은 하위 10% 평균입니다.",
        "",
        "| 중심 주파수 | 평균 감쇠 | 중앙값 | 최악 10% 평균 | 신뢰 |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in result["octave_rows"]:
        confidence = "O" if row["trusted"] else "낮음*"
        lines.append(
            f"| {row['center_hz']:.0f} Hz | {row['attenuation_mean_db']:+.2f} dB | "
            f"{row['attenuation_median_db']:+.2f} dB | "
            f"{row['attenuation_worst10_mean_db']:+.2f} dB | {confidence} |"
        )
    lines += [
        "",
        "*: S(z) 실측 유효대역 밖이므로 광대역 재보정 전에는 참고용입니다.",
        "",
        "이 평가는 저장된 오디오 파일만 읽는 오프라인 계산이며 실제 오디오를 출력하지 않습니다.",
    ]
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    with markdown_tmp.open("x", encoding="utf-8") as markdown_file:
        markdown_file.write("\n".join(lines) + "\n")
        markdown_file.flush()
        os.fsync(markdown_file.fileno())
    os.replace(markdown_tmp, markdown_path)

    source_rows = result["source_rows"]
    octave_rows = result["octave_rows"]
    # 비정규 center/대역은 모델 진단에는 유용할 수 있다. 다만 이 파일이 official
    # G4의 clear-pass 근거인지 여부를 사람이 읽어도 명확하게 남긴다. canonical
    # validator는 label만 믿지 않고 아래 raw center matrix를 다시 검사한다.
    g4_metric_scope = (
        "canonical_recorded_g4"
        if np.array_equal(
            np.asarray(result["octave_center_hz"], dtype=np.float64),
            np.asarray(OCTAVE_BAND_CENTERS_HZ, dtype=np.float64),
        )
        and np.array_equal(
            np.asarray(context.trusted_band_hz, dtype=np.float64),
            np.asarray(STRICT_TRUSTED_BAND_HZ, dtype=np.float64),
        )
        and context.physics_status == "measured_primary_path"
        and context.model_input_contract is not None
        and bool(context_model_input_sha)
        and not bool(allow_surrogate)
        and canonical_sampling
        and split in {"val", "test"}
        else "diagnostic_noncanonical"
    )
    npz_tmp = npz_path.with_suffix(".npz.tmp")
    with open(npz_tmp, "wb") as file_obj:
        np.savez_compressed(
            file_obj,
            checkpoint=np.asarray(str(Path(checkpoint)), dtype=np.str_),
            checkpoint_sha256=np.asarray(checkpoint_sha256, dtype=np.str_),
            experiment_contract_sha256=np.asarray(
                experiment_contract_sha256, dtype=np.str_
            ),
            model_input_contract_sha256=np.asarray(
                context_model_input_sha, dtype=np.str_
            ),
            selection_sha256=np.asarray(selection_sha256, dtype=np.str_),
            test_capability_sha256=np.asarray(
                test_capability_sha256, dtype=np.str_
            ),
            test_consumed_marker_sha256=np.asarray(
                test_consumed_marker_sha256, dtype=np.str_
            ),
            manifest=np.asarray(str(Path(manifest)), dtype=np.str_),
            manifest_sha256=np.asarray(manifest_sha256, dtype=np.str_),
            split=np.asarray(split, dtype=np.str_),
            g4_metric_scope=np.asarray(g4_metric_scope, dtype=np.str_),
            physics_status=np.asarray(context.physics_status, dtype=np.str_),
            allow_surrogate=np.asarray(bool(allow_surrogate)),
            sample_rate=np.asarray(context.sample_rate, dtype=np.int64),
            recorded_sampling_contract_schema=np.asarray(
                RECORDED_SAMPLING_CONTRACT_SCHEMA, dtype=np.str_
            ),
            recorded_sampling_canonical=np.asarray(
                canonical_sampling, dtype=np.bool_
            ),
            recorded_sampling_model_hop=np.asarray(
                resolved_model_hop, dtype=np.int64
            ),
            recorded_sampling_max_segments_per_session=np.asarray(
                resolved_max_segments, dtype=np.int64
            ),
            recorded_sampling_segment_seconds=np.asarray(
                resolved_segment_seconds, dtype=np.float64
            ),
            recorded_sampling_plant_settle_samples=np.asarray(
                plant_settle_samples, dtype=np.int64
            ),
            n_sessions=np.asarray(result["n_sessions"], dtype=np.int64),
            n_groups=np.asarray(result["n_groups"], dtype=np.int64),
            n_segments=np.asarray(result["n_segments"], dtype=np.int64),
            segment_samples=np.asarray(result["segment_samples"], dtype=np.int64),
            metric_samples_per_segment=np.asarray(
                result["metric_samples_per_segment"], dtype=np.int64
            ),
            trusted_band_hz=np.asarray(context.trusted_band_hz, dtype=np.float64),
            digital_reference_lead_samples=np.asarray(
                context.digital_reference_lead_samples, dtype=np.int64
            ),
            primary_delay_samples=np.asarray(
                -1
                if context.primary_delay_samples is None
                else context.primary_delay_samples,
                dtype=np.int64,
            ),
            secondary_delay_samples=np.asarray(
                context.secondary_path.delay_samples, dtype=np.int64
            ),
            secondary_handoff_samples=np.asarray(
                context.secondary_handoff_samples, dtype=np.int64
            ),
            feedback_delay_samples=np.asarray(
                feedback_delay_samples, dtype=np.int64
            ),
            edge_trim_samples=np.asarray(edge_trim_samples, dtype=np.int64),
            warmup_samples=np.asarray(warmup_samples, dtype=np.int64),
            g4_trusted_pass=np.asarray(trusted_pass, dtype=np.bool_),
            g4_fullband_pass=np.asarray(fullband_pass, dtype=np.bool_),
            g4_pass=np.asarray(g4_pass, dtype=np.bool_),
            # ---- strict trusted 150–1600Hz 내부 게이트 ----------------------------
            # 이 schema/배열이 없는 구형 metrics는 canonical completion에 쓸 수 없다.
            strict_trusted_subband_schema=np.asarray(
                STRICT_TRUSTED_SUBBAND_SCHEMA
                if strict_subband["schema_pass"]
                else "",
                dtype=np.str_,
            ),
            strict_trusted_subband_min_source_energy_density_ratio=np.asarray(
                result["strict_trusted_subband_min_source_energy_density_ratio"],
                dtype=np.float64,
            ),
            trusted_subband_hz=np.asarray(strict_subband["bands_hz"], dtype=np.float64),
            g4_trusted_subband_schema_pass=np.asarray(
                strict_subband["schema_pass"], dtype=np.bool_
            ),
            g4_trusted_subband_coverage_pass=np.asarray(
                strict_subband["g4_coverage_pass"], dtype=np.bool_
            ),
            g4_trusted_subband_power_pass=np.asarray(
                strict_subband["g4_power_pass"], dtype=np.bool_
            ),
            g4_trusted_subband_mean_pass=np.asarray(
                strict_subband["g4_mean_pass"], dtype=np.bool_
            ),
            g4_trusted_subband_worst10_pass=np.asarray(
                strict_subband["g4_worst10_pass"], dtype=np.bool_
            ),
            g4_trusted_subband_ci_pass=np.asarray(
                strict_subband["g4_ci_pass"], dtype=np.bool_
            ),
            g4_trusted_subband_pass=np.asarray(
                strict_subband["g4_pass"], dtype=np.bool_
            ),
            g4_upper_trusted_subband_pass=np.asarray(
                strict_subband["upper_pass"], dtype=np.bool_
            ),
            # 기능 2 판정을 npz 에도 남긴다 — finetune_readiness 가 이 값을 검증해야
            # "평균만 좋은" 모델이 게이트를 통과하지 못한다.
            g4_source_pass=np.asarray(source_pass, dtype=np.bool_),
            g4_worst_source_trusted_mean_db=np.asarray(
                worst_source_trusted_db, dtype=np.float64
            ),
            g4_worst_source_trusted_worst10_db=np.asarray(
                worst_source_trusted_worst10_db, dtype=np.float64
            ),
            g4_worst_source_family=np.asarray(worst_source_family),
            # ---- 3값 판정 (D3). INCONCLUSIVE 는 g4_pass=False 다 ----
            g4_verdict=np.asarray(verdict, dtype=np.str_),
            g4_verdict_reason=np.asarray(verdict_reason, dtype=np.str_),
            # ---- 대역 밖 do-no-harm (결함 3, 절대목표 1) ----
            g4_do_no_harm_pass=np.asarray(do_no_harm_pass, dtype=np.bool_),
            g4_max_out_of_band_amplification_db=np.asarray(
                MAX_OUT_OF_BAND_AMPLIFICATION_DB, dtype=np.float64
            ),
            g4_worst_octave_center_hz=np.asarray(
                float(worst_octave["center_hz"]) if worst_octave else -1.0,
                dtype=np.float64,
            ),
            g4_worst_octave_worst10_db=np.asarray(
                float(worst_octave["attenuation_worst10_mean_db"])
                if worst_octave
                else float("nan"),
                dtype=np.float64,
            ),
            # ---- 통계적 검정력 + 그룹 부트스트랩 CI (D3) ----
            g4_power_pass=np.asarray(power_pass, dtype=np.bool_),
            g4_ci_pass=np.asarray(ci_pass, dtype=np.bool_),
            g4_min_groups_per_family=np.asarray(MIN_GROUPS_PER_FAMILY, dtype=np.int64),
            g4_underpowered_families=np.asarray(
                [family for family, _ in underpowered], dtype=np.str_
            ),
            source_trusted_ci_lo_db=np.asarray(
                [lo for _, lo, _, _ in source_ci], dtype=np.float64
            ),
            source_trusted_ci_hi_db=np.asarray(
                [hi for _, _, hi, _ in source_ci], dtype=np.float64
            ),
            # ---- 플랜트 지문 (결함 5) ----
            # 지문을 **하나의 JSON 칸**으로 저장한다. 필드를 흩어 놓으면 비교할 때
            # 하나를 빠뜨리게 되고, 하필 그 빠뜨린 필드가 달랐던 것이 2026-08-04 사고다.
            plant_fingerprint_json=np.asarray(
                json.dumps(fingerprint.model_dump(), sort_keys=True, ensure_ascii=False),
                dtype=np.str_,
            ),
            plant_fingerprint_digest=np.asarray(fingerprint.digest(), dtype=np.str_),
            secondary_path_npz=np.asarray(
                str((context.cfg.get("duct", {}) or {}).get("secondary_path", {}).get("npz", "")),
                dtype=np.str_,
            ),
            secondary_path_sha256=np.asarray(
                fingerprint.secondary_sha256 or "", dtype=np.str_
            ),
            primary_path_sha256=np.asarray(fingerprint.primary_sha256 or "", dtype=np.str_),
            nmse_trusted_mean_db=np.asarray(trusted["mean_db"]),
            nmse_trusted_median_db=np.asarray(trusted["median_db"]),
            nmse_trusted_worst10_mean_db=np.asarray(
                trusted["worst10_mean_db"]
            ),
            nmse_fullband_mean_db=np.asarray(fullband["mean_db"]),
            nmse_fullband_median_db=np.asarray(fullband["median_db"]),
            nmse_fullband_worst10_mean_db=np.asarray(
                fullband["worst10_mean_db"]
            ),
            nmse_gap_trusted_minus_fullband_mean_db=np.asarray(
                result["gap_mean_db"]
            ),
            per_segment_trusted_db=result["per_segment_trusted_db"],
            per_segment_fullband_db=result["per_segment_fullband_db"],
            per_segment_gap_db=result["per_segment_gap_db"],
            # 집계 octave 행만으로는 do-no-harm scalar를 재검산할 수 없다. 각 열은
            # ``octave_center_hz``의 같은 인덱스이고, 각 행은 segment metadata의
            # 같은 인덱스다. persisted G4 contract가 이 raw matrix에서 mean/median/
            # worst10과 최악 octave/threshold 판정을 다시 만든다.
            per_segment_octave_attenuation_db=result[
                "per_segment_octave_attenuation_db"
            ],
            per_segment_trusted_subband_nmse_db=result[
                "per_segment_trusted_subband_nmse_db"
            ],
            per_segment_trusted_subband_coverage=result[
                "per_segment_trusted_subband_coverage"
            ],
            per_segment_trusted_subband_source_energy_density_ratio=result[
                "per_segment_trusted_subband_source_energy_density_ratio"
            ],
            segment_session_id=result["segment_session_id"],
            segment_group_id=result["segment_group_id"],
            segment_source_family=result["segment_source_family"],
            segment_start_sample=result["segment_start_sample"],
            segment_recorded_lead_samples=result[
                "segment_recorded_lead_samples"
            ],
            segment_recorded_delay_samples=result[
                "segment_recorded_delay_samples"
            ],
            segment_timing_contract_sha256=result[
                "segment_timing_contract_sha256"
            ],
            segment_source_timeline=result["segment_source_timeline"],
            source_family=np.asarray(
                [row["source_family"] for row in source_rows], dtype=np.str_
            ),
            source_n_segments=np.asarray(
                [row["n_segments"] for row in source_rows], dtype=np.int64
            ),
            source_n_sessions=np.asarray(
                [row["n_sessions"] for row in source_rows], dtype=np.int64
            ),
            source_n_groups=np.asarray(
                [row["n_groups"] for row in source_rows], dtype=np.int64
            ),
            source_nmse_trusted_mean_db=np.asarray(
                [row["trusted"]["mean_db"] for row in source_rows],
                dtype=np.float64,
            ),
            source_nmse_trusted_worst10_mean_db=np.asarray(
                [row["trusted"]["worst10_mean_db"] for row in source_rows],
                dtype=np.float64,
            ),
            source_nmse_fullband_mean_db=np.asarray(
                [row["fullband"]["mean_db"] for row in source_rows],
                dtype=np.float64,
            ),
            source_nmse_fullband_worst10_mean_db=np.asarray(
                [row["fullband"]["worst10_mean_db"] for row in source_rows],
                dtype=np.float64,
            ),
            source_gap_trusted_minus_fullband_mean_db=np.asarray(
                [row["gap_mean_db"] for row in source_rows], dtype=np.float64
            ),
            source_trusted_subband_n_segments=np.asarray(
                strict_subband["n_segments"], dtype=np.int64
            ),
            source_trusted_subband_n_groups=np.asarray(
                strict_subband["n_groups"], dtype=np.int64
            ),
            source_trusted_subband_coverage_fraction=np.asarray(
                strict_subband["coverage_fraction"], dtype=np.float64
            ),
            source_trusted_subband_source_energy_density_ratio_mean=np.asarray(
                strict_subband["density_ratio_mean"], dtype=np.float64
            ),
            source_trusted_subband_nmse_mean_db=np.asarray(
                strict_subband["mean_db"], dtype=np.float64
            ),
            source_trusted_subband_nmse_worst10_mean_db=np.asarray(
                strict_subband["worst10_db"], dtype=np.float64
            ),
            source_trusted_subband_ci_lo_db=np.asarray(
                strict_subband["ci_lo_db"], dtype=np.float64
            ),
            source_trusted_subband_ci_hi_db=np.asarray(
                strict_subband["ci_hi_db"], dtype=np.float64
            ),
            source_trusted_subband_coverage_pass=np.asarray(
                strict_subband["coverage_pass"], dtype=np.bool_
            ),
            source_trusted_subband_power_pass=np.asarray(
                strict_subband["power_pass"], dtype=np.bool_
            ),
            source_trusted_subband_mean_pass=np.asarray(
                strict_subband["mean_pass"], dtype=np.bool_
            ),
            source_trusted_subband_worst10_pass=np.asarray(
                strict_subband["worst10_pass"], dtype=np.bool_
            ),
            source_trusted_subband_ci_pass=np.asarray(
                strict_subband["ci_pass"], dtype=np.bool_
            ),
            source_trusted_subband_pass=np.asarray(
                strict_subband["source_pass"], dtype=np.bool_
            ),
            octave_center_hz=result["octave_center_hz"],
            octave_attenuation_mean_db=np.asarray(
                [row["attenuation_mean_db"] for row in octave_rows],
                dtype=np.float64,
            ),
            octave_attenuation_median_db=np.asarray(
                [row["attenuation_median_db"] for row in octave_rows],
                dtype=np.float64,
            ),
            octave_attenuation_worst10_mean_db=np.asarray(
                [row["attenuation_worst10_mean_db"] for row in octave_rows],
                dtype=np.float64,
            ),
            octave_trusted=np.asarray(
                [row["trusted"] for row in octave_rows], dtype=np.bool_
            ),
        )
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(npz_tmp, npz_path)
    directory_fd = os.open(
        out_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return markdown_path, npz_path


__all__ = [
    "G4_FAIL",
    "G4_INCONCLUSIVE",
    "G4_PASS",
    "MAX_OUT_OF_BAND_AMPLIFICATION_DB",
    "MIN_GROUPS_PER_FAMILY",
    "RecordedEvalContext",
    "RecordedSegment",
    "assert_comparable_metrics",
    "cluster_bootstrap_ci",
    "deterministic_segment_starts",
    "plant_fingerprint_from_metrics",
    "evaluate_recorded_segments",
    "iter_recorded_segments",
    "load_and_audit_recorded_manifest",
    "load_recorded_eval_context",
    "resolve_feedback_delay",
    "resolve_warmup_samples",
    "validate_resolved_checkpoint",
    "write_recorded_metrics",
]
