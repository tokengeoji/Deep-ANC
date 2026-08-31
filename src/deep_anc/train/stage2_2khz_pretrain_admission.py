"""Stage-2 scratch pretrain만 별도로 여는 typed admission.

recorded additions, 70:30 fine-tune transfer, 100k checkpoint를 요구하지 않는다.
새 P/S, public manifest bytes, lineage/coverage, deterministic sampler, output-y DNH
calibration과 전용 criterion source SHA가 모두 일치할 때 pretrain smoke와
scratch 100k 경로만 준비된다. test split은 manifest 계보 검사에만 쓰고
checkpoint 선택에는 절대 사용하지 않는다.

coverage receipt의 count는 authority가 아니다. actual manifest의 각 source bytes를
다시 decode해 split×family×5 octave와 1.6 kHz sentinel의 qualified dataset/component
identity 전체를 재구성하고 receipt와 exact 비교한다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import welch

from ..data.public_lineage import validate_recorded_clip_lineage
from ..data.stage2_pretrain_data_issuer import (
    stage2_recorded_public_intersection,
    validate_elice_bootstrap_inputs,
)
from ..dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_SOURCE_FAMILIES,
    Stage2TwoKilohertzContract,
)
from ..losses.stage2_2khz_loss import (
    STAGE2_2KHZ_LOSS_SCHEMA,
    Stage2TwoKilohertzLossConfig,
)
from .stage2_2khz_binding import (
    Stage2TwoKilohertzPlantBinding,
    load_stage2_2khz_plant_binding,
)
from .stage2_2khz_execution import (
    STAGE2_2KHZ_SAMPLER_SCHEMA,
    STAGE2_2KHZ_TRAINER_ADAPTER_SCHEMA,
    Stage2FamilyComponentBatchSampler,
    Stage2SamplerRecord,
    require_stage2_actuator_limit,
)
from .stage2_2khz_git_authority import (
    verify_source_commit_ancestor,
    verify_tracked_head_authority,
    verify_tracked_head_file,
)


STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA = "stage2_2khz_public_manifest_bundle_v1"
STAGE2_PUBLIC_LINEAGE_RECEIPT_SCHEMA = "stage2_2khz_public_lineage_receipt_v2"
STAGE2_PUBLIC_COVERAGE_RECEIPT_SCHEMA = "stage2_2khz_public_frequency_coverage_v2"
STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA = "stage2_2khz_transfer_bootstrap_receipt_v1"
# v1은 NPZ의 SHA만 기록해 typed admission이 실제 tensor를 다시 열어
# gradient를 재계산할 수 없었다. v2는 held NPZ의 path+SHA를 함께 봉인했고,
# v3는 calibration NPZ의 model config/initial-state SHA도 criterion/profile 및
# actual fresh-model state 재계산에 결속한다.
STAGE2_DNH_CALIBRATION_RECEIPT_SCHEMA = "stage2_2khz_dnh_gradient_calibration_v3"
STAGE2_CRITERION_IMPLEMENTATION_RECEIPT_SCHEMA = (
    "stage2_2khz_criterion_implementation_receipt_v2"
)
STAGE2_PRETRAIN_TYPED_ADMISSION_SCHEMA = "stage2_2khz_typed_pretrain_admission_v1"
STAGE2_PUBLIC_DATA_AUTHORITY_SCHEMA = "stage2_2khz_public_data_git_authority_v1"
STAGE2_PUBLIC_DATA_AUTHORITY_PATH = "authority/stage2_2khz_public_data.json"
STAGE2_PRETRAIN_AUTHORITY_SCHEMA = "stage2_2khz_pretrain_git_authority_v1"
STAGE2_PRETRAIN_AUTHORITY_PATH = "authority/stage2_2khz_pretrain.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = ("train", "val", "test")
_ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ = (1425.437949, 1795.939277)
_SOURCE_DENSITY_ALGORITHM = (
    "mono_mean_welch_nperseg8192_noverlap4096_detrend_false_v1"
)
_SOURCE_DENSITY_NPERSEG = 8192
_SOURCE_DENSITY_NOVERLAP = 4096
_RECORDED_HOLDOUT_PATH = "data/manifests/recorded_holdout.json"
_RECORDED_SYNTHETIC_INTERSECTION_ALGORITHM = (
    "transitive_basename_content_sha256_lineage_keys_v1"
)
_CANONICAL_PRETRAIN_PROFILE_PATH = "configs/stage2_2khz_train_pretrain.yaml"


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _inside(root: Path, raw: object, *, label: str) -> Path:
    relative = Path(str(raw or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            node = cursor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(node.st_mode):
            raise ValueError(f"{label} 경로에 symlink가 있습니다")
    return root / relative


def _snapshot(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"regular file만 허용합니다: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"snapshot 중 artifact가 바뀌었습니다: {path}")
    content = b"".join(chunks)
    return content, hashlib.sha256(content).hexdigest()


def _ref(
    root: Path,
    *,
    path: str,
    sha256: str,
    label: str,
) -> tuple[Path, bytes, str]:
    expected = _require_sha256(sha256, label=f"{label}.sha256")
    target = _inside(root, path, label=label)
    try:
        content, actual = _snapshot(target)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} artifact가 없습니다") from exc
    if actual != expected:
        raise ValueError(f"{label} bytes SHA가 profile/receipt와 다릅니다")
    return target, content, actual


def _json(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}는 UTF-8 JSON이어야 합니다") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root는 mapping이어야 합니다")
    return value


def _yaml_mapping(content: bytes, *, label: str) -> dict[str, Any]:
    """tracked profile/model YAML을 mapping으로만 읽는다."""

    try:
        value = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label}는 UTF-8 YAML이어야 합니다") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root는 mapping이어야 합니다")
    return value


def _artifact_ref(value: object, *, label: str) -> tuple[str, str]:
    """path/SHA 쌍의 모양과 SHA 문법을 한 곳에서 강제한다."""

    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} ref가 exact하지 않습니다")
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label}.path가 비었습니다")
    return path, _require_sha256(value.get("sha256"), label=f"{label}.sha256")


def _canonical_pretrain_model_ref(
    root: Path,
) -> tuple[tuple[str, str], int, int]:
    """actual tracked canonical profile에서 model/seed/batch를 재도출한다.

    criterion이 자기 자신을 가리키는 profile SHA를 들고 있으면 implementation
    receipt path 때문에 순환 결속이 생긴다. 대신 admission은 이미 exact-clean
    origin/dev authority를 확인한 동일 checkout에서 canonical pretrain profile의
    model ref, seed, batch를 다시 읽어 criterion과 비교한다.
    """

    profile_bytes, _profile_sha, _head = verify_tracked_head_file(
        root, _CANONICAL_PRETRAIN_PROFILE_PATH
    )
    profile = _yaml_mapping(profile_bytes, label="Stage-2 canonical pretrain profile")
    execution = profile.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Stage-2 canonical pretrain execution profile이 없습니다")
    model_ref = _artifact_ref(
        execution.get("model_config"),
        label="Stage-2 canonical pretrain model config",
    )
    seed = profile.get("seed")
    batch_size = execution.get("batch_size")
    if type(seed) is not int or int(seed) != 20260803:
        raise ValueError("Stage-2 canonical pretrain profile seed가 exact하지 않습니다")
    if type(batch_size) is not int or int(batch_size) != 96:
        raise ValueError("Stage-2 canonical pretrain profile batch_size가 exact하지 않습니다")
    return model_ref, int(seed), int(batch_size)


def _fresh_calibration_model(
    model_config: Mapping[str, Any], *, seed: int
) -> tuple[torch.nn.Module, str]:
    """actual model config+seed로 fresh scratch model/state digest를 만든다.

    JSON receipt의 state SHA 숫자를 믿지 않는다. CPU RNG state도 fork로 복원해
    admission caller의 RNG 상태를 바꾸지 않으며 CUDA context를 만들지 않는다.
    """

    if not isinstance(model_config, Mapping):
        raise ValueError("Stage-2 model config는 mapping이어야 합니다")
    if type(seed) is not int or int(seed) != 20260803:
        raise ValueError("Stage-2 model initialization seed가 canonical과 다릅니다")
    try:
        in_channels = int(model_config.get("in_channels", 0))
        hop = int(model_config.get("hop", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-2 model config channel/hop가 정수가 아닙니다") from exc
    if in_channels != 2 or hop != 128:
        raise ValueError("Stage-2 calibrated model은 2-channel/hop=128 Tiny여야 합니다")
    require_stage2_actuator_limit(model_config)

    # issuer -> admission constants import cycle을 module import 시 만들지 않는다.
    from ..models import build_model  # pylint: disable=import-outside-toplevel
    from .stage2_2khz_pretrain_issuer import (  # pylint: disable=import-outside-toplevel
        model_initial_state_sha256,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = build_model(dict(model_config)).cpu().eval()
        return model, model_initial_state_sha256(model)


def _rebuild_model_initial_state_sha256(
    model_config: Mapping[str, Any], *, seed: int
) -> str:
    """fresh scratch parameter/buffer SHA만 필요한 caller용 wrapper."""

    _model, digest = _fresh_calibration_model(model_config, seed=seed)
    return digest


def _require_rebuilt_model_initial_state_sha256(
    model_config: Mapping[str, Any], *, seed: int, expected_sha256: str
) -> None:
    """fresh model state가 declared calibration state와 exact한지 fail-closed 검사한다."""

    expected = _require_sha256(
        expected_sha256, label="criterion model initial state SHA"
    )
    actual = _rebuild_model_initial_state_sha256(model_config, seed=seed)
    if actual != expected:
        raise ValueError(
            "Stage-2 calibration model initial-state SHA가 actual config+seed 재계산과 다릅니다"
        )


def _validate_calibration_model_binding(
    *,
    root: Path,
    snapshot_metadata: Mapping[str, Any],
    criterion_model_config_ref: tuple[str, str],
    criterion_model_initial_state_sha256: str,
    criterion_seed: int,
    criterion_batch_size: int,
) -> None:
    """calibration NPZ의 model provenance를 profile/actual model로 재검증한다."""

    profile_model_ref, profile_seed, profile_batch_size = _canonical_pretrain_model_ref(
        root
    )
    if criterion_model_config_ref != profile_model_ref:
        raise ValueError(
            "Stage-2 criterion model config가 canonical pretrain profile과 다릅니다"
        )
    if criterion_seed != profile_seed or criterion_batch_size != profile_batch_size:
        raise ValueError("Stage-2 criterion seed/batch가 canonical pretrain profile과 다릅니다")

    model_path, model_bytes, model_sha = _ref(
        root,
        path=criterion_model_config_ref[0],
        sha256=criterion_model_config_ref[1],
        label="Stage-2 criterion model config",
    )
    # profile ref만 검증하고 working tree bytes를 재사용하지 않는다. exact-clean
    # tracked blob 자체와 held snapshot이 모두 같은 bytes여야 한다.
    tracked_bytes, tracked_sha, _tracked_head = verify_tracked_head_file(
        root, criterion_model_config_ref[0]
    )
    if tracked_sha != model_sha or tracked_bytes != model_bytes:
        raise ValueError("Stage-2 criterion model config tracked/held bytes가 다릅니다")
    del model_path
    model_config = _yaml_mapping(model_bytes, label="Stage-2 criterion model config")

    expected_config_sha = _require_sha256(
        criterion_model_config_ref[1], label="criterion model config SHA"
    )
    expected_initial_state_sha = _require_sha256(
        criterion_model_initial_state_sha256,
        label="criterion model initial state SHA",
    )
    if snapshot_metadata["model_config_sha256"] != expected_config_sha:
        raise ValueError("Stage-2 calibration batch model config SHA가 criterion/profile과 다릅니다")
    if snapshot_metadata["model_initial_state_sha256"] != expected_initial_state_sha:
        raise ValueError(
            "Stage-2 calibration batch model initial-state SHA가 criterion과 다릅니다"
        )
    _require_rebuilt_model_initial_state_sha256(
        model_config,
        seed=criterion_seed,
        expected_sha256=expected_initial_state_sha,
    )


def _contract_entry() -> dict[str, str]:
    contract = Stage2TwoKilohertzContract.canonical()
    return {"id": contract.contract_id, "sha256": contract.digest()}


def _density_ratios(
    frequency: np.ndarray,
    psd: np.ndarray,
    bands_hz: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    bands = tuple(tuple(float(value) for value in band) for band in bands_hz)
    baseline_mask = (frequency >= bands[0][0]) & (frequency < bands[-1][1])
    if not np.any(baseline_mask):
        raise ValueError("Stage-2 source density baseline에 Welch bin이 없습니다")
    baseline = float(np.mean(psd[baseline_mask]))
    if not math.isfinite(baseline):
        raise ValueError("Stage-2 source density baseline이 finite하지 않습니다")
    ratios: list[float] = []
    for lower, upper in bands:
        selected = (frequency >= lower) & (frequency < upper)
        if not np.any(selected):
            raise ValueError("Stage-2 source density octave에 Welch bin이 없습니다")
        density = float(np.mean(psd[selected]))
        ratio = (
            0.0
            if baseline <= np.finfo(np.float64).tiny
            else float(density / baseline)
        )
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError("Stage-2 source density ratio가 finite한 0 이상이 아닙니다")
        ratios.append(ratio)
    return tuple(ratios)


def _sentinel_density_ratio(
    frequency: np.ndarray,
    psd: np.ndarray,
    *,
    objective_bands_hz: Sequence[Sequence[float]],
) -> float:
    lower, upper = _ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ
    baseline_lower = float(objective_bands_hz[0][0])
    baseline_upper = float(objective_bands_hz[-1][1])
    selected = (frequency >= lower) & (frequency < upper)
    baseline_mask = (frequency >= baseline_lower) & (frequency < baseline_upper)
    if not np.any(selected) or not np.any(baseline_mask):
        raise ValueError("Stage-2 1.6 kHz sentinel/baseline에 Welch bin이 없습니다")
    baseline = float(np.mean(psd[baseline_mask]))
    density = float(np.mean(psd[selected]))
    ratio = (
        0.0
        if baseline <= np.finfo(np.float64).tiny
        else float(density / baseline)
    )
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("Stage-2 sentinel density ratio가 finite한 0 이상이 아닙니다")
    return ratio


def _source_density_from_held_bytes(
    content: bytes,
    *,
    expected_sample_rate: int,
    contract: Stage2TwoKilohertzContract,
) -> tuple[tuple[float, ...], float]:
    """manifest가 봉인한 동일 bytes를 decode해 source coverage를 다시 계산한다."""

    try:
        values, sample_rate = sf.read(
            io.BytesIO(content),
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Stage-2 public source bytes를 decode할 수 없습니다") from exc
    if int(sample_rate) != int(expected_sample_rate):
        raise ValueError("Stage-2 public source manifest/audio sample rate가 다릅니다")
    audio = np.asarray(values, dtype=np.float64)
    if (
        audio.ndim != 2
        or audio.shape[0] < 256
        or audio.shape[1] < 1
        or not np.all(np.isfinite(audio))
    ):
        raise ValueError("Stage-2 public source PCM이 density 계산에 부적격합니다")
    mono = np.mean(audio, axis=1, dtype=np.float64)
    nperseg = min(_SOURCE_DENSITY_NPERSEG, int(mono.size))
    noverlap = min(_SOURCE_DENSITY_NOVERLAP, nperseg - 1)
    frequency, psd = welch(
        mono,
        fs=int(sample_rate),
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
    )
    ratios = _density_ratios(
        frequency,
        psd,
        contract.octave_objective_bands_hz,
    )
    sentinel = _sentinel_density_ratio(
        frequency,
        psd,
        objective_bands_hz=contract.octave_objective_bands_hz,
    )
    return ratios, sentinel


@dataclass(frozen=True)
class Stage2PretrainSource:
    dataset_index: int
    relative_path: str
    content_sha256: str
    native_sample_rate: int


@dataclass(frozen=True)
class Stage2PretrainDataBinding:
    manifest_bundle_sha256: str
    lineage_receipt_sha256: str
    frequency_coverage_receipt_sha256: str
    transfer_bootstrap_receipt_sha256: str
    records: tuple[Stage2SamplerRecord, ...]
    sources: tuple[Stage2PretrainSource, ...]


@dataclass(frozen=True)
class Stage2PretrainTypedAdmission:
    plant_binding: Stage2TwoKilohertzPlantBinding
    data_binding: Stage2PretrainDataBinding
    sampler: Stage2FamilyComponentBatchSampler
    loss_config: Stage2TwoKilohertzLossConfig
    criterion_receipt_sha256: str
    dnh_calibration_receipt_sha256: str
    sampler_receipt_sha256: str
    status: str = "READY"
    schema_version: str = STAGE2_PRETRAIN_TYPED_ADMISSION_SCHEMA


def _coverage_entry(
    record: Stage2SamplerRecord,
    source: Stage2PretrainSource,
) -> dict[str, object]:
    return {
        "dataset_index": int(record.dataset_index),
        "component_id": str(record.component_id),
        "path": source.relative_path,
        "content_sha256": source.content_sha256,
    }


def _recompute_qualified_sources(
    root: Path,
    *,
    records: Sequence[Stage2SamplerRecord],
    sources: Sequence[Stage2PretrainSource],
    workers: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """actual manifest rows와 동일 source bytes에서 qualified set 전체를 유도한다."""

    contract = Stage2TwoKilohertzContract.canonical()
    records_by_index = {int(record.dataset_index): record for record in records}
    sources_by_index = {int(source.dataset_index): source for source in sources}
    if (
        len(records_by_index) != len(records)
        or len(sources_by_index) != len(sources)
        or set(records_by_index) != set(sources_by_index)
    ):
        raise ValueError("Stage-2 manifest record/source dataset_index 전단사가 깨졌습니다")

    octave: dict[str, dict[str, list[list[dict[str, object]]]]] = {
        split: {
            family: [[] for _ in contract.octave_objective_bands_hz]
            for family in STAGE2_2KHZ_SOURCE_FAMILIES
        }
        for split in _SPLITS
    }
    sentinel: dict[str, dict[str, list[dict[str, object]]]] = {
        split: {family: [] for family in STAGE2_2KHZ_SOURCE_FAMILIES}
        for split in _SPLITS
    }
    threshold = float(contract.minimum_source_density_ratio)
    ordered_indices = sorted(records_by_index)
    if workers is None:
        worker_count = min(16, max(1, int(os.cpu_count() or 1)))
    elif (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 16
    ):
        raise ValueError("Stage-2 admission workers는 1..16 integer여야 합니다")
    else:
        worker_count = workers

    def analyze(
        dataset_index: int,
    ) -> tuple[Stage2SamplerRecord, Stage2PretrainSource, tuple[float, ...], float]:
        record = records_by_index[dataset_index]
        source = sources_by_index[dataset_index]
        _, content, actual_sha = _ref(
            root,
            path=source.relative_path,
            sha256=source.content_sha256,
            label=f"Stage-2 coverage source #{dataset_index}",
        )
        if actual_sha != record.source_sha256:
            raise ValueError("Stage-2 sampler record/source bytes SHA가 다릅니다")
        ratios, sentinel_ratio = _source_density_from_held_bytes(
            content,
            expected_sample_rate=source.native_sample_rate,
            contract=contract,
        )
        return record, source, ratios, sentinel_ratio

    # 입력 dataset_index 순서를 map이 보존하므로 worker scheduling과 무관하게
    # receipt qualified-source 배열과 예외 surface 순서가 결정론적이다.
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="stage2-admission-density",
    ) as executor:
        for record, source, ratios, sentinel_ratio in executor.map(
            analyze, ordered_indices
        ):
            entry = _coverage_entry(record, source)
            for band_index, ratio in enumerate(ratios):
                if ratio >= threshold:
                    octave[record.split][record.source_family][band_index].append(entry)
            if sentinel_ratio >= threshold:
                sentinel[record.split][record.source_family].append(entry)

    minimum = int(contract.minimum_groups_per_family_octave)
    for split in _SPLITS:
        for family in STAGE2_2KHZ_SOURCE_FAMILIES:
            for band_index, entries in enumerate(octave[split][family]):
                components = {str(entry["component_id"]) for entry in entries}
                if len(components) < minimum:
                    raise ValueError(
                        "Stage-2 actual source bytes의 split×family×octave "
                        f"distinct component가 4개 미만입니다: {split}/{family}/{band_index}"
                    )
            sentinel_components = {
                str(entry["component_id"]) for entry in sentinel[split][family]
            }
            if len(sentinel_components) < minimum:
                raise ValueError(
                    "Stage-2 actual source bytes의 split×family×1.6 kHz sentinel "
                    f"distinct component가 4개 미만입니다: {split}/{family}"
                )
    return octave, sentinel


def _validate_manifest_bundle(
    root: Path,
    *,
    path: str,
    sha256: str,
) -> tuple[
    dict[str, Any],
    tuple[Stage2SamplerRecord, ...],
    tuple[Stage2PretrainSource, ...],
    str,
]:
    _, content, actual_sha = _ref(
        root, path=path, sha256=sha256, label="Stage-2 public manifest bundle"
    )
    payload = _json(content, label="Stage-2 public manifest bundle")
    expected_keys = {
        "schema",
        "status",
        "canonical_pretrain_eligible",
        "control_band_contract",
        "required_source_families",
        "required_splits",
        "recorded_artifacts_required_for_pretrain",
        "test_split_for_checkpoint_selection_allowed",
        "source_inventory_commit_sha",
        "items",
    }
    if set(payload) != expected_keys:
        raise ValueError("Stage-2 public manifest bundle key 집합이 exact하지 않습니다")
    exact = {
        "schema": STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract": _contract_entry(),
        "required_source_families": list(STAGE2_2KHZ_SOURCE_FAMILIES),
        "required_splits": list(_SPLITS),
        "recorded_artifacts_required_for_pretrain": False,
        "test_split_for_checkpoint_selection_allowed": False,
    }
    for key, expected in exact.items():
        if payload[key] != expected:
            raise ValueError(f"Stage-2 public manifest {key}가 canonical 계약과 다릅니다")
    source_inventory_commit_sha = str(payload["source_inventory_commit_sha"])
    if not re.fullmatch(r"[0-9a-f]{40}", source_inventory_commit_sha):
        raise ValueError("Stage-2 manifest source inventory commit이 40자리 SHA가 아닙니다")
    items = payload["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("Stage-2 public manifest item이 비었습니다")
    records: list[Stage2SamplerRecord] = []
    sources: list[Stage2PretrainSource] = []
    seen_indices: set[int] = set()
    seen_paths: set[str] = set()
    component_splits: dict[str, set[str]] = {}
    source_sha_splits: dict[str, set[str]] = {}
    source_sha_components: dict[str, set[str]] = {}
    lineage_splits: dict[str, set[str]] = {}
    lineage_components: dict[str, set[str]] = {}
    component_families: dict[str, set[str]] = {}
    for row in items:
        keys = {
            "dataset_index",
            "source_family",
            "component_id",
            "split",
            "path",
            "content_sha256",
            "content_size",
            "native_sample_rate",
            "native_nyquist_hz",
            "lineage_keys",
        }
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError("Stage-2 public manifest row key 집합이 exact하지 않습니다")
        index = row["dataset_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index in seen_indices:
            raise ValueError("Stage-2 public manifest dataset_index가 잘못됐습니다")
        seen_indices.add(index)
        relative = str(row["path"])
        if relative in seen_paths:
            raise ValueError("Stage-2 public manifest path가 중복됩니다")
        seen_paths.add(relative)
        source_path, _, source_sha = _ref(
            root,
            path=relative,
            sha256=str(row["content_sha256"]),
            label=f"Stage-2 public source #{index}",
        )
        if int(row["content_size"]) != source_path.stat().st_size:
            raise ValueError("Stage-2 public manifest content_size가 실제 bytes와 다릅니다")
        rate = row["native_sample_rate"]
        nyquist = row["native_nyquist_hz"]
        if (
            isinstance(rate, bool)
            or not isinstance(rate, int)
            or rate < 5657
            or isinstance(nyquist, bool)
            or not isinstance(nyquist, (int, float))
            or not math.isfinite(float(nyquist))
            or float(nyquist) < 2828.4271247462
            or not math.isclose(float(nyquist), float(rate) / 2.0, abs_tol=1.0e-9)
        ):
            raise ValueError("Stage-2 public source native Nyquist가 2 kHz octave 상단 미만입니다")
        lineage = row["lineage_keys"]
        if not isinstance(lineage, list) or not lineage or not all(
            isinstance(value, str) and value for value in lineage
        ):
            raise ValueError("Stage-2 public source lineage_keys가 비었습니다")
        if lineage != sorted(set(lineage)):
            raise ValueError("Stage-2 public source lineage_keys는 sorted unique여야 합니다")
        split = str(row["split"])
        component = str(row["component_id"])
        component_splits.setdefault(component, set()).add(split)
        component_families.setdefault(component, set()).add(str(row["source_family"]))
        source_sha_splits.setdefault(source_sha, set()).add(split)
        source_sha_components.setdefault(source_sha, set()).add(component)
        for key in lineage:
            lineage_splits.setdefault(key, set()).add(split)
            lineage_components.setdefault(key, set()).add(component)
        records.append(
            Stage2SamplerRecord(
                dataset_index=index,
                source_family=str(row["source_family"]),
                component_id=component,
                split=split,
                source_sha256=source_sha,
            )
        )
        sources.append(
            Stage2PretrainSource(
                dataset_index=index,
                relative_path=relative,
                content_sha256=source_sha,
                native_sample_rate=int(rate),
            )
        )
    for label, mapping in (
        ("component", component_splits),
        ("source SHA", source_sha_splits),
        ("original lineage", lineage_splits),
    ):
        crossing = sorted(key for key, splits in mapping.items() if len(splits) > 1)
        if crossing:
            raise ValueError(
                f"Stage-2 public manifest {label}가 split을 가로지릅니다: {crossing[0]}"
            )
    duplicate_component_identity = sorted(
        source_sha
        for source_sha, components in source_sha_components.items()
        if len(components) > 1
    )
    if duplicate_component_identity:
        raise ValueError(
            "Stage-2 public manifest 동일 source bytes SHA가 여러 component_id로 "
            f"분할됐습니다: {duplicate_component_identity[0]}"
        )
    crossing_family = sorted(
        component
        for component, families in component_families.items()
        if len(families) > 1
    )
    if crossing_family:
        raise ValueError(
            "Stage-2 public manifest component가 source family를 가로지릅니다: "
            f"{crossing_family[0]}"
        )
    duplicate_lineage_identity = sorted(
        key for key, components in lineage_components.items() if len(components) > 1
    )
    if duplicate_lineage_identity:
        raise ValueError(
            "Stage-2 public manifest original lineage key가 여러 component_id에 있습니다: "
            f"{duplicate_lineage_identity[0]}"
        )
    cells = {
        (split, family): {
            record.component_id
            for record in records
            if record.split == split and record.source_family == family
        }
        for split in _SPLITS
        for family in STAGE2_2KHZ_SOURCE_FAMILIES
    }
    if any(len(components) < 4 for components in cells.values()):
        raise ValueError("Stage-2 public manifest split×family independent component가 4개 미만입니다")
    return payload, tuple(records), tuple(sources), actual_sha


def _validate_recorded_synthetic_lineage(
    root: Path,
    *,
    manifest_payload: Mapping[str, Any],
    lineage_receipt: Mapping[str, Any],
) -> None:
    """actual recorded holdout bytes와 manifest row의 transitive 교집합을 재계산한다.

    receipt의 zero count는 판정 권위가 아니다. holdout 전체 file SHA와
    ``clip_lineage`` semantic SHA를 같이 결속한 뒤, actual manifest가 제공하는
    path basename/content SHA/lineage key만으로 교집합을 다시 계산한다.
    """

    holdout_ref = lineage_receipt.get("recorded_holdout")
    if not isinstance(holdout_ref, dict) or set(holdout_ref) != {"path", "sha256"}:
        raise ValueError("Stage-2 lineage recorded_holdout ref가 exact하지 않습니다")
    if holdout_ref["path"] != _RECORDED_HOLDOUT_PATH:
        raise ValueError("Stage-2 lineage는 canonical recorded holdout 경로만 허용합니다")
    _, holdout_bytes, holdout_sha = _ref(
        root,
        path=str(holdout_ref["path"]),
        sha256=str(holdout_ref["sha256"]),
        label="Stage-2 recorded holdout",
    )
    holdout = _json(holdout_bytes, label="Stage-2 recorded holdout")
    clip_lineage = holdout.get("clip_lineage")
    families = holdout.get("families")
    if not isinstance(clip_lineage, dict) or not isinstance(families, dict):
        raise ValueError("Stage-2 recorded holdout clip_lineage/families가 없습니다")
    recorded_rows = validate_recorded_clip_lineage(
        clip_lineage,
        families=families,
    )
    clip_lineage_sha = _require_sha256(
        clip_lineage.get("clips_sha256"),
        label="Stage-2 recorded clip lineage SHA",
    )

    items = manifest_payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Stage-2 public manifest item이 비었습니다")
    actual_intersection = stage2_recorded_public_intersection(
        recorded_rows=recorded_rows,
        public_items=items,
    )
    declared_intersection = lineage_receipt.get(
        "recorded_synthetic_lineage_intersection_count"
    )
    if (
        isinstance(declared_intersection, bool)
        or not isinstance(declared_intersection, int)
        or declared_intersection != actual_intersection
    ):
        raise ValueError(
            "Stage-2 lineage receipt zero count가 actual recorded/synthetic 교집합과 다릅니다"
        )
    if actual_intersection != 0:
        raise ValueError("Stage-2 actual recorded/synthetic lineage 교집합이 0이 아닙니다")

    declared_clip_count = lineage_receipt.get("recorded_clip_count")
    if isinstance(declared_clip_count, bool) or not isinstance(
        declared_clip_count, int
    ):
        raise ValueError("Stage-2 lineage recorded clip count가 정수가 아닙니다")
    exact = {
        "recorded_holdout": {
            "path": _RECORDED_HOLDOUT_PATH,
            "sha256": holdout_sha,
        },
        "recorded_clip_count": len(recorded_rows),
        "recorded_clip_lineage_sha256": clip_lineage_sha,
        "recorded_synthetic_intersection_algorithm": (
            _RECORDED_SYNTHETIC_INTERSECTION_ALGORITHM
        ),
        "actual_recorded_holdout_bytes_consumed": True,
    }
    if any(lineage_receipt.get(key) != value for key, value in exact.items()):
        raise ValueError(
            "Stage-2 lineage receipt가 actual recorded holdout bytes/lineage와 다릅니다"
        )


def _validate_data_receipts(
    root: Path,
    *,
    manifest_payload: Mapping[str, Any],
    manifest_sha: str,
    manifest_items: int,
    manifest_records: Sequence[Stage2SamplerRecord],
    manifest_sources: Sequence[Stage2PretrainSource],
    plant_binding_file_sha256: str,
    lineage_ref: tuple[str, str],
    coverage_ref: tuple[str, str],
    bootstrap_ref: tuple[str, str],
    source_inventory_commit_sha: str,
    workers: int | None = None,
    require_stage2_physical_transfer: bool = False,
) -> tuple[str, str, str]:
    contract = Stage2TwoKilohertzContract.canonical()
    _, lineage_bytes, lineage_sha = _ref(
        root, path=lineage_ref[0], sha256=lineage_ref[1], label="Stage-2 lineage receipt"
    )
    lineage = _json(lineage_bytes, label="Stage-2 lineage receipt")
    expected_lineage = {
        "schema": STAGE2_PUBLIC_LINEAGE_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "manifest_bundle_sha256": manifest_sha,
        "verified_item_count": manifest_items,
        "component_cross_split_count": 0,
        "source_sha_cross_split_count": 0,
        "original_lineage_cross_split_count": 0,
        "recorded_synthetic_lineage_intersection_count": 0,
        "actual_manifest_rows_consumed": True,
        "recorded_holdout": lineage.get("recorded_holdout"),
        "recorded_clip_count": lineage.get("recorded_clip_count"),
        "recorded_clip_lineage_sha256": lineage.get(
            "recorded_clip_lineage_sha256"
        ),
        "recorded_synthetic_intersection_algorithm": (
            _RECORDED_SYNTHETIC_INTERSECTION_ALGORITHM
        ),
        "actual_recorded_holdout_bytes_consumed": True,
        "source_inventory_commit_sha": source_inventory_commit_sha,
    }
    if lineage != expected_lineage:
        raise ValueError("Stage-2 lineage receipt가 actual manifest/zero-intersection과 exact하지 않습니다")
    _validate_recorded_synthetic_lineage(
        root,
        manifest_payload=manifest_payload,
        lineage_receipt=lineage,
    )

    _, coverage_bytes, coverage_sha = _ref(
        root, path=coverage_ref[0], sha256=coverage_ref[1], label="Stage-2 coverage receipt"
    )
    coverage = _json(coverage_bytes, label="Stage-2 coverage receipt")
    expected_keys = {
        "schema",
        "status",
        "canonical_pretrain_eligible",
        "control_band_contract_sha256",
        "manifest_bundle_sha256",
        "actual_source_bytes_recomputed",
        "plant_binding_file_sha256",
        "source_density_algorithm",
        "octave_objective_bands_hz",
        "minimum_source_density_ratio",
        "minimum_independent_components_per_family_octave",
        "qualified_sources_by_split_family_octave",
        "one_point_six_khz_sentinel_band_hz",
        "qualified_sources_by_split_family_one_point_six_khz_sentinel",
        "source_inventory_commit_sha",
    }
    if not isinstance(coverage, dict) or set(coverage) != expected_keys:
        raise ValueError("Stage-2 coverage receipt key 집합이 exact하지 않습니다")
    exact_coverage = {
        "schema": STAGE2_PUBLIC_COVERAGE_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "manifest_bundle_sha256": manifest_sha,
        "actual_source_bytes_recomputed": True,
        "plant_binding_file_sha256": _require_sha256(
            plant_binding_file_sha256, label="coverage plant binding SHA"
        ),
        "source_density_algorithm": _SOURCE_DENSITY_ALGORITHM,
        "octave_objective_bands_hz": [
            [float(lower), float(upper)]
            for lower, upper in contract.octave_objective_bands_hz
        ],
        "minimum_source_density_ratio": 0.25,
        "minimum_independent_components_per_family_octave": 4,
        "one_point_six_khz_sentinel_band_hz": list(
            _ONE_POINT_SIX_KHZ_SENTINEL_BAND_HZ
        ),
        "source_inventory_commit_sha": source_inventory_commit_sha,
    }
    for key, expected in exact_coverage.items():
        if coverage[key] != expected:
            raise ValueError(f"Stage-2 coverage {key}가 canonical 계약과 다릅니다")
    qualified_octave, qualified_sentinel = _recompute_qualified_sources(
        root,
        records=manifest_records,
        sources=manifest_sources,
        workers=workers,
    )
    if coverage["qualified_sources_by_split_family_octave"] != qualified_octave:
        raise ValueError(
            "Stage-2 coverage octave qualified IDs가 actual manifest/source bytes와 다릅니다"
        )
    if (
        coverage[
            "qualified_sources_by_split_family_one_point_six_khz_sentinel"
        ]
        != qualified_sentinel
    ):
        raise ValueError(
            "Stage-2 coverage 1.6 kHz qualified IDs가 actual manifest/source bytes와 다릅니다"
        )

    _, bootstrap_bytes, bootstrap_sha = _ref(
        root,
        path=bootstrap_ref[0],
        sha256=bootstrap_ref[1],
        label="Stage-2 transfer bootstrap receipt",
    )
    bootstrap = _json(bootstrap_bytes, label="Stage-2 transfer bootstrap receipt")
    bootstrap_keys = {
        "schema",
        "status",
        "canonical_pretrain_eligible",
        "control_band_contract_sha256",
        "manifest_bundle_sha256",
        "elice_bootstrap_receipt",
        "existing_instance_cache_reused",
        "all_declared_source_bytes_rehashed",
        "stale_run_or_checkpoint_auto_resume_allowed",
        "scratch_new_run_directory_required",
        "source_inventory_commit_sha",
    }
    if set(bootstrap) != bootstrap_keys:
        raise ValueError("Stage-2 transfer/bootstrap receipt key 집합이 exact하지 않습니다")
    origin_ref = bootstrap.get("elice_bootstrap_receipt")
    if (
        not isinstance(origin_ref, dict)
        or set(origin_ref) != {"path", "sha256"}
        or not isinstance(origin_ref.get("path"), str)
        or not origin_ref["path"]
        or type(bootstrap.get("existing_instance_cache_reused")) is not bool
    ):
        raise ValueError("Stage-2 transfer/bootstrap Elice cache provenance ref가 exact하지 않습니다")
    origin_sha = _require_sha256(
        origin_ref.get("sha256"), label="Elice bootstrap receipt SHA"
    )
    try:
        origin_bootstrap = validate_elice_bootstrap_inputs(
            root,
            source_inventory_commit_sha=source_inventory_commit_sha,
            bootstrap_receipt_path=str(origin_ref["path"]),
            expected_bootstrap_receipt_sha256=origin_sha,
            require_stage2_physical_transfer=require_stage2_physical_transfer,
        )
    except ValueError as exc:
        raise ValueError(
            "Stage-2 transfer/bootstrap Elice cache provenance를 재검증할 수 없습니다"
        ) from exc
    if origin_bootstrap["bootstrap_receipt"]["path"] != origin_ref["path"]:
        raise ValueError(
            "Stage-2 transfer/bootstrap Elice bootstrap receipt path가 canonical relative path가 아닙니다"
        )
    if require_stage2_physical_transfer:
        transferred_physical = origin_bootstrap.get("stage2_2khz_physical_transfer")
        if (
            not isinstance(transferred_physical, Mapping)
            or set(transferred_physical) != {"plant_binding", "physical_authority"}
            or not isinstance(transferred_physical.get("plant_binding"), Mapping)
            or set(transferred_physical["plant_binding"]) != {"path", "sha256"}
            or not isinstance(transferred_physical["physical_authority"], Mapping)
            or set(transferred_physical["physical_authority"]) != {"path", "sha256"}
            or transferred_physical["plant_binding"].get("sha256")
            != _require_sha256(
                plant_binding_file_sha256,
                label="Stage-2 transfer physical plant binding SHA",
            )
        ):
            raise ValueError(
                "Stage-2 transfer/bootstrap physical typed role이 canonical plant binding과 다릅니다"
            )
    expected_bootstrap = {
        "schema": STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "manifest_bundle_sha256": manifest_sha,
        "elice_bootstrap_receipt": {
            "path": str(origin_ref["path"]),
            "sha256": origin_sha,
        },
        "existing_instance_cache_reused": bool(
            origin_bootstrap["archive_cache_reused"]
        ),
        "all_declared_source_bytes_rehashed": True,
        "stale_run_or_checkpoint_auto_resume_allowed": False,
        "scratch_new_run_directory_required": True,
        "source_inventory_commit_sha": source_inventory_commit_sha,
    }
    if bootstrap != expected_bootstrap:
        raise ValueError("Stage-2 transfer/bootstrap receipt가 idempotent cache/scratch 계약과 다릅니다")
    return lineage_sha, coverage_sha, bootstrap_sha


def _load_self_attested_stage2_pretrain_data_binding_for_test(
    *,
    repository_root: str | Path,
    manifest_ref: tuple[str, str],
    lineage_ref: tuple[str, str],
    coverage_ref: tuple[str, str],
    bootstrap_ref: tuple[str, str],
    plant_binding_file_sha256: str,
    workers: int | None = None,
    require_stage2_physical_transfer: bool = False,
) -> Stage2PretrainDataBinding:
    root = Path(repository_root).resolve(strict=True)
    manifest, records, sources, manifest_sha = _validate_manifest_bundle(
        root, path=manifest_ref[0], sha256=manifest_ref[1]
    )
    lineage_sha, coverage_sha, bootstrap_sha = _validate_data_receipts(
        root,
        manifest_payload=manifest,
        manifest_sha=manifest_sha,
        manifest_items=len(manifest["items"]),
        manifest_records=records,
        manifest_sources=sources,
        plant_binding_file_sha256=plant_binding_file_sha256,
        source_inventory_commit_sha=str(manifest["source_inventory_commit_sha"]),
        lineage_ref=lineage_ref,
        coverage_ref=coverage_ref,
        bootstrap_ref=bootstrap_ref,
        workers=workers,
        require_stage2_physical_transfer=require_stage2_physical_transfer,
    )
    return Stage2PretrainDataBinding(
        manifest_bundle_sha256=manifest_sha,
        lineage_receipt_sha256=lineage_sha,
        frequency_coverage_receipt_sha256=coverage_sha,
        transfer_bootstrap_receipt_sha256=bootstrap_sha,
        records=records,
        sources=sources,
    )


def validate_stage2_pretrain_data_candidate(
    *,
    repository_root: str | Path,
    manifest_ref: tuple[str, str],
    lineage_ref: tuple[str, str],
    coverage_ref: tuple[str, str],
    bootstrap_ref: tuple[str, str],
    plant_binding_file_sha256: str,
    workers: int | None = None,
) -> Stage2PretrainDataBinding:
    """issuer 직후 candidate bytes를 재검증하되 학습 authority는 부여하지 않는다.

    production issuer는 아직 Git에 review/commit되지 않은 새 artifact를 검사해야 하므로
    tracked authority loader를 호출할 수 없다. 이 함수는 동일한 actual source decode,
    holdout lineage와 receipt exact 검증을 모두 수행하지만 반환값만으로 trainer를 열 수
    없으며, 이후 :func:`load_stage2_pretrain_data_binding`의 tracked human authority가
    별도로 필요하다.
    """

    return _load_self_attested_stage2_pretrain_data_binding_for_test(
        repository_root=repository_root,
        manifest_ref=manifest_ref,
        lineage_ref=lineage_ref,
        coverage_ref=coverage_ref,
        bootstrap_ref=bootstrap_ref,
        plant_binding_file_sha256=plant_binding_file_sha256,
        workers=workers,
    )


def load_stage2_pretrain_data_binding(
    *,
    repository_root: str | Path,
    manifest_ref: tuple[str, str],
    lineage_ref: tuple[str, str],
    coverage_ref: tuple[str, str],
    bootstrap_ref: tuple[str, str],
    plant_binding_file_sha256: str,
) -> Stage2PretrainDataBinding:
    """human-reviewed clean Git anchor가 승인한 public corpus만 적재한다."""

    root = Path(repository_root).resolve(strict=True)
    authority, _, head = verify_tracked_head_authority(
        root, STAGE2_PUBLIC_DATA_AUTHORITY_PATH
    )
    expected_keys = {
        "schema",
        "authority_kind",
        "status",
        "source_inventory_commit_sha",
        "control_band_contract_sha256",
        "plant_binding_file_sha256",
        "manifest_bundle",
        "lineage_receipt",
        "frequency_coverage_receipt",
        "transfer_bootstrap_receipt",
    }
    if not isinstance(authority, dict) or set(authority) != expected_keys:
        raise ValueError("Stage-2 public-data Git authority key 집합이 exact하지 않습니다")
    contract = Stage2TwoKilohertzContract.canonical()
    if (
        authority["schema"] != STAGE2_PUBLIC_DATA_AUTHORITY_SCHEMA
        or authority["authority_kind"] != "human_reviewed_public_corpus"
        or authority["status"] != "APPROVED"
        or authority["control_band_contract_sha256"] != contract.digest()
        or not re.fullmatch(r"[0-9a-f]{40}", str(authority["source_inventory_commit_sha"]))
    ):
        raise ValueError("Stage-2 public-data Git authority 의미가 canonical과 다릅니다")
    verify_source_commit_ancestor(
        root, str(authority["source_inventory_commit_sha"]), head=head
    )
    declared = {
        "plant_binding_file_sha256": _require_sha256(
            plant_binding_file_sha256, label="data authority plant binding SHA"
        ),
        "manifest_bundle": {"path": manifest_ref[0], "sha256": manifest_ref[1]},
        "lineage_receipt": {"path": lineage_ref[0], "sha256": lineage_ref[1]},
        "frequency_coverage_receipt": {
            "path": coverage_ref[0],
            "sha256": coverage_ref[1],
        },
        "transfer_bootstrap_receipt": {
            "path": bootstrap_ref[0],
            "sha256": bootstrap_ref[1],
        },
    }
    if any(authority[key] != value for key, value in declared.items()):
        raise ValueError("Stage-2 public-data authority가 profile artifact refs와 다릅니다")
    if authority["source_inventory_commit_sha"] != str(
        _validate_manifest_bundle(
            root, path=manifest_ref[0], sha256=manifest_ref[1]
        )[0]["source_inventory_commit_sha"]
    ):
        raise ValueError("Stage-2 public-data authority source commit이 manifest와 다릅니다")
    return _load_self_attested_stage2_pretrain_data_binding_for_test(
        repository_root=repository_root,
        manifest_ref=manifest_ref,
        lineage_ref=lineage_ref,
        coverage_ref=coverage_ref,
        bootstrap_ref=bootstrap_ref,
        plant_binding_file_sha256=plant_binding_file_sha256,
        require_stage2_physical_transfer=True,
    )


def load_stage2_pretrain_typed_admission(
    *,
    repository_root: str | Path,
    primary_path_sha256: str,
    secondary_path_sha256: str,
    plant_binding_ref: tuple[str, str],
    manifest_ref: tuple[str, str],
    lineage_ref: tuple[str, str],
    coverage_ref: tuple[str, str],
    bootstrap_ref: tuple[str, str],
    criterion_receipt_ref: tuple[str, str],
) -> Stage2PretrainTypedAdmission:
    """typed P/S·data·criterion을 읽어 pretrain-only READY 객체를 만든다."""

    root = Path(repository_root).resolve(strict=True)
    pretrain_authority, _, head = verify_tracked_head_authority(
        root, STAGE2_PRETRAIN_AUTHORITY_PATH
    )
    pretrain_authority_keys = {
        "schema",
        "authority_kind",
        "status",
        "artifact_source_commit_sha",
        "control_band_contract_sha256",
        "plant_binding",
        "manifest_bundle",
        "lineage_receipt",
        "frequency_coverage_receipt",
        "transfer_bootstrap_receipt",
        "criterion_implementation_receipt",
    }
    contract = Stage2TwoKilohertzContract.canonical()
    if not isinstance(pretrain_authority, dict) or set(pretrain_authority) != pretrain_authority_keys:
        raise ValueError("Stage-2 pretrain Git authority key 집합이 exact하지 않습니다")
    if (
        pretrain_authority["schema"] != STAGE2_PRETRAIN_AUTHORITY_SCHEMA
        or pretrain_authority["authority_kind"] != "human_reviewed_pretrain_admission"
        or pretrain_authority["status"] != "APPROVED"
        or pretrain_authority["control_band_contract_sha256"] != contract.digest()
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(pretrain_authority["artifact_source_commit_sha"])
        )
    ):
        raise ValueError("Stage-2 pretrain Git authority 의미가 canonical과 다릅니다")
    verify_source_commit_ancestor(
        root, str(pretrain_authority["artifact_source_commit_sha"]), head=head
    )
    declared_refs = {
        "plant_binding": {"path": plant_binding_ref[0], "sha256": plant_binding_ref[1]},
        "manifest_bundle": {"path": manifest_ref[0], "sha256": manifest_ref[1]},
        "lineage_receipt": {"path": lineage_ref[0], "sha256": lineage_ref[1]},
        "frequency_coverage_receipt": {"path": coverage_ref[0], "sha256": coverage_ref[1]},
        "transfer_bootstrap_receipt": {"path": bootstrap_ref[0], "sha256": bootstrap_ref[1]},
        "criterion_implementation_receipt": {
            "path": criterion_receipt_ref[0],
            "sha256": criterion_receipt_ref[1],
        },
    }
    if any(pretrain_authority[key] != value for key, value in declared_refs.items()):
        raise ValueError("Stage-2 pretrain authority가 campaign artifact refs와 다릅니다")
    binding = load_stage2_2khz_plant_binding(
        plant_binding_ref[0],
        repository_root=root,
        expected_binding_file_sha256=plant_binding_ref[1],
    )
    if binding.primary_path_sha256 != _require_sha256(
        primary_path_sha256, label="profile primary path SHA"
    ) or binding.secondary_path_sha256 != _require_sha256(
        secondary_path_sha256, label="profile secondary path SHA"
    ):
        raise ValueError("Stage-2 typed binding P/S SHA가 duct profile과 다릅니다")
    data = load_stage2_pretrain_data_binding(
        repository_root=root,
        manifest_ref=manifest_ref,
        lineage_ref=lineage_ref,
        coverage_ref=coverage_ref,
        bootstrap_ref=bootstrap_ref,
        plant_binding_file_sha256=binding.binding_file_sha256,
    )
    _, criterion_bytes, criterion_sha = _ref(
        root,
        path=criterion_receipt_ref[0],
        sha256=criterion_receipt_ref[1],
        label="Stage-2 criterion implementation receipt",
    )
    criterion = _json(criterion_bytes, label="Stage-2 criterion implementation receipt")
    keys = {
        "schema",
        "status",
        "canonical_pretrain_eligible",
        "control_band_contract_sha256",
        "plant_binding_file_sha256",
        "manifest_bundle_sha256",
        "loss_implementation",
        "trainer_adapter_implementation",
        "scratch_runner_implementation",
        "sampler_receipt",
        "dnh_calibration_receipt",
        "model_config",
        "model_initial_state_sha256",
        "batch_size",
        "seed",
        "generic_stage1_loss_used",
        "full_octave_v3_loss_used",
    }
    if not isinstance(criterion, dict) or set(criterion) != keys:
        raise ValueError("Stage-2 criterion receipt key 집합이 exact하지 않습니다")
    exact_criterion = {
        "schema": STAGE2_CRITERION_IMPLEMENTATION_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "plant_binding_file_sha256": plant_binding_ref[1],
        "manifest_bundle_sha256": data.manifest_bundle_sha256,
        "generic_stage1_loss_used": False,
        "full_octave_v3_loss_used": False,
    }
    for key, expected in exact_criterion.items():
        if criterion[key] != expected:
            raise ValueError(f"Stage-2 criterion {key}가 profile/artifact와 다릅니다")

    implementation_specs = (
        (
            criterion["loss_implementation"],
            "src/deep_anc/losses/stage2_2khz_loss.py",
            STAGE2_2KHZ_LOSS_SCHEMA,
            "Stage-2 loss implementation",
        ),
        (
            criterion["trainer_adapter_implementation"],
            "src/deep_anc/train/stage2_2khz_execution.py",
            STAGE2_2KHZ_TRAINER_ADAPTER_SCHEMA,
            "Stage-2 trainer adapter implementation",
        ),
        (
            criterion["scratch_runner_implementation"],
            "src/deep_anc/train/stage2_2khz_runner.py",
            "stage2_2khz_scratch_pretrain_runner_v1",
            "Stage-2 scratch runner implementation",
        ),
    )
    for entry, expected_path, expected_schema, label in implementation_specs:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "schema"}:
            raise ValueError(f"{label} ref가 exact하지 않습니다")
        if entry["path"] != expected_path or entry["schema"] != expected_schema:
            raise ValueError(f"{label} path/schema가 다릅니다")
        _ref(root, path=entry["path"], sha256=entry["sha256"], label=label)

    sampler_ref = criterion["sampler_receipt"]
    dnh_ref = criterion["dnh_calibration_receipt"]
    criterion_model_config_ref = _artifact_ref(
        criterion["model_config"], label="Stage-2 criterion model config"
    )
    criterion_model_initial_state_sha = _require_sha256(
        criterion["model_initial_state_sha256"],
        label="Stage-2 criterion model initial state SHA",
    )
    criterion_seed = criterion["seed"]
    criterion_batch_size = criterion["batch_size"]
    if type(criterion_seed) is not int or int(criterion_seed) != 20260803:
        raise ValueError("Stage-2 criterion seed가 canonical과 다릅니다")
    if type(criterion_batch_size) is not int or int(criterion_batch_size) != 96:
        raise ValueError("Stage-2 criterion batch_size가 canonical과 다릅니다")
    for entry, label in ((sampler_ref, "sampler receipt"), (dnh_ref, "DNH receipt")):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"Stage-2 {label} ref가 exact하지 않습니다")
    _, sampler_bytes, sampler_sha = _ref(
        root,
        path=sampler_ref["path"],
        sha256=sampler_ref["sha256"],
        label="Stage-2 sampler receipt",
    )
    sampler = Stage2FamilyComponentBatchSampler(
        data.records,
        batch_size=criterion_batch_size,
        seed=criterion_seed,
        split="train",
        manifest_bundle_sha256=data.manifest_bundle_sha256,
        sampler_receipt_sha256=sampler_sha,
    )
    if _json(sampler_bytes, label="Stage-2 sampler receipt") != sampler.expected_receipt_payload():
        raise ValueError("Stage-2 sampler receipt가 실제 manifest records/algorithm과 다릅니다")

    _, dnh_bytes, dnh_sha = _ref(
        root,
        path=dnh_ref["path"],
        sha256=dnh_ref["sha256"],
        label="Stage-2 DNH calibration receipt",
    )
    dnh = _json(dnh_bytes, label="Stage-2 DNH calibration receipt")
    expected_dnh_keys = {
        "schema",
        "status",
        "canonical_pretrain_eligible",
        "control_band_contract_sha256",
        "plant_binding_file_sha256",
        "manifest_bundle_sha256",
        "sampler_receipt_sha256",
        "actual_causal_secondary_output",
        "actual_family_balanced_batch",
        "model_config_sha256",
        "model_initial_state_sha256",
        "lambda_dnh",
        "output_y_gradient_share",
        "calibration_batch",
    }
    if not isinstance(dnh, dict) or set(dnh) != expected_dnh_keys:
        raise ValueError("Stage-2 DNH receipt key 집합이 exact하지 않습니다")
    fixed_dnh = {
        "schema": STAGE2_DNH_CALIBRATION_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "plant_binding_file_sha256": plant_binding_ref[1],
        "manifest_bundle_sha256": data.manifest_bundle_sha256,
        "sampler_receipt_sha256": sampler_sha,
        "actual_causal_secondary_output": True,
        "actual_family_balanced_batch": True,
        "model_config_sha256": criterion_model_config_ref[1],
        "model_initial_state_sha256": criterion_model_initial_state_sha,
    }
    for key, expected in fixed_dnh.items():
        if dnh[key] != expected:
            raise ValueError(f"Stage-2 DNH {key}가 typed artifact와 다릅니다")
    gradient_share = float(dnh["output_y_gradient_share"])
    lambda_dnh = float(dnh["lambda_dnh"])
    if not 0.2 <= gradient_share <= 0.4 or not math.isfinite(lambda_dnh) or lambda_dnh <= 0.0:
        raise ValueError("Stage-2 DNH gradient share/lambda가 admission 범위 밖입니다")
    calibration_ref = dnh["calibration_batch"]
    if not isinstance(calibration_ref, dict) or set(calibration_ref) != {"path", "sha256"}:
        raise ValueError("Stage-2 DNH calibration batch ref가 exact하지 않습니다")
    _batch_path, batch_bytes, batch_sha = _ref(
        root,
        path=calibration_ref["path"],
        sha256=calibration_ref["sha256"],
        label="Stage-2 DNH actual calibration batch",
    )

    # issuer가 만든 숫자를 믿지 않는다. _ref의 nofollow/stable snapshot bytes에서
    # tensor metadata, stored S*y, 그리고 full-objective gradient를 다시 계산한다.
    # local import는 issuer↔admission schema import cycle을 import-time에 만들지 않는다.
    from .stage2_2khz_pretrain_issuer import (  # pylint: disable=import-outside-toplevel
        calibrate_dnh_from_reloaded_batch,
        load_calibration_batch_bytes,
    )

    snapshot = load_calibration_batch_bytes(batch_bytes, expected_sha256=batch_sha)
    expected_snapshot_metadata = {
        "control_band_contract_sha256": contract.digest(),
        "plant_binding_file_sha256": plant_binding_ref[1],
        "plant_binding_runtime_sha256": binding.digest(),
        "primary_path_sha256": binding.primary_path_sha256,
        "secondary_path_sha256": binding.secondary_path_sha256,
        "manifest_bundle_sha256": data.manifest_bundle_sha256,
        "sampler_receipt_sha256": sampler_sha,
    }
    for key, expected in expected_snapshot_metadata.items():
        if snapshot.metadata[key] != expected:
            raise ValueError(f"Stage-2 DNH calibration batch {key}가 typed artifact와 다릅니다")
    if int(np.asarray(snapshot.arrays["y_target"]).shape[0]) != criterion_batch_size:
        raise ValueError("Stage-2 DNH calibration batch size가 criterion과 다릅니다")
    _validate_calibration_model_binding(
        root=root,
        snapshot_metadata=snapshot.metadata,
        criterion_model_config_ref=criterion_model_config_ref,
        criterion_model_initial_state_sha256=criterion_model_initial_state_sha,
        criterion_seed=criterion_seed,
        criterion_batch_size=criterion_batch_size,
    )
    recalibrated = calibrate_dnh_from_reloaded_batch(snapshot, binding=binding)
    recalibrated_share = float(recalibrated["output_y_gradient_share"])
    recalibrated_lambda = float(recalibrated["lambda_dnh"])
    if not math.isclose(
        gradient_share, recalibrated_share, rel_tol=1e-10, abs_tol=1e-12
    ) or not math.isclose(
        lambda_dnh, recalibrated_lambda, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(
            "Stage-2 DNH receipt scalar가 actual calibration NPZ 재계산과 다릅니다"
        )
    loss_config = Stage2TwoKilohertzLossConfig(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        lambda_dnh=lambda_dnh,
        dnh_calibration_receipt_sha256=dnh_sha,
        dnh_observed_gradient_share=gradient_share,
        family_balanced_sampler_receipt_sha256=sampler_sha,
    )
    return Stage2PretrainTypedAdmission(
        plant_binding=binding,
        data_binding=data,
        sampler=sampler,
        loss_config=loss_config,
        criterion_receipt_sha256=criterion_sha,
        dnh_calibration_receipt_sha256=dnh_sha,
        sampler_receipt_sha256=sampler_sha,
    )


__all__ = [
    "STAGE2_CRITERION_IMPLEMENTATION_RECEIPT_SCHEMA",
    "STAGE2_DNH_CALIBRATION_RECEIPT_SCHEMA",
    "STAGE2_PRETRAIN_TYPED_ADMISSION_SCHEMA",
    "STAGE2_PRETRAIN_AUTHORITY_PATH",
    "STAGE2_PRETRAIN_AUTHORITY_SCHEMA",
    "STAGE2_PUBLIC_DATA_AUTHORITY_PATH",
    "STAGE2_PUBLIC_DATA_AUTHORITY_SCHEMA",
    "STAGE2_PUBLIC_COVERAGE_RECEIPT_SCHEMA",
    "STAGE2_PUBLIC_LINEAGE_RECEIPT_SCHEMA",
    "STAGE2_PUBLIC_MANIFEST_BUNDLE_SCHEMA",
    "STAGE2_TRANSFER_BOOTSTRAP_RECEIPT_SCHEMA",
    "Stage2PretrainDataBinding",
    "Stage2PretrainSource",
    "Stage2PretrainTypedAdmission",
    "load_stage2_pretrain_data_binding",
    "load_stage2_pretrain_typed_admission",
]
