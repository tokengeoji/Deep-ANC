"""Stage-2 scratch pretrain sampler/DNH/criterion 불변 artifact 발행기.

이 모듈은 admission validator가 신뢰할 JSON 숫자를 사람이 채우는 경로를 제공하지
않는다. 실제 family-balanced sampler가 고른 source bytes에서 만든 tensor와 measured
causal P/S adapter 출력만 NPZ로 먼저 봉인하고, 그 NPZ를 다시 연 뒤 ``S*y``를
재계산해 output-y gradient budget을 얻는다. JSON receipt는 그 재계산 뒤에만 발행된다.

GPU, audio device, network는 이 모듈의 책임이 아니다. production CLI는 CPU만 쓴다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_SOURCE_FAMILIES,
    Stage2TwoKilohertzContract,
)
from ..losses.broadband_loss import CausalFIRPath
from ..losses.stage2_2khz_loss import (
    STAGE2_2KHZ_LOSS_SCHEMA,
    Stage2TwoKilohertzLoss,
    Stage2TwoKilohertzLossConfig,
)
from .stage2_2khz_binding import Stage2TwoKilohertzPlantBinding
from .stage2_2khz_execution import (
    STAGE2_2KHZ_SAMPLER_SCHEMA,
    STAGE2_2KHZ_TRAINER_ADAPTER_SCHEMA,
    Stage2ActualBatchIdentity,
    Stage2FamilyComponentBatchSampler,
    Stage2PrefixResult,
    Stage2TensorBatch,
)
from .stage2_2khz_pretrain_admission import (
    STAGE2_CRITERION_IMPLEMENTATION_RECEIPT_SCHEMA,
    STAGE2_DNH_CALIBRATION_RECEIPT_SCHEMA,
)
from .stage2_2khz_runner import STAGE2_PRETRAIN_RUNNER_SCHEMA


STAGE2_DNH_CALIBRATION_BATCH_SCHEMA = "stage2_2khz_dnh_actual_tensor_batch_v1"
STAGE2_DNH_GRADIENT_ALGORITHM = (
    "actual_output_y_l2_ratio_base_vs_lambda_dnh_two_point_decomposition_v1"
)
STAGE2_PRETRAIN_ISSUER_SUMMARY_SCHEMA = "stage2_2khz_pretrain_issuer_summary_v1"
STAGE2_EXTERNAL_CONTRACT_SCHEMA = "stage2_2khz_external_experiment_contract_v2"

DNH_SHARE_MIN = 0.2
DNH_SHARE_TARGET = 0.3
DNH_SHARE_MAX = 0.4

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{label}는 lowercase SHA-256이어야 합니다")
    return text


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - OS contract guard
                raise OSError("artifact write가 전진하지 않았습니다")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(content).hexdigest()


def publish_json_no_replace(path: str | Path, payload: Mapping[str, Any]) -> str:
    """canonical JSON을 O_EXCL로 발행한다."""

    content = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return _write_exclusive(Path(path), content)


def _tensor_array(value: torch.Tensor, *, label: str) -> np.ndarray:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or int(value.shape[0]) < 1
        or int(value.shape[1]) < 1
        or int(value.shape[-1]) < 1
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label}는 finite floating [B,C,T] tensor여야 합니다")
    return np.ascontiguousarray(value.detach().cpu().float().numpy(), dtype=np.float32)


@dataclass(frozen=True)
class Stage2DNHCalibrationSnapshot:
    """reload 뒤 gradient 계산에 사용하는 actual tensor snapshot."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        required = {
            "x_prefix",
            "x_target",
            "clean_playback_timeline",
            "y_prefix",
            "y_target",
            "primary_target",
            "secondary_target_observed",
            "dataset_indices",
            "global_sample_indices",
            "augmentation_seeds",
        }
        if set(self.arrays) != required:
            raise ValueError("Stage-2 calibration array key 집합이 exact하지 않습니다")
        metadata_keys = {
            "schema",
            "control_band_contract_sha256",
            "plant_binding_file_sha256",
            "plant_binding_runtime_sha256",
            "primary_path_sha256",
            "secondary_path_sha256",
            "manifest_bundle_sha256",
            "sampler_receipt_sha256",
            "model_config_sha256",
            "model_initial_state_sha256",
            "global_step",
            "source_families",
            "component_ids",
            "source_sha256",
            "manifest_row_sha256",
            "training_timing_contract_sha256",
            "prefix_samples",
            "target_samples",
            "array_sha256",
        }
        if set(self.metadata) != metadata_keys:
            raise ValueError("Stage-2 calibration metadata key 집합이 exact하지 않습니다")
        contract = Stage2TwoKilohertzContract.canonical()
        if (
            self.metadata["schema"] != STAGE2_DNH_CALIBRATION_BATCH_SCHEMA
            or self.metadata["control_band_contract_sha256"] != contract.digest()
        ):
            raise ValueError("Stage-2 calibration contract/schema가 canonical과 다릅니다")
        for label in (
            "plant_binding_file_sha256",
            "plant_binding_runtime_sha256",
            "primary_path_sha256",
            "secondary_path_sha256",
            "manifest_bundle_sha256",
            "sampler_receipt_sha256",
            "model_config_sha256",
            "model_initial_state_sha256",
            "training_timing_contract_sha256",
        ):
            _require_sha256(self.metadata[label], label=f"calibration {label}")
        array_hashes = self.metadata["array_sha256"]
        if not isinstance(array_hashes, Mapping) or set(array_hashes) != set(required):
            raise ValueError("Stage-2 calibration array SHA map이 exact하지 않습니다")
        for name, array in self.arrays.items():
            if _array_sha256(np.asarray(array)) != array_hashes[name]:
                raise ValueError(f"Stage-2 calibration {name} bytes SHA가 다릅니다")
        for name in (
            "x_prefix",
            "x_target",
            "clean_playback_timeline",
            "y_prefix",
            "y_target",
            "primary_target",
            "secondary_target_observed",
        ):
            values = np.asarray(self.arrays[name])
            if values.dtype != np.dtype("float32") or values.ndim != 3:
                raise ValueError(f"Stage-2 calibration {name}는 float32 [B,C,T]여야 합니다")
            if not bool(np.all(np.isfinite(values))):
                raise ValueError(f"Stage-2 calibration {name}에 non-finite 값이 있습니다")
        batch = int(np.asarray(self.arrays["y_target"]).shape[0])
        families = tuple(str(value) for value in self.metadata["source_families"])
        if len(families) != batch:
            raise ValueError("Stage-2 calibration family/tensor batch가 다릅니다")
        for key in ("component_ids", "source_sha256", "manifest_row_sha256"):
            values = tuple(str(value) for value in self.metadata[key])
            if len(values) != batch:
                raise ValueError(f"Stage-2 calibration {key}/tensor batch가 다릅니다")
            if key.endswith("sha256"):
                for value in values:
                    _require_sha256(value, label=f"calibration {key}")
        counts = Counter(families)
        expected = set(STAGE2_2KHZ_SOURCE_FAMILIES)
        if set(counts) != expected or len(set(counts.values())) != 1:
            raise ValueError("Stage-2 calibration batch가 actual family-balanced가 아닙니다")
        identity_arrays = (
            "dataset_indices",
            "global_sample_indices",
            "augmentation_seeds",
        )
        for name in identity_arrays:
            values = np.asarray(self.arrays[name])
            if values.ndim != 1 or int(values.size) != batch:
                raise ValueError(f"Stage-2 calibration {name} batch 길이가 다릅니다")
            if values.dtype.kind not in {"i", "u"}:
                raise ValueError(f"Stage-2 calibration {name}는 integer dtype이어야 합니다")
            if values.dtype.kind == "i" and bool(np.any(values < 0)):
                raise ValueError(f"Stage-2 calibration {name}에 음수가 있습니다")
        if len(set(np.asarray(self.arrays["dataset_indices"]).tolist())) != batch:
            raise ValueError("Stage-2 calibration dataset_indices가 중복됩니다")
        if len(set(np.asarray(self.arrays["global_sample_indices"]).tolist())) != batch:
            raise ValueError("Stage-2 calibration global_sample_indices가 중복됩니다")
        if len(set(np.asarray(self.arrays["augmentation_seeds"]).tolist())) != batch:
            raise ValueError("Stage-2 calibration augmentation_seeds가 중복됩니다")
        if len(set(self.metadata["manifest_row_sha256"])) != batch:
            raise ValueError("Stage-2 calibration manifest_row_sha256가 중복됩니다")
        prefix = int(self.metadata["prefix_samples"])
        target = int(self.metadata["target_samples"])
        for name in ("x_prefix", "y_prefix"):
            if np.asarray(self.arrays[name]).shape[-1] != prefix:
                raise ValueError(f"Stage-2 calibration {name} prefix 길이가 다릅니다")
        for name in (
            "x_target",
            "y_target",
            "primary_target",
            "secondary_target_observed",
        ):
            if np.asarray(self.arrays[name]).shape[-1] != target:
                raise ValueError(f"Stage-2 calibration {name} target 길이가 다릅니다")


def model_initial_state_sha256(model: torch.nn.Module) -> str:
    """serializer metadata에 의존하지 않는 initialized parameter/buffer digest."""

    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(list(value.shape)))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def snapshot_actual_stage2_batch(
    *,
    tensor_batch: Stage2TensorBatch,
    identity: Stage2ActualBatchIdentity,
    causal_result: Stage2PrefixResult,
    binding: Stage2TwoKilohertzPlantBinding,
    model_config_sha256: str,
    model_initial_state_sha256_value: str,
) -> Stage2DNHCalibrationSnapshot:
    """existing loader/model/measured adapter가 만든 실제 tensor를 immutable payload로 만든다."""

    if identity.dataset_indices != tensor_batch.dataset_indices:
        raise ValueError("Stage-2 calibration sampler/tensor dataset index가 다릅니다")
    if identity.manifest_row_sha256 != tensor_batch.manifest_row_sha256:
        raise ValueError("Stage-2 calibration sampler/tensor manifest row가 다릅니다")
    if identity.augmentation_seeds != tensor_batch.augmentation_seeds:
        raise ValueError("Stage-2 calibration sampler/tensor augmentation seed가 다릅니다")
    if causal_result.binding_sha256 != binding.digest():
        raise ValueError("Stage-2 calibration causal result/binding이 다릅니다")
    causal = tensor_batch.causal
    arrays: dict[str, np.ndarray] = {
        "x_prefix": _tensor_array(causal.x_prefix, label="x_prefix"),
        "x_target": _tensor_array(causal.x_target, label="x_target"),
        "clean_playback_timeline": _tensor_array(
            causal.clean_playback_timeline, label="clean_playback_timeline"
        ),
        "y_prefix": _tensor_array(causal_result.y_prefix, label="y_prefix"),
        "y_target": _tensor_array(causal_result.y_target, label="y_target"),
        "primary_target": _tensor_array(
            causal_result.primary_target, label="primary_target"
        ),
        "secondary_target_observed": _tensor_array(
            causal_result.secondary_target, label="secondary_target_observed"
        ),
        "dataset_indices": np.asarray(identity.dataset_indices, dtype="<i8"),
        "global_sample_indices": np.asarray(
            identity.global_sample_indices, dtype="<i8"
        ),
        "augmentation_seeds": np.asarray(identity.augmentation_seeds, dtype="<u8"),
    }
    metadata: dict[str, Any] = {
        "schema": STAGE2_DNH_CALIBRATION_BATCH_SCHEMA,
        "control_band_contract_sha256": binding.control_band_contract_sha256,
        "plant_binding_file_sha256": binding.binding_file_sha256,
        "plant_binding_runtime_sha256": binding.digest(),
        "primary_path_sha256": binding.primary_path_sha256,
        "secondary_path_sha256": binding.secondary_path_sha256,
        "manifest_bundle_sha256": identity.manifest_bundle_sha256,
        "sampler_receipt_sha256": identity.sampler_receipt_sha256,
        "model_config_sha256": _require_sha256(
            model_config_sha256, label="model config SHA"
        ),
        "model_initial_state_sha256": _require_sha256(
            model_initial_state_sha256_value, label="model initial state SHA"
        ),
        "global_step": int(identity.global_step),
        "source_families": list(identity.source_families),
        "component_ids": list(identity.component_ids),
        "source_sha256": list(identity.source_sha256),
        "manifest_row_sha256": list(identity.manifest_row_sha256),
        "training_timing_contract_sha256": (
            causal.training_timing_contract_sha256
        ),
        "prefix_samples": int(causal_result.prefix_samples),
        "target_samples": int(causal_result.target_samples),
        "array_sha256": {
            name: _array_sha256(value) for name, value in arrays.items()
        },
    }
    return Stage2DNHCalibrationSnapshot(arrays=arrays, metadata=metadata)


def publish_calibration_batch_no_replace(
    path: str | Path, snapshot: Stage2DNHCalibrationSnapshot
) -> str:
    """actual tensor NPZ를 O_EXCL로 쓰고 file SHA를 반환한다."""

    if not isinstance(snapshot, Stage2DNHCalibrationSnapshot):
        raise TypeError("Stage2DNHCalibrationSnapshot이 필요합니다")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        **snapshot.arrays,
        metadata_json=np.frombuffer(_canonical_json(snapshot.metadata), dtype=np.uint8),
    )
    return _write_exclusive(Path(path), buffer.getvalue())


def load_calibration_batch(
    path: str | Path, *, expected_sha256: str | None = None
) -> Stage2DNHCalibrationSnapshot:
    """allow_pickle 없이 NPZ를 reload하고 모든 array hash를 재검증한다."""

    target = Path(path)
    return load_calibration_batch_bytes(
        target.read_bytes(), expected_sha256=expected_sha256
    )


def load_calibration_batch_bytes(
    content: bytes, *, expected_sha256: str | None = None
) -> Stage2DNHCalibrationSnapshot:
    """이미 nofollow snapshot한 NPZ bytes를 decode해 DNH 입력을 재검증한다."""

    if not isinstance(content, bytes):
        raise TypeError("Stage-2 calibration NPZ content는 bytes여야 합니다")
    actual_sha = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and actual_sha != _require_sha256(
        expected_sha256, label="calibration batch file SHA"
    ):
        raise ValueError("Stage-2 calibration batch file SHA가 다릅니다")
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            expected = {
                "x_prefix",
                "x_target",
                "clean_playback_timeline",
                "y_prefix",
                "y_target",
                "primary_target",
                "secondary_target_observed",
                "dataset_indices",
                "global_sample_indices",
                "augmentation_seeds",
                "metadata_json",
            }
            if set(archive.files) != expected:
                raise ValueError("Stage-2 calibration NPZ member 집합이 exact하지 않습니다")
            metadata_raw = np.asarray(archive["metadata_json"], dtype=np.uint8)
            metadata = json.loads(metadata_raw.tobytes().decode("utf-8"))
            arrays = {
                name: np.ascontiguousarray(archive[name])
                for name in expected - {"metadata_json"}
            }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Stage-2 calibration NPZ를 안전하게 decode할 수 없습니다") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Stage-2 calibration metadata root는 mapping이어야 합니다")
    return Stage2DNHCalibrationSnapshot(arrays=arrays, metadata=metadata)


def _criterion_for_lambda(
    *, lambda_dnh: float, sampler_sha256: str, observed_share: float = DNH_SHARE_TARGET
) -> Stage2TwoKilohertzLoss:
    contract = Stage2TwoKilohertzContract.canonical()
    config = Stage2TwoKilohertzLossConfig(
        control_band_contract=contract,
        control_band_contract_sha256=contract.digest(),
        lambda_dnh=float(lambda_dnh),
        dnh_calibration_receipt_sha256="0" * 64,
        dnh_observed_gradient_share=float(observed_share),
        family_balanced_sampler_receipt_sha256=sampler_sha256,
    )
    return Stage2TwoKilohertzLoss(config)


def _gradient_at_lambda(
    *,
    lambda_dnh: float,
    y_prefix: torch.Tensor,
    y_target_value: torch.Tensor,
    primary_target: torch.Tensor,
    secondary_path: CausalFIRPath,
    source_families: Sequence[str],
    sampler_sha256: str,
) -> torch.Tensor:
    y_target = y_target_value.detach().clone().requires_grad_(True)
    y_full = torch.cat((y_prefix.detach(), y_target), dim=-1)
    prefix = int(y_prefix.shape[-1])
    secondary_target = secondary_path(y_full)[..., prefix:]
    criterion = _criterion_for_lambda(
        lambda_dnh=float(lambda_dnh), sampler_sha256=sampler_sha256
    )
    objective, _ = criterion(
        y_target,
        primary_target,
        secondary_target,
        source_families=source_families,
    )
    (gradient,) = torch.autograd.grad(objective, y_target, create_graph=False)
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("Stage-2 DNH calibration gradient가 non-finite입니다")
    return gradient.detach()


def calibrate_dnh_from_reloaded_batch(
    snapshot: Stage2DNHCalibrationSnapshot,
    *,
    binding: Stage2TwoKilohertzPlantBinding,
) -> dict[str, Any]:
    """actual NPZ tensor와 measured causal S로 λ/gradient share를 재계산한다.

    두 full-objective gradient ``g(λ=1)``, ``g(λ=2)``에서 선형 분해한다.
    ``g_base=2g(1)-g(2)``, ``g_dnh=g(2)-g(1)``이므로 별도 복제한 loss 식이나
    self-attested scalar가 계산에 들어오지 않는다.
    """

    if snapshot.metadata["plant_binding_file_sha256"] != binding.binding_file_sha256:
        raise ValueError("Stage-2 calibration batch/plant binding file SHA가 다릅니다")
    if snapshot.metadata["plant_binding_runtime_sha256"] != binding.digest():
        raise ValueError("Stage-2 calibration batch/runtime binding digest가 다릅니다")
    if snapshot.metadata["secondary_path_sha256"] != binding.secondary_path_sha256:
        raise ValueError("Stage-2 calibration batch/measured S SHA가 다릅니다")
    arrays = snapshot.arrays
    y_prefix = torch.from_numpy(np.asarray(arrays["y_prefix"], dtype=np.float32))
    y_target = torch.from_numpy(np.asarray(arrays["y_target"], dtype=np.float32))
    primary = torch.from_numpy(
        np.asarray(arrays["primary_target"], dtype=np.float32)
    )
    observed_secondary = torch.from_numpy(
        np.asarray(arrays["secondary_target_observed"], dtype=np.float32)
    )
    secondary_path = CausalFIRPath(binding.secondary_operator)
    with torch.no_grad():
        recomputed_secondary = secondary_path(torch.cat((y_prefix, y_target), dim=-1))[
            ..., int(y_prefix.shape[-1]) :
        ]
    if not torch.allclose(
        recomputed_secondary,
        observed_secondary,
        rtol=1.0e-6,
        atol=2.0e-7,
    ):
        maximum = float((recomputed_secondary - observed_secondary).abs().max())
        raise ValueError(
            "Stage-2 calibration stored S*y와 measured S reload 결과가 다릅니다: "
            f"max_abs={maximum:.9g}"
        )
    families = tuple(str(value) for value in snapshot.metadata["source_families"])
    sampler_sha = str(snapshot.metadata["sampler_receipt_sha256"])
    g_one = _gradient_at_lambda(
        lambda_dnh=1.0,
        y_prefix=y_prefix,
        y_target_value=y_target,
        primary_target=primary,
        secondary_path=secondary_path,
        source_families=families,
        sampler_sha256=sampler_sha,
    )
    g_two = _gradient_at_lambda(
        lambda_dnh=2.0,
        y_prefix=y_prefix,
        y_target_value=y_target,
        primary_target=primary,
        secondary_path=secondary_path,
        source_families=families,
        sampler_sha256=sampler_sha,
    )
    g_base = 2.0 * g_one - g_two
    g_dnh_unit = g_two - g_one
    base_norm = float(torch.linalg.vector_norm(g_base.double()))
    dnh_unit_norm = float(torch.linalg.vector_norm(g_dnh_unit.double()))
    if (
        not math.isfinite(base_norm)
        or not math.isfinite(dnh_unit_norm)
        or base_norm <= 0.0
        or dnh_unit_norm <= 0.0
    ):
        raise ValueError(
            "Stage-2 DNH calibration base/DNH output-y gradient가 0 또는 non-finite입니다"
        )
    candidates: list[dict[str, float]] = []
    for target_share in (DNH_SHARE_MIN, DNH_SHARE_TARGET, DNH_SHARE_MAX):
        candidate_lambda = target_share * base_norm / dnh_unit_norm
        if not math.isfinite(candidate_lambda) or candidate_lambda <= 0.0:
            raise ValueError("Stage-2 DNH candidate lambda가 finite 양수가 아닙니다")
        # 추천 숫자를 믿지 않고 같은 full objective를 이 λ와 2λ에서 다시 실행한다.
        g_lambda = _gradient_at_lambda(
            lambda_dnh=candidate_lambda,
            y_prefix=y_prefix,
            y_target_value=y_target,
            primary_target=primary,
            secondary_path=secondary_path,
            source_families=families,
            sampler_sha256=sampler_sha,
        )
        g_double = _gradient_at_lambda(
            lambda_dnh=2.0 * candidate_lambda,
            y_prefix=y_prefix,
            y_target_value=y_target,
            primary_target=primary,
            secondary_path=secondary_path,
            source_families=families,
            sampler_sha256=sampler_sha,
        )
        recomputed_base = 2.0 * g_lambda - g_double
        recomputed_weighted_dnh = g_double - g_lambda
        measured_base_norm = float(torch.linalg.vector_norm(recomputed_base.double()))
        measured_dnh_norm = float(
            torch.linalg.vector_norm(recomputed_weighted_dnh.double())
        )
        share = measured_dnh_norm / measured_base_norm
        if not math.isclose(share, target_share, rel_tol=2.0e-4, abs_tol=2.0e-6):
            raise ValueError(
                "Stage-2 DNH candidate 실제 재계산 share가 target과 다릅니다: "
                f"target={target_share}, actual={share}"
            )
        candidates.append(
            {
                "target_share": float(target_share),
                "lambda_dnh": float(candidate_lambda),
                "output_y_gradient_share": float(share),
                "base_gradient_l2": float(measured_base_norm),
                "weighted_dnh_gradient_l2": float(measured_dnh_norm),
            }
        )
    selected = candidates[1]
    if not DNH_SHARE_MIN <= selected["output_y_gradient_share"] <= DNH_SHARE_MAX:
        raise ValueError("Stage-2 selected DNH share가 admission 범위 밖입니다")
    return {
        "algorithm": STAGE2_DNH_GRADIENT_ALGORITHM,
        "gradient_domain": "model_output_y_target_after_measured_causal_secondary_path",
        "gradient_norm": "l2",
        "base_objective": "stage2_full_objective_with_lambda_dnh_removed",
        "candidate_recomputed_from_full_objective": True,
        "candidate_results": candidates,
        "selected_target_share": DNH_SHARE_TARGET,
        "lambda_dnh": selected["lambda_dnh"],
        "output_y_gradient_share": selected["output_y_gradient_share"],
    }


def build_dnh_receipt(
    *,
    snapshot: Stage2DNHCalibrationSnapshot,
    calibration_batch_path: str,
    calibration_batch_sha256: str,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """actual batch model provenance까지 묶는 typed-admission DNH receipt를 만든다."""

    _require_sha256(calibration_batch_sha256, label="DNH calibration batch SHA")
    if not isinstance(calibration_batch_path, str) or not calibration_batch_path:
        raise ValueError("DNH calibration batch path가 비었습니다")
    share = float(calibration["output_y_gradient_share"])
    lambda_dnh = float(calibration["lambda_dnh"])
    if not DNH_SHARE_MIN <= share <= DNH_SHARE_MAX:
        raise ValueError("DNH receipt share가 [0.2,0.4] 밖입니다")
    if not math.isfinite(lambda_dnh) or lambda_dnh <= 0.0:
        raise ValueError("DNH receipt lambda가 finite 양수가 아닙니다")
    return {
        "schema": STAGE2_DNH_CALIBRATION_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": snapshot.metadata[
            "control_band_contract_sha256"
        ],
        "plant_binding_file_sha256": snapshot.metadata[
            "plant_binding_file_sha256"
        ],
        "manifest_bundle_sha256": snapshot.metadata["manifest_bundle_sha256"],
        "sampler_receipt_sha256": snapshot.metadata["sampler_receipt_sha256"],
        "actual_causal_secondary_output": True,
        "actual_family_balanced_batch": True,
        # output-y gradient은 model output에서 계산된다. NPZ metadata의 model
        # config/state binding을 receipt에도 반복해 typed admission이 JSON 숫자만
        # 바꾼 우회를 막고 actual model을 다시 초기화할 수 있게 한다.
        "model_config_sha256": snapshot.metadata["model_config_sha256"],
        "model_initial_state_sha256": snapshot.metadata[
            "model_initial_state_sha256"
        ],
        "lambda_dnh": lambda_dnh,
        "output_y_gradient_share": share,
        "calibration_batch": {
            "path": calibration_batch_path,
            "sha256": calibration_batch_sha256,
        },
    }


def build_criterion_receipt(
    *,
    repository_root: str | Path,
    plant_binding_file_sha256: str,
    manifest_bundle_sha256: str,
    sampler_receipt_path: str,
    sampler_receipt_sha256: str,
    dnh_receipt_path: str,
    dnh_receipt_sha256: str,
    model_config_path: str,
    model_config_sha256: str,
    model_initial_state_sha256_value: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """실제 구현 bytes와 sampler/DNH/model scratch state를 묶는 criterion receipt."""

    root = Path(repository_root).resolve(strict=True)
    implementations = (
        (
            "loss_implementation",
            "src/deep_anc/losses/stage2_2khz_loss.py",
            STAGE2_2KHZ_LOSS_SCHEMA,
        ),
        (
            "trainer_adapter_implementation",
            "src/deep_anc/train/stage2_2khz_execution.py",
            STAGE2_2KHZ_TRAINER_ADAPTER_SCHEMA,
        ),
        (
            "scratch_runner_implementation",
            "src/deep_anc/train/stage2_2khz_runner.py",
            STAGE2_PRETRAIN_RUNNER_SCHEMA,
        ),
    )
    refs: dict[str, dict[str, str]] = {}
    for key, relative, schema in implementations:
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"Stage-2 implementation regular file이 없습니다: {relative}")
        refs[key] = {
            "path": relative,
            "sha256": _file_sha256(target),
            "schema": schema,
        }
    contract = Stage2TwoKilohertzContract.canonical()
    if not isinstance(model_config_path, str) or not model_config_path:
        raise ValueError("criterion model config path가 비었습니다")
    return {
        "schema": STAGE2_CRITERION_IMPLEMENTATION_RECEIPT_SCHEMA,
        "status": "PASS",
        "canonical_pretrain_eligible": True,
        "control_band_contract_sha256": contract.digest(),
        "plant_binding_file_sha256": _require_sha256(
            plant_binding_file_sha256, label="criterion plant binding SHA"
        ),
        "manifest_bundle_sha256": _require_sha256(
            manifest_bundle_sha256, label="criterion manifest SHA"
        ),
        **refs,
        "sampler_receipt": {
            "path": str(sampler_receipt_path),
            "sha256": _require_sha256(
                sampler_receipt_sha256, label="criterion sampler SHA"
            ),
        },
        "dnh_calibration_receipt": {
            "path": str(dnh_receipt_path),
            "sha256": _require_sha256(
                dnh_receipt_sha256, label="criterion DNH SHA"
            ),
        },
        "model_config": {
            "path": str(model_config_path),
            "sha256": _require_sha256(
                model_config_sha256, label="criterion model config SHA"
            ),
        },
        "model_initial_state_sha256": _require_sha256(
            model_initial_state_sha256_value,
            label="criterion model initial state SHA",
        ),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "generic_stage1_loss_used": False,
        "full_octave_v3_loss_used": False,
    }


def build_sampler_receipt(
    sampler: Stage2FamilyComponentBatchSampler,
) -> dict[str, Any]:
    payload = sampler.expected_receipt_payload()
    if payload.get("schema") != STAGE2_2KHZ_SAMPLER_SCHEMA:
        raise ValueError("Stage-2 sampler receipt schema가 다릅니다")
    return payload


def build_canonical_pretrain_external_contract(
    *,
    artifact_source_commit_sha: str,
    profile_sha256: Mapping[str, str],
    plant_sha256: Mapping[str, str],
    manifest_bundle_sha256: str,
    criterion_receipt_sha256: str,
) -> dict[str, Any]:
    """profile refs가 commit된 뒤 발행하는 scratch external contract payload."""

    commit = str(artifact_source_commit_sha)
    if not _COMMIT.fullmatch(commit):
        raise ValueError("external artifact source commit은 lowercase 40-hex여야 합니다")
    required_profiles = {
        "duct",
        "data",
        "evaluation",
        "canonical_pretrain",
        "canonical_finetune",
    }
    if set(profile_sha256) != required_profiles:
        raise ValueError("external profile SHA map key 집합이 exact하지 않습니다")
    required_plant = {
        "primary_path_sha256",
        "secondary_path_sha256",
        "plant_binding_sha256",
    }
    if set(plant_sha256) != required_plant:
        raise ValueError("external plant SHA map key 집합이 exact하지 않습니다")
    for key, value in {**profile_sha256, **plant_sha256}.items():
        _require_sha256(value, label=f"external {key}")
    contract = Stage2TwoKilohertzContract.canonical()
    return {
        "schema": STAGE2_EXTERNAL_CONTRACT_SCHEMA,
        "stage": "canonical_pretrain",
        "artifact_source_commit_sha": commit,
        "repository_clean_required": True,
        "control_band_contract": {
            "id": contract.contract_id,
            "sha256": contract.digest(),
        },
        "profile_sha256": dict(profile_sha256),
        "plant_sha256": dict(plant_sha256),
        "manifest_bundle_sha256": _require_sha256(
            manifest_bundle_sha256, label="external manifest SHA"
        ),
        "criterion_receipt_sha256": _require_sha256(
            criterion_receipt_sha256, label="external criterion SHA"
        ),
        "initialization_mode": "scratch",
        "init_checkpoint_sha256": None,
        "scratch_pretrain_origin_required": True,
        "legacy_artifacts_allowed": False,
        "automatic_resume_allowed": False,
        "training_eligible": True,
    }


__all__ = [
    "DNH_SHARE_MAX",
    "DNH_SHARE_MIN",
    "DNH_SHARE_TARGET",
    "STAGE2_DNH_CALIBRATION_BATCH_SCHEMA",
    "STAGE2_DNH_GRADIENT_ALGORITHM",
    "Stage2DNHCalibrationSnapshot",
    "build_canonical_pretrain_external_contract",
    "build_criterion_receipt",
    "build_dnh_receipt",
    "build_sampler_receipt",
    "calibrate_dnh_from_reloaded_batch",
    "load_calibration_batch",
    "load_calibration_batch_bytes",
    "model_initial_state_sha256",
    "publish_calibration_batch_no_replace",
    "publish_json_no_replace",
    "snapshot_actual_stage2_batch",
]
