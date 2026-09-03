#!/usr/bin/env python3
"""체크포인트 → 스트리밍 ONNX 내보내기 + ONNX Runtime(CPU) 등가성 검증.

규약: opset 17, 배치 1, 정적 shape, 상태 전부 명시 I/O, 블록 256(2프레임 내부 언롤).

  .venv/bin/python scripts/train/export_onnx.py --ckpt runs/pretrain_base_corrected/ckpt/best.pt \
      --out runs/export/model.onnx
"""

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.models import build_model                                   # noqa: E402
from deep_anc.dsp.timing import TrainingTimingContract                    # noqa: E402
from deep_anc.eval.broadband_runtime import (                             # noqa: E402
    RUNTIME_DEPLOYMENT_METADATA_SCHEMA,
)
from deep_anc.models.streaming import (                                   # noqa: E402
    ExportWrapper,
    flatten_states,
    state_names,
)
from deep_anc.realtime.engines import (                                   # noqa: E402
    checkpoint_digital_reference_lead_samples,
)
from deep_anc.train.experiment_contract import (                          # noqa: E402
    validate_embedded_experiment_contract,
)


def _sha256_handle(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_handle(handle)


def _load_checkpoint_snapshot(path: Path) -> tuple[object, str, int]:
    """한 held FD의 exact checkpoint bytes만 load/hash한다."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(
            f"checkpoint는 symlink가 아닌 readable regular file이어야 합니다: {path}: {exc}"
        ) from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"checkpoint는 regular file이어야 합니다: {path}")
        before_sha = _sha256_handle(handle)
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=False)
        after = os.fstat(handle.fileno())
        handle.seek(0)
        after_sha = _sha256_handle(handle)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before_sha != after_sha
    ):
        raise RuntimeError("checkpoint bytes가 load 중 변경됐습니다")
    return state, before_sha, int(before.st_size)


def _artifact_sha(contract: dict, name: str) -> str:
    artifacts = contract.get("artifacts")
    entry = artifacts.get(name) if isinstance(artifacts, dict) else None
    digest = entry.get("sha256") if isinstance(entry, dict) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"canonical experiment contract에 {name} SHA가 없습니다")
    return digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _staging_path(final: Path) -> Path:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{final.name}.", suffix=".partial", dir=final.parent
    )
    os.close(descriptor)
    path = Path(raw)
    path.chmod(0o644)
    return path


def _require_exact_regular(
    path: Path, *, expected_size: int, expected_sha256: str, label: str
) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{label}을 읽을 수 없습니다: {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise RuntimeError(f"{label}은 symlink가 아닌 regular file이어야 합니다: {path}")
    if int(info.st_size) != int(expected_size) or _sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} bytes가 staging 검증 결과와 다릅니다: {path}")
    return info


def _unlink_if_same_inode(path: Path, staged: os.stat_result) -> None:
    """이 transaction이 link한 exact inode만 rollback한다."""

    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(current.st_mode)
        and current.st_dev == staged.st_dev
        and current.st_ino == staged.st_ino
    ):
        path.unlink()


def _publish_one_noreplace(
    staged: Path,
    final: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> bool:
    """Hard-link로 no-replace 발행한다. exact race winner는 재사용한다."""

    try:
        os.link(staged, final, follow_symlinks=False)
    except FileExistsError:
        _require_exact_regular(
            final,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=f"기존 {label}",
        )
        return False
    _fsync_directory(final.parent)
    return True


def _publish_export_pair(
    *,
    staged_artifact: Path,
    staged_metadata: Path,
    artifact: Path,
    metadata: Path,
) -> str:
    """검증 완료 두 leaf를 no-replace로 발행하고 부분 실패를 복구한다.

    Artifact를 먼저, metadata를 마지막 completion leaf로 발행한다. Python 예외나
    SIGINT가 두 link 사이에 오면 이 호출이 만든 inode만 rollback한다. SIGKILL/전원
    중단으로 artifact-only가 남아도 다음 실행이 동일 bytes를 staging에서 다시 만든
    경우에만 metadata를 발행해 복구한다. 서로 다른 orphan은 절대 덮어쓰지 않는다.
    """

    if artifact == metadata:
        raise ValueError("ONNX artifact와 metadata 경로가 같을 수 없습니다")
    artifact_info = staged_artifact.lstat()
    metadata_info = staged_metadata.lstat()
    artifact_sha = _sha256(staged_artifact)
    metadata_sha = _sha256(staged_metadata)
    artifact_exists = os.path.lexists(artifact)
    metadata_exists = os.path.lexists(metadata)
    if artifact_exists and metadata_exists:
        raise FileExistsError(
            "canonical export artifact/metadata가 이미 모두 존재합니다: "
            f"{artifact}, {metadata}"
        )
    if artifact_exists:
        _require_exact_regular(
            artifact,
            expected_size=artifact_info.st_size,
            expected_sha256=artifact_sha,
            label="기존 orphan ONNX artifact",
        )
    if metadata_exists:
        _require_exact_regular(
            metadata,
            expected_size=metadata_info.st_size,
            expected_sha256=metadata_sha,
            label="기존 orphan ONNX metadata",
        )

    created_artifact = False
    created_metadata = False
    try:
        if not artifact_exists:
            created_artifact = _publish_one_noreplace(
                staged_artifact,
                artifact,
                expected_size=artifact_info.st_size,
                expected_sha256=artifact_sha,
                label="ONNX artifact",
            )
        if not metadata_exists:
            created_metadata = _publish_one_noreplace(
                staged_metadata,
                metadata,
                expected_size=metadata_info.st_size,
                expected_sha256=metadata_sha,
                label="ONNX metadata",
            )
        _require_exact_regular(
            artifact,
            expected_size=artifact_info.st_size,
            expected_sha256=artifact_sha,
            label="발행 ONNX artifact",
        )
        _require_exact_regular(
            metadata,
            expected_size=metadata_info.st_size,
            expected_sha256=metadata_sha,
            label="발행 ONNX metadata",
        )
        _fsync_directory(artifact.parent)
    except BaseException:
        # 같은 artifact bytes를 공유하는 다른 exporter가 metadata completion
        # leaf를 먼저 발행했을 수 있다. 이 경우 패배한 transaction이 자신이
        # 먼저 만든 artifact link를 지우면 승자의 완성 pair가 깨진다.
        foreign_completion = not created_metadata and os.path.lexists(metadata)
        if created_metadata:
            _unlink_if_same_inode(metadata, metadata_info)
        if created_artifact and not foreign_completion:
            _unlink_if_same_inode(artifact, artifact_info)
        _fsync_directory(artifact.parent)
        raise
    return "recovered" if artifact_exists or metadata_exists else "published"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="runs/export/model.onnx")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    raw_checkpoint_path = Path(args.ckpt).expanduser()
    if not raw_checkpoint_path.is_absolute():
        raw_checkpoint_path = Path.cwd() / raw_checkpoint_path
    checkpoint_path = Path(os.path.abspath(raw_checkpoint_path))
    state, checkpoint_sha, checkpoint_size = _load_checkpoint_snapshot(
        checkpoint_path
    )
    checkpoint_cfg = state.get("cfg") if isinstance(state, dict) else None
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError("checkpoint에 resolved cfg가 없습니다")
    experiment_contract = validate_embedded_experiment_contract(checkpoint_cfg)
    # training_timing_contract는 digital-reference 전용 timeline lead 유도에만
    # 쓰인다(runtime의 digital_reference_lead_samples와 달리, acoustic-reference
    # checkpoint의 data config에는 애초에 이 필드가 없다 — reference mic 입력에는
    # 이 lead 개념 자체가 전이되지 않는다, configs/runtime_tiny_mic_diagnostic.yaml
    # 참고). 이 SHA는 canonical Stage-1/Stage-2 admission에서만 대조되고, 이 스크립트가
    # 쓰는 단순 runtime 로딩 경로(engines.py)는 참조하지 않는다.
    reference_mode = str((checkpoint_cfg.get("data") or {}).get("reference_mode", "digital"))
    timing_sha = (
        TrainingTimingContract.from_data_config(checkpoint_cfg.get("data") or {}).digest()
        if reference_mode == "digital"
        else ""
    )
    # control_band_contract_sha256도 canonical Stage-1/Stage-2 admission이 통과한
    # checkpoint에만 stamp된다 — performance_pilot 등 admission을 거치지 않은 실험
    # role은 이 필드가 애초에 None이다(실측: acoustic pilot checkpoint에서 확인).
    # 값이 있는데 sha256 형식이 아니면(길이 불일치) 여전히 손상 신호이므로 거부한다.
    control_band_sha = str(checkpoint_cfg.get("control_band_contract_sha256") or "")
    if control_band_sha and len(control_band_sha) != 64:
        raise ValueError("checkpoint의 control_band_contract_sha256 형식이 올바르지 않습니다")
    model_cfg = state["cfg"]["model"]
    model = build_model(model_cfg)
    model.load_state_dict(state["model"])
    model.eval()

    wrapper = ExportWrapper(model, block_samples=args.block)
    names = state_names(model)
    init_states = flatten_states(model.init_states(1, "cpu"))
    x = torch.zeros(1, model.in_channels, args.block)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".json")
    if out_path == meta_path:
        raise ValueError("--out은 metadata .json과 다른 ONNX 경로여야 합니다")
    if os.path.lexists(out_path) and os.path.lexists(meta_path):
        raise FileExistsError(
            "canonical export artifact/metadata가 이미 모두 존재합니다: "
            f"{out_path}, {meta_path}"
        )

    staged_out = _staging_path(out_path)
    try:
        staged_meta = _staging_path(meta_path)
    except BaseException:
        staged_out.unlink(missing_ok=True)
        _fsync_directory(out_path.parent)
        raise

    input_names = ["x"] + names
    output_names = ["y"] + [f"{n}_out" for n in names]
    try:
        with torch.no_grad(), staged_out.open("wb") as output_handle:
            torch.onnx.export(
                wrapper,
                (x, *init_states),
                output_handle,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                dynamic_axes=None,          # 전부 정적 shape
            )
            output_handle.flush()
            os.fsync(output_handle.fileno())

        # ----- ORT(CPU) 등가성 검증: 임의 블록 20개 스트리밍 -----
        import onnxruntime as ort

        # Jetson(Tegra)에서 ORT 기본 스레드 affinity 설정이 크래시를 유발하므로 명시 지정
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            str(staged_out), so, providers=["CPUExecutionProvider"]
        )
        rng = np.random.default_rng(0)
        torch_states = model.init_states(1, "cpu")
        ort_states = [
            s.numpy().copy()
            for s in flatten_states(model.init_states(1, "cpu"))
        ]
        max_err = 0.0
        with torch.no_grad():
            for _ in range(20):
                blk = (
                    rng.standard_normal((1, model.in_channels, args.block)) * 0.02
                ).astype(np.float32)
                y_t, torch_states = model.streaming_step(
                    torch.from_numpy(blk), torch_states
                )
                feeds = {"x": blk}
                feeds.update({n: s for n, s in zip(names, ort_states)})
                outs = sess.run(None, feeds)
                y_o, ort_states = outs[0], outs[1:]
                max_err = max(
                    max_err, float(np.max(np.abs(y_t.numpy() - y_o)))
                )
        print(f"ORT 등가성 max err = {max_err:.3e} (허용 {args.tolerance})")
        if max_err > args.tolerance:
            print("검증 실패 — export 그래프를 점검하세요", file=sys.stderr)
            return 1

        _require_exact_regular(
            checkpoint_path,
            expected_size=checkpoint_size,
            expected_sha256=checkpoint_sha,
            label="현재 checkpoint",
        )
        meta = {
            "schema_version": RUNTIME_DEPLOYMENT_METADATA_SCHEMA,
            "engine": "ort",
            "model_name": model_cfg.get("name"),
            "experiment_contract_sha256": experiment_contract["sha256"],
            "control_band_contract_sha256": control_band_sha,
            "training_timing_contract_sha256": timing_sha,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "digital_reference_lead_samples": checkpoint_digital_reference_lead_samples(state),
            "deployment_artifact_path": str(out_path.resolve()),
            "deployment_artifact_sha256": _sha256(staged_out),
            "primary_path_sha256": _artifact_sha(experiment_contract, "primary_path"),
            "secondary_path_sha256": _artifact_sha(experiment_contract, "secondary_path"),
            # export는 runtime 검증을 위한 artifact를 만들 뿐이다. G4/Level-5/
            # physical runtime gate가 후속 PASS하기 전에 deployment 자격을 주지 않는다.
            "deployment_eligible": False,
            "block_samples": args.block,
            "hop": model.hop,
            "win": model.win,
            "in_channels": model.in_channels,
            "state_names": names,
            "ckpt": str(checkpoint_path),
            "ort_max_err": max_err,
        }
        with staged_meta.open("w", encoding="utf-8") as metadata_handle:
            json.dump(meta, metadata_handle, indent=2, ensure_ascii=False)
            metadata_handle.write("\n")
            metadata_handle.flush()
            os.fsync(metadata_handle.fileno())
        # metadata 조립 중 best.pt가 atomic replace된 경우에도 final 두 leaf를
        # 발행하지 않는다. 첫 검사는 metadata 내용의 SHA를 고정하고, 이 두 번째
        # 검사는 no-replace transaction 직전 lexical checkpoint authority를 고정한다.
        _require_exact_regular(
            checkpoint_path,
            expected_size=checkpoint_size,
            expected_sha256=checkpoint_sha,
            label="발행 직전 checkpoint",
        )
        state_label = _publish_export_pair(
            staged_artifact=staged_out,
            staged_metadata=staged_meta,
            artifact=out_path,
            metadata=meta_path,
        )
        print(f"ONNX 저장 ({state_label}): {out_path}")
        print(f"메타 저장 ({state_label}): {meta_path}")
        return 0
    finally:
        staged_out.unlink(missing_ok=True)
        staged_meta.unlink(missing_ok=True)
        _fsync_directory(out_path.parent)


if __name__ == "__main__":
    raise SystemExit(main())
