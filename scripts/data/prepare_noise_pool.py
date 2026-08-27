#!/usr/bin/env python3
"""노이즈 풀 인덱싱 → JSONL manifest 생성 (파일 단위 train/val/test 분할).

  .venv/bin/python scripts/data/prepare_noise_pool.py
리샘플/정규화는 학습 로더(NoisePool)가 실시간으로 수행하므로 여기서는 인덱스만 만든다.

2026-08-06 수정 — **선언한 태그를 전부 만들지 못하면 실패한다.**
------------------------------------------------------------
이전 판은 ``--root data/raw/noise`` 하나만 스캔했다. 그 아래에는 ``esc50`` 밖에 없고
``music``(data/raw/music/fma_small) 과 ``speech``(data/raw/speech/LibriSpeech) 는
**다른 루트**에 있다. 그래서 ``data/manifests`` 에는 ``esc50.jsonl`` 하나만 생겼다.

그 상태가 조용한 이유가 문제였다. ``src/deep_anc/data/synth_dataset.py`` 는 manifest 가
없는 태그를 로그 한 줄 없이 **합성원으로 폴백**한다. 즉 ``source_mix_ratio`` 가
speech 0.15 / music 0.10 을 선언해도 실제로는 그 0.25 가 전부 synthetic 으로 돌아가고,
학습 기록에는 선언된 혼합비가 남는다. 선언과 실행이 갈라진 것을 아무도 못 본다.

그래서 이 스크립트는 이제
  1. ``source_mix_ratio`` (+ acoustic 판)의 **태그 목록을 단일 출처로 읽고**,
  2. ``data/raw`` 전체를 그 태그로 매칭해 스캔하며,
  3. 비율 > 0 인데 소재가 없어 manifest 를 만들지 못한 태그가 하나라도 있으면
     **경고가 아니라 종료코드 1** 로 끝난다.

이 저장소에 실제로 있는 원본: ``music/fma_small``, ``noise/esc50``,
``speech/LibriSpeech``. ``dns_fullband`` / ``demand`` / ``machine`` 은 **유실됐다** —
있는 척하지 않는다. 세 태그가 선언에 남아 있는 한 이 스크립트는 실패하는 것이 옳다.
고치는 방법은 둘뿐이다: 원본을 다시 받거나, ``configs/data_sim.yaml`` 의 혼합비에서
그 태그를 지우는 것이다(그러면 게이트도 그 태그를 요구하지 않는다).
"""

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
import re
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_yaml                # noqa: E402
from deep_anc.data.holdout_contract import (                    # noqa: E402
    HoldoutContractError,
    read_regular_file_snapshot,
    reject_symlink_components,
    validate_holdout_contract,
)
from deep_anc.data.manifest import (                            # noqa: E402
    assign_splits,
    read_manifest_bytes,
    scan_wavs,
    write_manifest,
)
from deep_anc.data.public_lineage import (                      # noqa: E402
    PublicLineageError,
    build_public_lineage,
    canonical_json_sha256,
    validate_public_manifest_lineage,
)


_FROZEN = ConfigDict(frozen=True, extra="forbid")

DIAGNOSTIC_ONLY_EXIT = 3
GENERATION_SIDECAR = "manifest_generation.json"


class ManifestTransactionError(RuntimeError):
    """commit 실패와 rollback 실패를 모두 보존한다."""

    def __init__(
        self,
        original_error: BaseException,
        rollback_errors: list[BaseException],
        *,
        recovery_dir: Path,
    ) -> None:
        self.original_error = original_error
        self.rollback_errors = tuple(rollback_errors)
        self.recovery_dir = recovery_dir
        detail = "; ".join(f"{type(item).__name__}: {item}" for item in rollback_errors)
        super().__init__(
            f"원래 commit 오류: {type(original_error).__name__}: {original_error}; "
            f"rollback 오류 {len(rollback_errors)}건: {detail}; "
            f"수동 복구 자료: {recovery_dir}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _bind_audio_content_hashes(
    entries: list[dict], *, raw_roots: list[Path]
) -> list[dict]:
    """manifest 경로/헤더뿐 아니라 실제 raw audio bytes를 각 행에 결속한다."""

    bound: list[dict] = []
    lexical_roots = [Path(os.path.abspath(root)) for root in raw_roots]
    for index, entry in enumerate(entries):
        path = Path(str(entry.get("path") or ""))
        absolute = Path(os.path.abspath(path))
        allowed = next(
            (root for root in lexical_roots if absolute == root or root in absolute.parents),
            None,
        )
        if allowed is None:
            raise OSError(f"manifest entry #{index} raw audio가 선언 raw root 밖입니다: {path}")
        try:
            snapshot = read_regular_file_snapshot(
                absolute,
                root=allowed,
                label=f"manifest entry #{index} raw audio",
                capture_bytes=False,
            )
        except HoldoutContractError as exc:
            raise OSError(str(exc)) from exc
        item = dict(entry)
        item["path"] = str(snapshot.path)
        item["content_sha256"] = snapshot.sha256
        item["content_size"] = snapshot.size
        bound.append(item)
    return bound


def _recorded_source_pool_exclusion(
    values: list[str],
) -> tuple[set[str], list[dict[str, object]]]:
    """실측에 사용한 source-pool CSV의 clip basename을 exclusion set으로 읽는다.

    canonical holdout은 평가용 component를 보존하지만, source-pool CSV에는 실제 녹음
    세션 외에 예약된 clip도 함께 남을 수 있다. 이 목록을 manifest 생성 시점에 같이
    제외하지 않으면 readiness의 corpus-disjoint 게이트가 뒤늦게 실패한다. CSV bytes와
    basename digest를 generation sidecar에 남겨 어떤 pool 세대가 사용됐는지 추적한다.
    """

    basenames: set[str] = set()
    evidence: list[dict[str, object]] = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        snapshot = read_regular_file_snapshot(
            path,
            root=REPO_ROOT,
            label=f"recorded source-pool CSV {path}",
            capture_bytes=True,
        )
        if snapshot.data is None:
            raise OSError(f"recorded source-pool CSV bytes를 읽지 못했습니다: {path}")
        try:
            text = snapshot.data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"recorded source-pool CSV UTF-8 오류: {path}") from exc
        reader = csv.DictReader(text.splitlines())
        required = {"source_family", "path", "clips"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"recorded source-pool CSV 필드가 불완전합니다: {path} "
                f"(required={sorted(required)})"
            )
        file_basenames: set[str] = set()
        for number, row in enumerate(reader, start=2):
            raw_clips = str(row.get("clips") or "")
            try:
                clips = json.loads(raw_clips)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"recorded source-pool CSV clips JSON 오류: {path}:{number}"
                ) from exc
            if not isinstance(clips, list):
                raise ValueError(
                    f"recorded source-pool CSV clips가 목록이 아닙니다: {path}:{number}"
                )
            for clip in clips:
                item = str(clip).strip().replace("\\", "/")
                if not item:
                    raise ValueError(
                        f"recorded source-pool CSV 빈 clip: {path}:{number}"
                    )
                file_basenames.add(item.rsplit("/", 1)[-1].casefold())
        basenames.update(file_basenames)
        evidence.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": snapshot.sha256,
                "size": int(snapshot.size),
                "basename_count": len(file_basenames),
            }
        )
    return basenames, evidence


@contextmanager
def _generation_process_lock(out_dir: Path) -> Iterator[None]:
    """별도 lock 파일 없이 output parent directory inode를 process 단위로 잠근다."""

    descriptor = os.open(
        out_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ManifestTransactionError(
                RuntimeError(f"다른 manifest prepare가 실행 중입니다: {out_dir}"),
                [],
                recovery_dir=out_dir.parent,
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _verify_committed_generation(
    out_dir: Path,
    contract: dict,
    *,
    data_config: Path,
    holdout_path: Path | None,
) -> None:
    """backup 삭제 전 새 세대와 입력 계약의 commit postcondition을 재계산한다."""

    sidecar = read_regular_file_snapshot(
        out_dir / GENERATION_SIDECAR,
        root=REPO_ROOT,
        label="committed manifest generation sidecar",
    )
    if sidecar.data != _canonical_json_bytes(contract):
        raise RuntimeError("committed generation sidecar bytes가 staged contract와 다릅니다")
    contract_raw_roots = [REPO_ROOT / value for value in contract["raw_roots"]]
    committed_entries: dict[str, list[dict]] = {}
    for tag, metadata in sorted(contract["manifests"].items()):
        snapshot = read_regular_file_snapshot(
            out_dir / str(metadata["file"]),
            root=REPO_ROOT,
            label=f"committed {tag} manifest",
        )
        if snapshot.sha256 != metadata["sha256"]:
            raise RuntimeError(f"committed {tag} manifest SHA가 staged contract와 다릅니다")
        assert snapshot.data is not None
        entries = read_manifest_bytes(
            snapshot.data, manifest_path=out_dir / str(metadata["file"])
        )
        committed_entries[tag] = entries
        for index, entry in enumerate(entries):
            audio = Path(str(entry["path"]))
            absolute = Path(os.path.abspath(audio))
            allowed = next(
                (
                    root
                    for root in contract_raw_roots
                    if absolute == root or root in absolute.parents
                ),
                None,
            )
            if allowed is None:
                raise RuntimeError(f"committed {tag} raw path가 declared root 밖입니다: {audio}")
            audio_snapshot = read_regular_file_snapshot(
                absolute,
                root=allowed,
                label=f"committed {tag} raw audio #{index}",
                capture_bytes=False,
            )
            if audio_snapshot.sha256 != entry.get("content_sha256"):
                raise RuntimeError(f"prepare 중 {tag} raw audio bytes가 바뀌었습니다: {audio}")
    if contract.get("schema_version") == 3:
        lineage = contract.get("public_lineage")
        if not isinstance(lineage, dict):
            raise RuntimeError("committed 학습 세대에 public_lineage가 없습니다")
        summary = validate_public_manifest_lineage(committed_entries)
        if (
            summary["component_count"] != lineage.get("manifest_component_count")
            or summary["component_membership_sha256"]
            != lineage.get("manifest_component_membership_sha256")
        ):
            raise RuntimeError("committed public lineage component 증거가 staged 값과 다릅니다")
        metadata = lineage.get("metadata")
        if not isinstance(metadata, dict) or not metadata:
            raise RuntimeError("committed public lineage metadata가 비었습니다")
        for name, evidence in sorted(metadata.items()):
            if not isinstance(evidence, dict):
                raise RuntimeError(f"public lineage metadata.{name}가 mapping이 아닙니다")
            metadata_snapshot = read_regular_file_snapshot(
                REPO_ROOT / str(evidence.get("path") or ""),
                root=REPO_ROOT,
                label=f"public lineage metadata {name}",
                capture_bytes=False,
            )
            if (
                metadata_snapshot.sha256 != evidence.get("sha256")
                or metadata_snapshot.size != evidence.get("size")
            ):
                raise RuntimeError(f"prepare 중 public lineage metadata가 바뀌었습니다: {name}")
    config_snapshot = read_regular_file_snapshot(
        data_config,
        root=REPO_ROOT,
        label="manifest data config",
        capture_bytes=False,
    )
    if config_snapshot.sha256 != contract["data_config_sha256"]:
        raise RuntimeError("manifest commit 중 data config bytes가 바뀌었습니다")
    if holdout_path is not None:
        holdout_snapshot = read_regular_file_snapshot(
            holdout_path,
            root=REPO_ROOT,
            label="manifest holdout",
            capture_bytes=False,
        )
        if holdout_snapshot.sha256 != contract["holdout_sha256"]:
            raise RuntimeError("manifest commit 중 canonical holdout bytes가 바뀌었습니다")


@contextmanager
def _defer_termination_signals() -> Iterator[None]:
    """여러 manifest를 교체하는 짧은 commit 구간에는 INT/TERM을 지연한다.

    POSIX에는 여러 파일을 한 번에 rename하는 primitive가 없다. 따라서 모든 새 파일을
    먼저 같은 파일시스템에 staging하고, 기존 파일의 byte-for-byte 백업을 만든 다음
    교체한다. 일반 예외는 즉시 rollback하고 INT/TERM은 commit이 끝난 뒤 전달한다.
    SIGKILL/전원 장애는 막을 수 없으므로 generation sidecar의 전체 SHA 집합이 다음
    readiness에서 혼합 세대를 반드시 거부한다.
    """

    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    blocked = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    """root부터 path까지 현재 존재하는 component 중 symlink가 있는지 본다."""

    absolute = path.absolute()
    try:
        relative = absolute.relative_to(root.absolute())
    except ValueError:
        return True
    current = root.absolute()
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
        if not current.exists():
            break
    return False


def _paths_overlap(left: Path, right: Path) -> bool:
    a, b = left.resolve(), right.resolve()
    return a == b or a in b.parents or b in a.parents


def validate_output_destination(out_dir: Path, *, diagnostic_only: bool) -> None:
    """symlink/alias로 diagnostic 출력이 official manifest를 덮는 경로를 거부한다."""

    if _has_symlink_component(out_dir, root=REPO_ROOT):
        raise ValueError(f"manifest 출력 경로에 symlink/저장소 밖 component가 있습니다: {out_dir}")
    official = REPO_ROOT / "data/manifests"
    diagnostics = REPO_ROOT / "results/diagnostics"
    resolved_out = out_dir.resolve()
    resolved_official = official.resolve()
    if diagnostic_only:
        try:
            resolved_out.relative_to(diagnostics.resolve())
        except ValueError as exc:
            raise ValueError(
                "--allow-corpus-leak 출력은 symlink 해석 뒤에도 results/diagnostics 아래여야 하며 "
                "official data/manifests를 쓸 수 없습니다"
            ) from exc
        if _paths_overlap(resolved_out, resolved_official):
            raise ValueError("diagnostic 출력 경로가 official data/manifests와 겹칩니다")
        if out_dir.exists() and official.exists():
            try:
                if os.path.samefile(out_dir, official):
                    raise ValueError("diagnostic 출력이 official data/manifests와 같은 inode입니다")
            except FileNotFoundError:
                pass


def write_generation_transactionally(
    prepared: dict[str, tuple[list[dict], int, list[Path]]],
    *,
    out_dir: Path,
    data_config: Path,
    holdout_path: Path | None,
    seed: int,
    training_eligible: bool,
    raw_roots: list[Path] | None = None,
    public_lineage: dict | None = None,
) -> dict:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with _generation_process_lock(out_dir):
        return _write_generation_transactionally_locked(
            prepared,
            out_dir=out_dir,
            data_config=data_config,
            holdout_path=holdout_path,
            seed=seed,
            training_eligible=training_eligible,
            raw_roots=raw_roots,
            public_lineage=public_lineage,
        )


def _write_generation_transactionally_locked(
    prepared: dict[str, tuple[list[dict], int, list[Path]]],
    *,
    out_dir: Path,
    data_config: Path,
    holdout_path: Path | None,
    seed: int,
    training_eligible: bool,
    raw_roots: list[Path] | None,
    public_lineage: dict | None,
) -> dict:
    """검증된 manifest 세트와 provenance sidecar를 세대 단위로 교체한다.

    staging 중 하나라도 실패하면 대상 디렉터리는 전혀 바뀌지 않는다. commit 중 rename
    실패도 모든 기존 byte를 복구한다. sidecar는 각 manifest SHA와 holdout/config SHA를
    묶어 이후 소비자가 SIGKILL 같은 비정상 종료로 생긴 혼합 세대를 탐지하게 한다.
    """

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=".noise-manifest-stage-", dir=out_dir.parent)
    )
    staged = stage_root / "new"
    backups = stage_root / "old"
    staged.mkdir()
    backups.mkdir()
    cleanup_stage = True
    try:
        manifest_meta: dict[str, dict[str, object]] = {}
        for tag, (entries, _dropped, _sources) in sorted(prepared.items()):
            staged_path = staged / f"{tag}.jsonl"
            write_manifest(entries, staged_path)
            _fsync_file(staged_path)
            manifest_meta[tag] = {
                "file": staged_path.name,
                "entries": len(entries),
                "sha256": _sha256_file(staged_path),
            }

        roots_for_contract = raw_roots or [REPO_ROOT]
        if training_eligible and not isinstance(public_lineage, dict):
            raise ValueError("학습용 manifest 세대에는 public_lineage 증거가 필수입니다")
        contract = {
            "schema_version": 3 if training_eligible else 2,
            "training_eligible": bool(training_eligible),
            "seed": int(seed),
            "data_config": str(data_config.relative_to(REPO_ROOT)),
            "data_config_sha256": _sha256_file(data_config),
            "holdout": (
                str(holdout_path.relative_to(REPO_ROOT)) if holdout_path is not None else None
            ),
            "holdout_sha256": (
                _sha256_file(holdout_path) if holdout_path is not None else None
            ),
            "raw_roots": sorted(
                {
                    str(Path(root).relative_to(REPO_ROOT))
                    for root in roots_for_contract
                }
            ),
            "manifests": manifest_meta,
        }
        if public_lineage is not None:
            contract["public_lineage"] = public_lineage
        build_basis = _canonical_json_bytes(contract)
        contract["build_id"] = hashlib.sha256(build_basis).hexdigest()
        # 사람용 시각은 build identity에 포함하지 않는다. 같은 입력은 같은 build_id다.
        contract["created_at"] = datetime.now(timezone.utc).isoformat()
        sidecar_path = staged / GENERATION_SIDECAR
        sidecar_path.write_bytes(_canonical_json_bytes(contract))
        _fsync_file(sidecar_path)
        _fsync_directory(staged)
        _fsync_directory(stage_root)

        out_dir.mkdir(parents=True, exist_ok=True)
        _fsync_directory(out_dir.parent)
        names = [f"{tag}.jsonl" for tag in sorted(prepared)] + [GENERATION_SIDECAR]
        existed: dict[str, bool] = {}
        for name in names:
            target = out_dir / name
            existed[name] = target.is_file()
            if existed[name]:
                shutil.copy2(target, backups / name)
                _fsync_file(backups / name)
        _fsync_directory(backups)
        _fsync_directory(stage_root)

        installed: list[str] = []
        try:
            with _defer_termination_signals():
                for name in names:
                    os.replace(staged / name, out_dir / name)
                    installed.append(name)
                _fsync_directory(out_dir)
                _verify_committed_generation(
                    out_dir,
                    contract,
                    data_config=data_config,
                    holdout_path=holdout_path,
                )
        except BaseException as original_error:
            # 하나의 복구가 실패해도 나머지 대상 복구를 끝까지 시도한다. 실패한 backup은
            # 지우지 않아 수동 복구가 가능해야 하며, 원래 예외와 모든 rollback 오류를
            # 함께 보존한다.
            rollback_errors: list[BaseException] = []
            for name in reversed(installed):
                target = out_dir / name
                try:
                    if existed[name]:
                        os.replace(backups / name, target)
                    else:
                        target.unlink(missing_ok=True)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            try:
                _fsync_directory(out_dir)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
            if rollback_errors:
                cleanup_stage = False
                raise ManifestTransactionError(
                    original_error,
                    rollback_errors,
                    recovery_dir=stage_root,
                ) from original_error
            raise
        return contract
    finally:
        if cleanup_stage:
            shutil.rmtree(stage_root, ignore_errors=True)


class DeclaredPool(BaseModel):
    """혼합비가 선언한 소스 태그 하나. 태그 목록의 단일 출처는 ``data_sim.yaml`` 이다."""

    model_config = _FROZEN

    tag: str
    ratio: float
    """모든 혼합비 표에서 이 태그가 갖는 최대 비율. 0 이면 지금은 쓰이지 않는다."""

    @model_validator(mode="after")
    def _validate(self) -> "DeclaredPool":
        if not self.tag or self.tag != self.tag.strip():
            raise ValueError(f"태그가 비었거나 공백이 있습니다: {self.tag!r}")
        if any(ch in self.tag for ch in ("/", "\\", ".")):
            raise ValueError(f"태그는 파일명 조각이 아니라 이름이어야 합니다: {self.tag!r}")
        if not (0.0 <= self.ratio <= 1.0):
            raise ValueError(f"{self.tag}: 비율은 [0,1] 이어야 합니다: {self.ratio}")
        return self


class PoolPlan(BaseModel):
    """"무엇을 만들어야 하는가" 의 선언. 스캔 결과와 대조되는 기준이다."""

    model_config = _FROZEN

    pools: tuple[DeclaredPool, ...]
    roots: tuple[str, ...]

    @model_validator(mode="after")
    def _validate(self) -> "PoolPlan":
        seen = [item.tag for item in self.pools]
        if len(seen) != len(set(seen)):
            raise ValueError(f"태그가 중복됐습니다: {seen}")
        if not self.roots:
            raise ValueError("스캔할 루트가 없습니다")
        return self

    def required_tags(self) -> tuple[str, ...]:
        return tuple(item.tag for item in self.pools if item.ratio > 0.0)


def declared_pools(data_config: Path) -> tuple[DeclaredPool, ...]:
    """``source_mix_ratio`` (+ acoustic 판)에서 태그를 읽는다 — 유일한 경로.

    태그를 이 스크립트에 리터럴로 적으면 그것이 **두 번째 선언**이 되고, 설정과
    갈라진 순간 아무도 모른다(이 저장소가 반복한 발생기 A).
    """

    cfg = load_yaml(data_config)
    ratios: dict[str, float] = {}
    for key in ("source_mix_ratio", "source_mix_ratio_acoustic"):
        for tag, value in (cfg.get(key) or {}).items():
            if str(tag) == "synthetic":
                continue  # 파일 소재가 없는 즉석 생성원
            ratios[str(tag)] = max(float(value), ratios.get(str(tag), 0.0))
    return tuple(
        DeclaredPool(tag=tag, ratio=ratio) for tag, ratio in sorted(ratios.items())
    )


def discover_tag_dirs(root: Path, tags: frozenset[str]) -> dict[str, list[Path]]:
    """``root`` 아래에서 **선언된 태그 이름과 같은 디렉터리**를 찾는다.

    ``data/raw/music/fma_small`` 은 ``music`` 에서 멈추고(그 아래 전부가 music),
    ``data/raw/noise/esc50`` 은 ``noise`` 가 태그가 아니므로 한 단계 더 내려가
    ``esc50`` 에서 멈춘다. 즉 디렉터리 깊이를 스크립트가 가정하지 않는다 —
    ``--root data/raw/noise`` 하나만 보던 판이 music/speech 를 통째로 놓친 이유가
    바로 그 가정이었다.
    """

    found: dict[str, list[Path]] = {}
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return found
    if root.is_symlink() or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"raw root는 symlink가 아닌 directory여야 합니다: {root}")
    stack = [root]
    visited: set[tuple[int, int]] = set()
    while stack:
        current = stack.pop()
        current_stat = current.lstat()
        identity = (int(current_stat.st_dev), int(current_stat.st_ino))
        if identity in visited:
            raise ValueError(f"raw directory cycle/alias를 감지했습니다: {current}")
        visited.add(identity)
        for child in sorted(current.iterdir()):
            child_stat = child.lstat()
            if child.is_symlink():
                raise ValueError(f"raw tree의 symlink는 허용하지 않습니다: {child}")
            if not stat.S_ISDIR(child_stat.st_mode):
                continue
            if child.name in tags:
                found.setdefault(child.name, []).append(child)
            else:
                stack.append(child)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help=(
            "스캔할 원본 루트(반복 지정 가능). 기본은 data/raw 전체다 — "
            "music/noise/speech 가 서로 다른 루트에 있어서 하나만 보면 조용히 누락된다"
        ),
    )
    parser.add_argument("--out", default="data/manifests")
    parser.add_argument(
        "--data-config",
        default="configs/data_sim.yaml",
        help="source_mix_ratio 를 읽을 설정. 태그 목록의 단일 출처다",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--holdout",
        default="data/manifests/recorded_holdout.json",
        help=(
            "실측 재생에 쓴 원본 목록(JSON). 여기 있는 클립은 합성 manifest 에서 "
            "**구성 단계에서** 제외한다 (D1 코퍼스 누수). 파일이 없거나 비어 있으면 "
            "학습용 manifest 생성을 거부한다"
        ),
    )
    parser.add_argument(
        "--expected-holdout-sha256",
        default=None,
        help=(
            "신뢰한 canonical holdout의 64자리 SHA-256. 학습용 세대에서는 필수이며 "
            "스캔 시작/commit 직전/종료 후 다시 확인한다"
        ),
    )
    parser.add_argument(
        "--allow-corpus-leak",
        action="store_true",
        help="held-out 제외를 끈다. 진단 전용이며 학습 manifest 를 만들 때 쓰면 안 된다",
    )
    parser.add_argument(
        "--recorded-source-pool-csv",
        action="append",
        default=[],
        help=(
            "실측에 사용한 source-pool CSV(반복 지정 가능). 모든 clip basename을 "
            "합성 manifest에서 추가 제외해 예약/활성 원본 누수를 막는다"
        ),
    )
    args = parser.parse_args()

    plan = PoolPlan(
        pools=declared_pools(REPO_ROOT / args.data_config),
        roots=tuple(args.root or ["data/raw"]),
    )
    out_dir = REPO_ROOT / args.out
    tags = frozenset(item.tag for item in plan.pools)

    try:
        validate_output_destination(out_dir, diagnostic_only=bool(args.allow_corpus_leak))
    except ValueError as exc:
        print(f"[실패] manifest 출력 경로 계약 위반: {exc}", file=sys.stderr)
        return 2

    if not args.allow_corpus_leak and (
        not isinstance(args.expected_holdout_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_holdout_sha256) is None
    ):
        print(
            "[실패] 학습용 manifest에는 --expected-holdout-sha256 64자리가 필수입니다.",
            file=sys.stderr,
        )
        return 2
    expected_holdout_sha256 = (
        str(args.expected_holdout_sha256).lower()
        if args.expected_holdout_sha256 is not None
        else None
    )

    roots = [REPO_ROOT / value for value in plan.roots]
    missing_roots = [str(path) for path in roots if not path.exists()]
    if missing_roots:
        print(f"소스 루트 없음: {', '.join(missing_roots)}", file=sys.stderr)
        return 1
    try:
        for root in roots:
            reject_symlink_components(root, root=REPO_ROOT)
    except HoldoutContractError as exc:
        print(f"[실패] raw root 경로 계약 위반: {exc}", file=sys.stderr)
        return 1

    tag_dirs: dict[str, list[Path]] = {}
    try:
        for root in roots:
            for tag, paths in discover_tag_dirs(root, tags).items():
                tag_dirs.setdefault(tag, []).extend(paths)
    except (OSError, ValueError) as exc:
        print(f"[실패] raw tree 안전 검사 실패: {exc}", file=sys.stderr)
        return 1

    # ---- 실측과 겹치는 원본 제외 (D1) -------------------------------------------
    # 같은 오디오가 두 브랜치에 동시에 들어가면 모델은 같은 입력에 **상충하는 정답**을
    # 받는다. 합성은 이상적 P/S 라 −18 dB 까지 가능하고 실측은 실제 플랜트라 천장이
    # 훨씬 낮다. 실측(2026-08-05): 이 저장소의 data/raw 기준으로 실측 4계열 691 클립이
    # **전부** 합성 태그 디렉터리 안에 있다 (music 60 → raw/music, speech 218 →
    # raw/speech, machine 188 + environment 225 → raw/noise/esc50).
    # 사후 검사(check_corpus_disjoint)만으로는 부족하다 — 구성 단계에서 빼야 한다.
    holdout: set[str] = set()
    holdout_path: Path | None = None
    if args.allow_corpus_leak:
        print(
            "[진단 전용] --allow-corpus-leak: recorded holdout 검사를 생략합니다. "
            "이 실행의 manifest는 학습에 사용할 수 없습니다.",
            file=sys.stderr,
        )
    else:
        if not args.holdout:
            print(
                "[실패] --holdout 경로가 비어 있습니다. 학습 manifest는 recorded "
                "holdout 없이 만들 수 없습니다.",
                file=sys.stderr,
            )
            return 1
        holdout_path = REPO_ROOT / args.holdout
        if not holdout_path.is_file():
            print(
                f"[실패] held-out 목록이 없습니다: {holdout_path} — "
                "scripts/data/repair_source_pool_provenance.py --repair-csv "
                "--write-active-holdout 으로 historical provenance를 먼저 복구하세요",
                file=sys.stderr,
            )
            return 1
        try:
            holdout_summary = validate_holdout_contract(
                holdout_path,
                repo_root=REPO_ROOT,
                expected_sha256=expected_holdout_sha256,
            )
        except HoldoutContractError as exc:
            print(
                f"[실패] canonical recorded holdout 계약 위반: {exc}",
                file=sys.stderr,
            )
            return 1
        # validator가 hash/parse한 바로 그 fd snapshot을 exclusion에도 사용한다.
        families = holdout_summary.get("families")
        if not isinstance(families, dict) or not families:
            print(
                f"[실패] held-out JSON의 families가 비었거나 매핑이 아닙니다: {holdout_path}",
                file=sys.stderr,
            )
            return 1
        for family, values in families.items():
            if not isinstance(values, list):
                print(
                    f"[실패] held-out family {family!r} 값은 목록이어야 합니다: {holdout_path}",
                    file=sys.stderr,
                )
                return 1
            holdout.update(
                str(item).replace("\\", "/").rsplit("/", 1)[-1].casefold()
                for item in values
                if str(item).strip()
            )
        if not holdout:
            print(
                f"[실패] held-out 클립이 0개입니다: {holdout_path}. "
                "active recorded session provenance를 먼저 복구하세요.",
                file=sys.stderr,
            )
            return 1
        print(
            f"held-out 클립 {len(holdout)}개 제외 ({holdout_path}, "
            f"sha256={holdout_summary['sha256']})"
        )

    extra_excluded_basenames: set[str] = set()
    source_pool_exclusion_evidence: list[dict[str, object]] = []
    if not args.allow_corpus_leak and args.recorded_source_pool_csv:
        try:
            extra_excluded_basenames, source_pool_exclusion_evidence = (
                _recorded_source_pool_exclusion(args.recorded_source_pool_csv)
            )
        except (OSError, ValueError) as exc:
            print(f"[실패] recorded source-pool exclusion 읽기 실패: {exc}", file=sys.stderr)
            return 1
        print(
            f"recorded source-pool basename {len(extra_excluded_basenames)}개를 "
            "합성 manifest에서 추가 제외합니다"
        )

    # 모든 필수 태그를 먼저 메모리에서 준비한 뒤 한꺼번에 쓴다. 태그 하나가 없는데
    # 앞쪽 manifest만 새 버전으로 덮이면 디렉터리가 서로 다른 holdout 세대를 섞게 된다.
    # 실패 실행은 기존 manifest를 한 바이트도 바꾸지 않아야 한다.
    hashed_by_tag: dict[str, list[dict]] = {}
    sources_by_tag: dict[str, list[Path]] = {}
    for pool in plan.pools:
        sources = tag_dirs.get(pool.tag, [])
        entries: list[dict] = []
        for src in sources:
            entries.extend(scan_wavs(src, pool.tag))
        if not entries:
            continue
        try:
            entries = _bind_audio_content_hashes(entries, raw_roots=roots)
        except OSError as exc:
            print(f"[실패] {pool.tag} raw content hash 실패: {exc}", file=sys.stderr)
            return 1
        hashed_by_tag[pool.tag] = entries
        sources_by_tag[pool.tag] = sources

    prepared: dict[str, tuple[list[dict], int, list[Path]]] = {}
    public_lineage_evidence: dict | None = None
    if args.allow_corpus_leak:
        for tag, entries in sorted(hashed_by_tag.items()):
            assigned = assign_splits(
                entries, {"train": 0.9, "val": 0.05}, seed=args.seed
            )
            prepared[tag] = (assigned, 0, sources_by_tag[tag])
    elif hashed_by_tag:
        clip_lineage = holdout_summary.get("clip_lineage")
        if not isinstance(clip_lineage, dict):
            print(
                "[BLOCKED] canonical holdout에 content SHA + authoritative lineage가 없습니다",
                file=sys.stderr,
            )
            return 1
        try:
            lineage_kwargs: dict[str, object] = {}
            # 기존 fixture/diagnostic 호출은 인자를 주지 않아도 동작해야 한다. 실제
            # training 세대에서만 source-pool 추가 exclusion을 전달한다.
            if extra_excluded_basenames:
                lineage_kwargs["extra_excluded_basenames"] = extra_excluded_basenames
            lineage_build = build_public_lineage(
                hashed_by_tag,
                tag_roots=sources_by_tag,
                repo_root=REPO_ROOT,
                holdout_lineage=clip_lineage,
                **lineage_kwargs,
            )
            # transitive component가 tag를 가로지를 수 있으므로 모든 tag를 한 번에
            # 분할한다. tag별 shuffle은 같은 component를 서로 다른 split에 넣을 수 있다.
            combined: list[dict] = []
            for tag, entries in sorted(lineage_build.entries_by_tag.items()):
                for entry in entries:
                    item = dict(entry)
                    item["_public_lineage_tag"] = tag
                    combined.append(item)
            assigned_all = assign_splits(
                combined,
                {"train": 0.9, "val": 0.05},
                seed=args.seed,
                group_key="group_id",
            )
            assigned_by_tag: dict[str, list[dict]] = {
                tag: [] for tag in hashed_by_tag
            }
            for entry in assigned_all:
                item = dict(entry)
                tag = str(item.pop("_public_lineage_tag"))
                assigned_by_tag[tag].append(item)
            manifest_lineage = validate_public_manifest_lineage(assigned_by_tag)
        except PublicLineageError as exc:
            print(f"[BLOCKED] public corpus 계보를 증명할 수 없습니다: {exc}", file=sys.stderr)
            return 1
        for tag, entries in sorted(assigned_by_tag.items()):
            if entries:
                prepared[tag] = (
                    entries,
                    lineage_build.excluded_by_tag[tag],
                    sources_by_tag[tag],
                )
        public_lineage_evidence = dict(lineage_build.evidence)
        public_lineage_evidence.update(
            {
                "manifest_component_count": manifest_lineage["component_count"],
                "manifest_component_membership_sha256": manifest_lineage[
                    "component_membership_sha256"
                ],
            }
        )
        if source_pool_exclusion_evidence:
            ordered = sorted(extra_excluded_basenames)
            public_lineage_evidence["recorded_source_pool_exclusion"] = {
                "files": source_pool_exclusion_evidence,
                "basename_count": len(ordered),
                "basename_sha256": canonical_json_sha256(ordered),
            }

    # ---- 선언했는데 못 만든 태그 = 조용한 폴백의 씨앗 -----------------------------
    # 여기서 멈추지 않으면 synth_dataset 이 그 태그를 합성원으로 **로그 없이** 대체하고,
    # 학습은 선언한 혼합비와 다른 데이터로 돈다.
    missing = [tag for tag in plan.required_tags() if tag not in prepared]
    if missing:
        print("", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        print(
            "[실패] 선언된 소스 태그의 원본을 찾지 못했습니다: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "  스캔한 루트: " + ", ".join(str(p) for p in roots),
            file=sys.stderr,
        )
        print(
            "  이 상태로 학습하면 synth_dataset 이 없는 태그를 **조용히** 합성원으로\n"
            "  폴백하므로, 선언한 source_mix_ratio 와 다른 데이터로 돌게 됩니다.\n"
            f"  둘 중 하나를 하세요: (1) 원본을 {roots[0]} 아래 태그 이름 디렉터리로 받는다,\n"
            f"  (2) {args.data_config} 의 source_mix_ratio 에서 그 태그를 지운다.",
            file=sys.stderr,
        )
        print("=" * 78, file=sys.stderr)
        return 1

    if holdout_path is not None:
        try:
            before_commit = validate_holdout_contract(
                holdout_path,
                repo_root=REPO_ROOT,
                expected_sha256=expected_holdout_sha256,
            )
        except HoldoutContractError as exc:
            print(f"[실패] manifest commit 직전 holdout 재검증 실패: {exc}", file=sys.stderr)
            return 1
        if before_commit["sha256"] != holdout_summary["sha256"]:
            print("[실패] 준비 시작 후 canonical holdout bytes가 바뀌었습니다", file=sys.stderr)
            return 1

    try:
        generation = write_generation_transactionally(
            prepared,
            out_dir=out_dir,
            data_config=REPO_ROOT / args.data_config,
            holdout_path=holdout_path,
            seed=args.seed,
            training_eligible=not args.allow_corpus_leak,
            raw_roots=roots,
            public_lineage=public_lineage_evidence,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            f"[실패] manifest 세대 commit 실패 — 기존 세대를 복구했습니다: {exc}",
            file=sys.stderr,
        )
        return 1


    if holdout_path is not None:
        try:
            after_commit = validate_holdout_contract(
                holdout_path,
                repo_root=REPO_ROOT,
                expected_sha256=expected_holdout_sha256,
            )
        except HoldoutContractError as exc:
            print(f"[실패] manifest commit 후 holdout 재검증 실패: {exc}", file=sys.stderr)
            return 1
        if after_commit["sha256"] != holdout_summary["sha256"]:
            print("[실패] manifest commit 중 canonical holdout bytes가 바뀌었습니다", file=sys.stderr)
            return 1

    written: dict[str, int] = {}
    for tag, (entries, dropped, sources) in prepared.items():
        out = out_dir / f"{tag}.jsonl"
        written[tag] = len(entries)
        n_train = sum(1 for e in entries if e["split"] == "train")
        total_h = sum(e["duration_s"] for e in entries) / 3600.0
        suffix = f", held-out 제외 {dropped}" if dropped else ""
        where = ", ".join(str(p.relative_to(REPO_ROOT)) for p in sources)
        print(
            f"{tag}: {len(entries)}개 파일 ({total_h:.1f}h), train {n_train}{suffix} "
            f"← {where} → {out}"
        )

    print(
        f"완료: manifest {len(written)}개 ({', '.join(sorted(written))}), "
        f"build_id={generation['build_id']}"
    )
    if args.allow_corpus_leak:
        print(
            "[진단 전용 종료] holdout을 적용하지 않은 세대는 학습에 사용할 수 없습니다.",
            file=sys.stderr,
        )
        return DIAGNOSTIC_ONLY_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
