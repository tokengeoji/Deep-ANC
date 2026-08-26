#!/usr/bin/env python3
"""v1/v2 녹음 소스 풀의 잘린 ``clips`` provenance를 결정적으로 복구한다.

이 도구는 오디오 지문으로 원본을 *추정*하지 않는다. Git에 보존된 당시 builder를
그대로 읽어 당시 입력·seed·RNG 호출 경계·v2 exclusion을 재현한 뒤, 아래 조건을 모두
통과한 행에 한해서만 ``sources.csv``의 잘린 ``clips`` 열을 보충한다.

* historical CSV prefix, group_id, clip_count, path, sample rate가 모두 일치
* 당시 builder가 다시 만든 WAV와 보존된 WAV의 shape/rate/PCM이 exact 일치
  (정수 PCM인 경우에만 최대 1 LSB 허용; 현행 FLOAT WAV는 exact만 허용)
* v1과 v2 전체 행이 모두 통과 — 한 풀만 부분 복구하지 않음

풀 WAV와 ``data/recorded`` 아래 82개 세션은 읽기만 한다. 쓰기 대상은 명시한 CSV,
그 백업, active-session holdout, provenance report뿐이다. 기본 실행은 감사 전용이며,
실제 CSV 보충에는 ``--repair-csv``가 필요하다.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.holdout_contract import (  # noqa: E402
    HoldoutContractError,
    RECORDED_CONTENT_INTEGRITY_BOUNDARY,
    RECORDED_TREE_CONTENT_SNAPSHOT_ENCODING,
    RECORDED_TREE_SNAPSHOT_ENCODING,
    TreeMetadataSnapshot,
    read_regular_file_snapshot,
    reject_symlink_components,
    snapshot_regular_tree_metadata,
    validate_holdout_contract,
)
from deep_anc.data.public_lineage import (  # noqa: E402
    ESC50_METADATA,
    LIBRISPEECH_CHAPTERS,
    PublicLineageError,
    build_recorded_clip_lineage,
    esc50_lineage_keys,
    librispeech_lineage_keys,
    parse_esc50_metadata_bytes,
    parse_fma_tracks_bytes,
    parse_librispeech_chapters_bytes,
)

TARGET_RATE = 48_000
SESSION_SECONDS = 70.0
SESSIONS_PER_FAMILY = 20
EXPECTED_ACTIVE_SESSION_COUNT = 82
SEED = 20260804

V1_COMMIT = "7c7800fa94a8c5e156e049be896fd0b9586d983f"
V2_COMMIT = "0cb13b14e36c334783953aedd47aa0bc13d0fb6a"
BUILDER_PATH = "scripts/data/build_recording_sources.py"
BUILDER_SHA256 = {
    V1_COMMIT: "26d7fa6987310d6fd58f68a117a67a5e9397453aa96b61d1713838fc37452140",
    V2_COMMIT: "fc0f5fa428be4897291bcd486793ce1c08d2f5faa30306c463b8a04560fe71bc",
}

# 파일 mtime 경계와 CSV prefix 80/80 대조로 확정한 실제 호출 단위다. 한 tuple 안에서는
# RNG가 이어지고 tuple이 바뀌면 CLI 프로세스가 다시 시작돼 같은 seed로 reset된다.
V1_INVOCATIONS = (
    ("environment", "machine"),
    ("speech",),
    ("music",),
)
V2_INVOCATIONS = (("environment", "machine", "speech", "music"),)

# 입력 트리나 historical 호출 규약이 조용히 바뀌는 것을 막는 실측 기준선.
EXPECTED_POOL_ROWS = 80
EXPECTED_V1_PLACEMENTS = 983
EXPECTED_V1_UNIQUE = 854
EXPECTED_V1_TRUNCATED_EXCLUSION_UNIQUE = 691
EXPECTED_V2_PLACEMENTS = 416
FMA_TRACKS_CSV_SHA256 = "f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b"

REQUIRED_SYNTHETIC_MANIFESTS = (
    "dns_fullband.jsonl",
    "speech.jsonl",
    "music.jsonl",
    "demand.jsonl",
    "machine.jsonl",
    "esc50.jsonl",
)

CANONICAL_OUTPUTS = {
    "v1_csv": "data/source_pool/sources.csv",
    "v2_csv": "data/source_pool_v2/sources.csv",
    "report": "results/provenance",
    "active_holdout": "data/manifests/recorded_holdout.json",
    "regrouped_manifest": "data/manifests/recorded_regrouped.jsonl",
}


class ProvenanceError(RuntimeError):
    """복구 근거가 하나라도 불완전할 때 쓰는 fail-closed 예외."""


@dataclass(frozen=True)
class RowPlan:
    pool_name: str
    builder_commit: str
    source_family: str
    session_index: int
    group_id: str
    wav_path: str
    source_paths: tuple[str, ...]
    clips: tuple[str, ...]

    @property
    def key(self) -> tuple[str, int]:
        return self.source_family, self.session_index


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_builder_source(repo_root: Path, commit: str) -> str:
    """Git object와 고정 SHA를 모두 확인한 historical builder 원문을 반환한다."""

    try:
        raw = subprocess.check_output(
            ["git", "show", f"{commit}:{BUILDER_PATH}"],
            cwd=repo_root,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(
            f"historical builder를 읽을 수 없습니다: {commit}:{BUILDER_PATH}: {exc}"
        ) from exc
    expected = BUILDER_SHA256.get(commit)
    actual = sha256_bytes(raw)
    if expected is None or actual != expected:
        raise ProvenanceError(
            f"historical builder SHA 불일치: {commit} expected={expected} actual={actual}"
        )
    return raw.decode("utf-8")


def load_historical_builder(repo_root: Path, commit: str) -> types.ModuleType:
    """보존된 builder를 쓰기 함수는 호출하지 않는 전용 module로 로드한다."""

    source = _git_builder_source(repo_root, commit)
    module = types.ModuleType(f"historical_recording_sources_{commit[:8]}")
    module.__file__ = str(repo_root / BUILDER_PATH)
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    # historical 파일은 __file__에서 같은 값을 만들지만, worktree 경로를 명시해 감사한다.
    module.REPO_ROOT = repo_root
    return module


def _all_sources(module: types.ModuleType, repo_root: Path) -> dict[str, list[tuple[Path, str]]]:
    esc50 = module.collect_esc50(repo_root / "data/raw/noise/esc50")
    return {
        "environment": list(esc50.get("environment", [])),
        "machine": list(esc50.get("machine", [])),
        "speech": list(module.collect_tree(repo_root / "data/raw/speech", (".flac", ".wav"))),
        "music": list(module.collect_tree(repo_root / "data/raw/music", (".mp3", ".wav"))),
    }


def _select_family(
    module: types.ModuleType,
    *,
    pool_name: str,
    commit: str,
    family: str,
    candidates: list[tuple[Path, str]],
    rng: random.Random,
) -> list[RowPlan]:
    """historical ``build_family``의 선택 부분을 쓰기 없이 그대로 수행한다."""

    if not candidates:
        raise ProvenanceError(f"{pool_name}/{family}: 후보가 없습니다")
    by_group: dict[str, list[Path]] = {}
    for path, group in candidates:
        by_group.setdefault(str(group), []).append(path)
    groups = sorted(by_group)
    rng.shuffle(groups)
    if not groups:
        raise ProvenanceError(f"{pool_name}/{family}: group이 없습니다")

    total_samples = int(SESSION_SECONDS * TARGET_RATE)
    plans: list[RowPlan] = []
    for index in range(SESSIONS_PER_FAMILY):
        group = groups[index % len(groups)]
        pool = list(by_group[group])
        rng.shuffle(pool)
        used_paths: list[Path] = []
        collected = 0
        for path in pool:
            try:
                signal, rate = module.load_audio(path)
            except Exception:  # historical builder도 파일 하나만 건너뛴다.
                continue
            signal = module.to_target_rate(signal, rate)
            if signal.size < int(0.5 * TARGET_RATE):
                continue
            used_paths.append(path)
            collected += int(signal.size)
            if collected >= total_samples * 1.2:
                break
        if not used_paths:
            raise ProvenanceError(f"{pool_name}/{family}/{index}: 사용 가능한 clip이 없습니다")
        wav_path = f"data/{pool_name}/{family}/{family}_{index:03d}.wav"
        plans.append(
            RowPlan(
                pool_name=pool_name,
                builder_commit=commit,
                source_family=family,
                session_index=index,
                group_id=f"{family}-{group}".replace("_", "-").lower(),
                wav_path=wav_path,
                source_paths=tuple(str(path.resolve()) for path in used_paths),
                clips=tuple(path.name for path in used_paths),
            )
        )
    return plans


def _select_invocations(
    module: types.ModuleType,
    *,
    repo_root: Path,
    pool_name: str,
    commit: str,
    invocations: tuple[tuple[str, ...], ...],
    excluded_names: set[str] | None = None,
) -> list[RowPlan]:
    sources = _all_sources(module, repo_root)
    if excluded_names is not None:
        sources = {
            family: [(path, group) for path, group in values if path.name not in excluded_names]
            for family, values in sources.items()
        }
    plans: list[RowPlan] = []
    for invocation in invocations:
        rng = random.Random(SEED)
        for family in invocation:
            plans.extend(
                _select_family(
                    module,
                    pool_name=pool_name,
                    commit=commit,
                    family=family,
                    candidates=sources[family],
                    rng=rng,
                )
            )
    return plans


def reconstruct_plans(repo_root: Path) -> tuple[dict[str, list[RowPlan]], dict[str, Any]]:
    """v1 full list를 먼저 복원하고 그 historical 12-prefix로 v2 exclusion을 재현한다."""

    v1_module = load_historical_builder(repo_root, V1_COMMIT)
    v1 = _select_invocations(
        v1_module,
        repo_root=repo_root,
        pool_name="source_pool",
        commit=V1_COMMIT,
        invocations=V1_INVOCATIONS,
    )
    # v2 생성 당시 0cb13b1 clips_used_by_sessions()가 본 것은 v1 full list가 아니라
    # old builder의 used[:12] CSV였다. 복구 후 CSV를 읽으면 역사가 달라지므로, historical
    # v1 plan의 첫 12개로 그 당시 exclusion을 영구적으로 재구성한다.
    v1_truncated_exclusion = {clip for plan in v1 for clip in plan.clips[:12]}

    v2_module = load_historical_builder(repo_root, V2_COMMIT)
    v2 = _select_invocations(
        v2_module,
        repo_root=repo_root,
        pool_name="source_pool_v2",
        commit=V2_COMMIT,
        invocations=V2_INVOCATIONS,
        excluded_names=v1_truncated_exclusion,
    )

    counts = {
        "v1_rows": len(v1),
        "v1_placements": sum(len(plan.clips) for plan in v1),
        "v1_unique": len({clip for plan in v1 for clip in plan.clips}),
        "v1_historical_exclusion_unique": len(v1_truncated_exclusion),
        "v2_rows": len(v2),
        "v2_placements": sum(len(plan.clips) for plan in v2),
        "v2_unique": len({clip for plan in v2 for clip in plan.clips}),
    }
    expected = {
        "v1_rows": EXPECTED_POOL_ROWS,
        "v1_placements": EXPECTED_V1_PLACEMENTS,
        "v1_unique": EXPECTED_V1_UNIQUE,
        "v1_historical_exclusion_unique": EXPECTED_V1_TRUNCATED_EXCLUSION_UNIQUE,
        "v2_rows": EXPECTED_POOL_ROWS,
        "v2_placements": EXPECTED_V2_PLACEMENTS,
    }
    mismatches = {
        key: {"expected": value, "actual": counts.get(key)}
        for key, value in expected.items()
        if counts.get(key) != value
    }
    if mismatches:
        raise ProvenanceError(f"historical selection 기준선 불일치: {mismatches}")
    return {"source_pool": v1, "source_pool_v2": v2}, {
        "counts": counts,
        "expected": expected,
        "v1_invocations": [list(item) for item in V1_INVOCATIONS],
        "v2_invocations": [list(item) for item in V2_INVOCATIONS],
        "seed": SEED,
        "v2_exclusion_semantics": "historical v1 full plan의 used[:12] unique set",
    }


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProvenanceError(f"CSV header가 없습니다: {path}")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows, raw


def _parse_clips(row: dict[str, str], *, label: str) -> list[str]:
    try:
        value = json.loads(row.get("clips") or "[]")
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"{label}: clips JSON 오류: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProvenanceError(f"{label}: clips는 문자열 배열이어야 합니다")
    return value


def audit_csv_prefix(
    csv_path: Path, plans: list[RowPlan]
) -> tuple[dict[str, Any], list[str], list[dict[str, str]], bytes]:
    """CSV identity와 historical clips prefix를 전 행에서 검사한다."""

    fields, rows, raw = read_csv_rows(csv_path)
    expected = {plan.key: plan for plan in plans}
    seen: set[tuple[str, int]] = set()
    issues: list[str] = []
    details: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=2):
        label = f"{csv_path}:{position}"
        try:
            key = (str(row["source_family"]), int(row["session_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{label}: row key 오류: {exc}")
            continue
        if key in seen:
            issues.append(f"{label}: 중복 row {key}")
            continue
        seen.add(key)
        plan = expected.get(key)
        if plan is None:
            issues.append(f"{label}: historical plan에 없는 row {key}")
            continue
        try:
            declared_count = int(row["clip_count"])
            sample_rate = int(row["sample_rate_hz"])
            seconds = float(row["seconds"])
            stored = _parse_clips(row, label=label)
        except (KeyError, TypeError, ValueError, ProvenanceError) as exc:
            issues.append(str(exc))
            continue
        identity_checks = {
            "group_id": row.get("group_id") == plan.group_id,
            "path": PurePosixPath(str(row.get("path", ""))) == PurePosixPath(plan.wav_path),
            "sample_rate_hz": sample_rate == TARGET_RATE,
            "seconds": seconds == SESSION_SECONDS,
            "clip_count": declared_count == len(plan.clips),
            "clips_prefix": stored == list(plan.clips[: len(stored)]),
            "stored_not_longer": len(stored) <= len(plan.clips),
        }
        failed = [name for name, passed in identity_checks.items() if not passed]
        if failed:
            issues.append(f"{label}: {', '.join(failed)} 불일치")
        details.append(
            {
                "family": key[0],
                "session_index": key[1],
                "path": plan.wav_path,
                "declared_clips": len(stored),
                "reconstructed_clips": len(plan.clips),
                "missing_clips": max(0, len(plan.clips) - len(stored)),
                "prefix_pass": not failed,
            }
        )
    missing_rows = sorted(set(expected) - seen)
    if missing_rows:
        issues.append(f"historical row {len(missing_rows)}개 누락: {missing_rows[:8]}")
    return {
        "status": "PASS" if not issues else "FAIL",
        "csv_path": str(csv_path),
        "csv_sha256": sha256_bytes(raw),
        "row_count": len(rows),
        "expected_row_count": len(plans),
        "missing_clip_placements": sum(item["missing_clips"] for item in details),
        "issues": issues,
        "rows": details,
    }, fields, rows, raw


_WORKER_REPO: Path | None = None
_WORKER_MODULES: dict[str, types.ModuleType] = {}


def _worker_init(repo_root: str) -> None:
    global _WORKER_REPO, _WORKER_MODULES
    _WORKER_REPO = Path(repo_root)
    _WORKER_MODULES = {}


def _pcm_lsb(subtype: str) -> float | None:
    bits = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
    return 1.0 / (2 ** (bits[subtype] - 1)) if subtype in bits else None


def compare_pcm(expected: np.ndarray, actual: np.ndarray, subtype: str) -> dict[str, Any]:
    """FLOAT는 exact, 정수 PCM은 최대 1 LSB까지만 허용한다."""

    if expected.shape != actual.shape:
        return {
            "status": "FAIL",
            "shape_match": False,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "exact": False,
            "max_abs_error": None,
            "allowed_abs_error": 0.0,
        }
    exact = bool(np.array_equal(expected, actual))
    max_abs = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
    lsb = _pcm_lsb(subtype)
    allowed = 0.0 if lsb is None else lsb
    passed = exact or (lsb is not None and max_abs <= allowed)
    return {
        "status": "PASS" if passed else "FAIL",
        "shape_match": True,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "exact": exact,
        "max_abs_error": max_abs,
        "allowed_abs_error": allowed,
        "accepted_by": "exact" if exact else ("one_lsb" if passed else "none"),
    }


def verify_wav_plan(plan: RowPlan, repo_root: Path | None = None) -> dict[str, Any]:
    """한 행을 historical decode/resample/concatenate로 재합성해 WAV와 비교한다."""

    root = repo_root or _WORKER_REPO
    if root is None:
        raise RuntimeError("worker repo root가 설정되지 않았습니다")
    module = _WORKER_MODULES.get(plan.builder_commit)
    if module is None:
        module = load_historical_builder(root, plan.builder_commit)
        _WORKER_MODULES[plan.builder_commit] = module
    wav_path = root / plan.wav_path
    result: dict[str, Any] = {
        "family": plan.source_family,
        "session_index": plan.session_index,
        "path": plan.wav_path,
        "builder_commit": plan.builder_commit,
    }
    try:
        clips: list[np.ndarray] = []
        loaded_names: list[str] = []
        for value in plan.source_paths:
            path = Path(value)
            signal, rate = module.load_audio(path)
            signal = module.to_target_rate(signal, rate)
            clips.append(signal)
            loaded_names.append(path.name)
        if tuple(loaded_names) != plan.clips:
            raise ProvenanceError("검증 시 로드한 clip 순서가 plan과 달라졌습니다")
        expected = module.concatenate(clips, int(SESSION_SECONDS * TARGET_RATE))
        # 보존 WAV는 pathname으로 header/read/hash를 세 번 열지 않는다. 같은 fd에서
        # 고정한 한 byte snapshot을 header, PCM, SHA 증거 모두에 사용한다.
        wav_snapshot = read_regular_file_snapshot(
            wav_path,
            root=root,
            label=f"source-pool WAV {plan.wav_path}",
        )
        assert wav_snapshot.data is not None
        info = sf.info(io.BytesIO(wav_snapshot.data))
        actual, rate = sf.read(
            io.BytesIO(wav_snapshot.data), dtype="float32", always_2d=True
        )
        mono = actual[:, 0] if actual.ndim == 2 and actual.shape[1] == 1 else actual
        comparison = compare_pcm(expected, mono, str(info.subtype))
        result.update(
            {
                "status": "PASS"
                if rate == TARGET_RATE and int(info.channels) == 1 and comparison["status"] == "PASS"
                else "FAIL",
                "sample_rate_hz": int(rate),
                "channels": int(info.channels),
                "format": str(info.format),
                "subtype": str(info.subtype),
                "frames": int(info.frames),
                "wav_sha256": wav_snapshot.sha256,
                "pcm": comparison,
            }
        )
        if rate != TARGET_RATE:
            result["error"] = f"sample rate {rate} != {TARGET_RATE}"
        elif int(info.channels) != 1:
            result["error"] = f"channels {info.channels} != 1"
    except Exception as exc:  # worker 결과로 수집해 전체 transaction을 막는다.
        result.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    return result


def verify_all_wavs(
    plans_by_pool: dict[str, list[RowPlan]], repo_root: Path, *, jobs: int
) -> dict[str, list[dict[str, Any]]]:
    plans = [plan for pool in plans_by_pool.values() for plan in pool]
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in plans_by_pool}
    if jobs <= 1:
        _worker_init(str(repo_root))
        for index, plan in enumerate(plans, start=1):
            item = verify_wav_plan(plan)
            results[plan.pool_name].append(item)
            print(f"[PCM {index:3d}/{len(plans)}] {plan.wav_path}: {item['status']}", flush=True)
        return results

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs, initializer=_worker_init, initargs=(str(repo_root),)
    ) as executor:
        futures = {executor.submit(verify_wav_plan, plan): plan for plan in plans}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            plan = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # pragma: no cover - worker crash path
                item = {
                    "family": plan.source_family,
                    "session_index": plan.session_index,
                    "path": plan.wav_path,
                    "status": "FAIL",
                    "error": f"worker failure: {exc}",
                }
            results[plan.pool_name].append(item)
            done += 1
            print(f"[PCM {done:3d}/{len(plans)}] {plan.wav_path}: {item['status']}", flush=True)
    for values in results.values():
        values.sort(key=lambda item: (item["family"], int(item["session_index"])))
    return results


def render_repaired_csv(
    fields: list[str], rows: list[dict[str, str]], plans: list[RowPlan]
) -> bytes:
    """다른 열과 행 순서를 보존하고 clips만 full historical list로 바꾼다."""

    if "clips" not in fields:
        raise ProvenanceError("CSV에 clips 열이 없습니다")
    by_key = {plan.key: plan for plan in plans}
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for original in rows:
        row = dict(original)
        key = (str(row["source_family"]), int(row["session_index"]))
        if key not in by_key:
            raise ProvenanceError(f"repair 대상에 historical plan이 없습니다: {key}")
        row["clips"] = json.dumps(list(by_key[key].clips), ensure_ascii=False)
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _safe_write_target(path: Path) -> None:
    if path.suffix.lower() in {".wav", ".flac", ".mp3", ".npz"}:
        raise ProvenanceError(f"오디오/측정 파일은 쓸 수 없습니다: {path}")
    if "recorded" in {part.casefold() for part in path.parts} and path.suffix != ".json":
        raise ProvenanceError(f"recorded 세션 경로는 쓸 수 없습니다: {path}")


def _canonical_cli_path(value: str, *, field: str, expected: str, repo_root: Path) -> Path:
    """권위 복구 출력은 임의 경로나 symlink alias를 받지 않는다."""

    raw = PurePosixPath(value.replace("\\", "/"))
    if raw.is_absolute() or raw.as_posix() != expected or any(
        part in {"", ".", ".."} for part in raw.parts
    ):
        raise ProvenanceError(f"--{field.replace('_', '-')}는 {expected!r}만 허용합니다")
    candidate = Path(os.path.abspath(repo_root / expected))
    boundary = Path(os.path.abspath(repo_root))
    try:
        candidate.relative_to(boundary)
    except ValueError as exc:  # pragma: no cover - exact 상대경로가 이미 막지만 방어층 유지
        raise ProvenanceError(f"--{field.replace('_', '-')}가 저장소 밖입니다") from exc
    if candidate == boundary / "data/recorded" or boundary / "data/recorded" in candidate.parents:
        raise ProvenanceError("data/recorded 아래는 provenance 출력 대상으로 허용하지 않습니다")
    try:
        reject_symlink_components(
            candidate,
            root=boundary,
            allow_missing_leaf=not candidate.exists(),
        )
    except HoldoutContractError as exc:
        raise ProvenanceError(str(exc)) from exc
    return candidate


def validate_cli_output_contract(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Path]:
    """오래 걸리는 historical 재합성 전에 모든 권위 출력 경로를 고정한다."""

    values: dict[str, Path] = {}
    for field, expected in CANONICAL_OUTPUTS.items():
        values[field] = _canonical_cli_path(
            str(getattr(args, field)), field=field, expected=expected, repo_root=repo_root
        )
    return values


def publish_immutable_report(
    report_dir: Path,
    payload: bytes,
    *,
    canonical: bool,
    repo_root: Path,
) -> tuple[Path, str]:
    """fsync된 content-addressed report를 overwrite 없는 hard-link publish로 노출한다."""

    expected_dir = Path(os.path.abspath(repo_root / CANONICAL_OUTPUTS["report"]))
    if Path(os.path.abspath(report_dir)) != expected_dir:
        raise ProvenanceError(f"report directory는 {expected_dir}만 허용합니다")
    try:
        reject_symlink_components(report_dir, root=repo_root)
    except HoldoutContractError as exc:
        raise ProvenanceError(str(exc)) from exc
    digest = sha256_bytes(payload)
    stem = "source_pool_provenance_report" if canonical else "source_pool_provenance_audit"
    target = report_dir / f"{stem}.{digest}.json"
    if target.exists():
        snapshot = read_regular_file_snapshot(
            target, root=repo_root, label="immutable provenance report"
        )
        if snapshot.sha256 != digest or snapshot.data != payload:
            raise ProvenanceError(f"동일 content-addressed report 경로의 bytes가 다릅니다: {target}")
        return target, digest

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{stem}.", suffix=".tmp", dir=report_dir
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
            published = True
        except FileExistsError:
            snapshot = read_regular_file_snapshot(
                target, root=repo_root, label="immutable provenance report"
            )
            if snapshot.sha256 != digest or snapshot.data != payload:
                raise ProvenanceError(
                    f"경합으로 생긴 content-addressed report bytes가 다릅니다: {target}"
                )
        directory_fd = os.open(report_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
        if published:
            directory_fd = os.open(report_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    return target, digest


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write(path: Path, data: bytes) -> None:
    _safe_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def validate_then_atomic_write_holdout(
    path: Path, data: bytes, *, repo_root: Path
) -> dict[str, Any]:
    """기존 canonical holdout를 보존한 채 새 bytes를 먼저 full 검증한다."""

    _safe_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, candidate_name = tempfile.mkstemp(
        prefix=f".{path.name}.candidate.", dir=path.parent
    )
    candidate = Path(candidate_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        summary = validate_holdout_contract(candidate, repo_root=repo_root)
        os.replace(candidate, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return summary
    finally:
        candidate.unlink(missing_ok=True)


def repair_csvs_transactionally(
    audits: dict[str, dict[str, Any]],
    csv_inputs: dict[str, tuple[Path, list[str], list[dict[str, str]], bytes]],
    plans_by_pool: dict[str, list[RowPlan]],
) -> dict[str, Any]:
    """전 pool PASS와 TOCTOU SHA 재검사 뒤 CSV clips만 보충한다."""

    failed = [name for name, value in audits.items() if value["status"] != "PASS"]
    if failed:
        raise ProvenanceError(f"전체 provenance PASS 전에는 repair할 수 없습니다: {failed}")

    prepared: dict[str, bytes] = {}
    backups: dict[str, str] = {}
    for name, (path, fields, rows, before) in csv_inputs.items():
        current = path.read_bytes()
        if sha256_bytes(current) != sha256_bytes(before):
            raise ProvenanceError(f"감사 후 CSV가 바뀌었습니다(TOCTOU): {path}")
        prepared[name] = render_repaired_csv(fields, rows, plans_by_pool[name])

    # 모든 byte payload를 먼저 만들고, 원본 SHA 이름의 immutable backup을 확보한다.
    for name, (path, _fields, _rows, before) in csv_inputs.items():
        after = prepared[name]
        if after == before:
            continue
        backup = path.with_name(f"{path.name}.pre-repair.{sha256_bytes(before)[:12]}.bak")
        if backup.exists():
            if backup.read_bytes() != before:
                raise ProvenanceError(f"동명 backup 내용이 다릅니다: {backup}")
        else:
            _safe_write_target(backup)
            with backup.open("xb") as handle:
                handle.write(before)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(backup.parent)
        backups[name] = str(backup)

    # 교체 전 모든 target의 결과/원본을 먼저 등록한다. os.replace가 성공한 직후
    # directory fsync에서 예외가 나도 현재 target이 rollback 목록에서 빠지지 않는다.
    written: dict[str, Any] = {
        name: {
            "path": str(path),
            "changed": prepared[name] != before,
            "before_sha256": sha256_bytes(before),
            "after_sha256": sha256_bytes(prepared[name]),
            "backup": backups.get(name),
        }
        for name, (path, _fields, _rows, before) in csv_inputs.items()
    }
    try:
        for name, (path, _fields, _rows, before) in csv_inputs.items():
            after = prepared[name]
            if after != before:
                atomic_write(path, after)
        for name, (path, _fields, _rows, _before) in csv_inputs.items():
            snapshot = read_regular_file_snapshot(
                path,
                root=path.parent,
                label=f"post-repair CSV {name}",
            )
            if snapshot.sha256 != written[name]["after_sha256"]:
                raise ProvenanceError(
                    f"CSV transaction postcondition SHA 불일치: {path}"
                )
    except BaseException as original_error:
        rollback_errors: list[str] = []
        # written 상태가 아니라 prepared target 전체를 실제 current byte SHA로
        # 재감사한다. 각 복구가 실패해도 나머지 target 복구를 끝까지 시도한다.
        for name, (path, _fields, _rows, before) in csv_inputs.items():
            try:
                try:
                    current = read_regular_file_snapshot(
                        path,
                        root=path.parent,
                        label=f"rollback current CSV {name}",
                    )
                    current_sha = current.sha256
                except (FileNotFoundError, HoldoutContractError):
                    current_sha = None
                before_sha = sha256_bytes(before)
                if current_sha != before_sha:
                    atomic_write(path, before)
                restored = read_regular_file_snapshot(
                    path,
                    root=path.parent,
                    label=f"rollback restored CSV {name}",
                )
                if restored.sha256 != before_sha or restored.data != before:
                    raise ProvenanceError(
                        f"rollback bytes SHA/postcondition 불일치: {path}"
                    )
                # unchanged target도 transaction 실패 후 directory durability를
                # 명시적으로 완료한다.
                _fsync_directory(path.parent)
            except BaseException as rollback_error:
                rollback_errors.append(f"{name}: {rollback_error!r}")
        if rollback_errors:
            raise ProvenanceError(
                "CSV transaction 실패와 rollback 오류가 함께 발생했습니다; "
                f"original={original_error!r}; rollback={rollback_errors}"
            ) from original_error
        raise
    return written


def _normalise_repo_path(value: str, repo_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def collect_active_sessions(recorded_root: Path, repo_root: Path) -> list[dict[str, str]]:
    sessions: list[dict[str, str]] = []
    for directory in sorted(path for path in recorded_root.iterdir() if path.is_dir()):
        meta_path = directory / "session.json"
        if not meta_path.is_file():
            raise ProvenanceError(f"active session에 session.json이 없습니다: {directory}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProvenanceError(f"active session metadata 오류: {meta_path}: {exc}") from exc
        program = meta.get("program")
        if not isinstance(program, dict) or program.get("type") != "file" or not program.get("file"):
            raise ProvenanceError(f"active session program.file이 없습니다: {meta_path}")
        sessions.append(
            {
                "session_id": str(meta.get("session_id") or directory.name),
                "source_family": str(meta.get("source_family") or ""),
                "source_wav": _normalise_repo_path(str(program["file"]), repo_root),
                "session_dir": str(directory.resolve()),
            }
        )
    if len(sessions) != EXPECTED_ACTIVE_SESSION_COUNT:
        raise ProvenanceError(
            "canonical provenance는 active 82세션 exact tree만 허용합니다: "
            f"actual={len(sessions)}, root={recorded_root}"
        )
    return sessions


def build_active_holdout(
    sessions: list[dict[str, str]],
    plans_by_pool: dict[str, list[RowPlan]],
    *,
    csv_hashes: dict[str, str],
    repo_root: Path | None = None,
    report_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, RowPlan]]:
    """실제로 재생된 pool row만 포함하는 canonical holdout을 만든다."""

    by_wav = {plan.wav_path: plan for plans in plans_by_pool.values() for plan in plans}
    used_plans: dict[str, RowPlan] = {}
    families: dict[str, set[str]] = {}
    for session in sessions:
        plan = by_wav.get(session["source_wav"])
        if plan is None:
            raise ProvenanceError(
                f"active session이 historical pool 밖 WAV를 재생했습니다: "
                f"{session['session_id']} -> {session['source_wav']}"
            )
        if session["source_family"] and session["source_family"] != plan.source_family:
            raise ProvenanceError(
                f"source_family 불일치: {session['session_id']} metadata={session['source_family']} "
                f"pool={plan.source_family}"
            )
        used_plans[plan.wav_path] = plan
        families.setdefault(plan.source_family, set()).update(clip_key(item) for item in plan.clips)
    source_rows = sorted(used_plans)
    payload = {
        "purpose": (
            "active recorded session이 실제 재생한 source-pool row의 complete historical clip "
            "provenance. 합성 manifest 생성 전에 전부 제외한다."
        ),
        "scope": "active_sessions_only",
        "active_session_count": len(sessions),
        "active_source_row_count": len(source_rows),
        "sources_csv": [
            "data/source_pool/sources.csv",
            "data/source_pool_v2/sources.csv",
        ],
        "sources_csv_sha256": csv_hashes,
        "source_rows": source_rows,
        "families": {name: sorted(values) for name, values in sorted(families.items())},
        "total_clips": sum(len(values) for values in families.values()),
    }
    if report_path is not None:
        payload["provenance_report"] = report_path
    if repo_root is not None:
        try:
            payload["clip_lineage"] = build_recorded_clip_lineage(
                payload["families"], repo_root=repo_root
            )
        except PublicLineageError as exc:
            raise ProvenanceError(str(exc)) from exc
    return payload, used_plans


def bind_holdout_to_fixed_report(
    holdout: dict[str, Any], *, report_path: Path, repo_root: Path
) -> dict[str, Any]:
    """이미 기록·고정된 report의 byte SHA를 holdout에 결속한다.

    report에 holdout SHA를 넣지 않으므로 순환 hash가 생기지 않는다. report를 먼저
    atomic write하고 그 실제 bytes를 hash한 뒤에만 이 함수를 호출해야 한다.
    """

    if not report_path.is_file():
        raise ProvenanceError(f"고정할 provenance report가 없습니다: {report_path}")
    expected_relative = _normalise_repo_path(str(report_path), repo_root)
    bound = dict(holdout)
    bound["provenance_report"] = expected_relative
    bound["provenance_report_sha256"] = sha256_file(report_path)
    return bound


def clip_key(value: str) -> str:
    return str(value).strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _read_recorded_splits(manifest_path: Path) -> dict[str, str]:
    splits: dict[str, str] = {}
    for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            session_id, split = str(row["session_id"]), str(row["split"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ProvenanceError(f"{manifest_path}:{number}: split manifest 오류: {exc}") from exc
        if session_id in splits and splits[session_id] != split:
            raise ProvenanceError(f"session split 중복 충돌: {session_id}")
        splits[session_id] = split
    return splits


def audit_recorded_clip_split_leak(
    sessions: list[dict[str, str]],
    used_plans: dict[str, RowPlan],
    manifest_path: Path,
) -> dict[str, Any]:
    splits = _read_recorded_splits(manifest_path)
    by_clip: dict[str, dict[str, set[str]]] = {}
    missing_sessions: list[str] = []
    for session in sessions:
        split = splits.get(session["session_id"])
        if split is None:
            missing_sessions.append(session["session_id"])
            continue
        plan = used_plans[session["source_wav"]]
        for clip in plan.clips:
            item = by_clip.setdefault(clip_key(clip), {"splits": set(), "sessions": set()})
            item["splits"].add(split)
            item["sessions"].add(session["session_id"])
    violations = [
        {"clip": clip, "splits": sorted(item["splits"]), "sessions": sorted(item["sessions"])}
        for clip, item in sorted(by_clip.items())
        if len(item["splits"]) > 1
    ]
    passed = not missing_sessions and not violations
    return {
        "status": "PASS" if passed else "FAIL",
        "manifest": str(manifest_path),
        "missing_sessions": missing_sessions,
        "cross_split_clip_count": len(violations),
        "cross_split_clips": violations,
    }


def audit_synthetic_manifests(
    holdout: dict[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    holdout_keys = {
        clip_key(value)
        for values in holdout["families"].values()
        for value in values
    }
    missing: list[str] = []
    overlaps: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for name in REQUIRED_SYNTHETIC_MANIFESTS:
        path = manifest_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        keys: set[str] = set()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                keys.add(clip_key(str(row["path"])))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ProvenanceError(f"{path}:{number}: synthetic manifest 오류: {exc}") from exc
        counts[name] = len(keys)
        hit = sorted(keys & holdout_keys)
        if hit:
            overlaps[name] = hit
    passed = not missing and not overlaps
    return {
        "status": "PASS" if passed else ("BLOCKED" if missing else "FAIL"),
        "manifest_dir": str(manifest_dir),
        "required": list(REQUIRED_SYNTHETIC_MANIFESTS),
        "missing": missing,
        "unique_clip_counts": counts,
        "overlap_count": sum(len(values) for values in overlaps.values()),
        "overlaps": overlaps,
    }


def parse_fma_tracks(path: Path) -> dict[int, tuple[str, str]]:
    """FMA multi-row tracks.csv에서 track_id -> (artist_id, album_id)를 읽는다."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            level0 = next(reader)
            level1 = next(reader)
        except StopIteration as exc:
            raise ProvenanceError(f"FMA tracks.csv header가 불완전합니다: {path}") from exc
        width = max(len(level0), len(level1))
        level0 += [""] * (width - len(level0))
        level1 += [""] * (width - len(level1))

        def find_column(group: str, field: str) -> int:
            hits = [
                index
                for index, (a, b) in enumerate(zip(level0, level1))
                if a.strip().casefold() == group and b.strip().casefold() == field
            ]
            if len(hits) != 1:
                raise ProvenanceError(
                    f"FMA tracks.csv column {group}/{field}가 정확히 하나가 아닙니다: {hits}"
                )
            return hits[0]

        artist_col = find_column("artist", "id")
        album_col = find_column("album", "id")
        mapping: dict[int, tuple[str, str]] = {}
        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue
            track_id = int(row[0])
            if max(artist_col, album_col) >= len(row):
                raise ProvenanceError(f"FMA track {track_id}: metadata column 누락")
            artist, album = row[artist_col].strip(), row[album_col].strip()
            if not artist or not album:
                raise ProvenanceError(f"FMA track {track_id}: artist/album ID 누락")
            mapping[track_id] = (artist, album)
    if not mapping:
        raise ProvenanceError(f"FMA track mapping이 비었습니다: {path}")
    return mapping


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_lineage_component_plan(
    sessions: list[dict[str, str]],
    used_plans: dict[str, RowPlan],
    tracks_path: Path,
    chapters_path: Path | None = None,
    esc_metadata_path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """모든 active session을 원본 계보의 transitive component로 묶는다.

    edge는 세 종류다: 모든 family의 동일 원본 clip, music artist/album, speech
    reader/Gutenberg book 및 ESC-50 original src_file. 한 세션이 여러 계보를 섞으면 그 계보를 쓰는 세션 전체가 같은 component가
    된다. FMA metadata가 없으면 일부 component만 만들어 안전한 척하지 않고 차단한다.
    """

    if not tracks_path.is_file():
        return {
            "status": "BLOCKED",
            "tracks_csv": str(tracks_path),
            "reason": (
                "FMA tracks.csv가 없어 shared clip + music artist/album + speech speaker/book "
                "통합 component를 완성할 수 없습니다"
            ),
        }
    metadata_root = repo_root if repo_root is not None else tracks_path.parent
    try:
        tracks_snapshot = read_regular_file_snapshot(
            tracks_path,
            root=metadata_root,
            label="FMA tracks.csv",
        )
        assert tracks_snapshot.data is not None
        mapping = parse_fma_tracks_bytes(tracks_snapshot.data)
    except (OSError, ValueError, PublicLineageError) as exc:
        return {"status": "FAIL", "tracks_csv": str(tracks_path), "reason": str(exc)}
    chapters: dict[int, tuple[int, int]] | None = None
    chapters_sha256: str | None = None
    if any(session.get("source_family") == "speech" for session in sessions):
        if chapters_path is None or not chapters_path.is_file():
            return {
                "status": "BLOCKED",
                "tracks_csv": str(tracks_path),
                "librispeech_chapters_path": (
                    None if chapters_path is None else str(chapters_path)
                ),
                "reason": (
                    "LibriSpeech CHAPTERS.TXT가 없어 speech reader/Gutenberg book "
                    "component를 완성할 수 없습니다"
                ),
            }
        metadata_root = repo_root if repo_root is not None else chapters_path.parent
        try:
            chapters_snapshot = read_regular_file_snapshot(
                chapters_path,
                root=metadata_root,
                label="LibriSpeech CHAPTERS.TXT",
            )
            assert chapters_snapshot.data is not None
            chapters = parse_librispeech_chapters_bytes(chapters_snapshot.data)
            chapters_sha256 = chapters_snapshot.sha256
        except (OSError, ValueError, PublicLineageError) as exc:
            return {
                "status": "FAIL",
                "tracks_csv": str(tracks_path),
                "librispeech_chapters_path": str(chapters_path),
                "reason": str(exc),
            }

    esc_metadata: dict[str, str] | None = None
    esc_metadata_sha256: str | None = None
    if any(
        session.get("source_family") in {"environment", "machine"}
        for session in sessions
    ):
        if esc_metadata_path is None or not esc_metadata_path.is_file():
            return {
                "status": "BLOCKED",
                "tracks_csv": str(tracks_path),
                "esc50_metadata_path": (
                    None if esc_metadata_path is None else str(esc_metadata_path)
                ),
                "reason": (
                    "ESC-50 esc50.csv가 없어 environment/machine original src_file "
                    "component를 완성할 수 없습니다"
                ),
            }
        metadata_root = repo_root if repo_root is not None else esc_metadata_path.parent
        try:
            esc_snapshot = read_regular_file_snapshot(
                esc_metadata_path,
                root=metadata_root,
                label="ESC-50 esc50.csv",
            )
            assert esc_snapshot.data is not None
            esc_metadata = parse_esc50_metadata_bytes(esc_snapshot.data)
            esc_metadata_sha256 = esc_snapshot.sha256
        except (OSError, ValueError, PublicLineageError) as exc:
            return {
                "status": "FAIL",
                "tracks_csv": str(tracks_path),
                "esc50_metadata_path": str(esc_metadata_path),
                "reason": str(exc),
            }
    session_ids = [item["session_id"] for item in sessions]
    if len(session_ids) != len(set(session_ids)):
        return {"status": "FAIL", "reason": "active session_id가 중복됩니다"}
    dsu = _DisjointSet(session_ids)
    owner: dict[tuple[str, str], str] = {}
    missing_music: dict[str, list[int]] = {}
    lineage_by_session: dict[str, list[dict[str, str]]] = {}
    for session in sessions:
        try:
            plan = used_plans[session["source_wav"]]
        except KeyError:
            return {
                "status": "FAIL",
                "reason": f"active session source plan 누락: {session['session_id']}",
            }
        lineage: list[dict[str, str]] = []
        for clip in plan.clips:
            shared_key = ("clip", clip_key(clip))
            previous = owner.setdefault(shared_key, session["session_id"])
            dsu.union(previous, session["session_id"])
            lineage.append({"kind": "clip", "id": shared_key[1]})

            family = session["source_family"]
            keys: list[tuple[str, str]] = []
            if family == "music":
                stem = Path(clip).stem
                if not stem.isdigit() or int(stem) not in mapping:
                    missing_music.setdefault(session["session_id"], []).append(
                        int(stem) if stem.isdigit() else -1
                    )
                    continue
                artist, album = mapping[int(stem)]
                lineage.extend(
                    [
                        {"kind": "music_track", "id": stem},
                        {"kind": "music_artist", "id": artist},
                        {"kind": "music_album", "id": album},
                    ]
                )
                keys.extend((("music_artist", artist), ("music_album", album)))
            elif family == "speech":
                try:
                    assert chapters is not None
                    lineage_keys = librispeech_lineage_keys(clip, chapters)
                except (AssertionError, PublicLineageError) as exc:
                    return {"status": "FAIL", "reason": str(exc)}
                speaker = lineage_keys[0].split(":", 1)[1]
                book = lineage_keys[1].split(":", 1)[1]
                lineage.extend(
                    [
                        {"kind": "speech_speaker", "id": speaker},
                        {"kind": "speech_book", "id": book},
                    ]
                )
                keys.extend((("speech_speaker", speaker), ("speech_book", book)))
            elif family in {"environment", "machine"}:
                try:
                    assert esc_metadata is not None
                    source = esc50_lineage_keys(clip, esc_metadata)[0].split(":", 1)[1]
                except (AssertionError, PublicLineageError) as exc:
                    return {"status": "FAIL", "reason": str(exc)}
                lineage.append({"kind": "esc50_src_file", "id": source})
                keys.append(("esc50_src_file", source))
            for key in keys:
                previous = owner.setdefault(key, session["session_id"])
                dsu.union(previous, session["session_id"])
        lineage_by_session[session["session_id"]] = lineage
    if missing_music:
        return {
            "status": "FAIL",
            "tracks_csv": str(tracks_path),
            "missing_track_ids": missing_music,
            "reason": "active music source를 FMA artist/album에 전부 매핑하지 못했습니다",
        }
    components: dict[str, list[str]] = {}
    family_by_session = {item["session_id"]: item["source_family"] for item in sessions}
    for session in sessions:
        root = dsu.find(session["session_id"])
        components.setdefault(root, []).append(session["session_id"])
    canonical: dict[str, list[str]] = {}
    session_to_component: dict[str, str] = {}
    for members in sorted(components.values()):
        members = sorted(members)
        families = {family_by_session[item] for item in members}
        if len(families) != 1:
            return {
                "status": "FAIL",
                "reason": f"한 lineage component가 family를 가로지릅니다: {sorted(families)}",
                "sessions": members,
            }
        family = next(iter(families))
        component = (
            f"{family}-lineage-"
            + hashlib.sha256("\n".join(members).encode()).hexdigest()[:12]
        )
        canonical[component] = members
        for session_id in members:
            session_to_component[session_id] = component
    counts_by_family: dict[str, int] = {}
    for component, members in canonical.items():
        family = family_by_session[members[0]]
        counts_by_family[family] = counts_by_family.get(family, 0) + 1
    return {
        "status": "READY",
        "tracks_csv": str(tracks_path),
        "tracks_csv_sha256": tracks_snapshot.sha256,
        "librispeech_chapters_path": (
            None if chapters_path is None else str(chapters_path)
        ),
        "librispeech_chapters_sha256": chapters_sha256,
        "esc50_metadata_path": (
            None if esc_metadata_path is None else str(esc_metadata_path)
        ),
        "esc50_metadata_sha256": esc_metadata_sha256,
        "active_session_count": len(sessions),
        "component_count": len(canonical),
        "component_count_by_family": counts_by_family,
        "components": canonical,
        "session_to_component": session_to_component,
        "lineage_by_session": lineage_by_session,
        "note": "계획만 계산했으며 recorded manifest는 변경하지 않았습니다",
    }


def audit_fma_regroup_gate(
    sessions: list[dict[str, str]],
    used_plans: dict[str, RowPlan],
    tracks_path: Path,
) -> dict[str, Any]:
    """하위 호환 이름: 이제 music-only가 아니라 모든 lineage component를 감사한다."""

    return build_lineage_component_plan(sessions, used_plans, tracks_path)


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProvenanceError(f"{path}:{number}: manifest JSON 오류: {exc}") from exc
        if not isinstance(value, dict):
            raise ProvenanceError(f"{path}:{number}: manifest row는 객체여야 합니다")
        rows.append(value)
    return rows


def build_regrouped_manifest_entries(
    component_plan: dict[str, Any],
    sessions: list[dict[str, str]],
    *,
    input_manifest: Path,
    output_manifest: Path,
    seed: int = 20260803,
    ratios: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """lineage component를 원자 단위로 family-stratified deterministic split한다."""

    if component_plan.get("status") != "READY":
        raise ProvenanceError(
            "lineage component가 READY가 아니므로 regrouped manifest를 만들 수 없습니다: "
            f"{component_plan.get('status')}"
        )
    from deep_anc.data.manifest import (
        MANIFEST_PATH_BASE,
        assign_splits,
        manifest_relative_path,
        validate_group_splits,
    )
    from deep_anc.eval.recorded import MIN_GROUPS_PER_FAMILY

    source_rows = _read_manifest_rows(input_manifest)
    source_by_session: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        session_id = str(row.get("session_id") or "")
        if not session_id or session_id in source_by_session:
            raise ProvenanceError(f"입력 manifest session_id 누락/중복: {session_id!r}")
        source_by_session[session_id] = row
    session_by_id = {item["session_id"]: item for item in sessions}
    if set(source_by_session) != set(session_by_id):
        raise ProvenanceError(
            "입력 manifest와 active session 집합이 다릅니다: "
            f"manifest_only={sorted(set(source_by_session)-set(session_by_id))[:8]}, "
            f"active_only={sorted(set(session_by_id)-set(source_by_session))[:8]}"
        )

    session_to_component = component_plan["session_to_component"]
    entries: list[dict[str, Any]] = []
    for session_id in sorted(session_by_id):
        original = dict(source_by_session[session_id])
        component = session_to_component.get(session_id)
        if not component:
            raise ProvenanceError(f"component가 없는 active session: {session_id}")
        family = session_by_id[session_id]["source_family"]
        if str(original.get("source_family")) != family:
            raise ProvenanceError(
                f"입력 manifest family 불일치: {session_id} "
                f"{original.get('source_family')} != {family}"
            )
        old_group = str(original.get("group_id") or "")
        original["source_pool_group_id"] = old_group
        original["group_id"] = component
        original.pop("split", None)
        # 새 manifest가 다른 디렉터리에 있어도 portable relative path를 다시 계산한다.
        session_dir = session_by_id[session_id].get("session_dir")
        if session_dir:
            original["path"] = manifest_relative_path(Path(session_dir), output_manifest)
            original["path_base"] = MANIFEST_PATH_BASE
        original["lineage_schema"] = (
            "shared_clip+music_artist_album+speech_reader_gutenberg_book/v2"
        )
        entries.append(original)

    split_ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    assigned = assign_splits(
        entries,
        split_ratios,
        seed=seed,
        group_key="group_id",
        stratify_key="source_family",
        min_units_per_split={"val": MIN_GROUPS_PER_FAMILY, "test": MIN_GROUPS_PER_FAMILY},
    )
    validate_group_splits(assigned)

    # 모든 lineage key가 한 split에만 속하는지 독립적으로 다시 대조한다.
    split_by_session = {str(row["session_id"]): str(row["split"]) for row in assigned}
    key_splits: dict[tuple[str, str], set[str]] = {}
    for session_id, lineage in component_plan["lineage_by_session"].items():
        for item in lineage:
            key = (str(item["kind"]), str(item["id"]))
            key_splits.setdefault(key, set()).add(split_by_session[session_id])
    crossings = [
        {"kind": key[0], "id": key[1], "splits": sorted(values)}
        for key, values in sorted(key_splits.items())
        if len(values) > 1
    ]
    if crossings:
        raise ProvenanceError(f"lineage key가 split을 가로질렀습니다: {crossings[:8]}")

    stats: dict[str, dict[str, int]] = {}
    for family in sorted({str(row["source_family"]) for row in assigned}):
        stats[family] = {}
        for split in ("train", "val", "test"):
            stats[family][split] = len(
                {
                    str(row["group_id"])
                    for row in assigned
                    if row["source_family"] == family and row["split"] == split
                }
            )
    return assigned, {
        "status": "PASS",
        "seed": seed,
        "ratios": split_ratios,
        "session_count": len(assigned),
        "component_count": len({str(row["group_id"]) for row in assigned}),
        "groups_by_family_split": stats,
        "lineage_cross_split_count": 0,
    }


def write_regrouped_manifest(
    component_plan: dict[str, Any],
    sessions: list[dict[str, str]],
    *,
    input_manifest: Path,
    output_manifest: Path,
    seed: int = 20260803,
) -> dict[str, Any]:
    """READY component만 새 파일에 기록한다.

    이미 같은 canonical bytes가 있는 재감사는 idempotent PASS로 처리한다.
    다른 bytes는 절대 덮어쓰지 않고 실패한다. 이 구분이 없으면 권위
    provenance를 재검증하는 명령이 매번 새 split을 쓰거나, 반대로
    기존 파일이 바뀐 사실을 숨기게 된다.
    """

    if component_plan.get("status") != "READY":
        raise ProvenanceError(
            f"regroup gate {component_plan.get('status')}: manifest를 쓰지 않습니다"
        )
    if output_manifest.resolve() == input_manifest.resolve():
        raise ProvenanceError("기존 recorded manifest overwrite는 허용하지 않습니다")
    entries, audit = build_regrouped_manifest_entries(
        component_plan,
        sessions,
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        seed=seed,
    )
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries).encode()
    changed = True
    if output_manifest.exists():
        if output_manifest.is_symlink() or not output_manifest.is_file():
            raise ProvenanceError(
                f"regrouped manifest 기존 경로가 일반 파일이 아닙니다: {output_manifest}"
            )
        existing = output_manifest.read_bytes()
        if existing != payload:
            raise ProvenanceError(
                "기존 regrouped manifest bytes가 현재 lineage 계약과 다릅니다. "
                f"덮어쓰지 않습니다: {output_manifest}"
            )
        changed = False
    else:
        atomic_write(output_manifest, payload)
    audit.update(
        {
            "written": str(output_manifest),
            "changed": changed,
            "sha256": sha256_bytes(payload),
            "source_manifest": str(input_manifest),
            "source_manifest_sha256": sha256_file(input_manifest),
        }
    )
    return audit


def snapshot_recorded_tree(
    recorded_root: Path, *, repo_root: Path
) -> TreeMetadataSnapshot:
    """recorded tree를 symlink 없이 relpath/size/mtime_ns로 고정한다."""

    return snapshot_regular_tree_metadata(
        recorded_root,
        repo_root=repo_root,
        label="data/recorded provenance protection",
    )


def recorded_tree_protection_evidence(
    before: TreeMetadataSnapshot,
    after: TreeMetadataSnapshot,
) -> dict[str, Any]:
    """before/after 동일성을 실제 digest와 함께 canonical report에 남긴다."""

    unchanged = (
        before.entries == after.entries
        and before.sha256 == after.sha256
        and before.content_entries == after.content_entries
        and before.content_sha256 == after.content_sha256
    )
    return {
        "schema_version": 1,
        "status": "PASS" if unchanged else "FAIL",
        "root": "data/recorded",
        "file_count": before.file_count,
        "snapshot_encoding": RECORDED_TREE_SNAPSHOT_ENCODING,
        "before_sha256": before.sha256,
        "after_sha256": after.sha256,
        "content_snapshot_encoding": RECORDED_TREE_CONTENT_SNAPSHOT_ENCODING,
        "before_content_sha256": before.content_sha256,
        "after_content_sha256": after.content_sha256,
        "unchanged": unchanged,
        "content_integrity_boundary": RECORDED_CONTENT_INTEGRITY_BOUNDARY,
    }


def _report_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def downstream_is_blocked(downstream: dict[str, Any]) -> bool:
    """기존 split 진단을 제외한 canonical downstream gate를 합성한다."""

    statuses = [
        value.get("status")
        for name, value in downstream.items()
        if name != "legacy_recorded_clip_split" and isinstance(value, dict)
    ]
    return any(status not in {"PASS", "READY"} for status in statuses)


def build_canonical_provenance_report(
    *,
    plans_by_pool: dict[str, list[RowPlan]],
    pool_audits: dict[str, dict[str, Any]],
    selection_evidence: dict[str, Any],
    csv_hashes: dict[str, str],
    active_holdout_gate: dict[str, Any],
    recorded_before: TreeMetadataSnapshot,
    recorded_after: TreeMetadataSnapshot,
    component_plan: dict[str, Any],
    regrouped_manifest: dict[str, Any],
    recorded_clip_split: dict[str, Any],
) -> dict[str, Any]:
    """위치/시각/사전 repair 상태/downstream corpus와 무관한 결정적 권위 증거."""

    canonical_pools: dict[str, Any] = {}
    for pool_name, plans in sorted(plans_by_pool.items()):
        csv_rows = [
            {
                "family": plan.source_family,
                "session_index": plan.session_index,
                "path": plan.wav_path,
                "declared_clips": len(plan.clips),
                "reconstructed_clips": len(plan.clips),
                "missing_clips": 0,
                "prefix_pass": True,
            }
            for plan in sorted(plans, key=lambda item: (item.source_family, item.session_index))
        ]
        # PCM rows는 이미 한 fd의 보존 WAV snapshot과 historical 재합성을 결속했다.
        pcm = json.loads(json.dumps(pool_audits[pool_name]["pcm"]))
        canonical_pools[pool_name] = {
            "status": "PASS",
            "csv": {
                "status": "PASS",
                "csv_sha256": csv_hashes[pool_name],
                "row_count": len(csv_rows),
                "expected_row_count": len(plans),
                "missing_clip_placements": 0,
                "issues": [],
                "rows": csv_rows,
            },
            "pcm": pcm,
        }
    if component_plan.get("status") != "READY":
        raise ProvenanceError("canonical lineage component plan이 READY가 아닙니다")
    if component_plan.get("tracks_csv_sha256") != FMA_TRACKS_CSV_SHA256:
        raise ProvenanceError(
            "FMA tracks.csv SHA-256이 고정 canonical metadata와 다릅니다: "
            f"{component_plan.get('tracks_csv_sha256')}"
        )
    chapters_sha = component_plan.get("librispeech_chapters_sha256")
    if not isinstance(chapters_sha, str) or re.fullmatch(r"[0-9a-f]{64}", chapters_sha) is None:
        raise ProvenanceError("LibriSpeech CHAPTERS.TXT SHA-256 증거가 없습니다")
    clip_metadata = active_holdout_gate.get("clip_lineage_metadata")
    clip_lineage_sha = active_holdout_gate.get("clip_lineage_sha256")
    if (
        not isinstance(clip_metadata, dict)
        or clip_metadata.get("librispeech_chapters", {}).get("sha256") != chapters_sha
        or clip_metadata.get("fma_tracks", {}).get("sha256") != FMA_TRACKS_CSV_SHA256
        or not isinstance(clip_lineage_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", clip_lineage_sha) is None
    ):
        raise ProvenanceError("active holdout clip lineage와 component metadata 결속이 다릅니다")
    esc_metadata = clip_metadata.get("esc50")
    if (
        not isinstance(esc_metadata, dict)
        or component_plan.get("esc50_metadata_sha256") != esc_metadata.get("sha256")
    ):
        raise ProvenanceError("active holdout ESC-50 metadata 증거가 없습니다")
    if regrouped_manifest.get("status") != "PASS":
        raise ProvenanceError("canonical recorded_regrouped manifest가 PASS가 아닙니다")
    if (
        regrouped_manifest.get("lineage_cross_split_count") != 0
        or recorded_clip_split.get("status") != "PASS"
        or recorded_clip_split.get("cross_split_clip_count") != 0
    ):
        raise ProvenanceError("canonical regrouped split에 lineage/clip crossing이 있습니다")
    components = {
        str(component): sorted(str(session) for session in members)
        for component, members in sorted(component_plan["components"].items())
    }
    membership_bytes = json.dumps(
        components, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    lineage_contract = {
        "schema_version": 2,
        "tracks_csv": "data/raw/music/fma_metadata/tracks.csv",
        "tracks_csv_sha256": FMA_TRACKS_CSV_SHA256,
        "librispeech_chapters_path": LIBRISPEECH_CHAPTERS,
        "librispeech_chapters_sha256": chapters_sha,
        "esc50_metadata_path": ESC50_METADATA,
        "esc50_metadata_sha256": esc_metadata["sha256"],
        "holdout_clip_lineage_sha256": clip_lineage_sha,
        "active_session_count": component_plan["active_session_count"],
        "component_count": component_plan["component_count"],
        "component_count_by_family": component_plan["component_count_by_family"],
        "components": components,
        "component_membership_sha256": hashlib.sha256(membership_bytes).hexdigest(),
        "regrouped_manifest": "data/manifests/recorded_regrouped.jsonl",
        "regrouped_manifest_sha256": regrouped_manifest["sha256"],
        "regrouped_row_count": regrouped_manifest["session_count"],
        "regrouped_component_count": regrouped_manifest["component_count"],
        "groups_by_family_split": regrouped_manifest["groups_by_family_split"],
        "lineage_cross_split_count": 0,
        "source_clip_cross_split_count": 0,
    }
    tree_evidence = recorded_tree_protection_evidence(recorded_before, recorded_after)
    if tree_evidence["status"] != "PASS":
        raise ProvenanceError("data/recorded tree가 provenance repair 전후 변경됐습니다")
    return {
        "schema_version": 1,
        "status": "PASS",
        "mode": "repair",
        "authority": "historical_builder_reproduction_plus_pcm_validation",
        "historical_builders": {
            "v1": {
                "commit": V1_COMMIT,
                "path": BUILDER_PATH,
                "source_sha256": BUILDER_SHA256[V1_COMMIT],
            },
            "v2": {
                "commit": V2_COMMIT,
                "path": BUILDER_PATH,
                "source_sha256": BUILDER_SHA256[V2_COMMIT],
            },
        },
        "selection": selection_evidence,
        "pools": canonical_pools,
        "repair": {
            "requested": True,
            "performed": True,
            "files": {
                name: {"after_sha256": csv_hashes[name]}
                for name in sorted(csv_hashes)
            },
        },
        "post_repair_csv_sha256": {
            name: csv_hashes[name] for name in sorted(csv_hashes)
        },
        # validator가 권위 자료에서 소비하는 downstream 증거는 active-session
        # mapping뿐이다. synthetic/legacy/FMA 상태는 diagnostic report로 분리한다.
        "downstream_gates": {"active_holdout": active_holdout_gate},
        "lineage_contract": lineage_contract,
        "recorded_tree_protection": tree_evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-csv", default="data/source_pool/sources.csv")
    parser.add_argument("--v2-csv", default="data/source_pool_v2/sources.csv")
    parser.add_argument("--recorded-root", default="data/recorded")
    parser.add_argument("--recorded-manifest", default="data/manifests/recorded_train.jsonl")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument(
        "--fma-tracks", default="data/raw/music/fma_metadata/tracks.csv"
    )
    parser.add_argument(
        "--report",
        default="results/provenance",
        help="content-addressed report를 publish할 고정 directory",
    )
    parser.add_argument(
        "--active-holdout", default="data/manifests/recorded_holdout.json"
    )
    parser.add_argument(
        "--regrouped-manifest",
        default="data/manifests/recorded_regrouped.jsonl",
        help="lineage component split의 새 출력. 기존 recorded_train.jsonl은 덮어쓰지 않음",
    )
    parser.add_argument("--regroup-seed", type=int, default=20260803)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--repair-csv",
        action="store_true",
        help="전 행 prefix+PCM 검증이 PASS일 때만 두 CSV의 clips를 보충",
    )
    parser.add_argument(
        "--write-active-holdout",
        action="store_true",
        help="검증된 full provenance에서 active 82세션 전용 holdout을 원자적으로 기록",
    )
    parser.add_argument(
        "--write-regrouped-manifest",
        action="store_true",
        help=(
            "shared clip + music artist/album + speech speaker/book component를 새 manifest에 "
            "분할. FMA metadata/그룹 하한 미충족 시 아무것도 쓰지 않음"
        ),
    )
    parser.add_argument(
        "--require-downstream-gates",
        action="store_true",
        help="synthetic manifest/FMA/recorded split gate가 막히면 provenance PASS 후에도 EXIT=3",
    )
    args = parser.parse_args(argv)

    repo_root = REPO_ROOT
    try:
        canonical_paths = validate_cli_output_contract(args, repo_root=repo_root)
    except ProvenanceError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    if args.write_active_holdout and not args.repair_csv:
        print(
            "[FAIL] canonical holdout 기록에는 --repair-csv가 함께 필요합니다. "
            "audit-only report는 canonical bundle의 권위 자료가 아닙니다.",
            file=sys.stderr,
        )
        return 2
    if args.write_active_holdout and not args.write_regrouped_manifest:
        print(
            "[FAIL] canonical holdout 기록에는 --write-regrouped-manifest가 함께 필요합니다. "
            "lineage component/split bytes를 같은 canonical report에 결속해야 합니다.",
            file=sys.stderr,
        )
        return 2

    csv_paths = {
        "source_pool": canonical_paths["v1_csv"],
        "source_pool_v2": canonical_paths["v2_csv"],
    }
    recorded_root = repo_root / args.recorded_root
    report_dir = canonical_paths["report"]
    try:
        recorded_before = snapshot_recorded_tree(recorded_root, repo_root=repo_root)
    except HoldoutContractError as exc:
        print(f"[FAIL] recorded tree 보호 snapshot 실패: {exc}", file=sys.stderr)
        return 2

    print("[1/5] historical selection 재현", flush=True)
    try:
        plans_by_pool, selection_evidence = reconstruct_plans(repo_root)
    except ProvenanceError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print("[2/5] CSV prefix/identity 검사", flush=True)
    csv_inputs: dict[str, tuple[Path, list[str], list[dict[str, str]], bytes]] = {}
    pool_audits: dict[str, dict[str, Any]] = {}
    for name, plans in plans_by_pool.items():
        path = csv_paths[name]
        try:
            audit, fields, rows, raw = audit_csv_prefix(path, plans)
        except (OSError, ProvenanceError) as exc:
            audit = {"status": "FAIL", "csv_path": str(path), "issues": [str(exc)]}
            fields, rows, raw = [], [], b""
        pool_audits[name] = {"csv": audit}
        csv_inputs[name] = (path, fields, rows, raw)

    if any(value["csv"]["status"] != "PASS" for value in pool_audits.values()):
        wav_results = {name: [] for name in plans_by_pool}
    else:
        print("[3/5] historical PCM 전수 재합성/대조", flush=True)
        wav_results = verify_all_wavs(plans_by_pool, repo_root, jobs=max(1, args.jobs))

    for name in plans_by_pool:
        values = wav_results[name]
        pcm_pass = len(values) == len(plans_by_pool[name]) and all(
            item.get("status") == "PASS" for item in values
        )
        pool_audits[name]["pcm"] = {
            "status": "PASS" if pcm_pass else "FAIL",
            "passed_rows": sum(item.get("status") == "PASS" for item in values),
            "expected_rows": len(plans_by_pool[name]),
            "rows": values,
        }
        pool_audits[name]["status"] = (
            "PASS"
            if pool_audits[name]["csv"]["status"] == "PASS" and pcm_pass
            else "FAIL"
        )

    provenance_pass = all(value["status"] == "PASS" for value in pool_audits.values())
    repair_result: dict[str, Any] = {"requested": bool(args.repair_csv), "performed": False}
    if provenance_pass and args.repair_csv:
        print("[4/5] 검증된 clips만 transaction repair", flush=True)
        try:
            repair_result.update(
                {
                    "performed": True,
                    "files": repair_csvs_transactionally(
                        pool_audits, csv_inputs, plans_by_pool
                    ),
                }
            )
        except ProvenanceError as exc:
            provenance_pass = False
            repair_result.update({"status": "FAIL", "error": str(exc)})

    csv_hashes = {
        name: sha256_file(path) for name, path in csv_paths.items() if path.is_file()
    }
    downstream: dict[str, Any] = {}
    regroup_write_failed = False
    holdout_payload: dict[str, Any] | None = None
    component_plan: dict[str, Any] = {"status": "BLOCKED", "reason": "not evaluated"}
    if provenance_pass:
        print("[5/5] active holdout/누수/FMA gate", flush=True)
        try:
            sessions = collect_active_sessions(recorded_root, repo_root)
            holdout_payload, used_plans = build_active_holdout(
                sessions,
                plans_by_pool,
                csv_hashes=csv_hashes,
                repo_root=repo_root,
            )
            downstream["active_holdout"] = {
                "status": "PASS",
                "active_session_count": holdout_payload["active_session_count"],
                "active_source_row_count": holdout_payload["active_source_row_count"],
                "total_clips": holdout_payload["total_clips"],
                "clip_lineage_sha256": holdout_payload["clip_lineage"]["clips_sha256"],
                "clip_lineage_metadata": holdout_payload["clip_lineage"]["metadata"],
            }
            # 기존 manifest의 누수는 진단으로 보존하되 신규 canonical
            # split의 진입 게이트로 사용하지 않는다. 구형 split의 6개
            # clip 누수를 새 component split으로 복구했는데도 둘을 같은
            # 필수 게이트로 합산하면 복구 명령은 영원히 EXIT=3이 된다.
            downstream["legacy_recorded_clip_split"] = audit_recorded_clip_split_leak(
                sessions, used_plans, repo_root / args.recorded_manifest
            )
            downstream["synthetic_corpus"] = audit_synthetic_manifests(
                holdout_payload, repo_root / args.manifest_dir
            )
            component_plan = build_lineage_component_plan(
                sessions,
                used_plans,
                repo_root / args.fma_tracks,
                repo_root / LIBRISPEECH_CHAPTERS,
                repo_root / ESC50_METADATA,
                repo_root=repo_root,
            )
            downstream["lineage_components"] = component_plan
            if args.write_regrouped_manifest:
                try:
                    downstream["regrouped_manifest"] = write_regrouped_manifest(
                        component_plan,
                        sessions,
                        input_manifest=repo_root / args.recorded_manifest,
                        output_manifest=canonical_paths["regrouped_manifest"],
                        seed=args.regroup_seed,
                    )
                    downstream["recorded_clip_split"] = audit_recorded_clip_split_leak(
                        sessions,
                        used_plans,
                        canonical_paths["regrouped_manifest"],
                    )
                except ProvenanceError as exc:
                    regroup_write_failed = True
                    downstream["regrouped_manifest"] = {
                        "status": (
                            "BLOCKED"
                            if component_plan.get("status") == "BLOCKED"
                            else "FAIL"
                        ),
                        "written": False,
                        "error": str(exc),
                    }
                    downstream["recorded_clip_split"] = {
                        "status": "BLOCKED",
                        "error": "regrouped manifest가 검증되지 않았습니다",
                    }
            else:
                regrouped_path = canonical_paths["regrouped_manifest"]
                if regrouped_path.is_file() and not regrouped_path.is_symlink():
                    downstream["recorded_clip_split"] = audit_recorded_clip_split_leak(
                        sessions, used_plans, regrouped_path
                    )
                else:
                    downstream["recorded_clip_split"] = {
                        "status": "BLOCKED",
                        "error": f"canonical regrouped manifest가 없습니다: {regrouped_path}",
                    }
        except (OSError, ProvenanceError) as exc:
            downstream["active_holdout"] = {"status": "FAIL", "error": str(exc)}

    try:
        recorded_after = snapshot_recorded_tree(recorded_root, repo_root=repo_root)
    except HoldoutContractError as exc:
        print(f"[FAIL] recorded tree 재검증 snapshot 실패: {exc}", file=sys.stderr)
        return 2
    recorded_unchanged = (
        recorded_before.entries == recorded_after.entries
        and recorded_before.sha256 == recorded_after.sha256
        and recorded_before.content_entries == recorded_after.content_entries
        and recorded_before.content_sha256 == recorded_after.content_sha256
    )
    if not recorded_unchanged:
        provenance_pass = False

    diagnostic_report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if provenance_pass else "FAIL",
        "mode": "repair" if args.repair_csv else "audit_only",
        "authority": "historical_builder_reproduction_plus_pcm_validation",
        "historical_builders": {
            "v1": {
                "commit": V1_COMMIT,
                "path": BUILDER_PATH,
                "source_sha256": BUILDER_SHA256[V1_COMMIT],
            },
            "v2": {
                "commit": V2_COMMIT,
                "path": BUILDER_PATH,
                "source_sha256": BUILDER_SHA256[V2_COMMIT],
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "soundfile": sf.__version__,
        },
        "selection": selection_evidence,
        "pools": pool_audits,
        "repair": repair_result,
        "post_repair_csv_sha256": csv_hashes,
        "downstream_gates": downstream,
        "recorded_tree_protection": recorded_tree_protection_evidence(
            recorded_before, recorded_after
        ),
    }
    canonical_publish = bool(
        provenance_pass
        and args.repair_csv
        and args.write_active_holdout
        and holdout_payload is not None
        and downstream.get("active_holdout", {}).get("status") == "PASS"
        and component_plan.get("status") == "READY"
        and component_plan.get("tracks_csv_sha256") == FMA_TRACKS_CSV_SHA256
        and downstream.get("regrouped_manifest", {}).get("status") == "PASS"
        and downstream.get("recorded_clip_split", {}).get("status") == "PASS"
    )
    canonical_report: dict[str, Any] | None = None
    if canonical_publish:
        canonical_report = build_canonical_provenance_report(
            plans_by_pool=plans_by_pool,
            pool_audits=pool_audits,
            selection_evidence=selection_evidence,
            csv_hashes=csv_hashes,
            active_holdout_gate=downstream["active_holdout"],
            recorded_before=recorded_before,
            recorded_after=recorded_after,
            component_plan=component_plan,
            regrouped_manifest=downstream["regrouped_manifest"],
            recorded_clip_split=downstream["recorded_clip_split"],
        )
        diagnostic_report["canonical_candidate_sha256"] = sha256_bytes(
            _report_bytes(canonical_report)
        )
    try:
        diagnostic_path, _diagnostic_sha = publish_immutable_report(
            report_dir,
            _report_bytes(diagnostic_report),
            canonical=False,
            repo_root=repo_root,
        )
        report_path = diagnostic_path
        if canonical_report is not None:
            report_path, _report_sha = publish_immutable_report(
                report_dir,
                _report_bytes(canonical_report),
                canonical=True,
                repo_root=repo_root,
            )
    except (OSError, ProvenanceError, HoldoutContractError) as exc:
        print(f"[FAIL] immutable provenance report publish 실패: {exc}", file=sys.stderr)
        return 2
    print(f"diagnostic provenance report: {diagnostic_path}")
    if canonical_report is not None:
        print(f"canonical provenance report: {report_path}")
    if not provenance_pass:
        print("[FAIL] provenance 검증 실패 — CSV/holdout을 권위 자료로 사용할 수 없습니다")
        return 2

    if args.write_active_holdout:
        if not canonical_publish:
            print(
                "[FAIL] active holdout은 fixed FMA metadata + deterministic lineage component + "
                "canonical recorded_regrouped manifest까지 PASS해야 기록합니다",
                file=sys.stderr,
            )
            return 2
        active_holdout_path = canonical_paths["active_holdout"]
        try:
            bound_holdout = bind_holdout_to_fixed_report(
                holdout_payload, report_path=report_path, repo_root=repo_root
            )
            validate_then_atomic_write_holdout(
                active_holdout_path,
                _report_bytes(bound_holdout),
                repo_root=repo_root,
            )
        except (OSError, ProvenanceError, HoldoutContractError) as exc:
            print(
                "[FAIL] immutable report→holdout 결속/재검증 실패. 기존 holdout은 "
                f"그대로이며 새 report는 content-addressed 증거로 보존됩니다: {exc}",
                file=sys.stderr,
            )
            return 2
        print(
            "canonical holdout: "
            f"{active_holdout_path} (report_sha256={bound_holdout['provenance_report_sha256']})"
        )

    blocked = downstream_is_blocked(downstream)
    print("[PASS] v1/v2 historical prefix + PCM provenance")
    if blocked:
        print("[BLOCKED] downstream corpus/split/FMA gate는 report를 확인하세요")
        return 3 if args.require_downstream_gates or regroup_write_failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
