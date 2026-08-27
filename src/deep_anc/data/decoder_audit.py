"""공개 음원 디코더 적합성 감사.

MP3 같은 압축 공개 코퍼스는 헤더가 정상이어도 특정 seek 위치에서 decoder 경고나
비정상 PCM을 낼 수 있다. ``NoisePool``의 재시도는 학습을 멈추지 않게 하지만,
canonical 학습 세대에서는 어떤 raw 파일이 실제 표본 분포에 들어갈 수 있는지를
먼저 고정해야 한다. 이 모듈은 raw를 변경하지 않고, 그 결정을 재현 가능한 JSON
inventory로 만든다.

이 모듈은 manifest 생성과 의도적으로 분리돼 있다. 따라서 audit 결과는 raw 파일의
content SHA와 decoder fingerprint로만 결속할 수 있고, manifest 쪽은 이 JSON을
검증한 뒤에야 accepted row를 사용할 수 있다.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
import tempfile
from typing import Any, Iterable, Iterator, Sequence
import warnings

import numpy as np
import soundfile as sf


AUDIT_SCHEMA_VERSION = 1
DEFAULT_AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3"})
DEFAULT_SEQUENTIAL_CHUNK_FRAMES = (65_536, 262_144)
DEFAULT_SEGMENT_FRAMES = 65_536
DEFAULT_SEGMENT_GRID_DENOMINATOR = 7
MAX_DECODED_PCM_ABS = 2.0
MIN_DECODED_RMS = 1e-8


def canonical_json_bytes(payload: object) -> bytes:
    """JSON 값을 플랫폼 독립적인 canonical byte열로 직렬화한다."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    """canonical JSON SHA-256을 반환한다."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _libmpg123_fingerprint() -> dict[str, str | None]:
    """현재 process가 찾을 수 있는 libmpg123 식별자를 best-effort로 읽는다.

    libsndfile 배포판에 따라 MP3 decoder가 정적으로 결합될 수도 있으므로, 찾지
    못한 경우도 fingerprint에 명시한다. audit 자체가 이 정보 부재 때문에 raw를
    거부하면 안 된다.
    """

    library = ctypes.util.find_library("mpg123")
    version: str | None = None
    if library:
        try:
            handle = ctypes.CDLL(library)
            distversion = handle.mpg123_distversion
            distversion.restype = ctypes.c_char_p
            raw = distversion()
            if raw:
                version = raw.decode("utf-8", errors="replace")
        except (AttributeError, OSError):
            # 일부 배포판은 함수 심볼을 내보내지 않는다. library 이름 자체가
            # decoder fingerprint에 남으므로 조용히 계속한다.
            pass
    return {"library": library, "version": version}


def decoder_fingerprint() -> dict[str, object]:
    """현재 decode 결과에 영향을 줄 수 있는 라이브러리/런타임 지문을 만든다."""

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "soundfile": getattr(sf, "__version__", None),
        "libsndfile": getattr(sf, "__libsndfile_version__", None),
        "libmpg123": _libmpg123_fingerprint(),
    }


@dataclass
class _CapturedStderr:
    data: bytes = b""


@contextmanager
def _capture_decoder_stderr() -> Iterator[_CapturedStderr]:
    """C decoder가 fd=2에 쓰는 경고까지 캡처한다.

    ``contextlib.redirect_stderr``만으로는 libsndfile/libmpg123의 C-level stderr를
    잡지 못한다. audit은 단일 process/단일 thread로 실행하는 도구이므로 짧은
    decode 구간에만 fd 2를 임시 파일로 바꾼다.
    """

    captured = _CapturedStderr()
    saved_fd: int | None = None
    with tempfile.TemporaryFile(mode="w+b") as sink:
        try:
            try:
                sys.stderr.flush()
            except (AttributeError, OSError):
                pass
            saved_fd = os.dup(2)
            os.dup2(sink.fileno(), 2)
            yield captured
        finally:
            if saved_fd is not None:
                try:
                    try:
                        sys.stderr.flush()
                    except (AttributeError, OSError):
                        pass
                    os.dup2(saved_fd, 2)
                finally:
                    os.close(saved_fd)
            sink.seek(0)
            captured.data = sink.read()


def _normalise_text(value: str, *, root: Path) -> str:
    """host 절대경로를 report에 고정하지 않도록 오류 문자열을 정규화한다."""

    result = value.replace("\r\n", "\n").replace("\r", "\n")
    root_text = str(root)
    if root_text:
        result = result.replace(root_text, "$ROOT")
    return result


def _stderr_evidence(data: bytes, *, root: Path) -> dict[str, object] | None:
    if not data:
        return None
    text = _normalise_text(data.decode("utf-8", errors="replace"), root=root)
    # 파일별 decoder 경고는 수 KB를 넘지 않는 것이 정상이다. 비정상적으로 긴
    # stderr가 report를 오염시키지 않도록 원문 digest와 앞부분만 보존한다.
    return {
        "byte_count": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "excerpt": text[:2048],
        "truncated": len(text) > 2048,
    }


def _python_warning_evidence(
    captured: Sequence[warnings.WarningMessage], *, root: Path
) -> list[dict[str, str]]:
    """``warnings.warn`` 경로도 native stderr와 별도로 보존한다."""

    return [
        {
            "category": item.category.__name__,
            "message": _normalise_text(str(item.message), root=root),
        }
        for item in captured
    ]


def _finding(
    code: str,
    *,
    phase: str,
    detail: str | None = None,
    **metadata: object,
) -> dict[str, object]:
    result: dict[str, object] = {"code": code, "phase": phase}
    if detail:
        result["detail"] = detail
    result.update(metadata)
    return result


def _file_snapshot(path: Path) -> tuple[dict[str, int], str]:
    """regular raw 파일을 read-only로 hash하고 같은 inode/stat인지 확인한다."""

    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("regular file이 아니거나 symlink입니다")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("open한 raw 대상이 regular file이 아닙니다")
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        opened_identity = (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_size),
            int(opened.st_mtime_ns),
            int(opened.st_ctime_ns),
        )
        if opened_identity != before_identity:
            raise RuntimeError("hash 시작 전 raw inode/stat가 변경됐습니다")
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
        after_open = os.fstat(handle.fileno())
    after = os.lstat(path)
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if after_open != opened or after_identity != before_identity:
        raise RuntimeError("hash 중 raw inode/stat가 변경됐습니다")
    return (
        {
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
            "size": int(before.st_size),
            "mtime_ns": int(before.st_mtime_ns),
            "ctime_ns": int(before.st_ctime_ns),
        },
        digest.hexdigest(),
    )


def _header(path: Path, *, root: Path) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    findings: list[dict[str, object]] = []
    info: sf._SoundFileInfo | None = None
    error: Exception | None = None
    with warnings.catch_warnings(record=True) as python_warnings:
        warnings.simplefilter("always")
        with _capture_decoder_stderr() as captured:
            try:
                info = sf.info(str(path))
            except Exception as exc:  # soundfile can expose several C exception types.
                error = exc
    warning = _stderr_evidence(captured.data, root=root)
    if warning is not None:
        findings.append(_finding("decoder_stderr", phase="header", stderr=warning))
    warning_messages = _python_warning_evidence(python_warnings, root=root)
    if warning_messages:
        findings.append(
            _finding("decoder_warning", phase="header", warnings=warning_messages)
        )
    if error is not None:
        findings.append(
            _finding(
                "header_decode_error",
                phase="header",
                detail=_normalise_text(str(error), root=root),
                exception=type(error).__name__,
            )
        )
        return None, findings
    assert info is not None
    result: dict[str, object] = {
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "format": str(info.format),
        "subtype": str(info.subtype),
        "endian": str(info.endian),
        "sections": int(info.sections),
        # SoundFile 0.12의 _SoundFileInfo에는 seekable이 없고 0.14에서는
        # backend에 따라 제공될 수 있다. full/seek scan 자체가 아래에서
        # 검증하므로 여기서는 호환 가능한 header field만 보조적으로 남긴다.
        "seekable": bool(getattr(info, "seekable", True)),
    }
    if int(info.frames) <= 0 or int(info.samplerate) <= 0 or int(info.channels) <= 0:
        findings.append(
            _finding(
                "invalid_header",
                phase="header",
                detail=(
                    f"frames={info.frames}, sample_rate={info.samplerate}, "
                    f"channels={info.channels}"
                ),
            )
        )
    return result, findings


def _scan_stream(
    path: Path,
    *,
    phase: str,
    root: Path,
    start: int | None,
    frames: int | None,
    chunk_frames: int,
    expected_frames: int | None,
    max_decoded_pcm_abs: float,
    min_decoded_rms: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """하나의 full stream 또는 deterministic seek segment를 수치 검사한다."""

    findings: list[dict[str, object]] = []
    chunks = 0
    frames_read = 0
    peak = 0.0
    square_sum = 0.0
    sample_count = 0
    error: Exception | None = None
    nonfinite = False
    with warnings.catch_warnings(record=True) as python_warnings:
        warnings.simplefilter("always")
        with _capture_decoder_stderr() as captured:
            try:
                with sf.SoundFile(str(path), mode="r") as handle:
                    if start is not None:
                        handle.seek(start)
                    remaining = frames
                    while remaining is None or remaining > 0:
                        request = chunk_frames if remaining is None else min(chunk_frames, remaining)
                        if request <= 0:
                            break
                        block = handle.read(
                            frames=request,
                            dtype="float32",
                            always_2d=True,
                        )
                        if block.shape[0] == 0:
                            break
                        chunks += 1
                        frames_read += int(block.shape[0])
                        if remaining is not None:
                            remaining -= int(block.shape[0])
                        if not np.isfinite(block).all():
                            nonfinite = True
                            break
                        values = block.astype(np.float64, copy=False)
                        peak = max(peak, float(np.max(np.abs(values))))
                        square_sum += float(np.sum(values * values, dtype=np.float64))
                        sample_count += int(values.size)
            except Exception as exc:  # soundfile's error classes vary by backend.
                error = exc

    result: dict[str, object] = {
        "phase": phase,
        "chunk_frames": int(chunk_frames),
        "start_frame": None if start is None else int(start),
        "requested_frames": None if frames is None else int(frames),
        "expected_frames": None if expected_frames is None else int(expected_frames),
        "frames_read": int(frames_read),
        "chunks": int(chunks),
        "peak": float(peak),
        "rms": None,
    }
    warning = _stderr_evidence(captured.data, root=root)
    if warning is not None:
        result["stderr"] = warning
        findings.append(_finding("decoder_stderr", phase=phase, stderr=warning))
    warning_messages = _python_warning_evidence(python_warnings, root=root)
    if warning_messages:
        result["warnings"] = warning_messages
        findings.append(
            _finding("decoder_warning", phase=phase, warnings=warning_messages)
        )
    if error is not None:
        result["error"] = {
            "type": type(error).__name__,
            "message": _normalise_text(str(error), root=root),
        }
        findings.append(
            _finding(
                "decode_error",
                phase=phase,
                detail=_normalise_text(str(error), root=root),
                exception=type(error).__name__,
            )
        )
    if nonfinite:
        findings.append(_finding("nonfinite_pcm", phase=phase))
    if sample_count == 0:
        findings.append(_finding("empty_decode", phase=phase))
    else:
        rms = math.sqrt(square_sum / sample_count)
        result["rms"] = float(rms)
        if peak > max_decoded_pcm_abs:
            findings.append(
                _finding(
                    "pcm_peak_exceeds_limit",
                    phase=phase,
                    peak=float(peak),
                    limit=float(max_decoded_pcm_abs),
                )
            )
        if rms <= min_decoded_rms:
            findings.append(
                _finding(
                    "pcm_rms_below_limit",
                    phase=phase,
                    rms=float(rms),
                    limit=float(min_decoded_rms),
                )
            )
    if expected_frames is not None and frames_read != expected_frames:
        findings.append(
            _finding(
                "frame_count_mismatch",
                phase=phase,
                expected_frames=int(expected_frames),
                frames_read=int(frames_read),
            )
        )
    return result, findings


def _segment_starts(total_frames: int, *, segment_frames: int, denominator: int) -> list[int]:
    """파일 앞/뒤와 내부를 모두 덮는 seek grid를 반환한다."""

    actual_frames = min(total_frames, segment_frames)
    last_start = max(0, total_frames - actual_frames)
    return sorted(
        {
            int((last_start * numerator) // denominator)
            for numerator in range(denominator + 1)
        }
    )


def _relative_path(path: Path, *, root: Path) -> str:
    raw_root = Path(os.path.abspath(root))
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(raw_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"audit path가 root 밖입니다: {path}") from exc


def discover_audio_files(
    roots: Iterable[str | Path],
    *,
    extensions: Iterable[str] = DEFAULT_AUDIO_EXTENSIONS,
) -> list[Path]:
    """여러 tree에서 확장자 기준 후보를 stable order로 발견한다.

    symlink도 inventory에 남겨 audit이 reject하게 한다. directory symlink는 따라가지
    않아 원본 tree 밖을 우연히 스캔하지 않는다.
    """

    normalised_extensions = frozenset(value.lower() for value in extensions)
    candidates: dict[str, Path] = {}
    for raw_root in roots:
        root = Path(raw_root)
        if root.is_file() or root.is_symlink():
            values = [root]
        elif root.is_dir():
            values = root.rglob("*")
        else:
            raise FileNotFoundError(f"audit scan root가 없습니다: {root}")
        for path in values:
            if path.suffix.lower() not in normalised_extensions:
                continue
            if not (path.is_file() or path.is_symlink()):
                continue
            absolute = str(Path(os.path.abspath(path)))
            candidates.setdefault(absolute, path)
    return [candidates[key] for key in sorted(candidates)]


def _audit_one(
    path: Path,
    *,
    root: Path,
    sequential_chunk_frames: Sequence[int],
    segment_frames: int,
    segment_grid_denominator: int,
    max_decoded_pcm_abs: float,
    min_decoded_rms: float,
) -> dict[str, object]:
    relative_path = _relative_path(path, root=root)
    record: dict[str, object] = {
        "relative_path": relative_path,
        "content_sha256": None,
        "content_size": None,
        "header": None,
        "scan": {"sequential": [], "segment_grid": []},
        "findings": [],
        "decision": "reject",
    }
    findings: list[dict[str, object]] = []
    snapshot: dict[str, int]
    try:
        snapshot, content_sha256 = _file_snapshot(path)
        record["content_sha256"] = content_sha256
        record["content_size"] = snapshot["size"]
    except Exception as exc:
        findings.append(
            _finding(
                "raw_snapshot_error",
                phase="raw_snapshot",
                detail=_normalise_text(str(exc), root=root),
                exception=type(exc).__name__,
            )
        )
        record["findings"] = findings
        return record

    header, header_findings = _header(path, root=root)
    record["header"] = header
    findings.extend(header_findings)
    expected_total = int(header["frames"]) if header is not None else None

    for chunk_frames in sequential_chunk_frames:
        scan, scan_findings = _scan_stream(
            path,
            phase=f"sequential_{chunk_frames}",
            root=root,
            start=None,
            frames=None,
            chunk_frames=chunk_frames,
            expected_frames=expected_total,
            max_decoded_pcm_abs=max_decoded_pcm_abs,
            min_decoded_rms=min_decoded_rms,
        )
        assert isinstance(record["scan"], dict)
        sequential = record["scan"]["sequential"]
        assert isinstance(sequential, list)
        sequential.append(scan)
        findings.extend(scan_findings)

    if expected_total is not None and expected_total > 0:
        expected_segment = min(expected_total, segment_frames)
        for start in _segment_starts(
            expected_total,
            segment_frames=segment_frames,
            denominator=segment_grid_denominator,
        ):
            scan, scan_findings = _scan_stream(
                path,
                phase="segment_grid",
                root=root,
                start=start,
                frames=expected_segment,
                chunk_frames=min(segment_frames, 65_536),
                expected_frames=expected_segment,
                max_decoded_pcm_abs=max_decoded_pcm_abs,
                min_decoded_rms=min_decoded_rms,
            )
            assert isinstance(record["scan"], dict)
            segments = record["scan"]["segment_grid"]
            assert isinstance(segments, list)
            segments.append(scan)
            findings.extend(scan_findings)

    # decode하는 동안 raw가 바뀌면 header/PCM/SHA 조합의 provenance가 사라진다.
    # full re-hash 대신 inode/stat 변화 자체를 hard fail로 기록한다. 처음 hash 단계가
    # 같은 stat를 이미 검증했으므로 일반적인 atomic replacement도 놓치지 않는다.
    try:
        current = os.lstat(path)
        current_identity = {
            "device": int(current.st_dev),
            "inode": int(current.st_ino),
            "size": int(current.st_size),
            "mtime_ns": int(current.st_mtime_ns),
            "ctime_ns": int(current.st_ctime_ns),
        }
        if not stat.S_ISREG(current.st_mode) or current_identity != snapshot:
            raise RuntimeError("decode 후 raw inode/stat가 변경됐습니다")
    except Exception as exc:
        findings.append(
            _finding(
                "raw_changed_during_audit",
                phase="post_decode_snapshot",
                detail=_normalise_text(str(exc), root=root),
                exception=type(exc).__name__,
            )
        )

    record["findings"] = findings
    record["decision"] = "accept" if not findings else "reject"
    return record


def _validate_policy(
    sequential_chunk_frames: Sequence[int],
    segment_frames: int,
    segment_grid_denominator: int,
    max_decoded_pcm_abs: float,
    min_decoded_rms: float,
) -> tuple[int, ...]:
    chunks = tuple(int(value) for value in sequential_chunk_frames)
    if not chunks or any(value <= 0 for value in chunks) or len(set(chunks)) != len(chunks):
        raise ValueError("sequential_chunk_frames는 중복 없는 양의 정수여야 합니다")
    required = set(DEFAULT_SEQUENTIAL_CHUNK_FRAMES)
    if not required.issubset(chunks):
        raise ValueError(
            "canonical decoder audit에는 full sequential 65536 및 262144 frame scan이 필수입니다"
        )
    if segment_frames <= 0 or segment_grid_denominator <= 0:
        raise ValueError("segment frame/grid denominator는 양의 정수여야 합니다")
    if not math.isfinite(max_decoded_pcm_abs) or max_decoded_pcm_abs <= 0.0:
        raise ValueError("max_decoded_pcm_abs는 유한한 양수여야 합니다")
    if not math.isfinite(min_decoded_rms) or min_decoded_rms < 0.0:
        raise ValueError("min_decoded_rms는 유한한 0 이상 값이어야 합니다")
    return tuple(sorted(chunks))


def audit_audio_paths(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
    root_label: str = ".",
    sequential_chunk_frames: Sequence[int] = DEFAULT_SEQUENTIAL_CHUNK_FRAMES,
    segment_frames: int = DEFAULT_SEGMENT_FRAMES,
    segment_grid_denominator: int = DEFAULT_SEGMENT_GRID_DENOMINATOR,
    max_decoded_pcm_abs: float = MAX_DECODED_PCM_ABS,
    min_decoded_rms: float = MIN_DECODED_RMS,
) -> dict[str, object]:
    """정렬된 raw audio 후보를 전체 디코드/seek grid로 감사한다.

    반환 JSON은 timestamp·절대경로를 담지 않아 같은 raw와 decoder 환경에서는 byte
    단위로 재현된다. ``status=complete``는 inventory 작성을 끝냈다는 뜻이며, 개별
    rejection 유무는 ``summary``와 ``decision``으로 판단한다.
    """

    chunks = _validate_policy(
        sequential_chunk_frames,
        segment_frames,
        segment_grid_denominator,
        max_decoded_pcm_abs,
        min_decoded_rms,
    )
    base_root = Path(os.path.abspath(Path(root)))
    if not base_root.is_dir():
        raise FileNotFoundError(f"audit root가 directory가 아닙니다: {base_root}")

    ordered: dict[str, Path] = {}
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = base_root / path
        relative = _relative_path(path, root=base_root)
        ordered.setdefault(relative, path)
    inventory = [
        _audit_one(
            path,
            root=base_root,
            sequential_chunk_frames=chunks,
            segment_frames=int(segment_frames),
            segment_grid_denominator=int(segment_grid_denominator),
            max_decoded_pcm_abs=float(max_decoded_pcm_abs),
            min_decoded_rms=float(min_decoded_rms),
        )
        for _relative, path in sorted(ordered.items())
    ]
    accepted = [
        {
            "relative_path": row["relative_path"],
            "content_sha256": row["content_sha256"],
            "content_size": row["content_size"],
        }
        for row in inventory
        if row["decision"] == "accept"
    ]
    fingerprint = decoder_fingerprint()
    report: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "root_label": str(root_label),
        "audit_policy": {
            "audio_extensions": sorted(DEFAULT_AUDIO_EXTENSIONS),
            "sequential_chunk_frames": list(chunks),
            "segment_frames": int(segment_frames),
            "segment_grid_denominator": int(segment_grid_denominator),
            "max_decoded_pcm_abs": float(max_decoded_pcm_abs),
            "min_decoded_rms": float(min_decoded_rms),
        },
        "decoder_fingerprint": fingerprint,
        "decoder_fingerprint_sha256": canonical_json_sha256(fingerprint),
        "inventory": inventory,
        "inventory_sha256": canonical_json_sha256(inventory),
        "accepted_inventory_sha256": canonical_json_sha256(accepted),
        "summary": {
            "candidate_count": len(inventory),
            "accepted_count": len(accepted),
            "rejected_count": len(inventory) - len(accepted),
        },
    }
    # top-level sha는 자기 자신을 제외한 canonical report의 digest다. downstream은
    # immutable output file SHA와 함께 이 값을 보조 evidence로 보관할 수 있다.
    report["audit_sha256"] = canonical_json_sha256(report)
    return report


def audit_audio_tree(
    root: str | Path,
    *,
    root_label: str = ".",
    extensions: Iterable[str] = DEFAULT_AUDIO_EXTENSIONS,
    **kwargs: object,
) -> dict[str, object]:
    """하나의 raw tree를 발견부터 audit까지 수행하는 편의 함수."""

    base_root = Path(os.path.abspath(Path(root)))
    paths = discover_audio_files([base_root], extensions=extensions)
    return audit_audio_paths(paths, root=base_root, root_label=root_label, **kwargs)


def write_audit_report(report: dict[str, object], path: str | Path) -> Path:
    """canonical JSON을 atomic rename으로 저장한다. raw tree에는 절대 쓰지 않는다."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "DEFAULT_AUDIO_EXTENSIONS",
    "DEFAULT_SEQUENTIAL_CHUNK_FRAMES",
    "DEFAULT_SEGMENT_FRAMES",
    "DEFAULT_SEGMENT_GRID_DENOMINATOR",
    "MAX_DECODED_PCM_ABS",
    "MIN_DECODED_RMS",
    "audit_audio_paths",
    "audit_audio_tree",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "decoder_fingerprint",
    "discover_audio_files",
    "write_audit_report",
]
