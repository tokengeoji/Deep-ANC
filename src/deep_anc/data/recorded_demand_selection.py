"""추가 덕트 녹음용 DEMAND 환경음의 immutable 선택 번들 계약.

선택 당시의 public DEMAND manifest와 선택 source bytes를 별도 no-replace 번들에
보존한다. 이후 recorded generation을 synthetic corpus에서 제외해 live manifest가
재발행되어도 이 validator는 mutable live path를 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from . import public_lineage
from .holdout_contract import read_regular_file_snapshot, validate_holdout_contract
from .manifest import read_manifest_bytes
from .recorded_dns_selection import (
    DNS_SELECTION_GENERATION_ID,
    DNS_MIN_DENSITY_RATIO,
    DNS_STRICT_SUBBANDS_HZ,
    _band_density,
    _strict_primary,
)
from .source_trust import (
    SourceTrustError,
    exact_clean_source_evidence,
    validate_environment_freeze_source_commit,
)


DEMAND_SELECTION_SCHEMA_VERSION = 2
DEMAND_SELECTION_KIND = "recorded_demand_environment_selection"
DEMAND_SELECTION_GENERATION_ID = DNS_SELECTION_GENERATION_ID
DEMAND_SELECTION_BUNDLE_ROOT = (
    "data/source_plans/recorded_additions/demand_environment_selections/"
    "stage1-coverage-v4-gainprobe006/canonical"
)
DEMAND_SELECTION_RECEIPT = f"{DEMAND_SELECTION_BUNDLE_ROOT}/selection_receipt.json"
DEMAND_SELECTION_PARENT_MANIFEST = (
    f"{DEMAND_SELECTION_BUNDLE_ROOT}/inputs/demand.selection-parent.jsonl"
)
DEMAND_SELECTION_PARENT_GENERATION = (
    f"{DEMAND_SELECTION_BUNDLE_ROOT}/inputs/manifest_generation.selection-parent.json"
)
DEMAND_SELECTION_PARENT_BOOTSTRAP = (
    f"{DEMAND_SELECTION_BUNDLE_ROOT}/inputs/elice_bootstrap_receipt.selection-parent.json"
)
DEMAND_SELECTION_PARENT_FREEZE = (
    f"{DEMAND_SELECTION_BUNDLE_ROOT}/inputs/environment-freeze.selection-parent.txt"
)
DEMAND_SELECTION_PARENT_HOLDOUT = (
    f"{DEMAND_SELECTION_BUNDLE_ROOT}/inputs/recorded_holdout.selection-parent.json"
)
DEMAND_BOOTSTRAP_RECEIPT = "data/manifests/elice_bootstrap_receipt.json"
DEMAND_ENVIRONMENT_FREEZE = ".venv/environment-freeze.txt"
# 원본의 일반적인 ch01.wav basename을 source plan/exclusion에 노출하지 않는다.
# component exclusion은 아래 public group 하나로만 수행한다. full origin과 실제 재생
# composite를 서로 다른 immutable 파일로 보존해야, window/gain transform을 원본에서
# 재계산하면서도 source plan은 실제 재생 bytes만 가리킬 수 있다.
DEMAND_SELECTION_ORIGIN_SOURCE = (
    f"{DEMAND_SELECTION_BUNDLE_ROOT}/sources/"
    "origin-environment-demand-dkitchen-ch01-f7e2a2868219.wav"
)
DEMAND_SELECTION_SOURCE = (
    f"{DEMAND_SELECTION_BUNDLE_ROOT}/sources/"
    "environment-demand-dkitchen-ch01-f7e2a2868219-185600ms-peaknorm.wav"
)
DEMAND_SOURCE_ORIGIN = "data/raw/noise/demand/DKITCHEN/ch01.wav"
DEMAND_SOURCE_SHA256 = (
    "f7e2a2868219da6294749c2f63fcfe9d7dd17a91f70dd97ac844e8ca0dcf92c6"
)
DEMAND_SOURCE_SIZE = 28_800_428
DEMAND_PREEXCLUSION_MANIFEST = "data/manifests/canonical_v4/demand.jsonl"
DEMAND_PREEXCLUSION_MANIFEST_SHA256 = (
    "71298426ff05baa7d2021509a537c14c98578b4bb33307397835aac0e101cb0c"
)
DEMAND_PREEXCLUSION_ROW_COUNT = 96
DEMAND_PUBLIC_GROUP_ID = (
    "public-lineage-68586695ced7cf3017c43f05c409bf965f782a70677215d77b43747d8959dffa"
)
DEMAND_PUBLIC_LINEAGE_KEY = "demand_environment:DKITCHEN"
DEMAND_PUBLIC_GROUP_MEMBER_COUNT = 16
DEMAND_RECORDED_SPLIT = "test"
# 0.1초 격자 0.0--285.0초 exact 전수 scan에서 기존 strict-P density와
# flatness/entropy/stationarity를 모두 통과한 창 중 150--1600 Hz absolute playback
# level이 가장 큰 창이다. 원본 excerpt와 실제 composite 재생 시작을 구분한다.
DEMAND_ORIGIN_WINDOW_START_SECONDS = 185.6
DEMAND_ORIGIN_WINDOW_START_FRAME = 8_908_800
DEMAND_WINDOW_START_SECONDS = 0.0
DEMAND_WINDOW_SECONDS = 15.0
DEMAND_SAMPLE_RATE = 48_000
DEMAND_WINDOW_START_FRAME = 0
DEMAND_WINDOW_FRAMES = 720_000
DEMAND_TRANSFORM = "mono_48000_pcm16_window_peak_normalize_720000/v1"
DEMAND_PLAYBACK_AMPLITUDE = 0.06

# 실제 2026-08-30 input-only probe의 가장 높은 quiet ERR는 -64.33 dBFS였다. -64.0을
# 보수적 ceiling으로 두고 coh²=0.90에 필요한 SNR 10log10(.9/.1)=9.542 dB를 요구한다.
# selection은 물리 meter를 대신하지 않는다. 대신 공식 meter와 같은 digital band-RMS
# 정의에서 playback이 meter보다 2 dB 넘게 낮아지지 않게 해, fresh meter 최저
# -52.1 dBFS에서도 predicted ERR >= -54.1 dBFS가 되도록 한다.
DEMAND_CONSERVATIVE_QUIET_FLOOR_DBFS = -64.0
DEMAND_REQUIRED_CAPTURE_COHERENCE_SQUARED = 0.90
DEMAND_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK = 2.0
DEMAND_MIN_RENDERED_RMS_DBFS = -50.0
DEMAND_MAX_RENDERED_RMS_DBFS = -32.0
DEMAND_MIN_SPECTRAL_FLATNESS = 0.75
DEMAND_MIN_SPECTRAL_ENTROPY = 0.94
DEMAND_MAX_ONE_SECOND_RMS_PEAK_TO_PEAK_DB = 6.0
DEMAND_STATIONARITY_BAND_HZ = (600.0, 1600.0)
DEMAND_WELCH_NPERSEG = 8192
DEMAND_WELCH_NOVERLAP = 4096
DEMAND_SOURCE_KIND = "external_demand_environment_file"
DEMAND_LINEAGE_KEY = "environment-demand-lineage-fb39879ec061"
DEMAND_PARENT82_HOLDOUT = "data/manifests/recorded_holdout.json"
DEMAND_PARENT82_CLIP_COUNT = 682
DEMAND_MANIFEST_GENERATION = (
    "data/manifests/canonical_v4/manifest_generation.json"
)
DEMAND_MANIFEST_SCHEMA_VERSION = 4
DEMAND_SELECTION_STRICT_PRIMARY_PATH = (
    "assets/measured/primary_path_il_strict_5dc06fdd.npz"
)
DEMAND_SELECTION_STRICT_PRIMARY_SHA256 = (
    "23fa43f1ec46d5bca6bdad53938b81bb2d2c85afc4eee35e83c555b6c4f0c598"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class DemandSelectionError(ValueError):
    """DEMAND immutable selection 계약 위반."""


class DemandSelectionBlocked(DemandSelectionError):
    """필수 pre-exclusion 번들이 없어 source plan을 발행할 수 없다."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _without_evidence_sha(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "evidence_sha256"}


def _snapshot(
    repo_root: Path, relative: str, *, label: str, capture_bytes: bool = True
) -> Any:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DemandSelectionError(f"{label} 경로는 저장소 상대경로여야 합니다")
    try:
        return read_regular_file_snapshot(
            repo_root / relative,
            root=repo_root,
            label=label,
            capture_bytes=capture_bytes,
        )
    except (OSError, ValueError) as exc:
        raise DemandSelectionError(str(exc)) from exc


def _file_ref(snapshot: Any, *, repo_root: Path) -> dict[str, Any]:
    try:
        relative = snapshot.path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise DemandSelectionError(f"번들 입력이 저장소 밖입니다: {snapshot.path}") from exc
    return {"path": relative, "sha256": snapshot.sha256, "size": int(snapshot.size)}


def _validate_file_ref(
    repo_root: Path,
    value: object,
    *,
    expected_path: str,
    label: str,
    capture_bytes: bool = True,
) -> Any:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size"}
        or value.get("path") != expected_path
        or _SHA256_RE.fullmatch(str(value.get("sha256") or "")) is None
        or isinstance(value.get("size"), bool)
        or not isinstance(value.get("size"), int)
        or int(value["size"]) <= 0
    ):
        raise DemandSelectionError(f"{label} file ref가 exact 계약과 다릅니다")
    snapshot = _snapshot(
        repo_root,
        expected_path,
        label=label,
        capture_bytes=capture_bytes,
    )
    if snapshot.sha256 != value["sha256"] or snapshot.size != value["size"]:
        raise DemandSelectionError(f"{label} path/SHA/size가 receipt와 다릅니다")
    return snapshot


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DemandSelectionError(f"{label} JSON 중복 key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DemandSelectionError(f"{label} JSON 오류: {exc}") from exc
    if not isinstance(value, dict):
        raise DemandSelectionError(f"{label} 최상위는 object여야 합니다")
    return value


def _load_jsonl_objects(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DemandSelectionError(f"{label} UTF-8 오류: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        rows.append(
            _load_json_object(
                line.encode("utf-8"), label=f"{label} row #{line_number}"
            )
        )
    return rows


def _git_head(repo_root: Path) -> str:
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        replace = subprocess.run(
            ["git", "replace", "-l"],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DemandSelectionError(f"no-replace git HEAD 확인 실패: {exc}") from exc
    if replace or _COMMIT_RE.fullmatch(head) is None:
        raise DemandSelectionError("git replace가 있거나 HEAD가 전체 40자리 SHA가 아닙니다")
    return head


def _bootstrap_environment(
    repo_root: Path,
    *,
    bootstrap_receipt: str,
    bootstrap_receipt_sha256: str,
    expected_commit: str,
) -> tuple[Any, dict[str, Any], Any]:
    if _SHA256_RE.fullmatch(str(bootstrap_receipt_sha256).lower()) is None:
        raise DemandSelectionError("Elice bootstrap receipt 외부 SHA-256가 필요합니다")
    bootstrap = _snapshot(repo_root, bootstrap_receipt, label="Elice bootstrap receipt")
    if bootstrap.sha256 != str(bootstrap_receipt_sha256).lower():
        raise DemandSelectionError("Elice bootstrap receipt SHA가 외부 anchor와 다릅니다")
    assert bootstrap.data is not None
    payload = _load_json_object(bootstrap.data, label="Elice bootstrap receipt")
    environment = payload.get("environment")
    if (
        payload.get("expected_commit") != expected_commit
        or not isinstance(environment, Mapping)
        or set(environment)
        != {
            "freeze_receipt",
            "freeze_receipt_sha256",
            "torch_version",
            "torch_cuda",
        }
        or environment.get("freeze_receipt") != DEMAND_ENVIRONMENT_FREEZE
        or _SHA256_RE.fullmatch(str(environment.get("freeze_receipt_sha256") or ""))
        is None
    ):
        raise DemandSelectionError(
            "bootstrap receipt expected_commit/environment freeze 결속이 없습니다"
        )
    freeze = _snapshot(
        repo_root,
        str(environment["freeze_receipt"]),
        label="Elice environment freeze",
    )
    if freeze.sha256 != environment["freeze_receipt_sha256"]:
        raise DemandSelectionError("environment freeze SHA가 bootstrap receipt와 다릅니다")
    assert freeze.data is not None
    try:
        validate_environment_freeze_source_commit(
            freeze.data, expected_commit=expected_commit
        )
    except SourceTrustError as exc:
        raise DemandSelectionError(f"environment freeze source commit 오류: {exc}") from exc
    return bootstrap, payload, freeze


def _parent82_evidence(raw: bytes) -> dict[str, Any]:
    payload = _load_json_object(raw, label="parent82 holdout")
    families = payload.get("families")
    lineage = payload.get("clip_lineage")
    if not isinstance(families, Mapping) or not isinstance(lineage, Mapping):
        raise DemandSelectionError("parent82 holdout families/clip_lineage가 없습니다")
    try:
        rows = public_lineage.validate_recorded_clip_lineage(
            lineage, families=families
        )
    except ValueError as exc:
        raise DemandSelectionError(f"parent82 clip lineage 오류: {exc}") from exc
    if len(rows) != DEMAND_PARENT82_CLIP_COUNT:
        raise DemandSelectionError(
            f"parent82 clip authority는 exact {DEMAND_PARENT82_CLIP_COUNT}행이어야 합니다"
        )
    overlaps = {
        "basename": sorted(
            {
                str(row["clip"]).casefold()
                for row in rows
                if str(row["clip"]).casefold() == "ch01.wav"
            }
        ),
        "content_sha256": sorted(
            {
                str(row["content_sha256"])
                for row in rows
                if row["content_sha256"] == DEMAND_SOURCE_SHA256
            }
        ),
        "lineage_keys": sorted(
            {
                DEMAND_PUBLIC_LINEAGE_KEY
                for row in rows
                if DEMAND_PUBLIC_LINEAGE_KEY in row["lineage_keys"]
            }
        ),
    }
    if any(overlaps.values()):
        raise DemandSelectionError(
            f"DKITCHEN selection이 parent82 basename/content/lineage와 겹칩니다: {overlaps}"
        )
    clips_sha256 = lineage.get("clips_sha256")
    if _SHA256_RE.fullmatch(str(clips_sha256 or "")) is None:
        raise DemandSelectionError("parent82 clips_sha256가 canonical SHA-256가 아닙니다")
    return {
        "clip_count": len(rows),
        "clips_sha256": clips_sha256,
        "selected_overlap": overlaps,
    }


def _parent82_authority(repo_root: Path) -> tuple[Any, dict[str, Any]]:
    holdout = _snapshot(
        repo_root,
        DEMAND_PARENT82_HOLDOUT,
        label="DEMAND selection parent82 holdout",
    )
    try:
        summary = validate_holdout_contract(
            holdout.path,
            repo_root=repo_root,
            expected_sha256=holdout.sha256,
        )
    except (OSError, ValueError) as exc:
        raise DemandSelectionError(f"parent82 holdout 검증 실패: {exc}") from exc
    assert holdout.data is not None
    if summary.get("_validated_holdout_bytes") != holdout.data:
        raise DemandSelectionError("parent82 holdout가 검증 도중 변경됐습니다")
    return holdout, _parent82_evidence(holdout.data)


def _manifest_rows(raw: bytes, *, manifest_path: Path) -> list[dict[str, Any]]:
    try:
        rows = read_manifest_bytes(raw, manifest_path=manifest_path)
        lineage = public_lineage.validate_public_manifest_lineage({"demand": rows})
    except ValueError as exc:
        raise DemandSelectionError(f"DEMAND pre-exclusion manifest 오류: {exc}") from exc
    if lineage.get("component_count") != 6:
        raise DemandSelectionError("DEMAND pre-exclusion manifest는 exact 6개 environment여야 합니다")
    return [dict(row) for row in rows]


def _selected_manifest_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != DEMAND_PREEXCLUSION_ROW_COUNT:
        raise DemandSelectionError(
            "DEMAND pre-exclusion manifest row 수 불일치: "
            f"{len(rows)} != {DEMAND_PREEXCLUSION_ROW_COUNT}"
        )
    members = [row for row in rows if row.get("group_id") == DEMAND_PUBLIC_GROUP_ID]
    if len(members) != DEMAND_PUBLIC_GROUP_MEMBER_COUNT:
        raise DemandSelectionError(
            "DKITCHEN public component는 정확히 16개 channel이어야 합니다"
        )
    expected_names = {f"ch{index:02d}.wav" for index in range(1, 17)}
    actual_names = {Path(str(row.get("path") or "")).name for row in members}
    if (
        actual_names != expected_names
        or any(row.get("lineage_keys") != [DEMAND_PUBLIC_LINEAGE_KEY] for row in members)
        or any(row.get("split") != "train" for row in members)
        or any(row.get("tag") != "demand" for row in members)
    ):
        raise DemandSelectionError(
            "DKITCHEN 16-channel component의 path/lineage/source split이 다릅니다"
        )
    selected = [
        row
        for row in members
        if Path(str(row.get("path") or "")).as_posix().endswith("/DKITCHEN/ch01.wav")
    ]
    if len(selected) != 1:
        raise DemandSelectionError("DKITCHEN/ch01.wav manifest row는 정확히 하나여야 합니다")
    row = {key: value for key, value in selected[0].items() if not str(key).startswith("_")}
    if (
        row.get("content_sha256") != DEMAND_SOURCE_SHA256
        or row.get("content_size") != DEMAND_SOURCE_SIZE
        or row.get("sample_rate") != DEMAND_SAMPLE_RATE
        or row.get("channels") != 1
    ):
        raise DemandSelectionError("DKITCHEN/ch01.wav SHA/size/audio metadata가 다릅니다")
    return {
        "manifest_index": rows.index(selected[0]),
        "manifest_row": row,
        "manifest_row_sha256": _canonical_json_sha256(row),
        "public_group_id": DEMAND_PUBLIC_GROUP_ID,
        "public_group_member_count": len(members),
        "public_source_split": "train",
        "recorded_split": DEMAND_RECORDED_SPLIT,
        "lineage_keys": [DEMAND_PUBLIC_LINEAGE_KEY],
    }


def _decode_source(raw: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(raw))
        values, sample_rate = sf.read(
            io.BytesIO(raw), dtype="float64", always_2d=True
        )
    except RuntimeError as exc:
        raise DemandSelectionError(f"DEMAND selected source decode 실패: {exc}") from exc
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or sample_rate != DEMAND_SAMPLE_RATE
        or info.samplerate != DEMAND_SAMPLE_RATE
        or info.channels != 1
        or values.shape[1] != 1
        or info.frames != values.shape[0]
        or info.frames < 1
        or not bool(np.isfinite(values).all())
    ):
        raise DemandSelectionError(
            "DEMAND selected source는 non-empty mono48k PCM16 WAV여야 합니다"
        )
    return np.asarray(values[:, 0], dtype=np.float64), {
        "sample_rate": DEMAND_SAMPLE_RATE,
        "channels": 1,
        "frames": int(info.frames),
        "subtype": "PCM_16",
    }


def _canonical_composite_bytes(values: np.ndarray) -> bytes:
    """full origin에서 exact 15초를 잘라 window-peak-normalized PCM16으로 만든다.

    ``NoiseProgram(file)``은 파일 전체 peak를 playback amplitude의 기준으로 쓴다.
    원본 DKITCHEN의 300초 global peak는 150.65초에 있고 선택 창과 무관해서, origin을
    그대로 넘기면 amplitude=0.06이어도 실제 150--1600 Hz playback이 -78 dBFS까지
    내려간다. transform을 receipt에 명시하고 bytes를 원본에서 재계산해 이 숨은 gain을
    제거한다. 선택 창 peak가 0이거나 너무 작으면 절대 증폭하지 않고 fail-closed한다.
    """

    source = np.asarray(values, dtype=np.float64).reshape(-1)
    start = DEMAND_ORIGIN_WINDOW_START_FRAME
    stop = start + DEMAND_WINDOW_FRAMES
    window = np.asarray(source[start:stop], dtype=np.float64)
    if window.shape != (DEMAND_WINDOW_FRAMES,) or not np.all(np.isfinite(window)):
        raise DemandSelectionError(
            "DEMAND origin이 canonical "
            f"{DEMAND_ORIGIN_WINDOW_START_SECONDS:.1f}--"
            f"{DEMAND_ORIGIN_WINDOW_START_SECONDS + DEMAND_WINDOW_SECONDS:.1f}초 "
            "window를 포함하지 않습니다"
        )
    peak = float(np.max(np.abs(window)))
    if not math.isfinite(peak) or peak <= 1.0e-7:
        raise DemandSelectionError("DEMAND origin window peak가 normalization에 너무 작습니다")
    quantized = np.rint(np.clip(window / peak, -1.0, 1.0) * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(DEMAND_SAMPLE_RATE)
        handle.writeframes(quantized.tobytes())
    return output.getvalue()


def _rendered_level_metrics(composite_raw: bytes) -> dict[str, Any]:
    """실제 ``NoiseProgram``+fade bytes의 absolute excitation/safety를 계산한다."""

    from deep_anc.dsp.measurement_level import (
        OFFICIAL_MEASUREMENT_CHANNEL_MAP,
        OFFICIAL_MEASUREMENT_LEVEL,
        band_rms_dbfs,
        expected_meter_output_pcm,
    )
    from deep_anc.realtime.noise_gen import (
        NoiseProgram,
        render_recording_file_window,
    )

    program = NoiseProgram(
        {
            "type": "file",
            "file": DEMAND_SELECTION_SOURCE,
            "file_start_seconds": DEMAND_WINDOW_START_SECONDS,
            "amplitude": DEMAND_PLAYBACK_AMPLITUDE,
        },
        DEMAND_SAMPLE_RATE,
        file_bytes=composite_raw,
    )
    rendered = np.asarray(
        render_recording_file_window(
            program,
            DEMAND_WINDOW_FRAMES,
            sample_rate=DEMAND_SAMPLE_RATE,
        ),
        dtype=np.float64,
    )
    floor = np.finfo(np.float64).tiny
    peak = float(np.max(np.abs(rendered)))
    rms_dbfs = float(
        20.0 * np.log10(np.sqrt(np.mean(np.square(rendered))) + floor)
    )
    trusted_dbfs = float(band_rms_dbfs(rendered))
    meter_pcm = expected_meter_output_pcm(
        noise_channel=OFFICIAL_MEASUREMENT_CHANNEL_MAP["noise_out"]
    )
    meter_float = np.asarray(meter_pcm[:, 0], dtype=np.float64) / 32768.0
    meter_playback_dbfs = float(band_rms_dbfs(meter_float))
    minimum_trusted_dbfs = float(
        meter_playback_dbfs - DEMAND_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK
    )
    meter_min_dbfs = float(OFFICIAL_MEASUREMENT_LEVEL.meter_min_dbfs)
    predicted_err_min_dbfs = float(
        meter_min_dbfs + trusted_dbfs - meter_playback_dbfs
    )
    predicted_snr_db = float(
        predicted_err_min_dbfs - DEMAND_CONSERVATIVE_QUIET_FLOOR_DBFS
    )
    required_snr_db = float(
        10.0
        * math.log10(
            DEMAND_REQUIRED_CAPTURE_COHERENCE_SQUARED
            / (1.0 - DEMAND_REQUIRED_CAPTURE_COHERENCE_SQUARED)
        )
    )
    passed = bool(
        peak <= DEMAND_PLAYBACK_AMPLITUDE + 1.0e-12
        and DEMAND_MIN_RENDERED_RMS_DBFS <= rms_dbfs <= DEMAND_MAX_RENDERED_RMS_DBFS
        and trusted_dbfs >= minimum_trusted_dbfs
        and predicted_snr_db >= required_snr_db
    )
    if not passed:
        raise DemandSelectionError(
            "DEMAND composite absolute playback level/SNR safety gate 실패: "
            f"peak={peak:.9f}, rms={rms_dbfs:.3f} dBFS, trusted="
            f"{trusted_dbfs:.3f} dBFS, predicted_snr={predicted_snr_db:.3f} dB"
        )
    return {
        "schema_version": 1,
        "metric_definition": "measurement_level.band_rms_dbfs_hann_v1",
        "playback_amplitude": DEMAND_PLAYBACK_AMPLITUDE,
        "frames": DEMAND_WINDOW_FRAMES,
        "peak_linear": peak,
        "peak_dbfs": float(20.0 * np.log10(max(peak, floor))),
        "rms_dbfs": rms_dbfs,
        "trusted_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
        "trusted_band_rms_dbfs": trusted_dbfs,
        "official_meter_playback_trusted_band_dbfs": meter_playback_dbfs,
        "meter_target_min_dbfs": meter_min_dbfs,
        "conservative_quiet_floor_dbfs": DEMAND_CONSERVATIVE_QUIET_FLOOR_DBFS,
        "predicted_err_trusted_band_min_dbfs": predicted_err_min_dbfs,
        "predicted_signal_to_quiet_db": predicted_snr_db,
        "thresholds": {
            "maximum_peak_linear": DEMAND_PLAYBACK_AMPLITUDE,
            "minimum_rms_dbfs": DEMAND_MIN_RENDERED_RMS_DBFS,
            "maximum_rms_dbfs": DEMAND_MAX_RENDERED_RMS_DBFS,
            "maximum_db_below_official_meter_playback": (
                DEMAND_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK
            ),
            "minimum_trusted_band_rms_dbfs": minimum_trusted_dbfs,
            "required_capture_coherence_squared": (
                DEMAND_REQUIRED_CAPTURE_COHERENCE_SQUARED
            ),
            "minimum_predicted_signal_to_quiet_db": required_snr_db,
        },
        "passed": True,
    }


def _validate_rendered_level_metadata(value: Any) -> dict[str, Any]:
    """receipt의 absolute playback/SNR 수치를 수식까지 다시 검증한다."""

    from deep_anc.dsp.measurement_level import (
        OFFICIAL_MEASUREMENT_CHANNEL_MAP,
        OFFICIAL_MEASUREMENT_LEVEL,
        band_rms_dbfs,
        expected_meter_output_pcm,
    )

    if not isinstance(value, Mapping):
        raise DemandSelectionError("DEMAND rendered_level mapping이 필요합니다")
    required = {
        "schema_version",
        "metric_definition",
        "playback_amplitude",
        "frames",
        "peak_linear",
        "peak_dbfs",
        "rms_dbfs",
        "trusted_band_hz",
        "trusted_band_rms_dbfs",
        "official_meter_playback_trusted_band_dbfs",
        "meter_target_min_dbfs",
        "conservative_quiet_floor_dbfs",
        "predicted_err_trusted_band_min_dbfs",
        "predicted_signal_to_quiet_db",
        "thresholds",
        "passed",
    }
    if set(value) != required:
        raise DemandSelectionError("DEMAND rendered_level 필드 집합이 다릅니다")
    thresholds = value.get("thresholds")
    threshold_keys = {
        "maximum_peak_linear",
        "minimum_rms_dbfs",
        "maximum_rms_dbfs",
        "maximum_db_below_official_meter_playback",
        "minimum_trusted_band_rms_dbfs",
        "required_capture_coherence_squared",
        "minimum_predicted_signal_to_quiet_db",
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != threshold_keys:
        raise DemandSelectionError("DEMAND rendered_level thresholds가 다릅니다")
    numeric_keys = {
        "playback_amplitude",
        "peak_linear",
        "peak_dbfs",
        "rms_dbfs",
        "trusted_band_rms_dbfs",
        "official_meter_playback_trusted_band_dbfs",
        "meter_target_min_dbfs",
        "conservative_quiet_floor_dbfs",
        "predicted_err_trusted_band_min_dbfs",
        "predicted_signal_to_quiet_db",
    }
    if any(
        isinstance(value.get(key), bool)
        or not isinstance(value.get(key), (int, float))
        or not math.isfinite(float(value[key]))
        for key in numeric_keys
    ) or any(
        isinstance(thresholds.get(key), bool)
        or not isinstance(thresholds.get(key), (int, float))
        or not math.isfinite(float(thresholds[key]))
        for key in threshold_keys
    ):
        raise DemandSelectionError("DEMAND rendered_level에 non-finite 수치가 있습니다")
    required_snr = 10.0 * math.log10(
        DEMAND_REQUIRED_CAPTURE_COHERENCE_SQUARED
        / (1.0 - DEMAND_REQUIRED_CAPTURE_COHERENCE_SQUARED)
    )
    meter_pcm = expected_meter_output_pcm(
        noise_channel=OFFICIAL_MEASUREMENT_CHANNEL_MAP["noise_out"]
    )
    meter_float = np.asarray(meter_pcm[:, 0], dtype=np.float64) / 32768.0
    expected_meter_playback = float(band_rms_dbfs(meter_float))
    expected_meter_min = float(OFFICIAL_MEASUREMENT_LEVEL.meter_min_dbfs)
    expected_min_trusted = (
        expected_meter_playback
        - DEMAND_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK
    )
    predicted_err = (
        expected_meter_min
        + float(value["trusted_band_rms_dbfs"])
        - expected_meter_playback
    )
    predicted_snr = predicted_err - DEMAND_CONSERVATIVE_QUIET_FLOOR_DBFS
    exact = {
        "maximum_peak_linear": DEMAND_PLAYBACK_AMPLITUDE,
        "minimum_rms_dbfs": DEMAND_MIN_RENDERED_RMS_DBFS,
        "maximum_rms_dbfs": DEMAND_MAX_RENDERED_RMS_DBFS,
        "maximum_db_below_official_meter_playback": (
            DEMAND_MAX_DB_BELOW_OFFICIAL_METER_PLAYBACK
        ),
        "minimum_trusted_band_rms_dbfs": expected_min_trusted,
        "required_capture_coherence_squared": (
            DEMAND_REQUIRED_CAPTURE_COHERENCE_SQUARED
        ),
        "minimum_predicted_signal_to_quiet_db": required_snr,
    }
    if (
        value.get("schema_version") != 1
        or value.get("metric_definition")
        != "measurement_level.band_rms_dbfs_hann_v1"
        or value.get("frames") != DEMAND_WINDOW_FRAMES
        or value.get("trusted_band_hz") != [150.0, 1600.0]
        or value.get("passed") is not True
        or not math.isclose(
            float(value["playback_amplitude"]),
            DEMAND_PLAYBACK_AMPLITUDE,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(value["conservative_quiet_floor_dbfs"]),
            DEMAND_CONSERVATIVE_QUIET_FLOOR_DBFS,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(value["official_meter_playback_trusted_band_dbfs"]),
            expected_meter_playback,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(value["meter_target_min_dbfs"]),
            expected_meter_min,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or float(value["peak_linear"]) <= 0.0
        or not math.isclose(
            float(value["peak_dbfs"]),
            20.0 * math.log10(float(value["peak_linear"])),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or any(
            not math.isclose(
                float(thresholds[key]), float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
            for key, expected in exact.items()
        )
        or not math.isclose(
            float(value["predicted_err_trusted_band_min_dbfs"]),
            predicted_err,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(value["predicted_signal_to_quiet_db"]),
            predicted_snr,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or float(value["peak_linear"]) > DEMAND_PLAYBACK_AMPLITUDE + 1e-12
        or not (
            DEMAND_MIN_RENDERED_RMS_DBFS
            <= float(value["rms_dbfs"])
            <= DEMAND_MAX_RENDERED_RMS_DBFS
        )
        or float(value["trusted_band_rms_dbfs"]) < expected_min_trusted
        or predicted_snr < required_snr
    ):
        raise DemandSelectionError("DEMAND rendered_level absolute/SNR 계약 위반")
    return json.loads(_canonical_json_bytes(dict(value)).decode("utf-8"))


def _selection_metrics(values: np.ndarray, fir: np.ndarray) -> dict[str, Any]:
    from scipy.signal import welch

    window = np.asarray(
        values[
            DEMAND_WINDOW_START_FRAME : DEMAND_WINDOW_START_FRAME
            + DEMAND_WINDOW_FRAMES
        ],
        dtype=np.float64,
    )
    if window.shape != (DEMAND_WINDOW_FRAMES,):
        raise DemandSelectionError("DEMAND selected 15초 window가 source 범위를 넘습니다")
    densities, covered = _band_density(window, fir)
    frequencies, power = welch(
        window,
        fs=DEMAND_SAMPLE_RATE,
        nperseg=DEMAND_WELCH_NPERSEG,
        noverlap=DEMAND_WELCH_NOVERLAP,
        scaling="density",
    )
    mask = (frequencies >= DEMAND_STATIONARITY_BAND_HZ[0]) & (
        frequencies <= DEMAND_STATIONARITY_BAND_HZ[1]
    )
    band_power = np.asarray(power[mask], dtype=np.float64)
    if band_power.size < 2 or not np.all(np.isfinite(band_power)) or float(band_power.sum()) <= 0.0:
        raise DemandSelectionError("DEMAND stationarity spectrum이 유효하지 않습니다")
    floor = np.finfo(np.float64).tiny
    flatness = float(np.exp(np.mean(np.log(np.maximum(band_power, floor)))) / np.mean(band_power))
    probability = band_power / float(np.sum(band_power, dtype=np.float64))
    entropy = float(-np.sum(probability * np.log(probability)) / np.log(band_power.size))
    one_second = window.reshape(-1, DEMAND_SAMPLE_RATE)
    rms_db = 20.0 * np.log10(
        np.sqrt(np.mean(np.square(one_second), axis=1)) + floor
    )
    peak_to_peak = float(np.max(rms_db) - np.min(rms_db))
    if (
        covered != len(DNS_STRICT_SUBBANDS_HZ)
        or any(float(value) < DNS_MIN_DENSITY_RATIO for value in densities)
        or flatness < DEMAND_MIN_SPECTRAL_FLATNESS
        or entropy < DEMAND_MIN_SPECTRAL_ENTROPY
        or peak_to_peak > DEMAND_MAX_ONE_SECOND_RMS_PEAK_TO_PEAK_DB
    ):
        raise DemandSelectionError(
            "DEMAND selected source가 strict-P density/stationarity gate를 통과하지 못했습니다"
        )
    return {
        "strict_p_coverage": {
            "subbands_hz": [list(value) for value in DNS_STRICT_SUBBANDS_HZ],
            "minimum_density_ratio": DNS_MIN_DENSITY_RATIO,
            "density_ratios": [float(value) for value in densities],
            "covered_subband_count": int(covered),
        },
        "stationarity": {
            "band_hz": list(DEMAND_STATIONARITY_BAND_HZ),
            "welch_nperseg": DEMAND_WELCH_NPERSEG,
            "welch_noverlap": DEMAND_WELCH_NOVERLAP,
            "spectral_flatness": flatness,
            "minimum_spectral_flatness": DEMAND_MIN_SPECTRAL_FLATNESS,
            "spectral_entropy": entropy,
            "minimum_spectral_entropy": DEMAND_MIN_SPECTRAL_ENTROPY,
            "one_second_rms_peak_to_peak_db": peak_to_peak,
            "maximum_one_second_rms_peak_to_peak_db": (
                DEMAND_MAX_ONE_SECOND_RMS_PEAK_TO_PEAK_DB
            ),
        },
    }


def build_demand_selection_payload(
    *,
    repo_root: str | Path,
    bootstrap_receipt: str,
    bootstrap_receipt_sha256: str,
    expected_commit: str,
    expected_manifest_generation_sha256: str,
    public_manifest: str | None = None,
    strict_primary: str | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """clean exact Elice 세대에서 deterministic DKITCHEN 번들을 만든다."""

    root = Path(os.path.abspath(Path(repo_root)))
    commit = _git_head(root)
    expected_commit = str(expected_commit).lower()
    if _COMMIT_RE.fullmatch(expected_commit) is None or commit != expected_commit:
        raise DemandSelectionError(
            f"DEMAND selection expected commit과 HEAD가 다릅니다: {expected_commit} != {commit}"
        )
    try:
        clean_source = exact_clean_source_evidence(
            root,
            expected_commit=commit,
            reject_runtime_bytecode=True,
        )
    except SourceTrustError as exc:
        raise DemandSelectionError(f"DEMAND selection clean exact source 오류: {exc}") from exc
    if bootstrap_receipt != DEMAND_BOOTSTRAP_RECEIPT:
        raise DemandSelectionError(
            "DEMAND selection bootstrap receipt는 canonical 경로여야 합니다"
        )
    bootstrap, _bootstrap_payload, freeze = _bootstrap_environment(
        root,
        bootstrap_receipt=bootstrap_receipt,
        bootstrap_receipt_sha256=bootstrap_receipt_sha256,
        expected_commit=commit,
    )
    public_manifest = public_manifest or DEMAND_PREEXCLUSION_MANIFEST
    strict_primary = strict_primary or DEMAND_SELECTION_STRICT_PRIMARY_PATH
    if public_manifest != DEMAND_PREEXCLUSION_MANIFEST:
        raise DemandSelectionError(
            "DEMAND selection public manifest는 canonical pre-exclusion 경로여야 합니다"
        )
    generation_relative = (
        Path(public_manifest).parent / "manifest_generation.json"
    ).as_posix()
    if generation_relative != DEMAND_MANIFEST_GENERATION:
        raise DemandSelectionError("DEMAND manifest_generation 경로가 canonical이 아닙니다")
    generation_snapshot = _snapshot(
        root, generation_relative, label="DEMAND pre-exclusion manifest generation"
    )
    if (
        _SHA256_RE.fullmatch(str(expected_manifest_generation_sha256).lower())
        is None
        or generation_snapshot.sha256
        != str(expected_manifest_generation_sha256).lower()
    ):
        raise DemandSelectionError(
            "DEMAND manifest_generation 외부 SHA anchor가 다릅니다"
        )
    # manifest_contract -> recorded_generation_exclusion -> recorded_generation이
    # 이 모듈을 다시 import하므로 top-level import는 순환한다.
    from .manifest_contract import validate_manifest_generation

    manifest_generation = validate_manifest_generation(
        root / Path(public_manifest).parent,
        required_tags={"demand"},
        repo_root=root,
    )
    if (
        manifest_generation.get("_validated_sidecar_bytes")
        != generation_snapshot.data
        or manifest_generation.get("_validated_recorded_generation_exclusion")
        is not None
        or manifest_generation.get("training_eligible") is not True
        or _SHA256_RE.fullmatch(str(manifest_generation.get("build_id") or ""))
        is None
    ):
        raise DemandSelectionError(
            "DEMAND selection은 exclusion 전 validated manifest generation에서만 허용됩니다"
        )
    manifest = _snapshot(root, public_manifest, label="DEMAND pre-exclusion manifest")
    if manifest.sha256 != DEMAND_PREEXCLUSION_MANIFEST_SHA256:
        raise DemandSelectionError("DEMAND pre-exclusion manifest SHA가 승인값과 다릅니다")
    assert manifest.data is not None
    validated_manifest_bytes = manifest_generation.get("_validated_manifest_bytes")
    if (
        not isinstance(validated_manifest_bytes, Mapping)
        or validated_manifest_bytes.get("demand") != manifest.data
    ):
        raise DemandSelectionError(
            "validate_manifest_generation이 결속한 DEMAND bytes와 selection input이 다릅니다"
        )
    rows = _manifest_rows(manifest.data, manifest_path=manifest.path)
    selected = _selected_manifest_evidence(rows)
    source = _snapshot(root, DEMAND_SOURCE_ORIGIN, label="DEMAND selected origin")
    if source.sha256 != DEMAND_SOURCE_SHA256 or source.size != DEMAND_SOURCE_SIZE:
        raise DemandSelectionError("DEMAND selected origin SHA/size가 승인값과 다릅니다")
    assert source.data is not None
    origin_values, origin_audio = _decode_source(source.data)
    if origin_audio["frames"] < (
        DEMAND_ORIGIN_WINDOW_START_FRAME + DEMAND_WINDOW_FRAMES
    ):
        raise DemandSelectionError(
            "DEMAND selected origin이 canonical excerpt 끝까지 포함하지 않습니다"
        )
    composite_raw = _canonical_composite_bytes(origin_values)
    composite_values, composite_audio = _decode_source(composite_raw)
    if composite_audio["frames"] != DEMAND_WINDOW_FRAMES:
        raise DemandSelectionError("DEMAND composite는 exact 15초여야 합니다")
    primary, fir, primary_metadata = _strict_primary(root, strict_primary)
    if primary.sha256 != DEMAND_SELECTION_STRICT_PRIMARY_SHA256:
        raise DemandSelectionError("DEMAND selection strict P SHA가 승인값과 다릅니다")
    holdout, parent82 = _parent82_authority(root)
    assert holdout.data is not None
    metrics = _selection_metrics(composite_values, fir)
    rendered_level = _rendered_level_metrics(composite_raw)
    origin_bundle_source = {
        "path": DEMAND_SELECTION_ORIGIN_SOURCE,
        "sha256": source.sha256,
        "size": source.size,
        **origin_audio,
    }
    bundle_source = {
        "path": DEMAND_SELECTION_SOURCE,
        "sha256": hashlib.sha256(composite_raw).hexdigest(),
        "size": len(composite_raw),
        **composite_audio,
    }
    selected.update(
        {
            "origin_source": _file_ref(source, repo_root=root),
            "origin_bundle_source": origin_bundle_source,
            "bundle_source": bundle_source,
            "origin_window_start_seconds": DEMAND_ORIGIN_WINDOW_START_SECONDS,
            "origin_window_start_frame": DEMAND_ORIGIN_WINDOW_START_FRAME,
            "window_start_seconds": DEMAND_WINDOW_START_SECONDS,
            "window_start_frame": DEMAND_WINDOW_START_FRAME,
            "window_seconds": DEMAND_WINDOW_SECONDS,
            "window_frames": DEMAND_WINDOW_FRAMES,
            "transform": DEMAND_TRANSFORM,
            "repeat_count": 1,
            **metrics,
            "rendered_level": rendered_level,
        }
    )
    payload: dict[str, Any] = {
        "schema_version": DEMAND_SELECTION_SCHEMA_VERSION,
        "kind": DEMAND_SELECTION_KIND,
        "generation_id": DEMAND_SELECTION_GENERATION_ID,
        "source_commit": commit,
        "clean_source": clean_source,
        "bootstrap_receipt_origin": _file_ref(bootstrap, repo_root=root),
        "bootstrap_receipt": {
            "path": DEMAND_SELECTION_PARENT_BOOTSTRAP,
            "sha256": bootstrap.sha256,
            "size": bootstrap.size,
        },
        "environment_freeze_origin": _file_ref(freeze, repo_root=root),
        "environment_freeze": {
            "path": DEMAND_SELECTION_PARENT_FREEZE,
            "sha256": freeze.sha256,
            "size": freeze.size,
        },
        "manifest_generation_origin": _file_ref(
            generation_snapshot, repo_root=root
        ),
        "manifest_generation": {
            "path": DEMAND_SELECTION_PARENT_GENERATION,
            "sha256": generation_snapshot.sha256,
            "size": generation_snapshot.size,
            "build_id": manifest_generation.get("build_id"),
            "training_eligible": manifest_generation.get("training_eligible"),
        },
        "public_manifest_origin": {
            **_file_ref(manifest, repo_root=root),
            "row_count": len(rows),
        },
        "public_manifest": {
            "path": DEMAND_SELECTION_PARENT_MANIFEST,
            "sha256": manifest.sha256,
            "size": manifest.size,
            "row_count": len(rows),
        },
        "strict_primary": primary_metadata,
        "parent82": {
            "holdout_origin": _file_ref(holdout, repo_root=root),
            "holdout": {
                "path": DEMAND_SELECTION_PARENT_HOLDOUT,
                "sha256": holdout.sha256,
                "size": holdout.size,
            },
            **parent82,
        },
        "selected": selected,
    }
    try:
        final_clean_source = exact_clean_source_evidence(
            root,
            expected_commit=commit,
            reject_runtime_bytecode=True,
        )
    except SourceTrustError as exc:
        raise DemandSelectionError(
            f"DEMAND selection 종료 clean exact source 오류: {exc}"
        ) from exc
    if final_clean_source != clean_source:
        raise DemandSelectionError("DEMAND selection 도중 exact source가 변경됐습니다")
    payload["evidence_sha256"] = _canonical_json_sha256(payload)
    return payload, {
        "inputs/elice_bootstrap_receipt.selection-parent.json": bootstrap.data,
        "inputs/environment-freeze.selection-parent.txt": freeze.data,
        "inputs/demand.selection-parent.jsonl": manifest.data,
        "inputs/manifest_generation.selection-parent.json": generation_snapshot.data,
        "inputs/recorded_holdout.selection-parent.json": holdout.data,
        f"sources/{Path(DEMAND_SELECTION_ORIGIN_SOURCE).name}": source.data,
        f"sources/{Path(DEMAND_SELECTION_SOURCE).name}": composite_raw,
    }


def validate_demand_selection_receipt(
    *,
    repo_root: str | Path,
    receipt_path: str = DEMAND_SELECTION_RECEIPT,
    expected_receipt_sha256: str | None = None,
    require_source_files: bool = True,
    verify_current_commit: bool = True,
) -> dict[str, Any]:
    """live manifest와 독립적으로 immutable DEMAND bundle을 재검증한다."""

    root = Path(os.path.abspath(Path(repo_root)))
    try:
        receipt = _snapshot(root, receipt_path, label="DEMAND selection receipt")
    except DemandSelectionError as exc:
        raise DemandSelectionBlocked(
            f"BLOCKED: DEMAND selection receipt가 없습니다/유효하지 않습니다: {exc}"
        ) from exc
    if expected_receipt_sha256 is not None and (
        _SHA256_RE.fullmatch(str(expected_receipt_sha256).lower()) is None
        or receipt.sha256 != str(expected_receipt_sha256).lower()
    ):
        raise DemandSelectionError("DEMAND selection receipt 외부 SHA가 다릅니다")
    assert receipt.data is not None
    payload = _load_json_object(receipt.data, label="DEMAND selection receipt")
    required = {
        "schema_version",
        "kind",
        "generation_id",
        "source_commit",
        "clean_source",
        "bootstrap_receipt_origin",
        "bootstrap_receipt",
        "environment_freeze_origin",
        "environment_freeze",
        "manifest_generation_origin",
        "manifest_generation",
        "public_manifest_origin",
        "public_manifest",
        "strict_primary",
        "parent82",
        "selected",
        "evidence_sha256",
    }
    if set(payload) != required or (
        payload.get("schema_version") != DEMAND_SELECTION_SCHEMA_VERSION
        or payload.get("kind") != DEMAND_SELECTION_KIND
        or payload.get("generation_id") != DEMAND_SELECTION_GENERATION_ID
        or payload.get("evidence_sha256")
        != _canonical_json_sha256(_without_evidence_sha(payload))
    ):
        raise DemandSelectionError("DEMAND selection receipt schema/self-seal 불일치")

    commit = str(payload.get("source_commit") or "")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise DemandSelectionError("DEMAND selection source_commit이 유효하지 않습니다")
    clean_source = payload.get("clean_source")
    clean_policy = (
        clean_source.get("policy") if isinstance(clean_source, Mapping) else None
    )
    if (
        not isinstance(clean_source, Mapping)
        or clean_source.get("commit") != commit
        or not isinstance(clean_policy, Mapping)
        or clean_policy.get("protected_runtime_bytecode") != "forbidden"
    ):
        raise DemandSelectionError("DEMAND selection clean_source evidence가 없습니다")
    if verify_current_commit:
        try:
            current_source = exact_clean_source_evidence(root, expected_commit=commit)
        except SourceTrustError as exc:
            raise DemandSelectionError(
                f"DEMAND selection clean exact source 재검증 실패: {exc}"
            ) from exc
        common_fields = {
            "schema",
            "commit",
            "head_tree_object_id",
            "git_object_format",
            "tracked_file_count",
            "tracked_inventory_sha256",
        }
        current_policy = current_source.get("policy")
        if (
            any(
                clean_source.get(key) != current_source.get(key)
                for key in common_fields
            )
            or not isinstance(current_policy, Mapping)
            or any(
                clean_policy.get(key) != current_policy.get(key)
                for key in set(clean_policy) - {"protected_runtime_bytecode"}
            )
        ):
            raise DemandSelectionError(
                "DEMAND selection clean_source evidence가 현재 exact source와 다릅니다"
            )

    bootstrap_origin = payload.get("bootstrap_receipt_origin")
    bootstrap_ref = payload.get("bootstrap_receipt")
    if (
        not isinstance(bootstrap_origin, Mapping)
        or set(bootstrap_origin) != {"path", "sha256", "size"}
        or bootstrap_origin.get("path") != DEMAND_BOOTSTRAP_RECEIPT
        or not isinstance(bootstrap_ref, Mapping)
        or bootstrap_origin.get("sha256") != bootstrap_ref.get("sha256")
        or bootstrap_origin.get("size") != bootstrap_ref.get("size")
    ):
        raise DemandSelectionError(
            "DEMAND immutable bootstrap와 origin receipt 결속이 다릅니다"
        )
    bootstrap = _validate_file_ref(
        root,
        bootstrap_ref,
        expected_path=DEMAND_SELECTION_PARENT_BOOTSTRAP,
        label="DEMAND immutable bootstrap receipt",
    )
    assert bootstrap.data is not None
    bootstrap_payload = _load_json_object(
        bootstrap.data, label="DEMAND immutable bootstrap receipt"
    )
    bootstrap_environment = bootstrap_payload.get("environment")
    if (
        bootstrap_payload.get("expected_commit") != commit
        or not isinstance(bootstrap_environment, Mapping)
        or set(bootstrap_environment)
        != {
            "freeze_receipt",
            "freeze_receipt_sha256",
            "torch_version",
            "torch_cuda",
        }
        or bootstrap_environment.get("freeze_receipt")
        != DEMAND_ENVIRONMENT_FREEZE
    ):
        raise DemandSelectionError(
            "DEMAND bootstrap expected_commit/environment 계약이 다릅니다"
        )

    freeze_origin = payload.get("environment_freeze_origin")
    freeze_ref = payload.get("environment_freeze")
    if (
        not isinstance(freeze_origin, Mapping)
        or set(freeze_origin) != {"path", "sha256", "size"}
        or freeze_origin.get("path") != DEMAND_ENVIRONMENT_FREEZE
        or not isinstance(freeze_ref, Mapping)
        or freeze_origin.get("sha256") != freeze_ref.get("sha256")
        or freeze_origin.get("size") != freeze_ref.get("size")
    ):
        raise DemandSelectionError(
            "DEMAND immutable environment freeze와 origin 결속이 다릅니다"
        )
    freeze = _validate_file_ref(
        root,
        freeze_ref,
        expected_path=DEMAND_SELECTION_PARENT_FREEZE,
        label="DEMAND immutable environment freeze",
    )
    assert freeze.data is not None
    if bootstrap_environment.get("freeze_receipt_sha256") != freeze.sha256:
        raise DemandSelectionError(
            "DEMAND environment freeze가 immutable bootstrap와 다릅니다"
        )
    try:
        validate_environment_freeze_source_commit(
            freeze.data, expected_commit=commit
        )
    except SourceTrustError as exc:
        raise DemandSelectionError(
            f"DEMAND environment freeze source 결속 실패: {exc}"
        ) from exc

    generation_origin = payload.get("manifest_generation_origin")
    generation_ref = payload.get("manifest_generation")
    if (
        not isinstance(generation_origin, Mapping)
        or set(generation_origin) != {"path", "sha256", "size"}
        or generation_origin.get("path") != DEMAND_MANIFEST_GENERATION
        or not isinstance(generation_ref, Mapping)
        or set(generation_ref)
        != {"path", "sha256", "size", "build_id", "training_eligible"}
        or generation_ref.get("path") != DEMAND_SELECTION_PARENT_GENERATION
        or generation_origin.get("sha256") != generation_ref.get("sha256")
        or generation_origin.get("size") != generation_ref.get("size")
        or generation_ref.get("training_eligible") is not True
        or _SHA256_RE.fullmatch(str(generation_ref.get("build_id") or ""))
        is None
    ):
        raise DemandSelectionError(
            "DEMAND immutable manifest_generation과 origin 결속이 다릅니다"
        )
    generation_snapshot = _validate_file_ref(
        root,
        {key: generation_ref[key] for key in ("path", "sha256", "size")},
        expected_path=DEMAND_SELECTION_PARENT_GENERATION,
        label="DEMAND immutable manifest generation",
    )
    assert generation_snapshot.data is not None
    generation_payload = _load_json_object(
        generation_snapshot.data, label="DEMAND immutable manifest generation"
    )
    # top-level import는 manifest_contract -> recorded_generation -> 이 모듈의
    # 순환을 만들므로 완전히 초기화된 검증 시점에 단일 출처 helper를 불러온다.
    from .manifest_contract import manifest_generation_build_id

    derived_build_id = manifest_generation_build_id(generation_payload)
    if (
        generation_payload.get("schema_version")
        != DEMAND_MANIFEST_SCHEMA_VERSION
        or generation_payload.get("training_eligible") is not True
        or generation_payload.get("build_id") != derived_build_id
        or generation_ref.get("build_id") != derived_build_id
    ):
        raise DemandSelectionError(
            "DEMAND immutable manifest_generation schema/build_id가 다릅니다"
        )

    manifest_ref = payload.get("public_manifest")
    origin_ref = payload.get("public_manifest_origin")
    if (
        not isinstance(manifest_ref, Mapping)
        or set(manifest_ref) != {"path", "sha256", "size", "row_count"}
        or not isinstance(origin_ref, Mapping)
        or set(origin_ref) != {"path", "sha256", "size", "row_count"}
        or manifest_ref.get("path") != DEMAND_SELECTION_PARENT_MANIFEST
        or origin_ref.get("path") != DEMAND_PREEXCLUSION_MANIFEST
        or manifest_ref.get("sha256") != DEMAND_PREEXCLUSION_MANIFEST_SHA256
        or manifest_ref.get("row_count") != DEMAND_PREEXCLUSION_ROW_COUNT
        or {key: origin_ref.get(key) for key in ("sha256", "size", "row_count")}
        != {key: manifest_ref.get(key) for key in ("sha256", "size", "row_count")}
    ):
        raise DemandSelectionError("DEMAND immutable manifest ref/origin 결속이 다릅니다")
    manifest = _snapshot(
        root, str(manifest_ref["path"]), label="DEMAND immutable pre-exclusion manifest"
    )
    if manifest.sha256 != manifest_ref.get("sha256") or manifest.size != manifest_ref.get("size"):
        raise DemandSelectionError("DEMAND immutable manifest path/SHA/size 불일치")
    assert manifest.data is not None
    rows = _manifest_rows(manifest.data, manifest_path=manifest.path)
    derived_selected = _selected_manifest_evidence(rows)
    raw_parent_rows = _load_jsonl_objects(
        manifest.data, label="DEMAND immutable pre-exclusion manifest"
    )
    retained_parent_rows = [
        row
        for row in raw_parent_rows
        if row.get("group_id") != DEMAND_PUBLIC_GROUP_ID
    ]
    if (
        len(raw_parent_rows) != DEMAND_PREEXCLUSION_ROW_COUNT
        or len(retained_parent_rows)
        != DEMAND_PREEXCLUSION_ROW_COUNT - DEMAND_PUBLIC_GROUP_MEMBER_COUNT
    ):
        raise DemandSelectionError(
            "DEMAND immutable parent에서 DKITCHEN16을 뺀 exact 80행을 유도하지 못했습니다"
        )

    parent = payload.get("parent82")
    if not isinstance(parent, Mapping) or set(parent) != {
        "holdout_origin",
        "holdout",
        "clip_count",
        "clips_sha256",
        "selected_overlap",
    }:
        raise DemandSelectionError("DEMAND parent82 receipt schema가 다릅니다")
    holdout_origin = parent.get("holdout_origin")
    holdout_ref = parent.get("holdout")
    if (
        not isinstance(holdout_origin, Mapping)
        or set(holdout_origin) != {"path", "sha256", "size"}
        or holdout_origin.get("path") != DEMAND_PARENT82_HOLDOUT
        or not isinstance(holdout_ref, Mapping)
        or holdout_origin.get("sha256") != holdout_ref.get("sha256")
        or holdout_origin.get("size") != holdout_ref.get("size")
    ):
        raise DemandSelectionError(
            "DEMAND immutable parent82 holdout와 origin 결속이 다릅니다"
        )
    holdout = _validate_file_ref(
        root,
        holdout_ref,
        expected_path=DEMAND_SELECTION_PARENT_HOLDOUT,
        label="DEMAND immutable parent82 holdout",
    )
    assert holdout.data is not None
    try:
        holdout_summary = validate_holdout_contract(
            holdout.path,
            repo_root=root,
            expected_sha256=holdout.sha256,
        )
    except (OSError, ValueError) as exc:
        raise DemandSelectionError(
            f"DEMAND immutable parent82 holdout 검증 실패: {exc}"
        ) from exc
    if holdout_summary.get("_validated_holdout_bytes") != holdout.data:
        raise DemandSelectionError("DEMAND parent82 holdout가 검증 도중 변경됐습니다")
    derived_parent = {
        "holdout_origin": dict(holdout_origin),
        "holdout": dict(holdout_ref),
        **_parent82_evidence(holdout.data),
    }
    if dict(parent) != derived_parent:
        raise DemandSelectionError(
            "DEMAND parent82 lineage/exclusion evidence가 재유도값과 다릅니다"
        )

    selected = payload.get("selected")
    selected_keys = {
        "manifest_index",
        "manifest_row",
        "manifest_row_sha256",
        "public_group_id",
        "public_group_member_count",
        "public_source_split",
        "recorded_split",
        "lineage_keys",
        "origin_source",
        "origin_bundle_source",
        "bundle_source",
        "origin_window_start_seconds",
        "origin_window_start_frame",
        "window_start_seconds",
        "window_start_frame",
        "window_seconds",
        "window_frames",
        "transform",
        "repeat_count",
        "strict_p_coverage",
        "stationarity",
        "rendered_level",
    }
    if not isinstance(selected, Mapping) or set(selected) != selected_keys:
        raise DemandSelectionError("DEMAND selected evidence 필드 집합이 다릅니다")
    for key, value in derived_selected.items():
        if selected.get(key) != value:
            raise DemandSelectionError(f"DEMAND selected manifest evidence 불일치: {key}")
    if (
        selected.get("origin_window_start_seconds")
        != DEMAND_ORIGIN_WINDOW_START_SECONDS
        or selected.get("origin_window_start_frame")
        != DEMAND_ORIGIN_WINDOW_START_FRAME
        or selected.get("window_start_seconds") != DEMAND_WINDOW_START_SECONDS
        or selected.get("window_start_frame") != DEMAND_WINDOW_START_FRAME
        or selected.get("window_seconds") != DEMAND_WINDOW_SECONDS
        or selected.get("window_frames") != DEMAND_WINDOW_FRAMES
        or selected.get("transform") != DEMAND_TRANSFORM
        or selected.get("repeat_count") != 1
    ):
        raise DemandSelectionError(
            "DEMAND selected origin/playback start/duration/transform 계약 불일치"
        )
    origin_source = selected.get("origin_source")
    origin_bundle_source = selected.get("origin_bundle_source")
    bundle_source = selected.get("bundle_source")
    if (
        origin_source
        != {
            "path": DEMAND_SOURCE_ORIGIN,
            "sha256": DEMAND_SOURCE_SHA256,
            "size": DEMAND_SOURCE_SIZE,
        }
        or not isinstance(origin_bundle_source, Mapping)
        or set(origin_bundle_source)
        != {"path", "sha256", "size", "sample_rate", "channels", "frames", "subtype"}
        or origin_bundle_source.get("path") != DEMAND_SELECTION_ORIGIN_SOURCE
        or origin_bundle_source.get("sha256") != DEMAND_SOURCE_SHA256
        or origin_bundle_source.get("size") != DEMAND_SOURCE_SIZE
        or origin_bundle_source.get("sample_rate") != DEMAND_SAMPLE_RATE
        or origin_bundle_source.get("channels") != 1
        or isinstance(origin_bundle_source.get("frames"), bool)
        or not isinstance(origin_bundle_source.get("frames"), int)
        or int(origin_bundle_source["frames"])
        < DEMAND_ORIGIN_WINDOW_START_FRAME + DEMAND_WINDOW_FRAMES
        or origin_bundle_source.get("subtype") != "PCM_16"
        or not isinstance(bundle_source, Mapping)
        or set(bundle_source)
        != {"path", "sha256", "size", "sample_rate", "channels", "frames", "subtype"}
        or bundle_source.get("path") != DEMAND_SELECTION_SOURCE
        or Path(str(bundle_source.get("path"))).name != Path(DEMAND_SELECTION_SOURCE).name
        or Path(str(bundle_source.get("path"))).name == "ch01.wav"
        or _SHA256_RE.fullmatch(str(bundle_source.get("sha256") or "")) is None
        or isinstance(bundle_source.get("size"), bool)
        or not isinstance(bundle_source.get("size"), int)
        or int(bundle_source["size"]) <= 44
        or bundle_source.get("sample_rate") != DEMAND_SAMPLE_RATE
        or bundle_source.get("channels") != 1
        or isinstance(bundle_source.get("frames"), bool)
        or not isinstance(bundle_source.get("frames"), int)
        or int(bundle_source["frames"]) != DEMAND_WINDOW_FRAMES
        or bundle_source.get("subtype") != "PCM_16"
    ):
        raise DemandSelectionError(
            "DEMAND origin/composite exact ref 또는 unique basename 불일치"
        )

    rendered_level = _validate_rendered_level_metadata(
        selected.get("rendered_level")
    )

    coverage = selected.get("strict_p_coverage")
    stationarity = selected.get("stationarity")
    if (
        not isinstance(coverage, Mapping)
        or set(coverage)
        != {
            "subbands_hz",
            "minimum_density_ratio",
            "density_ratios",
            "covered_subband_count",
        }
        or coverage.get("subbands_hz")
        != [list(value) for value in DNS_STRICT_SUBBANDS_HZ]
        or coverage.get("minimum_density_ratio") != DNS_MIN_DENSITY_RATIO
        or coverage.get("covered_subband_count") != len(DNS_STRICT_SUBBANDS_HZ)
        or not isinstance(coverage.get("density_ratios"), list)
        or len(coverage["density_ratios"]) != len(DNS_STRICT_SUBBANDS_HZ)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < DNS_MIN_DENSITY_RATIO
            for value in coverage["density_ratios"]
        )
    ):
        raise DemandSelectionError("DEMAND strict-P density evidence schema가 다릅니다")
    stationarity_keys = {
        "band_hz",
        "welch_nperseg",
        "welch_noverlap",
        "spectral_flatness",
        "minimum_spectral_flatness",
        "spectral_entropy",
        "minimum_spectral_entropy",
        "one_second_rms_peak_to_peak_db",
        "maximum_one_second_rms_peak_to_peak_db",
    }
    if (
        not isinstance(stationarity, Mapping)
        or set(stationarity) != stationarity_keys
        or stationarity.get("band_hz") != list(DEMAND_STATIONARITY_BAND_HZ)
        or stationarity.get("welch_nperseg") != DEMAND_WELCH_NPERSEG
        or stationarity.get("welch_noverlap") != DEMAND_WELCH_NOVERLAP
        or stationarity.get("minimum_spectral_flatness")
        != DEMAND_MIN_SPECTRAL_FLATNESS
        or stationarity.get("minimum_spectral_entropy")
        != DEMAND_MIN_SPECTRAL_ENTROPY
        or stationarity.get("maximum_one_second_rms_peak_to_peak_db")
        != DEMAND_MAX_ONE_SECOND_RMS_PEAK_TO_PEAK_DB
        or any(
            isinstance(stationarity.get(key), bool)
            or not isinstance(stationarity.get(key), (int, float))
            or not math.isfinite(float(stationarity[key]))
            for key in (
                "spectral_flatness",
                "spectral_entropy",
                "one_second_rms_peak_to_peak_db",
            )
        )
        or float(stationarity["spectral_flatness"])
        < DEMAND_MIN_SPECTRAL_FLATNESS
        or float(stationarity["spectral_entropy"])
        < DEMAND_MIN_SPECTRAL_ENTROPY
        or float(stationarity["one_second_rms_peak_to_peak_db"])
        > DEMAND_MAX_ONE_SECOND_RMS_PEAK_TO_PEAK_DB
    ):
        raise DemandSelectionError(
            "DEMAND stationarity peak-to-peak evidence schema가 다릅니다"
        )

    strict = payload.get("strict_primary")
    if (
        not isinstance(strict, Mapping)
        or strict.get("path") != DEMAND_SELECTION_STRICT_PRIMARY_PATH
    ):
        raise DemandSelectionError("DEMAND strict primary ref가 유효하지 않습니다")
    primary, fir, primary_metadata = _strict_primary(root, str(strict["path"]))
    if (
        primary.sha256 != DEMAND_SELECTION_STRICT_PRIMARY_SHA256
        or dict(strict) != primary_metadata
    ):
        raise DemandSelectionError("DEMAND strict primary metadata/SHA가 다릅니다")
    origin_source_snapshot = None
    source_snapshot = None
    if require_source_files:
        origin_source_snapshot = _snapshot(
            root,
            str(origin_bundle_source["path"]),
            label="DEMAND immutable full origin source",
        )
        if (
            origin_source_snapshot.sha256 != origin_bundle_source.get("sha256")
            or origin_source_snapshot.size != origin_bundle_source.get("size")
        ):
            raise DemandSelectionError(
                "DEMAND immutable full origin source SHA/size 불일치"
            )
        assert origin_source_snapshot.data is not None
        origin_values, origin_audio = _decode_source(origin_source_snapshot.data)
        if any(
            origin_bundle_source.get(key) != value
            for key, value in origin_audio.items()
        ):
            raise DemandSelectionError("DEMAND immutable origin audio metadata 불일치")
        source_snapshot = _snapshot(
            root,
            str(bundle_source["path"]),
            label="DEMAND immutable playback composite",
        )
        if (
            source_snapshot.sha256 != bundle_source.get("sha256")
            or source_snapshot.size != bundle_source.get("size")
        ):
            raise DemandSelectionError("DEMAND immutable composite SHA/size 불일치")
        assert source_snapshot.data is not None
        expected_composite = _canonical_composite_bytes(origin_values)
        if source_snapshot.data != expected_composite:
            raise DemandSelectionError(
                "DEMAND playback composite가 immutable origin window/transform에서 "
                "재유도한 bytes와 다릅니다"
            )
        values, audio = _decode_source(source_snapshot.data)
        if any(bundle_source.get(key) != value for key, value in audio.items()):
            raise DemandSelectionError("DEMAND immutable composite audio metadata 불일치")
        derived_metrics = _selection_metrics(values, fir)
        for observed, expected in zip(
            coverage["density_ratios"],
            derived_metrics["strict_p_coverage"]["density_ratios"],
            strict=True,
        ):
            if not math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12):
                raise DemandSelectionError("DEMAND strict-P density 재계산값 불일치")
        if {
            key: value
            for key, value in coverage.items()
            if key != "density_ratios"
        } != {
            key: value
            for key, value in derived_metrics["strict_p_coverage"].items()
            if key != "density_ratios"
        }:
            raise DemandSelectionError("DEMAND strict-P coverage schema 불일치")
        for key, expected in derived_metrics["stationarity"].items():
            observed = stationarity.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(observed), expected, rel_tol=1e-10, abs_tol=1e-12):
                    raise DemandSelectionError(f"DEMAND stationarity 재계산값 불일치: {key}")
            elif observed != expected:
                raise DemandSelectionError(f"DEMAND stationarity schema 불일치: {key}")
        derived_level = _rendered_level_metrics(source_snapshot.data)
        for key, expected in derived_level.items():
            observed = rendered_level.get(key)
            if isinstance(expected, float):
                if not math.isclose(
                    float(observed), expected, rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise DemandSelectionError(
                        f"DEMAND rendered_level 재계산값 불일치: {key}"
                    )
            elif isinstance(expected, dict):
                if not isinstance(observed, Mapping) or set(observed) != set(expected):
                    raise DemandSelectionError(
                        f"DEMAND rendered_level nested schema 불일치: {key}"
                    )
                for nested_key, nested_expected in expected.items():
                    nested_observed = observed.get(nested_key)
                    if isinstance(nested_expected, float):
                        if not math.isclose(
                            float(nested_observed),
                            nested_expected,
                            rel_tol=1e-10,
                            abs_tol=1e-12,
                        ):
                            raise DemandSelectionError(
                                "DEMAND rendered_level threshold 재계산값 불일치: "
                                f"{nested_key}"
                            )
                    elif nested_observed != nested_expected:
                        raise DemandSelectionError(
                            "DEMAND rendered_level threshold schema 불일치: "
                            f"{nested_key}"
                        )
            elif observed != expected:
                raise DemandSelectionError(
                    f"DEMAND rendered_level schema 불일치: {key}"
                )

    bundle_files = sorted(
        [
            {
                "path": receipt_path,
                "sha256": receipt.sha256,
                "size": receipt.size,
            },
            {
                "path": str(bootstrap_ref["path"]),
                "sha256": bootstrap.sha256,
                "size": bootstrap.size,
            },
            {
                "path": str(freeze_ref["path"]),
                "sha256": freeze.sha256,
                "size": freeze.size,
            },
            {
                "path": str(generation_ref["path"]),
                "sha256": generation_snapshot.sha256,
                "size": generation_snapshot.size,
            },
            {
                "path": str(manifest_ref["path"]),
                "sha256": manifest.sha256,
                "size": manifest.size,
            },
            {
                "path": str(holdout_ref["path"]),
                "sha256": holdout.sha256,
                "size": holdout.size,
            },
            {
                "path": str(origin_bundle_source["path"]),
                "sha256": str(origin_bundle_source["sha256"]),
                "size": int(origin_bundle_source["size"]),
            },
            {
                "path": str(bundle_source["path"]),
                "sha256": str(bundle_source["sha256"]),
                "size": int(bundle_source["size"]),
            },
        ],
        key=lambda item: str(item["path"]),
    )
    if len(bundle_files) != 8 or len({item["path"] for item in bundle_files}) != 8:
        raise DemandSelectionError(
            "DEMAND selection bundle은 receipt+bootstrap+freeze+generation+manifest+"
            "holdout+origin+composite exact 8개여야 합니다"
        )
    return {
        "receipt_path": receipt_path,
        "receipt_sha256": receipt.sha256,
        "receipt_size": receipt.size,
        "evidence_sha256": payload["evidence_sha256"],
        "source_commit": commit,
        "bootstrap_receipt_path": str(bootstrap_ref["path"]),
        "bootstrap_receipt_sha256": bootstrap.sha256,
        "bootstrap_receipt_size": bootstrap.size,
        "environment_freeze_path": str(freeze_ref["path"]),
        "environment_freeze_sha256": freeze.sha256,
        "environment_freeze_size": freeze.size,
        "manifest_generation_path": str(generation_ref["path"]),
        "manifest_generation_sha256": generation_snapshot.sha256,
        "manifest_generation_size": generation_snapshot.size,
        "manifest_generation_build_id": derived_build_id,
        "public_manifest_path": str(manifest_ref["path"]),
        "public_manifest_sha256": manifest.sha256,
        "public_manifest_size": manifest.size,
        "strict_primary_path": str(strict["path"]),
        "strict_primary_sha256": primary.sha256,
        "selected": dict(selected),
        "origin_source_snapshot": origin_source_snapshot,
        "source_snapshot": source_snapshot,
        "bundle_files": bundle_files,
        # training entry guard에서 validator가 이미 결속한 live manifest bytes와
        # exact 80 retained row identity/order를 대조하는 process-local evidence다.
        "_expected_live_demand_rows": retained_parent_rows,
    }


def require_demand_selection_excluded_from_manifest_generation(
    generation: Mapping[str, Any], *, repo_root: str | Path
) -> None:
    """번들이 존재하면 exclusion sidecar 이전 synthetic pretrain을 막는다.

    source plan 발행과 live manifest 재발행 사이에는 DKITCHEN 16채널이 아직
    synthetic corpus에 남아 있다. 이 짧은 중간 상태를 학습 가능한 상태로 취급하지
    않는다.
    """

    root = Path(os.path.abspath(Path(repo_root)))
    receipt_path = root / DEMAND_SELECTION_RECEIPT
    if not receipt_path.exists():
        return
    summary = validate_demand_selection_receipt(
        repo_root=root,
        require_source_files=False,
    )
    entries = generation.get("_validated_entries")
    manifest_bytes = generation.get("_validated_manifest_bytes")
    exclusion = generation.get("_validated_recorded_generation_exclusion")
    if (
        not isinstance(entries, Mapping)
        or not isinstance(manifest_bytes, Mapping)
        or exclusion is None
    ):
        raise DemandSelectionBlocked(
            "BLOCKED: DEMAND selection 뒤 recorded_generation_exclusion을 결속한 "
            "public manifest를 재발행하기 전에는 synthetic pretrain을 시작할 수 없습니다"
        )
    demand_rows = entries.get("demand")
    demand_raw = manifest_bytes.get("demand")
    if not isinstance(demand_rows, list) or not isinstance(demand_raw, bytes):
        raise DemandSelectionBlocked("BLOCKED: validated DEMAND manifest가 없습니다")
    selected = summary["selected"]
    public_group = str(selected["public_group_id"])
    if any(str(row.get("group_id")) == public_group for row in demand_rows):
        raise DemandSelectionBlocked(
            "BLOCKED: recorded DKITCHEN public group 16개가 synthetic manifest에 남아 있습니다"
        )
    live_rows = _load_jsonl_objects(demand_raw, label="validated live DEMAND manifest")
    expected_rows = summary["_expected_live_demand_rows"]
    if (
        len(demand_rows) != 80
        or len(live_rows) != 80
        or live_rows != expected_rows
    ):
        raise DemandSelectionBlocked(
            "BLOCKED: live DEMAND manifest는 immutable 96행 parent에서 DKITCHEN16만 "
            "제외한 exact 80행과 동일해야 합니다"
        )


__all__ = [
    "DEMAND_BOOTSTRAP_RECEIPT",
    "DEMAND_ENVIRONMENT_FREEZE",
    "DEMAND_LINEAGE_KEY",
    "DEMAND_MANIFEST_GENERATION",
    "DEMAND_PREEXCLUSION_MANIFEST",
    "DEMAND_PUBLIC_GROUP_ID",
    "DEMAND_PUBLIC_GROUP_MEMBER_COUNT",
    "DEMAND_PUBLIC_LINEAGE_KEY",
    "DEMAND_RECORDED_SPLIT",
    "DEMAND_SELECTION_BUNDLE_ROOT",
    "DEMAND_SELECTION_GENERATION_ID",
    "DEMAND_SELECTION_ORIGIN_SOURCE",
    "DEMAND_SELECTION_RECEIPT",
    "DEMAND_SELECTION_SOURCE",
    "DEMAND_SOURCE_KIND",
    "DEMAND_SOURCE_ORIGIN",
    "DEMAND_TRANSFORM",
    "DEMAND_ORIGIN_WINDOW_START_SECONDS",
    "DEMAND_WINDOW_SECONDS",
    "DEMAND_WINDOW_START_SECONDS",
    "DemandSelectionBlocked",
    "DemandSelectionError",
    "build_demand_selection_payload",
    "require_demand_selection_excluded_from_manifest_generation",
    "validate_demand_selection_receipt",
]
