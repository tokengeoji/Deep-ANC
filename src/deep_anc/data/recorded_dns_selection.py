"""외부 DNS speech 추가 녹음용 2단계 선택 receipt 계약.

선택기는 Elice의 canonical public speech manifest 전체를 strict P(z)로 스캔하고,
서로 다른 public lineage component 다섯 개를 고른다. 출력 receipt와 10.333초
PCM16 raw/15초 repeat-trim composite는 no-replace bundle로 발행한다. Jetson 쪽
validator는 선택 점수를 신뢰하지 않고 manifest 전체 계보, parent82 보수적 숫자
alias, 선택 raw/composite bytes를 다시 검증한다.

Jetson에는 선택되지 않은 DNS raw 8천여 개를 복제하지 않는다. 따라서 local
validator는 full-scan inventory/ranking의 self-seal과 외부 receipt 파일 SHA를
검증하고, 선택된 다섯 raw의 strict-P 점수만 독립 재계산한다. 최종 권위 coverage는
실제 덕트에서 녹음한 additions를 포함해 다시 만드는 recorded subband report다.

이 모듈은 오디오 장치를 열지 않는다. 저장된 WAV/NPZ만 읽는다.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .holdout_contract import read_regular_file_snapshot
from .manifest import read_manifest_bytes
from . import public_lineage
from .recording_source_preflight import (
    SOURCE_PREFLIGHT_FRAMES,
    SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO,
    SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE,
    SOURCE_PREFLIGHT_SCHEMA,
    TIMELINE_FEASIBILITY_SCHEMA,
    RecordingSourcePreflightError,
    rendered_source_preflight,
    validate_rendered_source_preflight,
)
from .timeline import TimelineSettings
from deep_anc.realtime.noise_gen import RECORDING_FILE_FADE_SECONDS
from .source_trust import (
    SELECTOR_PYCACHE_PREFIX,
    SELECTOR_RUNTIME_SCHEMA,
    SourceTrustError,
    exact_clean_source_evidence,
    exact_selector_runtime_evidence,
    validate_environment_freeze_source_commit,
)


DNS_SELECTION_SCHEMA_VERSION = 3
DNS_SELECTION_KIND = "recorded_dns_speech_selection"
DNS_SELECTION_GENERATION_ID = "stage1-coverage-v2"
DNS_SELECTION_RECEIPT = (
    "data/source_plans/recorded_additions/dns_speech_selections/"
    "stage1-coverage-v2/selection_receipt.json"
)
DNS_SELECTION_BUNDLE_ROOT = (
    "data/source_plans/recorded_additions/dns_speech_selections/"
    "stage1-coverage-v2"
)
DNS_SOURCE_KIND = "external_dns_speech_composite"
DNS_RAW_SAMPLE_RATE = 48_000
DNS_RAW_FRAMES = 496_000  # 10 + 1/3초
DNS_RAW_SECONDS = DNS_RAW_FRAMES / DNS_RAW_SAMPLE_RATE
DNS_COMPOSITE_FRAMES = 720_000
DNS_COMPOSITE_SECONDS = DNS_COMPOSITE_FRAMES / DNS_RAW_SAMPLE_RATE
DNS_REPEAT_COUNT = 2
DNS_TRANSFORM = "mono_48000_pcm16_window_repeat_trim_720000/v1"
DNS_SELECTION_COUNT = 5
DNS_SPLIT_QUOTAS = {"train": 2, "val": 1, "test": 2}
DNS_RECORDED_SPLIT_ASSIGNMENT = ("train", "train", "val", "test", "test")
DNS_STRICT_SUBBANDS_HZ = (
    (150.0, 300.0),
    (300.0, 600.0),
    (600.0, 1000.0),
    (1000.0, 1600.0),
)
DNS_MIN_DENSITY_RATIO = 0.25
DNS_PLAYBACK_AMPLITUDE = SOURCE_PREFLIGHT_PLAYBACK_AMPLITUDE
_DNS_TIMELINE_SETTINGS = TimelineSettings(sample_rate=DNS_RAW_SAMPLE_RATE)
DNS_TIMELINE_SOURCE_SPAN_FRAMES = (
    _DNS_TIMELINE_SETTINGS.window_samples
    + 2 * _DNS_TIMELINE_SETTINGS.coarse_search_samples
)
DNS_TIMELINE_HOP_FRAMES = _DNS_TIMELINE_SETTINGS.hop_samples
DNS_TIMELINE_MIN_WINDOW_RMS = _DNS_TIMELINE_SETTINGS.min_window_rms
DNS_TIMELINE_MIN_ELIGIBLE_RATIO = SOURCE_PREFLIGHT_MIN_ELIGIBLE_RATIO
DNS_SCAN_ALGORITHM = {
    "name": "strict_primary_dns_distinct_group_coverage",
    "version": 3,
    "source_contract": {
        "channels": 1,
        "sample_rate": DNS_RAW_SAMPLE_RATE,
        "minimum_frames": DNS_RAW_FRAMES,
    },
    "window_frames": DNS_RAW_FRAMES,
    "window_rule": "nonoverlap_starts_plus_exact_tail",
    "pcm16_quantization": "clip_rint_x32767_to_int16_then_decode_div32768",
    "primary_application": (
        "numpy_rfft_irfft_power2_full_convolution_then_causal_crop_to_window_frames"
    ),
    "third_party_convolution": "forbidden",
    "spectrum": "numpy_hanning_rfft_squared_magnitude_sum",
    "output_raw": "selected_window_pcm16_le_wav",
    "output_composite": DNS_TRANSFORM,
    "subbands_hz": [list(value) for value in DNS_STRICT_SUBBANDS_HZ],
    "minimum_density_ratio": DNS_MIN_DENSITY_RATIO,
    "rendered_source_preflight": {
        "schema": SOURCE_PREFLIGHT_SCHEMA,
        "timeline_schema": TIMELINE_FEASIBILITY_SCHEMA,
        "playback_amplitude": DNS_PLAYBACK_AMPLITUDE,
        "composite_frames": DNS_COMPOSITE_FRAMES,
        "fade_seconds": RECORDING_FILE_FADE_SECONDS,
        "source_span_samples": DNS_TIMELINE_SOURCE_SPAN_FRAMES,
        "hop_samples": DNS_TIMELINE_HOP_FRAMES,
        "minimum_window_rms": DNS_TIMELINE_MIN_WINDOW_RMS,
        "minimum_eligible_ratio": DNS_TIMELINE_MIN_ELIGIBLE_RATIO,
        "rule": (
            "full_trusted_band_level_predicted_snr_and_all_sliding_starts_"
            "without_tail_injection"
        ),
    },
    "ranking": (
        "covered_subband_count_desc,highband_1000_1600_density_desc,"
        "minimum_subband_density_desc,total_density_desc,"
        "timeline_eligible_ratio_desc,manifest_index_asc"
    ),
    "group_rule": "best_manifest_window_per_public_group_then_global_top5",
    "recorded_split_assignment": list(DNS_RECORDED_SPLIT_ASSIGNMENT),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_NUMERIC_ALIAS_PREFIXES = (
    "conservative_speech_reader_numeric:",
    "conservative_speech_book_numeric:",
)
_SCAN_RESULT_KEYS = {
    "manifest_index",
    "manifest_row_sha256",
    "path",
    "content_sha256",
    "group_id",
    "public_source_split",
    "lineage_keys",
    "status",
    "reason",
    "coverage_scan",
    "source_preflight",
    "selected_window_start_frame",
}
_SELECTED_ITEM_KEYS = {
    "order",
    "manifest_index",
    "manifest_row",
    "manifest_row_sha256",
    "public_group_id",
    "public_source_split",
    "recorded_split",
    "lineage_keys",
    "source_content_sha256",
    "source_window_start_frame",
    "coverage_scan",
    "source_preflight",
    "raw_output",
    "composite_output",
}


class DNSSelectionError(ValueError):
    """DNS selection receipt 또는 그 입력이 canonical 계약과 다르다."""


class DNSSelectionBlocked(DNSSelectionError):
    """필수 receipt/후보가 없어 recorded generation을 발행할 수 없다."""


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


def _file_ref(snapshot: Any, *, repo_root: Path) -> dict[str, Any]:
    try:
        relative = snapshot.path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise DNSSelectionError(f"receipt 입력이 저장소 밖입니다: {snapshot.path}") from exc
    return {"path": relative, "sha256": snapshot.sha256, "size": int(snapshot.size)}


def _snapshot(
    repo_root: Path, relative: str, *, label: str, capture_bytes: bool = True
) -> Any:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DNSSelectionError(f"{label} 경로는 저장소 상대경로여야 합니다")
    try:
        return read_regular_file_snapshot(
            repo_root / relative,
            root=repo_root,
            label=label,
            capture_bytes=capture_bytes,
        )
    except (OSError, ValueError) as exc:
        raise DNSSelectionError(str(exc)) from exc


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DNSSelectionError(f"{label}에 중복 JSON key가 있습니다: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DNSSelectionError(f"{label} JSON 오류: {exc}") from exc
    if not isinstance(value, dict):
        raise DNSSelectionError(f"{label} 최상위는 object여야 합니다")
    return value


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
        raise DNSSelectionError(f"no-replace git HEAD를 확인할 수 없습니다: {exc}") from exc
    if replace:
        raise DNSSelectionError("git replace ref가 있어 selection commit을 신뢰할 수 없습니다")
    if _COMMIT_RE.fullmatch(head) is None:
        raise DNSSelectionError("selection git HEAD가 전체 40자리 SHA가 아닙니다")
    return head


def _parent_speech_authority(repo_root: Path) -> dict[str, Any]:
    holdout = _snapshot(
        repo_root,
        "data/manifests/recorded_holdout.json",
        label="DNS selection parent82 holdout",
    )
    assert holdout.data is not None
    payload = _load_json_object(holdout.data, label="parent82 holdout")
    families = payload.get("families")
    lineage = payload.get("clip_lineage")
    if not isinstance(families, Mapping) or not isinstance(lineage, Mapping):
        raise DNSSelectionError("parent82 holdout families/clip_lineage가 없습니다")
    try:
        rows = public_lineage.validate_recorded_clip_lineage(
            lineage, families=families
        )
    except ValueError as exc:
        raise DNSSelectionError(f"parent82 clip lineage 검증 실패: {exc}") from exc
    keys: set[str] = set()
    for row in rows:
        if row.get("family") != "speech":
            continue
        try:
            keys.update(
                public_lineage.conservative_cross_corpus_speech_lineage_keys(
                    row.get("lineage_keys", ())
                )
            )
        except ValueError as exc:
            raise DNSSelectionError(f"parent82 speech alias 재유도 실패: {exc}") from exc
    numeric_aliases = sorted(
        key for key in keys if key.startswith(_NUMERIC_ALIAS_PREFIXES)
    )
    if not numeric_aliases:
        raise DNSSelectionError("parent82 speech 보수적 reader/book alias가 비었습니다")
    return {
        "holdout": _file_ref(holdout, repo_root=repo_root),
        "speech_lineage_keys": sorted(keys),
        "speech_lineage_keys_sha256": _canonical_json_sha256(sorted(keys)),
        "numeric_aliases": numeric_aliases,
        "numeric_aliases_sha256": _canonical_json_sha256(numeric_aliases),
    }


def _read_public_speech_manifest(
    repo_root: Path, relative: str
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    manifest = _snapshot(repo_root, relative, label="DNS public speech manifest")
    assert manifest.data is not None
    try:
        rows = read_manifest_bytes(manifest.data, manifest_path=manifest.path)
        lineage = public_lineage.validate_public_manifest_lineage({"speech": rows})
    except ValueError as exc:
        raise DNSSelectionError(f"public speech manifest lineage 검증 실패: {exc}") from exc
    if not rows or any(row.get("tag") != "speech" for row in rows):
        raise DNSSelectionError("public speech manifest가 비었거나 speech 외 tag가 있습니다")
    return manifest, [dict(row) for row in rows], lineage


def _strict_primary(repo_root: Path, relative: str) -> tuple[Any, np.ndarray, dict[str, Any]]:
    primary = _snapshot(repo_root, relative, label="DNS selection strict primary")
    assert primary.data is not None
    try:
        with np.load(io.BytesIO(primary.data), allow_pickle=False) as archive:
            fir = np.asarray(archive["fir"], dtype=np.float64).reshape(-1)
            sample_rate = int(np.asarray(archive["sample_rate"]).reshape(-1)[0])
            delay = int(np.asarray(archive["delay_samples"]).reshape(-1)[0])
            band = [
                float(value)
                for value in np.asarray(archive["consistency_band_hz"]).reshape(-1)
            ]
    except (KeyError, ValueError, OSError) as exc:
        raise DNSSelectionError(f"strict primary NPZ를 읽을 수 없습니다: {exc}") from exc
    if (
        sample_rate != DNS_RAW_SAMPLE_RATE
        or delay < 0
        or fir.size < 1
        or not np.all(np.isfinite(fir))
        or float(np.max(np.abs(fir))) <= 0.0
        or band != [150.0, 1600.0]
    ):
        raise DNSSelectionError("strict primary sample-rate/FIR/trusted-band 계약 불일치")
    metadata = {
        **_file_ref(primary, repo_root=repo_root),
        "sample_rate": sample_rate,
        "delay_samples": delay,
        "fir_taps": int(fir.size),
        "consistency_band_hz": band,
    }
    return primary, np.ascontiguousarray(fir), metadata


def _pcm16_wav_bytes(values: np.ndarray) -> bytes:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size < 1 or not np.all(np.isfinite(samples)):
        raise DNSSelectionError("PCM16 변환 입력이 비었거나 non-finite입니다")
    quantized = _pcm16_samples(samples)
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(DNS_RAW_SAMPLE_RATE)
        handle.writeframes(quantized.tobytes())
    return output.getvalue()


def _pcm16_samples(values: np.ndarray) -> np.ndarray:
    """WAV writer와 scan이 공유하는 deterministic PCM16 quantization."""

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size < 1 or not np.all(np.isfinite(samples)):
        raise DNSSelectionError("PCM16 변환 입력이 비었거나 non-finite입니다")
    return np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")


def _pcm16_decoded_values(values: np.ndarray) -> np.ndarray:
    """libsndfile의 PCM16→float64 scale과 같은 배열을 bytes 생성 없이 만든다."""

    return _pcm16_samples(values).astype(np.float64) / 32768.0


def dns_composite_bytes_from_raw(raw_bytes: bytes) -> bytes:
    """exact 10.333초 mono48k PCM16 raw를 두 번 반복하고 15초에서 자른다."""

    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(raw_bytes))
        values, sample_rate = sf.read(
            io.BytesIO(raw_bytes), dtype="float64", always_2d=True
        )
    except RuntimeError as exc:
        raise DNSSelectionError(f"DNS selected raw decode 실패: {exc}") from exc
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.channels != 1
        or info.samplerate != DNS_RAW_SAMPLE_RATE
        or info.frames != DNS_RAW_FRAMES
        or sample_rate != DNS_RAW_SAMPLE_RATE
        or values.shape != (DNS_RAW_FRAMES, 1)
    ):
        raise DNSSelectionError(
            "DNS selected raw는 exact 496000-frame mono48k PCM16 WAV여야 합니다"
        )
    composite = np.tile(values[:, 0], DNS_REPEAT_COUNT)[:DNS_COMPOSITE_FRAMES]
    if composite.size != DNS_COMPOSITE_FRAMES:
        raise DNSSelectionError("DNS repeat-trim 결과가 720000 frame이 아닙니다")
    return _pcm16_wav_bytes(composite)


def _apply_dns_playback_gain_and_fade(mono: np.ndarray) -> np.ndarray:
    """decoded mono PCM에 실제 file gain과 수집 fade를 같은 dtype 순서로 적용한다."""

    values = np.asarray(mono, dtype=np.float32)
    if values.shape != (DNS_COMPOSITE_FRAMES,) or not np.all(np.isfinite(values)):
        raise DNSSelectionError("DNS composite playback float32 배열이 유효하지 않습니다")
    peak = float(np.max(np.abs(values)) + 1.0e-9)
    rendered = np.asarray(values / peak * DNS_PLAYBACK_AMPLITUDE, dtype=np.float32)
    ramp_frames = min(
        int(RECORDING_FILE_FADE_SECONDS * DNS_RAW_SAMPLE_RATE),
        rendered.size // 2,
    )
    if ramp_frames > 0:
        ramp = np.linspace(0.0, 1.0, ramp_frames, dtype=np.float32)
        rendered[:ramp_frames] *= ramp
        rendered[-ramp_frames:] *= ramp[::-1]
    return np.ascontiguousarray(rendered)


def _render_dns_composite_playback(composite_raw: bytes) -> np.ndarray:
    """``NoiseProgram(file)``+수집 fade의 exact bytes 경로를 벡터 연산으로 재현한다.

    실제 generator의 file branch는 같은 mono float32 배열을 Python loop로 한 sample씩
    복사한다. selector는 8천여 후보를 전수 스캔하므로 그 loop를 반복할 수 없다. 아래
    계산은 같은 decode/peak/float32/fade 순서를 보존하며 회귀 테스트가 실제 generator
    출력과 bit-exact 동등성을 강제한다.
    """

    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(composite_raw))
        data, sample_rate = sf.read(
            io.BytesIO(composite_raw), dtype="float32", always_2d=True
        )
    except RuntimeError as exc:
        raise DNSSelectionError(f"DNS composite playback decode 실패: {exc}") from exc
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.channels != 1
        or info.samplerate != DNS_RAW_SAMPLE_RATE
        or info.frames != DNS_COMPOSITE_FRAMES
        or sample_rate != DNS_RAW_SAMPLE_RATE
        or data.shape != (DNS_COMPOSITE_FRAMES, 1)
    ):
        raise DNSSelectionError(
            "DNS composite playback은 exact 720000-frame mono48k PCM16 WAV여야 합니다"
        )
    return _apply_dns_playback_gain_and_fade(data.mean(axis=1))


def _render_dns_window_playback_fast(values: np.ndarray) -> np.ndarray:
    """scan 후보를 WAV encode/decode 없이 최종 playback float32로 만든다.

    canonical bytes 경로의 산술 순서를 생략하지 않는다: source를 PCM16으로 한 번
    양자화하고 ``q/32768``로 decode한 뒤 repeat/trim, composite PCM16으로 다시
    양자화·decode한다. 그 다음에만 file peak gain과 fade를 적용한다. selected 5개
    validator는 이 fast path를 신뢰하지 않고 계속 실제 WAV bytes에서 재계산한다.
    """

    source = np.asarray(values, dtype=np.float64).reshape(-1)
    if source.shape != (DNS_RAW_FRAMES,) or not np.all(np.isfinite(source)):
        raise DNSSelectionError(
            "DNS fast timeline scan 입력은 exact 496000-frame finite mono여야 합니다"
        )
    raw_pcm = _pcm16_samples(source)
    raw_decoded = raw_pcm.astype(np.float64) / 32768.0
    remaining = DNS_COMPOSITE_FRAMES - DNS_RAW_FRAMES
    if remaining < 0 or remaining > DNS_RAW_FRAMES:
        raise DNSSelectionError("DNS repeat/trim frame 계약이 fast path 범위를 벗어납니다")
    composite_source = np.concatenate(
        [raw_decoded, raw_decoded[:remaining]]
    )
    composite_pcm = _pcm16_samples(composite_source)
    composite_decoded = composite_pcm.astype(np.float32) / 32768.0
    return _apply_dns_playback_gain_and_fade(composite_decoded)


def _rendered_source_preflight(composite_raw: bytes) -> dict[str, Any]:
    """실제 15초 file playback의 timeline·trusted-band·SNR을 함께 계산한다.

    timeline coarse stage가 source에서 읽는 것은 0.25초 capture 창뿐 아니라 양쪽
    ``coarse_search_samples``까지 합친 13,200-frame span이다. 따라서 단순 전체 RMS나
    strict-P spectrum만으로는 긴 무음/간헐음을 막을 수 없다. 선택 composite bytes를
    실제 ``NoiseProgram`` peak normalization과 수집 fade로 렌더링한 뒤, coarse stage와
    같은 span/hop/RMS 하한과 공식 측정 레벨 기반 absolute trusted-band/SNR 하한을
    함께 적용한다. 이 함수는 오디오 장치를 열지 않는다.
    """

    rendered = _render_dns_composite_playback(composite_raw)
    if (
        DNS_COMPOSITE_FRAMES != SOURCE_PREFLIGHT_FRAMES
        or rendered.shape != (SOURCE_PREFLIGHT_FRAMES,)
        or not np.all(np.isfinite(rendered))
    ):
        raise DNSSelectionError("DNS composite rendered timeline이 유효하지 않습니다")
    try:
        return rendered_source_preflight(rendered)
    except RecordingSourcePreflightError as exc:
        raise DNSSelectionError(
            f"DNS composite rendered source preflight 계산 실패: {exc}"
        ) from exc


def _validate_source_preflight(value: object, *, label: str) -> dict[str, Any]:
    try:
        return validate_rendered_source_preflight(value)
    except RecordingSourcePreflightError as exc:
        raise DNSSelectionError(f"{label} rendered source preflight 오류: {exc}") from exc


def _window_source_preflight(values: np.ndarray) -> dict[str, Any]:
    """scan window를 encode/decode 없는 exact-equivalent full preflight로 평가한다."""

    try:
        return rendered_source_preflight(
            _render_dns_window_playback_fast(values)
        )
    except RecordingSourcePreflightError as exc:
        raise DNSSelectionError(
            f"DNS scan window rendered source preflight 계산 실패: {exc}"
        ) from exc


def _decode_source(raw: bytes, *, label: str) -> np.ndarray:
    import soundfile as sf

    try:
        values, sample_rate = sf.read(
            io.BytesIO(raw), dtype="float64", always_2d=True
        )
    except RuntimeError as exc:
        raise DNSSelectionError(f"{label} decode 실패: {exc}") from exc
    if sample_rate != DNS_RAW_SAMPLE_RATE or values.shape[1] != 1:
        raise DNSSelectionError(f"{label}는 mono 48kHz여야 합니다")
    values = np.asarray(values[:, 0], dtype=np.float64)
    if values.size < DNS_RAW_FRAMES or not np.all(np.isfinite(values)):
        raise DNSSelectionError(f"{label}가 10.333초보다 짧거나 non-finite입니다")
    return values


def _window_starts(frame_count: int) -> tuple[int, ...]:
    starts = list(range(0, frame_count - DNS_RAW_FRAMES + 1, DNS_RAW_FRAMES))
    tail = frame_count - DNS_RAW_FRAMES
    starts.append(tail)
    return tuple(sorted(set(starts)))


def _band_density(values: np.ndarray, fir: np.ndarray) -> tuple[list[float], int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    fir = np.asarray(fir, dtype=np.float64).reshape(-1)
    if values.size < 1 or fir.size < 1:
        raise DNSSelectionError("strict-P density 입력/FIR이 비었습니다")
    full_length = values.size + fir.size - 1
    fft_length = 1 << (full_length - 1).bit_length()
    filtered = np.fft.irfft(
        np.fft.rfft(values, n=fft_length) * np.fft.rfft(fir, n=fft_length),
        n=fft_length,
    )[: values.size]
    window = np.hanning(filtered.size)
    spectrum = np.fft.rfft(filtered * window)
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(filtered.size, 1.0 / DNS_RAW_SAMPLE_RATE)

    def integrate(lo: float, hi: float, *, include_hi: bool) -> float:
        mask = (frequencies >= lo) & (
            frequencies <= hi if include_hi else frequencies < hi
        )
        return float(np.sum(power[mask], dtype=np.float64))

    trusted = integrate(150.0, 1600.0, include_hi=True)
    if not math.isfinite(trusted) or trusted <= 0.0:
        return [0.0 for _ in DNS_STRICT_SUBBANDS_HZ], 0
    densities: list[float] = []
    covered = 0
    trusted_width = 1600.0 - 150.0
    for index, (lo, hi) in enumerate(DNS_STRICT_SUBBANDS_HZ):
        band_power = integrate(lo, hi, include_hi=index == len(DNS_STRICT_SUBBANDS_HZ) - 1)
        density = (band_power / (hi - lo)) / (trusted / trusted_width)
        density = float(density) if math.isfinite(density) else 0.0
        densities.append(density)
        if density >= DNS_MIN_DENSITY_RATIO:
            covered += 1
    return densities, covered


def _rank_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    scan = result["coverage_scan"]
    preflight = result["source_preflight"]
    timeline = preflight["timeline_feasibility"]
    densities = [float(value) for value in scan["density_ratios"]]
    return (
        -int(scan["covered_subband_count"]),
        -densities[3],
        -min(densities),
        -sum(densities),
        -float(timeline["eligible_ratio"]),
        int(result["manifest_index"]),
    )


def _scan_manifest_rows(
    *,
    repo_root: Path,
    rows: Sequence[Mapping[str, Any]],
    lineage_summary: Mapping[str, Any],
    parent_keys: set[str],
    fir: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    components = lineage_summary.get("components")
    if not isinstance(components, Mapping):
        raise DNSSelectionError("public speech component summary가 없습니다")
    for index, raw_row in enumerate(rows):
        row = {key: value for key, value in raw_row.items() if not str(key).startswith("_")}
        path = Path(str(row.get("path") or ""))
        group = str(row.get("group_id") or "")
        split = str(row.get("split") or "")
        content = str(row.get("content_sha256") or "")
        keys_raw = row.get("lineage_keys")
        result: dict[str, Any] = {
            "manifest_index": index,
            "manifest_row_sha256": _canonical_json_sha256(row),
            "path": str(row.get("path") or ""),
            "content_sha256": content,
            "group_id": group,
            "public_source_split": split,
            "lineage_keys": list(keys_raw) if isinstance(keys_raw, list) else [],
            "status": "ineligible",
            "reason": "",
            "coverage_scan": None,
            "source_preflight": None,
            "selected_window_start_frame": None,
        }
        try:
            if _SHA256_RE.fullmatch(content) is None:
                raise DNSSelectionError("manifest content SHA가 유효하지 않습니다")
            if split not in DNS_SPLIT_QUOTAS:
                raise DNSSelectionError("manifest split이 canonical 값이 아닙니다")
            if not isinstance(keys_raw, list):
                raise DNSSelectionError("manifest lineage_keys가 없습니다")
            try:
                dns_keys = public_lineage.dns_speech_lineage_keys(path.name)
            except public_lineage.PublicLineageBlocked:
                # schema-v4 speech에는 DNS와 LibriSpeech가 함께 있을 수 있다.
                # generation validator가 둘의 raw를 모두 감사한 뒤, selector는
                # external DNS 이름만 eligible population으로 제한한다.
                result["reason"] = "not_dns_read_speech"
                results.append(result)
                continue
            expected_keys = list(
                public_lineage.conservative_cross_corpus_speech_lineage_keys(
                    dns_keys
                )
            )
            if list(keys_raw) != expected_keys:
                raise DNSSelectionError("DNS filename에서 재유도한 reader/book alias와 다릅니다")
            component = components.get(group)
            if not isinstance(component, Mapping):
                raise DNSSelectionError("public group component가 없습니다")
            component_keys = {str(value) for value in component.get("lineage_keys", ())}
            overlap = sorted(component_keys & parent_keys)
            if overlap:
                result["reason"] = f"parent82_lineage_overlap:{','.join(overlap)}"
                results.append(result)
                continue
            source_path = path if path.is_absolute() else repo_root / path
            source_path = Path(os.path.abspath(source_path))
            if source_path != repo_root and repo_root not in source_path.parents:
                raise DNSSelectionError("public DNS source path가 Elice repository 밖입니다")
            source = read_regular_file_snapshot(
                source_path,
                root=repo_root,
                label=f"DNS public source #{index}",
            )
            assert source.data is not None
            if (
                source.sha256 != content
                or source.size != row.get("content_size")
            ):
                raise DNSSelectionError(
                    "public source bytes SHA/size가 manifest와 다릅니다"
                )
            values = _decode_source(source.data, label=f"DNS public source #{index}")
            best: tuple[
                tuple[Any, ...], int, list[float], int, dict[str, Any]
            ] | None = None
            for start in _window_starts(values.size):
                window_values = values[start : start + DNS_RAW_FRAMES]
                # receipt가 보존하는 것은 float decoder 배열이 아니라 PCM16 raw
                # bytes다. 전체 8천여 후보의 WAV/composite를 만들지 않고도 같은
                # libsndfile decode scale을 재현해 점수만 보존한다. top5 bytes는
                # 선택 뒤 source를 다시 열어 별도로 materialize한다.
                quantized_values = _pcm16_decoded_values(window_values)
                densities, covered = _band_density(quantized_values, fir)
                preflight = _window_source_preflight(window_values)
                timeline = preflight["timeline_feasibility"]
                key = (
                    not bool(preflight["passed"]),
                    -covered,
                    -densities[3],
                    -min(densities),
                    -sum(densities),
                    -float(timeline["eligible_ratio"]),
                    start,
                )
                candidate = (key, start, densities, covered, preflight)
                if best is None or key < best[0]:
                    best = candidate
            if best is None:
                raise DNSSelectionError("DNS source에서 scan window를 만들지 못했습니다")
            _key, start, densities, covered, preflight = best
            passed = bool(preflight["passed"])
            result.update(
                {
                    "status": "eligible" if passed else "ineligible",
                    "reason": (
                        ""
                        if passed
                        else "rendered_source_preflight_below_minimum"
                    ),
                    "coverage_scan": {
                        "density_ratios": densities,
                        "covered_subband_count": covered,
                    },
                    "source_preflight": preflight,
                    "selected_window_start_frame": start,
                }
            )
        except (OSError, ValueError) as exc:
            # canonical manifest raw의 손상/decoder 오류는 조용히 후보 제외하지 않는다.
            raise DNSSelectionError(f"public speech manifest row #{index} scan 실패: {exc}") from exc
        results.append(result)
    return results


def _materialize_selected_bytes(
    *, repo_root: Path, row: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[bytes, bytes]:
    """top5가 확정된 뒤 선택 source만 다시 열어 exact output bytes를 만든다."""

    path = Path(str(row.get("path") or ""))
    source_path = path if path.is_absolute() else repo_root / path
    source_path = Path(os.path.abspath(source_path))
    if source_path != repo_root and repo_root not in source_path.parents:
        raise DNSSelectionError("selected DNS source path가 repository 밖입니다")
    source = read_regular_file_snapshot(
        source_path, root=repo_root, label="selected DNS source rematerialization"
    )
    assert source.data is not None
    if (
        source.sha256 != result.get("content_sha256")
        or source.size != row.get("content_size")
    ):
        raise DNSSelectionError("selected DNS source SHA/size가 scan 이후 변경됐습니다")
    values = _decode_source(source.data, label="selected DNS source rematerialization")
    start = result.get("selected_window_start_frame")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise DNSSelectionError("selected DNS window start frame이 유효하지 않습니다")
    window = values[start : start + DNS_RAW_FRAMES]
    if window.shape != (DNS_RAW_FRAMES,):
        raise DNSSelectionError("selected DNS window가 source 범위를 넘습니다")
    raw_bytes = _pcm16_wav_bytes(window)
    composite_bytes = dns_composite_bytes_from_raw(raw_bytes)
    return raw_bytes, composite_bytes


def _validate_scan_inventory(
    *,
    rows: Sequence[Mapping[str, Any]],
    lineage_summary: Mapping[str, Any],
    parent_keys: set[str],
    scan_results: Sequence[object],
) -> None:
    """raw 재스캔 없이도 full inventory의 정적 lineage/상태를 exact 검증한다."""

    components = lineage_summary.get("components")
    if not isinstance(components, Mapping):
        raise DNSSelectionError("public speech component summary가 없습니다")
    if len(scan_results) != len(rows):
        raise DNSSelectionError("DNS scan inventory와 public manifest 행 수가 다릅니다")
    for index, (raw_row, raw_result) in enumerate(zip(rows, scan_results)):
        row = {
            key: value
            for key, value in raw_row.items()
            if not str(key).startswith("_")
        }
        if not isinstance(raw_result, Mapping) or set(raw_result) != _SCAN_RESULT_KEYS:
            raise DNSSelectionError(f"DNS scan result #{index} 필드 집합이 다릅니다")
        result = raw_result
        keys = row.get("lineage_keys")
        static_expected = {
            "manifest_index": index,
            "manifest_row_sha256": _canonical_json_sha256(row),
            "path": row.get("path"),
            "content_sha256": row.get("content_sha256"),
            "group_id": row.get("group_id"),
            "public_source_split": row.get("split"),
            "lineage_keys": keys,
        }
        if (
            type(result.get("manifest_index")) is not int
            or any(
                result.get(field) != value
                for field, value in static_expected.items()
            )
        ):
            raise DNSSelectionError(
                f"DNS scan result #{index}가 immutable manifest row와 다릅니다"
            )
        if not isinstance(keys, list):
            raise DNSSelectionError(f"DNS manifest row #{index} lineage_keys가 없습니다")
        try:
            dns_keys = public_lineage.dns_speech_lineage_keys(
                Path(str(row.get("path") or "")).name
            )
        except public_lineage.PublicLineageBlocked:
            if (
                result.get("status") != "ineligible"
                or result.get("reason") != "not_dns_read_speech"
                or result.get("coverage_scan") is not None
                or result.get("source_preflight") is not None
                or result.get("selected_window_start_frame") is not None
            ):
                raise DNSSelectionError(
                    f"DNS scan result #{index} non-DNS 상태가 canonical이 아닙니다"
                )
            continue
        expected_keys = list(
            public_lineage.conservative_cross_corpus_speech_lineage_keys(dns_keys)
        )
        if keys != expected_keys:
            raise DNSSelectionError(
                f"DNS scan result #{index} reader/book alias가 filename과 다릅니다"
            )
        component = components.get(str(row.get("group_id") or ""))
        if not isinstance(component, Mapping):
            raise DNSSelectionError(f"DNS scan result #{index} public component가 없습니다")
        component_keys = {str(value) for value in component.get("lineage_keys", ())}
        overlap = sorted(component_keys & parent_keys)
        if overlap:
            expected_reason = f"parent82_lineage_overlap:{','.join(overlap)}"
            if (
                result.get("status") != "ineligible"
                or result.get("reason") != expected_reason
                or result.get("coverage_scan") is not None
                or result.get("source_preflight") is not None
                or result.get("selected_window_start_frame") is not None
            ):
                raise DNSSelectionError(
                    f"DNS scan result #{index} parent82 overlap 상태가 다릅니다"
                )
            continue
        scan = result.get("coverage_scan")
        preflight = result.get("source_preflight")
        start = result.get("selected_window_start_frame")
        if (
            not isinstance(scan, Mapping)
            or set(scan) != {"density_ratios", "covered_subband_count"}
            or not isinstance(scan.get("density_ratios"), list)
            or len(scan["density_ratios"]) != len(DNS_STRICT_SUBBANDS_HZ)
            or any(
                type(value) is not float
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in scan["density_ratios"]
            )
            or isinstance(scan.get("covered_subband_count"), bool)
            or not isinstance(scan.get("covered_subband_count"), int)
            or scan["covered_subband_count"]
            != sum(
                float(value) >= DNS_MIN_DENSITY_RATIO
                for value in scan["density_ratios"]
            )
            or isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
        ):
            raise DNSSelectionError(
                f"DNS scan result #{index} eligible coverage schema가 다릅니다"
            )
        validated_preflight = _validate_source_preflight(
            preflight, label=f"DNS scan result #{index}"
        )
        expected_status = "eligible" if validated_preflight["passed"] else "ineligible"
        expected_reason = (
            ""
            if validated_preflight["passed"]
            else "rendered_source_preflight_below_minimum"
        )
        if (
            result.get("status") != expected_status
            or result.get("reason") != expected_reason
        ):
            raise DNSSelectionError(
                f"DNS scan result #{index} source preflight 상태가 다릅니다"
            )


def _select_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, Mapping[str, Any]] = {}
    for result in results:
        preflight = result.get("source_preflight")
        if (
            result.get("status") != "eligible"
            or not isinstance(preflight, Mapping)
            or preflight.get("passed") is not True
        ):
            continue
        group = str(result["group_id"])
        previous = by_group.get(group)
        if previous is None or _rank_key(result) < _rank_key(previous):
            by_group[group] = result
    candidates = sorted(by_group.values(), key=_rank_key)
    covered = [
        item
        for item in candidates
        if int(item["coverage_scan"]["covered_subband_count"])
        == len(DNS_STRICT_SUBBANDS_HZ)
    ]
    if len(covered) < DNS_SELECTION_COUNT:
        raise DNSSelectionBlocked(
            "BLOCKED: DNS speech의 strict 네 부대역을 모두 cover하는 독립 public "
            f"group이 부족합니다: required={DNS_SELECTION_COUNT}, actual={len(covered)}"
        )
    selected = [dict(item) for item in covered[:DNS_SELECTION_COUNT]]
    if (
        len(selected) != DNS_SELECTION_COUNT
        or len({row["group_id"] for row in selected}) != DNS_SELECTION_COUNT
    ):
        raise DNSSelectionBlocked("BLOCKED: DNS selection은 정확히 5개 독립 group이어야 합니다")
    return selected


def build_dns_selection_payload(
    *,
    repo_root: str | Path,
    public_manifest: str,
    bootstrap_receipt: str,
    bootstrap_receipt_sha256: str,
    strict_primary: str,
    expected_commit: str | None = None,
    expected_public_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Elice에서 receipt payload와 bundle 파일 bytes를 결정론적으로 만든다."""

    root = Path(os.path.abspath(Path(repo_root)))
    commit = _git_head(root)
    if expected_commit is not None and commit != str(expected_commit).lower():
        raise DNSSelectionError(
            f"selection expected commit과 HEAD가 다릅니다: {expected_commit} != {commit}"
        )
    try:
        clean_source = exact_clean_source_evidence(
            root,
            expected_commit=commit,
            reject_runtime_bytecode=True,
        )
    except SourceTrustError as exc:
        raise DNSSelectionError(f"selection clean exact source 검증 실패: {exc}") from exc
    if _SHA256_RE.fullmatch(str(bootstrap_receipt_sha256).lower()) is None:
        raise DNSSelectionError("bootstrap receipt 외부 SHA-256 anchor가 필요합니다")
    bootstrap = _snapshot(root, bootstrap_receipt, label="Elice bootstrap receipt")
    if bootstrap.sha256 != str(bootstrap_receipt_sha256).lower():
        raise DNSSelectionError("Elice bootstrap receipt SHA가 외부 anchor와 다릅니다")
    assert bootstrap.data is not None
    bootstrap_payload = _load_json_object(bootstrap.data, label="Elice bootstrap receipt")
    if bootstrap_payload.get("expected_commit") != commit:
        raise DNSSelectionError("bootstrap receipt expected_commit과 selection HEAD가 다릅니다")
    environment = bootstrap_payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment)
        != {
            "freeze_receipt",
            "freeze_receipt_sha256",
            "torch_version",
            "torch_cuda",
        }
        or environment.get("freeze_receipt") != ".venv/environment-freeze.txt"
        or _SHA256_RE.fullmatch(
            str(environment.get("freeze_receipt_sha256") or "")
        )
        is None
    ):
        raise DNSSelectionError(
            "bootstrap receipt의 exact environment freeze 결속이 없습니다"
        )
    freeze = _snapshot(
        root,
        str(environment["freeze_receipt"]),
        label="Elice environment freeze receipt",
    )
    if freeze.sha256 != environment["freeze_receipt_sha256"]:
        raise DNSSelectionError(
            "Elice environment freeze bytes가 bootstrap receipt SHA와 다릅니다"
        )
    assert freeze.data is not None
    try:
        validate_environment_freeze_source_commit(
            freeze.data, expected_commit=commit
        )
    except SourceTrustError as exc:
        raise DNSSelectionError(
            f"Elice environment freeze source 결속 실패: {exc}"
        ) from exc
    try:
        selector_runtime = exact_selector_runtime_evidence(
            root,
            freeze_receipt=str(environment["freeze_receipt"]),
            expected_freeze_sha256=str(environment["freeze_receipt_sha256"]),
        )
    except SourceTrustError as exc:
        raise DNSSelectionError(f"selection isolated runtime 검증 실패: {exc}") from exc

    manifest, rows, lineage = _read_public_speech_manifest(root, public_manifest)
    if expected_public_manifest_sha256 is not None and (
        _SHA256_RE.fullmatch(str(expected_public_manifest_sha256).lower()) is None
        or manifest.sha256 != str(expected_public_manifest_sha256).lower()
    ):
        raise DNSSelectionError(
            "selector가 다시 연 public speech manifest가 schema-v4 validator snapshot과 다릅니다"
        )
    _primary, fir, primary_metadata = _strict_primary(root, strict_primary)
    parent = _parent_speech_authority(root)
    parent_keys = set(parent["speech_lineage_keys"])
    scan_results = _scan_manifest_rows(
        repo_root=root,
        rows=rows,
        lineage_summary=lineage,
        parent_keys=parent_keys,
        fir=fir,
    )
    chosen = _select_results(scan_results)
    files: dict[str, bytes] = {}
    selected: list[dict[str, Any]] = []
    for order, result in enumerate(chosen):
        manifest_index = int(result["manifest_index"])
        raw_bytes, composite_bytes = _materialize_selected_bytes(
            repo_root=root,
            row=rows[manifest_index],
            result=result,
        )
        suffix = str(result["group_id"]).removeprefix("public-lineage-")[:12]
        raw_relative = f"sources/speech-dns-{order + 1:02d}-{suffix}-raw.wav"
        composite_relative = f"sources/speech-dns-{order + 1:02d}-{suffix}-repeat-trim.wav"
        files[raw_relative] = raw_bytes
        files[composite_relative] = composite_bytes
        row = {key: value for key, value in rows[manifest_index].items() if not str(key).startswith("_")}
        selected.append(
            {
                "order": order,
                "manifest_index": manifest_index,
                "manifest_row": row,
                "manifest_row_sha256": _canonical_json_sha256(row),
                "public_group_id": result["group_id"],
                "public_source_split": result["public_source_split"],
                "recorded_split": DNS_RECORDED_SPLIT_ASSIGNMENT[order],
                "lineage_keys": result["lineage_keys"],
                "source_content_sha256": result["content_sha256"],
                "source_window_start_frame": result["selected_window_start_frame"],
                "coverage_scan": result["coverage_scan"],
                "source_preflight": result["source_preflight"],
                "raw_output": {
                    "path": f"{DNS_SELECTION_BUNDLE_ROOT}/{raw_relative}",
                    "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "size": len(raw_bytes),
                    "sample_rate": DNS_RAW_SAMPLE_RATE,
                    "channels": 1,
                    "frames": DNS_RAW_FRAMES,
                    "subtype": "PCM_16",
                },
                "composite_output": {
                    "path": f"{DNS_SELECTION_BUNDLE_ROOT}/{composite_relative}",
                    "sha256": hashlib.sha256(composite_bytes).hexdigest(),
                    "size": len(composite_bytes),
                    "sample_rate": DNS_RAW_SAMPLE_RATE,
                    "channels": 1,
                    "frames": DNS_COMPOSITE_FRAMES,
                    "subtype": "PCM_16",
                    "transform": DNS_TRANSFORM,
                    "repeat_count": DNS_REPEAT_COUNT,
                },
            }
        )
    assert manifest.data is not None and bootstrap.data is not None and freeze.data is not None
    bootstrap_parent_relative = "inputs/elice_bootstrap_receipt.selection-parent.json"
    files[bootstrap_parent_relative] = bootstrap.data
    freeze_parent_relative = "inputs/environment-freeze.selection-parent.txt"
    files[freeze_parent_relative] = freeze.data
    selection_parent_relative = "inputs/speech.selection-parent.jsonl"
    files[selection_parent_relative] = manifest.data
    immutable_manifest_ref = {
        "path": f"{DNS_SELECTION_BUNDLE_ROOT}/{selection_parent_relative}",
        "sha256": manifest.sha256,
        "size": manifest.size,
        "row_count": len(rows),
    }
    payload: dict[str, Any] = {
        "schema_version": DNS_SELECTION_SCHEMA_VERSION,
        "kind": DNS_SELECTION_KIND,
        "generation_id": DNS_SELECTION_GENERATION_ID,
        "source_commit": commit,
        "clean_source": clean_source,
        "bootstrap_receipt_origin": _file_ref(bootstrap, repo_root=root),
        "bootstrap_receipt": {
            "path": f"{DNS_SELECTION_BUNDLE_ROOT}/{bootstrap_parent_relative}",
            "sha256": bootstrap.sha256,
            "size": bootstrap.size,
        },
        "environment_freeze_origin": _file_ref(freeze, repo_root=root),
        "environment_freeze": {
            "path": f"{DNS_SELECTION_BUNDLE_ROOT}/{freeze_parent_relative}",
            "sha256": freeze.sha256,
            "size": freeze.size,
        },
        "selector_runtime": selector_runtime,
        "public_manifest_origin": {
            **_file_ref(manifest, repo_root=root),
            "row_count": len(rows),
        },
        "public_manifest": immutable_manifest_ref,
        "public_lineage": {
            "schema": public_lineage.PUBLIC_LINEAGE_SCHEMA,
            "component_count": lineage["component_count"],
            "component_membership_sha256": lineage[
                "component_membership_sha256"
            ],
            "crosswalk_policy_sha256": _canonical_json_sha256(
                public_lineage.PUBLIC_CROSSWALK_POLICY
            ),
        },
        "parent82": parent,
        "strict_primary": primary_metadata,
        "algorithm": DNS_SCAN_ALGORITHM,
        "scan_results": scan_results,
        "scan_results_sha256": _canonical_json_sha256(scan_results),
        "selected": selected,
        "selected_sha256": _canonical_json_sha256(selected),
    }
    try:
        final_clean_source = exact_clean_source_evidence(
            root,
            expected_commit=commit,
            reject_runtime_bytecode=True,
        )
    except SourceTrustError as exc:
        raise DNSSelectionError(
            f"selection scan 종료 clean exact source 재검증 실패: {exc}"
        ) from exc
    if final_clean_source != clean_source:
        raise DNSSelectionError("selection scan 도중 exact source evidence가 변경됐습니다")
    try:
        final_selector_runtime = exact_selector_runtime_evidence(
            root,
            freeze_receipt=str(environment["freeze_receipt"]),
            expected_freeze_sha256=str(environment["freeze_receipt_sha256"]),
        )
    except SourceTrustError as exc:
        raise DNSSelectionError(
            f"selection scan 종료 isolated runtime 재검증 실패: {exc}"
        ) from exc
    if final_selector_runtime != selector_runtime:
        raise DNSSelectionError("selection scan 도중 selector runtime evidence가 변경됐습니다")
    payload["evidence_sha256"] = _canonical_json_sha256(payload)
    return payload, files


def _validate_ref(
    repo_root: Path, value: object, *, label: str, capture_bytes: bool = True
) -> Any:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size"}
        or not isinstance(value.get("path"), str)
        or _SHA256_RE.fullmatch(str(value.get("sha256") or "")) is None
        or type(value.get("size")) is not int
        or int(value["size"]) <= 0
    ):
        raise DNSSelectionError(f"{label} file ref가 유효하지 않습니다")
    snapshot = _snapshot(
        repo_root, str(value["path"]), label=label, capture_bytes=capture_bytes
    )
    if snapshot.sha256 != value["sha256"] or snapshot.size != value["size"]:
        raise DNSSelectionError(f"{label} path/SHA/size가 receipt와 다릅니다")
    return snapshot


def _freeze_package_versions(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DNSSelectionError("DNS immutable environment freeze가 UTF-8이 아닙니다") from exc
    versions: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        canonical = name.strip().lower().replace("_", "-")
        if canonical in versions:
            raise DNSSelectionError(
                f"DNS immutable environment freeze package 중복: {canonical}"
            )
        versions[canonical] = version.strip()
    return versions


def _validate_selector_runtime_receipt(
    value: object, *, freeze_raw: bytes, freeze_sha256: str
) -> None:
    required = {
        "schema",
        "python_executable",
        "python_executable_realpath",
        "python_executable_sha256",
        "python_executable_size",
        "python_base_prefix",
        "python_version",
        "flags",
        "pycache_prefix",
        "sys_path",
        "environment_freeze_sha256",
        "modules",
        "libsndfile",
        "scipy_policy",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise DNSSelectionError("DNS selector runtime evidence 필드가 exact 계약과 다릅니다")
    flags = value.get("flags")
    if (
        value.get("schema") != SELECTOR_RUNTIME_SCHEMA
        or value.get("environment_freeze_sha256") != freeze_sha256
        or value.get("pycache_prefix") != SELECTOR_PYCACHE_PREFIX
        or value.get("scipy_policy")
        != "provenance_recorded_never_called_by_dns_numpy_power2_fft"
        or flags
        != {
            "isolated": 1,
            "ignore_environment": 1,
            "no_user_site": 1,
            "no_site": 1,
            "dont_write_bytecode": 1,
        }
    ):
        raise DNSSelectionError("DNS selector isolated runtime 계약이 다릅니다")
    executable = value.get("python_executable")
    executable_realpath = value.get("python_executable_realpath")
    base_prefix = value.get("python_base_prefix")
    paths = value.get("sys_path")
    if (
        not isinstance(executable, str)
        or not Path(executable).is_absolute()
        or not executable.endswith("/.venv/bin/python")
        or not isinstance(executable_realpath, str)
        or not Path(executable_realpath).is_absolute()
        or _SHA256_RE.fullmatch(
            str(value.get("python_executable_sha256") or "")
        )
        is None
        or type(value.get("python_executable_size")) is not int
        or int(value["python_executable_size"]) <= 0
        or not isinstance(base_prefix, str)
        or not Path(base_prefix).is_absolute()
        or base_prefix == "/"
        or not isinstance(value.get("python_version"), str)
        or not re.fullmatch(r"\d+\.\d+\.\d+", str(value["python_version"]))
        or not isinstance(paths, list)
        or not paths
        or len(paths) != len(set(paths))
        or any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths)
    ):
        raise DNSSelectionError("DNS selector interpreter/sys.path evidence가 유효하지 않습니다")
    issue_root = Path(executable).parents[2]
    if Path(base_prefix) not in Path(executable_realpath).parents:
        raise DNSSelectionError("DNS selector interpreter realpath가 base prefix 밖입니다")
    if paths[0] != str(issue_root / "src") or not any(
        Path(path) == issue_root / ".venv/lib" / f"python{str(value['python_version']).rsplit('.', 1)[0]}" / "site-packages"
        for path in paths
    ):
        raise DNSSelectionError("DNS selector sys.path가 issue repository/venv와 결속되지 않았습니다")
    if any("/.local/" in f"/{path.strip('/')}" for path in paths):
        raise DNSSelectionError("DNS selector sys.path에 user-site가 포함됐습니다")

    modules = value.get("modules")
    required_modules = {
        "numpy",
        "numpy.fft",
        "soundfile",
        "_soundfile",
        "_cffi_backend",
        "scipy",
        "scipy.signal",
    }
    if (
        not isinstance(modules, Mapping)
        or not required_modules.issubset(modules)
        or any(
            name not in required_modules
            and not name.startswith("numpy.fft.")
            and not (
                name.startswith("numpy")
                and isinstance(modules[name], Mapping)
                and modules[name].get("origin_kind") == "native_extension"
            )
            and not (
                name.startswith("scipy")
                and isinstance(modules[name], Mapping)
                and modules[name].get("origin_kind") == "native_extension"
            )
            for name in modules
        )
    ):
        raise DNSSelectionError("DNS selector module provenance 집합이 다릅니다")
    for name, reference in modules.items():
        if (
            not isinstance(reference, Mapping)
            or set(reference)
            != {
                "name",
                "path",
                "sha256",
                "size",
                "version",
                "loader",
                "origin_kind",
                "cached_path",
            }
            or reference.get("name") != name
            or not isinstance(reference.get("path"), str)
            or not Path(str(reference["path"])).is_absolute()
            or _SHA256_RE.fullmatch(str(reference.get("sha256") or "")) is None
            or type(reference.get("size")) is not int
            or int(reference["size"]) <= 0
            or reference.get("version") is not None
            and not isinstance(reference.get("version"), str)
            or reference.get("origin_kind")
            not in {"source", "native_extension"}
            or reference.get("loader")
            != (
                "SourceFileLoader"
                if reference.get("origin_kind") == "source"
                else "ExtensionFileLoader"
            )
        ):
            raise DNSSelectionError(f"DNS selector module provenance가 유효하지 않습니다: {name}")
        module_path = Path(str(reference["path"]))
        cached_path = reference.get("cached_path")
        if name == "_cffi_backend" and reference.get("origin_kind") != "native_extension":
            raise DNSSelectionError("DNS selector cffi backend이 native extension이 아닙니다")
        if reference.get("origin_kind") == "source":
            if (
                not isinstance(cached_path, str)
                or not cached_path.startswith(SELECTOR_PYCACHE_PREFIX + "/")
                or module_path.suffix not in {".py", ".pyw"}
            ):
                raise DNSSelectionError(
                    f"DNS selector source bytecode isolation evidence가 다릅니다: {name}"
                )
        elif cached_path is not None and (
            not isinstance(cached_path, str)
            or not cached_path.startswith(SELECTOR_PYCACHE_PREFIX + "/")
        ):
            raise DNSSelectionError(
                f"DNS selector native module cached path가 다릅니다: {name}"
            )
        if name == "_cffi_backend":
            if (
                issue_root / ".venv" not in module_path.parents
                and Path(base_prefix) not in module_path.parents
            ):
                raise DNSSelectionError(
                    "DNS selector cffi backend가 issue venv/base prefix 밖입니다"
                )
        elif issue_root / ".venv" not in module_path.parents:
            raise DNSSelectionError(
                f"DNS selector package가 issue venv 밖에서 로드됐습니다: {name}"
            )
    if not any(
        name.startswith("numpy.fft.")
        and reference.get("origin_kind") == "native_extension"
        for name, reference in modules.items()
    ) or not any(
        name.startswith("numpy")
        and reference.get("origin_kind") == "native_extension"
        for name, reference in modules.items()
    ):
        raise DNSSelectionError("DNS selector NumPy FFT/native backend bytes evidence가 없습니다")

    libsndfile = value.get("libsndfile")
    if (
        not isinstance(libsndfile, Mapping)
        or set(libsndfile) != {"path", "sha256", "size", "version"}
        or not isinstance(libsndfile.get("path"), str)
        or not Path(str(libsndfile["path"])).is_absolute()
        or issue_root / ".venv" not in Path(str(libsndfile["path"])).parents
        or not Path(str(libsndfile["path"])).name.startswith("libsndfile_")
        or Path(str(libsndfile["path"])).suffix not in {".so", ".dylib", ".dll"}
        or _SHA256_RE.fullmatch(str(libsndfile.get("sha256") or "")) is None
        or type(libsndfile.get("size")) is not int
        or int(libsndfile["size"]) <= 0
        or not isinstance(libsndfile.get("version"), str)
        or re.fullmatch(r"\d+\.\d+(?:\.\d+)?", str(libsndfile["version"])) is None
    ):
        raise DNSSelectionError("DNS selector actual libsndfile backend evidence가 유효하지 않습니다")
    versions = _freeze_package_versions(freeze_raw)
    for package in ("numpy", "soundfile", "scipy"):
        if modules[package].get("version") != versions.get(package):
            raise DNSSelectionError(
                f"DNS selector live {package} version과 immutable freeze가 다릅니다"
            )


def validate_dns_selection_receipt(
    *,
    repo_root: str | Path,
    receipt_path: str = DNS_SELECTION_RECEIPT,
    expected_receipt_sha256: str | None = None,
    require_source_files: bool = True,
    verify_current_commit: bool = True,
) -> dict[str, Any]:
    """선택 receipt를 manifest/parent/raw/composite에서 독립 재검증한다."""

    root = Path(os.path.abspath(Path(repo_root)))
    try:
        receipt = _snapshot(root, receipt_path, label="DNS selection receipt")
    except DNSSelectionError as exc:
        raise DNSSelectionBlocked(
            f"BLOCKED: external DNS speech selection receipt가 없습니다/유효하지 않습니다: {exc}"
        ) from exc
    if expected_receipt_sha256 is not None and (
        _SHA256_RE.fullmatch(str(expected_receipt_sha256).lower()) is None
        or receipt.sha256 != str(expected_receipt_sha256).lower()
    ):
        raise DNSSelectionError("DNS selection receipt 외부 SHA anchor가 다릅니다")
    assert receipt.data is not None
    payload = _load_json_object(receipt.data, label="DNS selection receipt")
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
        "selector_runtime",
        "public_manifest_origin",
        "public_manifest",
        "public_lineage",
        "parent82",
        "strict_primary",
        "algorithm",
        "scan_results",
        "scan_results_sha256",
        "selected",
        "selected_sha256",
        "evidence_sha256",
    }
    if set(payload) != required:
        raise DNSSelectionError("DNS selection receipt 필드 집합이 exact 계약과 다릅니다")
    if (
        payload.get("schema_version") != DNS_SELECTION_SCHEMA_VERSION
        or payload.get("kind") != DNS_SELECTION_KIND
        or payload.get("generation_id") != DNS_SELECTION_GENERATION_ID
        or payload.get("algorithm") != DNS_SCAN_ALGORITHM
        or payload.get("evidence_sha256")
        != _canonical_json_sha256(_without_evidence_sha(payload))
    ):
        raise DNSSelectionError("DNS selection receipt schema/algorithm/self-seal 불일치")
    commit = str(payload.get("source_commit") or "")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise DNSSelectionError("DNS selection source_commit이 유효하지 않습니다")
    clean_source = payload.get("clean_source")
    clean_policy = clean_source.get("policy") if isinstance(clean_source, Mapping) else None
    if (
        not isinstance(clean_source, Mapping)
        or not isinstance(clean_policy, Mapping)
        or clean_policy.get("protected_runtime_bytecode") != "forbidden"
    ):
        raise DNSSelectionError("DNS selection clean_source evidence가 없습니다")
    if verify_current_commit:
        try:
            current_source = exact_clean_source_evidence(
                root, expected_commit=commit
            )
        except SourceTrustError as exc:
            raise DNSSelectionError(
                f"DNS selection clean exact source 재검증 실패: {exc}"
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
            any(clean_source.get(key) != current_source.get(key) for key in common_fields)
            or not isinstance(current_policy, Mapping)
            or any(
                clean_policy.get(key) != current_policy.get(key)
                for key in set(clean_policy) - {"protected_runtime_bytecode"}
            )
        ):
            raise DNSSelectionError(
                "DNS selection clean_source evidence가 현재 exact source와 다릅니다"
            )

    bootstrap_origin = payload.get("bootstrap_receipt_origin")
    bootstrap_ref = payload.get("bootstrap_receipt")
    if (
        not isinstance(bootstrap_origin, Mapping)
        or not isinstance(bootstrap_ref, Mapping)
        or set(bootstrap_origin) != {"path", "sha256", "size"}
        or bootstrap_origin.get("sha256")
        != bootstrap_ref.get("sha256")
        or bootstrap_origin.get("size")
        != bootstrap_ref.get("size")
    ):
        raise DNSSelectionError("DNS immutable bootstrap와 origin receipt 결속이 다릅니다")
    bootstrap = _validate_ref(root, bootstrap_ref, label="DNS bootstrap receipt")
    assert bootstrap.data is not None
    bootstrap_payload = _load_json_object(bootstrap.data, label="DNS bootstrap receipt")
    if bootstrap_payload.get("expected_commit") != commit:
        raise DNSSelectionError("DNS bootstrap expected_commit과 receipt commit이 다릅니다")

    freeze_origin = payload.get("environment_freeze_origin")
    freeze_ref = payload.get("environment_freeze")
    if (
        not isinstance(freeze_origin, Mapping)
        or not isinstance(freeze_ref, Mapping)
        or set(freeze_origin) != {"path", "sha256", "size"}
        or freeze_origin.get("sha256") != freeze_ref.get("sha256")
        or freeze_origin.get("size") != freeze_ref.get("size")
    ):
        raise DNSSelectionError("DNS immutable environment freeze와 origin 결속이 다릅니다")
    freeze = _validate_ref(root, freeze_ref, label="DNS environment freeze receipt")
    assert freeze.data is not None
    bootstrap_environment = bootstrap_payload.get("environment")
    if (
        not isinstance(bootstrap_environment, Mapping)
        or bootstrap_environment.get("freeze_receipt") != freeze_origin.get("path")
        or bootstrap_environment.get("freeze_receipt_sha256") != freeze.sha256
    ):
        raise DNSSelectionError(
            "DNS environment freeze가 immutable bootstrap receipt와 다릅니다"
        )
    try:
        validate_environment_freeze_source_commit(
            freeze.data, expected_commit=commit
        )
    except SourceTrustError as exc:
        raise DNSSelectionError(
            f"DNS environment freeze source 결속 실패: {exc}"
        ) from exc
    _validate_selector_runtime_receipt(
        payload.get("selector_runtime"),
        freeze_raw=freeze.data,
        freeze_sha256=freeze.sha256,
    )

    manifest_ref = payload.get("public_manifest")
    if not isinstance(manifest_ref, Mapping) or set(manifest_ref) != {
        "path", "sha256", "size", "row_count"
    }:
        raise DNSSelectionError("DNS public manifest ref가 유효하지 않습니다")
    manifest_origin = payload.get("public_manifest_origin")
    if (
        not isinstance(manifest_origin, Mapping)
        or set(manifest_origin) != {"path", "sha256", "size", "row_count"}
        or manifest_origin.get("sha256") != manifest_ref.get("sha256")
        or manifest_origin.get("size") != manifest_ref.get("size")
        or manifest_origin.get("row_count") != manifest_ref.get("row_count")
    ):
        raise DNSSelectionError("DNS immutable selection-parent와 origin manifest 결속이 다릅니다")
    manifest = _snapshot(root, str(manifest_ref["path"]), label="DNS public speech manifest")
    if (
        manifest.sha256 != manifest_ref.get("sha256")
        or manifest.size != manifest_ref.get("size")
    ):
        raise DNSSelectionError("DNS public manifest path/SHA/size 불일치")
    assert manifest.data is not None
    rows = read_manifest_bytes(manifest.data, manifest_path=manifest.path)
    if len(rows) != manifest_ref.get("row_count"):
        raise DNSSelectionError("DNS public manifest row_count 불일치")
    try:
        lineage = public_lineage.validate_public_manifest_lineage({"speech": rows})
    except ValueError as exc:
        raise DNSSelectionError(f"DNS public manifest 전체 lineage 검증 실패: {exc}") from exc
    expected_lineage = {
        "schema": public_lineage.PUBLIC_LINEAGE_SCHEMA,
        "component_count": lineage["component_count"],
        "component_membership_sha256": lineage["component_membership_sha256"],
        "crosswalk_policy_sha256": _canonical_json_sha256(
            public_lineage.PUBLIC_CROSSWALK_POLICY
        ),
    }
    if payload.get("public_lineage") != expected_lineage:
        raise DNSSelectionError("DNS receipt public lineage 증거가 manifest 재유도값과 다릅니다")

    parent = _parent_speech_authority(root)
    if payload.get("parent82") != parent:
        raise DNSSelectionError("DNS receipt parent82 numeric alias 증거가 현재 holdout과 다릅니다")
    parent_keys = set(parent["speech_lineage_keys"])

    strict = payload.get("strict_primary")
    if not isinstance(strict, Mapping) or not isinstance(strict.get("path"), str):
        raise DNSSelectionError("DNS strict primary ref가 유효하지 않습니다")
    _primary, fir, strict_metadata = _strict_primary(root, str(strict["path"]))
    if dict(strict) != strict_metadata:
        raise DNSSelectionError("DNS strict primary metadata가 현재 NPZ와 다릅니다")

    scan_results = payload.get("scan_results")
    selected = payload.get("selected")
    if (
        not isinstance(scan_results, list)
        or len(scan_results) != len(rows)
        or payload.get("scan_results_sha256") != _canonical_json_sha256(scan_results)
        or not isinstance(selected, list)
        or len(selected) != DNS_SELECTION_COUNT
        or payload.get("selected_sha256") != _canonical_json_sha256(selected)
    ):
        raise DNSSelectionError("DNS scan/selected inventory SHA 또는 개수가 다릅니다")
    _validate_scan_inventory(
        rows=rows,
        lineage_summary=lineage,
        parent_keys=parent_keys,
        scan_results=scan_results,
    )
    # receipt score를 다시 seal하는 것만으로 winner를 바꿀 수 없도록 저장된 전체
    # scan 결과에서 canonical ranking과 quota를 재적용한다.
    derived_selected = _select_results(scan_results)
    derived_indices = [int(item["manifest_index"]) for item in derived_selected]
    actual_indices = [int(item.get("manifest_index", -1)) for item in selected]
    if actual_indices != derived_indices:
        raise DNSSelectionError("DNS selected items가 전체 scan의 canonical winner와 다릅니다")

    groups: set[str] = set()
    composites: set[str] = set()
    validated_items: list[dict[str, Any]] = []
    for order, item in enumerate(selected):
        if (
            not isinstance(item, Mapping)
            or set(item) != _SELECTED_ITEM_KEYS
            or type(item.get("order")) is not int
            or item.get("order") != order
        ):
            raise DNSSelectionError("DNS selected item order가 canonical이 아닙니다")
        raw_index = item.get("manifest_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise DNSSelectionError("DNS selected manifest index type이 다릅니다")
        index = raw_index
        if index < 0 or index >= len(rows):
            raise DNSSelectionError("DNS selected manifest index가 범위 밖입니다")
        row = {key: value for key, value in rows[index].items() if not str(key).startswith("_")}
        derived_scan = scan_results[index]
        if (
            item.get("manifest_row") != row
            or item.get("manifest_row_sha256") != _canonical_json_sha256(row)
            or item.get("public_group_id") != row.get("group_id")
            or item.get("public_source_split") != row.get("split")
            or item.get("recorded_split") != DNS_RECORDED_SPLIT_ASSIGNMENT[order]
            or item.get("lineage_keys") != row.get("lineage_keys")
            or item.get("source_content_sha256") != row.get("content_sha256")
            or item.get("source_window_start_frame")
            != derived_scan.get("selected_window_start_frame")
            or isinstance(item.get("source_window_start_frame"), bool)
            or not isinstance(item.get("source_window_start_frame"), int)
            or item.get("coverage_scan") != derived_scan.get("coverage_scan")
            or item.get("source_preflight")
            != derived_scan.get("source_preflight")
        ):
            raise DNSSelectionError("DNS selected item이 public manifest row와 다릅니다")
        group = str(item["public_group_id"])
        if group in groups:
            raise DNSSelectionError("DNS selected public group이 중복됩니다")
        groups.add(group)
        try:
            derived_keys = list(
                public_lineage.conservative_cross_corpus_speech_lineage_keys(
                    public_lineage.dns_speech_lineage_keys(Path(str(row["path"])).name)
                )
            )
        except ValueError as exc:
            raise DNSSelectionError(f"DNS selected filename lineage 재유도 실패: {exc}") from exc
        if derived_keys != item["lineage_keys"] or parent_keys.intersection(derived_keys):
            raise DNSSelectionError("DNS selected reader/book alias가 선언과 다르거나 parent82와 겹칩니다")
        raw_ref = item.get("raw_output")
        composite_ref = item.get("composite_output")
        expected_raw_keys = {"path", "sha256", "size", "sample_rate", "channels", "frames", "subtype"}
        expected_composite_keys = expected_raw_keys | {"transform", "repeat_count"}
        if (
            not isinstance(raw_ref, Mapping)
            or set(raw_ref) != expected_raw_keys
            or not isinstance(composite_ref, Mapping)
            or set(composite_ref) != expected_composite_keys
            or raw_ref.get("sample_rate") != DNS_RAW_SAMPLE_RATE
            or raw_ref.get("channels") != 1
            or raw_ref.get("frames") != DNS_RAW_FRAMES
            or raw_ref.get("subtype") != "PCM_16"
            or composite_ref.get("sample_rate") != DNS_RAW_SAMPLE_RATE
            or composite_ref.get("channels") != 1
            or composite_ref.get("frames") != DNS_COMPOSITE_FRAMES
            or composite_ref.get("subtype") != "PCM_16"
            or composite_ref.get("transform") != DNS_TRANSFORM
            or composite_ref.get("repeat_count") != DNS_REPEAT_COUNT
        ):
            raise DNSSelectionError("DNS raw/composite transform metadata가 다릅니다")
        if require_source_files:
            raw_snapshot = _validate_ref(
                root,
                {key: raw_ref[key] for key in ("path", "sha256", "size")},
                label=f"DNS selected raw #{order}",
            )
            composite_snapshot = _validate_ref(
                root,
                {
                    key: composite_ref[key]
                    for key in ("path", "sha256", "size")
                },
                label=f"DNS selected composite #{order}",
            )
            assert raw_snapshot.data is not None and composite_snapshot.data is not None
            expected_composite = dns_composite_bytes_from_raw(raw_snapshot.data)
            if expected_composite != composite_snapshot.data:
                raise DNSSelectionError("DNS composite bytes가 raw repeat-trim 재생성과 다릅니다")
            if composite_snapshot.sha256 in composites:
                raise DNSSelectionError("DNS selected composite SHA가 중복됩니다")
            composites.add(composite_snapshot.sha256)
            # strict P coverage를 selected raw에서 다시 계산해 receipt scan과 대조한다.
            source_values = _decode_source(raw_snapshot.data, label=f"DNS selected raw #{order}")
            densities, covered = _band_density(source_values, fir)
            scan = item.get("coverage_scan")
            if (
                not isinstance(scan, Mapping)
                or scan.get("covered_subband_count") != covered
                or len(scan.get("density_ratios", ())) != len(densities)
                or any(
                    not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
                    for left, right in zip(scan["density_ratios"], densities)
                )
                or covered != len(DNS_STRICT_SUBBANDS_HZ)
            ):
                raise DNSSelectionError("DNS selected strict-P coverage scan 재계산값이 다릅니다")
            preflight = _validate_source_preflight(
                item.get("source_preflight"),
                label=f"DNS selected #{order}",
            )
            derived_preflight = _rendered_source_preflight(
                composite_snapshot.data
            )
            if preflight != derived_preflight or preflight["passed"] is not True:
                raise DNSSelectionError(
                    "DNS selected rendered source preflight 재계산값이 다릅니다"
                )
        validated_items.append(dict(item))
    split_counts = {
        split: sum(item["recorded_split"] == split for item in selected)
        for split in DNS_SPLIT_QUOTAS
    }
    if split_counts != DNS_SPLIT_QUOTAS:
        raise DNSSelectionError(f"DNS selected split quota 불일치: {split_counts}")
    return {
        "receipt_path": receipt_path,
        "receipt_sha256": receipt.sha256,
        "receipt_size": receipt.size,
        "evidence_sha256": payload["evidence_sha256"],
        "source_commit": commit,
        "bootstrap_receipt_path": str(payload["bootstrap_receipt"]["path"]),
        "bootstrap_receipt_sha256": bootstrap.sha256,
        "bootstrap_receipt_size": bootstrap.size,
        "environment_freeze_path": str(payload["environment_freeze"]["path"]),
        "environment_freeze_sha256": freeze.sha256,
        "environment_freeze_size": freeze.size,
        "selector_runtime": dict(payload["selector_runtime"]),
        "public_manifest_sha256": manifest.sha256,
        "public_manifest_path": str(manifest_ref["path"]),
        "public_manifest_size": manifest.size,
        "strict_primary_path": str(strict_metadata["path"]),
        "strict_primary_sha256": str(strict_metadata["sha256"]),
        "selected": validated_items,
        "selected_group_ids": sorted(groups),
    }


__all__ = [
    "DNS_COMPOSITE_FRAMES",
    "DNS_COMPOSITE_SECONDS",
    "DNS_RAW_FRAMES",
    "DNS_REPEAT_COUNT",
    "DNS_RECORDED_SPLIT_ASSIGNMENT",
    "DNS_SCAN_ALGORITHM",
    "DNS_SELECTION_BUNDLE_ROOT",
    "DNS_SELECTION_GENERATION_ID",
    "DNS_SELECTION_RECEIPT",
    "DNS_SOURCE_KIND",
    "DNS_SPLIT_QUOTAS",
    "DNS_TRANSFORM",
    "DNSSelectionBlocked",
    "DNSSelectionError",
    "build_dns_selection_payload",
    "dns_composite_bytes_from_raw",
    "validate_dns_selection_receipt",
]
