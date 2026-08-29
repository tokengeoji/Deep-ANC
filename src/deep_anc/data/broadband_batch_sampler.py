"""실측 ERR 기반 광대역 batch 자격 증거와 결정적 계층 sampler.

광대역 loss는 각 batch의 일곱 제어대역마다 target ``d`` density를 통과한 item이
최소 4개 필요하다. group-level campaign coverage는 이 local 조건을 보장하지 않으므로,
이 모듈은 실제 ``mics.wav`` ch0(ERR)의 **학습 길이 그대로인 segment**를 FFT로 다시
검산하여 exact start와 7-bit 자격을 봉인한다. source 파일의 spectrum은 자격 판정에
사용하지 않는다.

PASS receipt만 sampler로 열 수 있다. family×band 독립 component가 부족하거나 batch가
네 family를 정확히 균등 배치하면서 band별 4개를 만들 수 없으면 BLOCKED receipt를
만들고 학습 진입은 예외로 닫는다. 오디오 장치는 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from ..dsp.control_band_contract import ControlBandContract
from .resumable_stream import indexed_rng


BROADBAND_BATCH_RECEIPT_SCHEMA = "recorded_err_subband_batch_eligibility_v2"
QUALIFIED_SAMPLING_MODE = "family_lineage_session_subband_qualified"
REQUIRED_FAMILIES = ("environment", "machine", "music", "speech")
MIN_TARGET_D_DENSITY_RATIO = 0.25
MIN_VALID_ITEMS_PER_BAND = 4
MIN_COMPONENTS_PER_FAMILY_BAND = 4


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_d_density_ratios(
    target_d: np.ndarray,
    *,
    sample_rate: int,
    bands_hz: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """``BroadbandANCLoss._target_density_by_subband``와 같은 bin 산술."""

    values = np.asarray(target_d, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("target d segment는 유한한 1-D 2샘플 이상이어야 합니다")
    samples = int(values.size)
    power = np.abs(np.fft.rfft(values, norm="ortho")) ** 2
    densities: list[float] = []
    total_power = 0.0
    total_bins = 0
    for index, raw_band in enumerate(bands_hz):
        lo, hi = (float(raw_band[0]), float(raw_band[1]))
        # ANCLoss._band_bins: ceil(lo*N/fs), floor(hi*N/fs). 마지막 외에는
        # [lo, hi)로 고정한다.
        lo_bin = max(0, int(math.ceil(lo * samples / sample_rate)))
        hi_bin = min(power.size - 1, int(math.floor(hi * samples / sample_rate)))
        if index != len(bands_hz) - 1:
            hi_bin = min(hi_bin, int(math.ceil(hi * samples / sample_rate)) - 1)
        if lo_bin > hi_bin:
            raise ValueError(f"segment FFT에 subband bin이 없습니다: {(lo, hi)}")
        selected = power[lo_bin : hi_bin + 1]
        densities.append(float(np.mean(selected)))
        total_power += float(np.sum(selected))
        total_bins += int(selected.size)
    flat = total_power / total_bins
    if flat <= np.finfo(np.float64).tiny:
        return tuple(0.0 for _ in densities)
    return tuple(float(value / flat) for value in densities)


def seal_broadband_batch_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop("evidence_sha256", None)
    sealed["evidence_sha256"] = hashlib.sha256(_canonical_json(sealed)).hexdigest()
    return sealed


def build_broadband_batch_receipt(
    *,
    manifest_path: str | Path,
    entries: Sequence[Mapping[str, Any]],
    sample_rate: int,
    segment_samples: int,
    batch_size: int,
    valid_prefix_samples: int,
    split: str = "train",
    edge_trim_samples: int | None = None,
    max_segments_per_session: int = 64,
    contract: ControlBandContract | None = None,
) -> dict[str, Any]:
    """현재 train manifest와 raw ERR로 PASS/BLOCKED receipt를 만든다.

    segment start는 edge trim 뒤의 비중첩 grid다. 세션별 최대 개수를 균일 간격으로
    고르므로 긴 세션 하나가 후보 pool을 독점하지 않는다.
    """

    control = contract or ControlBandContract.broadband_point_control()
    fs = int(sample_rate)
    length = int(segment_samples)
    size = int(batch_size)
    prefix = int(valid_prefix_samples)
    selected_split = str(split)
    trim = prefix if edge_trim_samples is None else int(edge_trim_samples)
    maximum = int(max_segments_per_session)
    if control.role != "broadband_point_control" or fs != int(control.sample_rate):
        raise ValueError("광대역 point-control 48k 계약이 필요합니다")
    if (
        length < 256
        or length % 256
        or size < 1
        or prefix <= 0
        or prefix % 256
        or trim < prefix
        or maximum < 1
        or selected_split not in {"train", "val"}
    ):
        raise ValueError("segment/batch/trim/max_segments 계약이 유효하지 않습니다")
    manifest = Path(manifest_path).expanduser().absolute()
    manifest_bytes = manifest.read_bytes()
    bands = tuple(
        tuple(float(value) for value in band)
        for band in control.point_control_subbands_hz
    )

    sessions: list[dict[str, Any]] = []
    blockers: list[str] = []
    component_assignment: dict[str, tuple[str, str]] = {}
    seen_sessions: set[str] = set()
    for entry in entries:
        component = str(entry.get("group_id") or "").strip()
        split = str(entry.get("split") or "").strip()
        family = str(entry.get("source_family") or "").strip()
        if not component or not split or not family:
            raise ValueError("manifest split/family/component가 비었습니다")
        assignment = (split, family)
        if component in component_assignment and component_assignment[component] != assignment:
            raise ValueError(f"component가 split/family를 넘나듭니다: {component}")
        component_assignment[component] = assignment
    for entry in entries:
        if str(entry.get("split")) != selected_split:
            continue
        session_id = str(entry.get("session_id") or "").strip()
        family = str(entry.get("source_family") or "").strip()
        component = str(entry.get("group_id") or "").strip()
        lineage = str(entry.get("source_pool_group_id") or "").strip()
        if not session_id or session_id in seen_sessions or not family or not component:
            raise ValueError(
                "train manifest session/family/component가 비었거나 중복입니다"
            )
        if not lineage or lineage == component:
            raise ValueError("qualified sampler에는 lineage regroup 증거가 필요합니다")
        seen_sessions.add(session_id)
        mics_path = Path(str(entry["path"])).expanduser().absolute() / "mics.wav"
        info = sf.info(mics_path)
        if int(info.samplerate) != fs or int(info.channels) < 1:
            raise ValueError(f"{session_id}: mics.wav sample rate/channel이 잘못됐습니다")
        target, read_fs = sf.read(mics_path, dtype="float64", always_2d=True)
        if int(read_fs) != fs:
            raise ValueError(f"{session_id}: mics.wav read sample rate가 다릅니다")
        err = np.asarray(target[:, 0], dtype=np.float64)
        first = trim
        last = err.size - trim - length
        starts = (
            np.arange(first, last + 1, length, dtype=np.int64)
            if last >= first
            else np.empty(0, dtype=np.int64)
        )
        if starts.size > maximum:
            chosen = np.linspace(0, starts.size - 1, num=maximum, dtype=np.int64)
            starts = starts[chosen]
        segment_rows: list[dict[str, Any]] = []
        for start in starts.tolist():
            ratios = target_d_density_ratios(
                err[start : start + length], sample_rate=fs, bands_hz=bands
            )
            valid = tuple(value >= MIN_TARGET_D_DENSITY_RATIO for value in ratios)
            if any(valid):
                segment_rows.append(
                    {
                        "start_frame": int(start),
                        "density_ratios": list(ratios),
                        "valid_bands": list(valid),
                    }
                )
        sessions.append(
            {
                "session_id": session_id,
                "source_family": family,
                "component_id": component,
                "mics": {
                    "path": str(mics_path),
                    "size_bytes": mics_path.stat().st_size,
                    "sha256": _sha256_file(mics_path),
                },
                "segments": segment_rows,
            }
        )

    family_set = {str(row["source_family"]) for row in sessions}
    if family_set != set(REQUIRED_FAMILIES):
        blockers.append(
            "train family 집합 불일치: "
            f"actual={sorted(family_set)}, required={list(REQUIRED_FAMILIES)}"
        )
    if size % len(REQUIRED_FAMILIES):
        blockers.append(
            f"batch_size {size}는 4개 family로 정확히 나누어지지 않습니다"
        )
    quota = size // len(REQUIRED_FAMILIES)
    if quota < 1:
        blockers.append("family quota가 1보다 작습니다")
    coverage_rows: list[dict[str, Any]] = []
    for family in REQUIRED_FAMILIES:
        family_rows = [row for row in sessions if row["source_family"] == family]
        unique_segments = sum(len(row["segments"]) for row in family_rows)
        if unique_segments < quota:
            blockers.append(
                f"{family}: unique eligible ERR segments {unique_segments} < quota {quota}"
            )
        for band_index, band in enumerate(bands):
            components = {
                str(row["component_id"])
                for row in family_rows
                if any(bool(segment["valid_bands"][band_index]) for segment in row["segments"])
            }
            passed = len(components) >= MIN_COMPONENTS_PER_FAMILY_BAND
            coverage_rows.append(
                {
                    "source_family": family,
                    "band_hz": list(band),
                    "eligible_components": len(components),
                    "passed": passed,
                }
            )
            if not passed:
                blockers.append(
                    f"{family}/{band[0]:g}-{band[1]:g}Hz eligible components "
                    f"{len(components)} < {MIN_COMPONENTS_PER_FAMILY_BAND}"
                )
    payload = {
        "schema": BROADBAND_BATCH_RECEIPT_SCHEMA,
        "role": (
            "training_batch_admission_not_diagnostic"
            if selected_split == "train"
            else "model_selection_batch_admission_not_diagnostic"
        ),
        "split": selected_split,
        "control_band_contract_sha256": control.digest(),
        "manifest": {
            "path": str(manifest),
            "size_bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "sample_rate": fs,
        "segment_samples": length,
        "batch_size": size,
        "valid_prefix_samples": prefix,
        "edge_trim_samples": trim,
        "max_segments_per_session": maximum,
        "subbands_hz": [list(band) for band in bands],
        "minimum_target_d_density_ratio": MIN_TARGET_D_DENSITY_RATIO,
        "minimum_valid_items_per_band": MIN_VALID_ITEMS_PER_BAND,
        "minimum_components_per_family_band": MIN_COMPONENTS_PER_FAMILY_BAND,
        "required_families": list(REQUIRED_FAMILIES),
        "target_evidence": "mics.wav ch0 ERR only; source spectrum is not used",
        "sessions": sessions,
        "summary": {
            "status": "PASS" if not blockers else "BLOCKED",
            "coverage": coverage_rows,
            "blockers": blockers,
        },
    }
    return seal_broadband_batch_receipt(payload)


@dataclass(frozen=True)
class QualifiedSegment:
    session_id: str
    source_family: str
    component_id: str
    start_frame: int
    valid_bands: tuple[bool, ...]
    required_post_augment_bands: tuple[int, ...] = ()


class BroadbandQualifiedBatchPlanner:
    """PASS receipt에서 global batch index의 순수 함수인 item 계획을 만든다."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        verify_raw: bool = True,
        expected_split: str | None = None,
        expected_valid_prefix_samples: int | None = None,
    ) -> None:
        receipt = dict(payload)
        evidence = receipt.pop("evidence_sha256", None)
        if evidence != hashlib.sha256(_canonical_json(receipt)).hexdigest():
            raise ValueError("broadband batch receipt evidence SHA가 다릅니다")
        receipt["evidence_sha256"] = evidence
        if receipt.get("schema") != BROADBAND_BATCH_RECEIPT_SCHEMA:
            raise ValueError("broadband batch receipt schema가 다릅니다")
        expected_keys = {
            "schema", "role", "split", "control_band_contract_sha256", "manifest",
            "sample_rate", "segment_samples", "batch_size", "edge_trim_samples",
            "valid_prefix_samples",
            "max_segments_per_session", "subbands_hz",
            "minimum_target_d_density_ratio", "minimum_valid_items_per_band",
            "minimum_components_per_family_band", "required_families",
            "target_evidence", "sessions", "summary", "evidence_sha256",
        }
        if set(receipt) != expected_keys:
            raise ValueError("broadband batch receipt field 집합이 정확하지 않습니다")
        expected_role = (
            "training_batch_admission_not_diagnostic"
            if receipt.get("split") == "train"
            else "model_selection_batch_admission_not_diagnostic"
        )
        if (
            receipt.get("role") != expected_role
            or receipt.get("target_evidence")
            != "mics.wav ch0 ERR only; source spectrum is not used"
        ):
            raise ValueError("source-only/diagnostic evidence는 batch 자격이 아닙니다")
        if receipt.get("summary", {}).get("status") != "PASS":
            blockers = receipt.get("summary", {}).get("blockers", [])
            raise ValueError(
                "광대역 recorded batch receipt가 BLOCKED입니다: "
                + "; ".join(map(str, blockers))
            )
        self.batch_size = int(receipt["batch_size"])
        self.segment_samples = int(receipt["segment_samples"])
        self.sample_rate = int(receipt["sample_rate"])
        self.split = str(receipt["split"])
        self.valid_prefix_samples = int(receipt["valid_prefix_samples"])
        if self.split not in {"train", "val"}:
            raise ValueError("broadband batch receipt split이 train/val이 아닙니다")
        if expected_split is not None and self.split != str(expected_split):
            raise ValueError("broadband batch receipt split이 dataset split과 다릅니다")
        if (
            self.valid_prefix_samples <= 0
            or self.valid_prefix_samples % 256
            or int(receipt["edge_trim_samples"]) < self.valid_prefix_samples
        ):
            raise ValueError("broadband batch receipt prefix/edge trim 계약 위반")
        if (
            expected_valid_prefix_samples is not None
            and self.valid_prefix_samples != int(expected_valid_prefix_samples)
        ):
            raise ValueError("broadband batch receipt prefix가 authority prefix와 다릅니다")
        self.bands = tuple(tuple(map(float, band)) for band in receipt["subbands_hz"])
        control = ControlBandContract.broadband_point_control()
        if (
            receipt.get("control_band_contract_sha256") != control.digest()
            or self.bands
            != tuple(tuple(map(float, band)) for band in control.point_control_subbands_hz)
        ):
            raise ValueError(
                "broadband batch control-band 계약이 current authority와 다릅니다"
            )
        if (
            float(receipt["minimum_target_d_density_ratio"]) != MIN_TARGET_D_DENSITY_RATIO
            or int(receipt["minimum_valid_items_per_band"]) != MIN_VALID_ITEMS_PER_BAND
            or int(receipt["minimum_components_per_family_band"]) != MIN_COMPONENTS_PER_FAMILY_BAND
            or tuple(receipt["required_families"]) != REQUIRED_FAMILIES
            or self.batch_size % len(REQUIRED_FAMILIES)
        ):
            raise ValueError("broadband batch hard floor가 canonical 값과 다릅니다")
        manifest = receipt["manifest"]
        manifest_path = Path(manifest["path"])
        if (
            not manifest_path.is_file()
            or manifest_path.stat().st_size != int(manifest["size_bytes"])
            or _sha256_file(manifest_path) != str(manifest["sha256"])
        ):
            raise ValueError("broadband batch receipt manifest bytes가 다릅니다")

        manifest_entries = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest_selected = {
            str(entry.get("session_id")): entry
            for entry in manifest_entries
            if str(entry.get("split")) == self.split
        }
        if len(manifest_selected) != sum(
            str(entry.get("split")) == self.split for entry in manifest_entries
        ):
            raise ValueError(f"{self.split} manifest session_id가 중복됐습니다")
        all_component_assignments: dict[str, tuple[str, str]] = {}
        for entry in manifest_entries:
            component = str(entry.get("group_id") or "")
            assignment = (str(entry.get("split") or ""), str(entry.get("source_family") or ""))
            if not component or not all(assignment):
                raise ValueError("manifest split/family/component가 비었습니다")
            if (
                component in all_component_assignments
                and all_component_assignments[component] != assignment
            ):
                raise ValueError(f"component가 split/family를 넘나듭니다: {component}")
            all_component_assignments[component] = assignment
        receipt_sessions = receipt["sessions"]
        if not isinstance(receipt_sessions, list) or {
            str(session.get("session_id")) for session in receipt_sessions
        } != set(manifest_selected):
            raise ValueError(
                f"receipt session 집합이 current {self.split} manifest와 다릅니다"
            )
        component_assignments: dict[str, str] = {}
        seen_ranges: set[tuple[str, int]] = set()
        for session in receipt_sessions:
            session_id = str(session.get("session_id"))
            entry = manifest_selected[session_id]
            family = str(session.get("source_family"))
            component = str(session.get("component_id"))
            expected_mics = Path(str(entry["path"])).expanduser().absolute() / "mics.wav"
            if (
                family != str(entry.get("source_family"))
                or component != str(entry.get("group_id"))
                or Path(str(session.get("mics", {}).get("path", ""))).expanduser().absolute()
                != expected_mics
            ):
                raise ValueError(f"{session_id}: receipt lineage/path가 manifest와 다릅니다")
            if component in component_assignments and component_assignments[component] != family:
                raise ValueError("component가 여러 family로 갈라졌습니다")
            component_assignments[component] = family
            for segment in session.get("segments", []):
                key = (session_id, int(segment.get("start_frame", -1)))
                if key in seen_ranges:
                    raise ValueError(f"{session_id}: duplicate eligible segment가 있습니다")
                seen_ranges.add(key)
                if (
                    len(segment.get("density_ratios", [])) != len(self.bands)
                    or len(segment.get("valid_bands", [])) != len(self.bands)
                ):
                    raise ValueError(f"{session_id}: density/valid vector 길이가 다릅니다")

        self._segments: tuple[QualifiedSegment, ...] = tuple(
            QualifiedSegment(
                session_id=str(session["session_id"]),
                source_family=str(session["source_family"]),
                component_id=str(session["component_id"]),
                start_frame=int(segment["start_frame"]),
                valid_bands=tuple(bool(value) for value in segment["valid_bands"]),
            )
            for session in receipt["sessions"]
            for segment in session["segments"]
        )
        if verify_raw:
            self._verify_raw(receipt)
        self._by_family: dict[str, tuple[QualifiedSegment, ...]] = {
            family: tuple(row for row in self._segments if row.source_family == family)
            for family in REQUIRED_FAMILIES
        }
        coverage = []
        for family in REQUIRED_FAMILIES:
            for band_index, band in enumerate(self.bands):
                components = {
                    row.component_id
                    for row in self._by_family[family]
                    if row.valid_bands[band_index]
                }
                passed = len(components) >= MIN_COMPONENTS_PER_FAMILY_BAND
                coverage.append(
                    {
                        "source_family": family,
                        "band_hz": list(band),
                        "eligible_components": len(components),
                        "passed": passed,
                    }
                )
                if not passed:
                    raise ValueError(
                        f"{family}/{band} eligible component가 4보다 적습니다"
                    )
        if receipt["summary"] != {
            "status": "PASS",
            "coverage": coverage,
            "blockers": [],
        }:
            raise ValueError("broadband batch PASS summary가 segment evidence와 다릅니다")
        # Constructor에서도 실제 plan 하나를 만들어 receipt가 허위 PASS여도 fail-close.
        self.batch(0, seed=0)

    @property
    def session_ids(self) -> frozenset[str]:
        return frozenset(row.session_id for row in self._segments)

    def _verify_raw(self, receipt: Mapping[str, Any]) -> None:
        for session in receipt["sessions"]:
            reference = session["mics"]
            path = Path(reference["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(reference["size_bytes"])
                or _sha256_file(path) != str(reference["sha256"])
            ):
                raise ValueError(
                    f"{session['session_id']}: mics.wav bytes가 receipt와 다릅니다"
                )
            audio, fs = sf.read(path, dtype="float64", always_2d=True)
            if int(fs) != self.sample_rate:
                raise ValueError(f"{session['session_id']}: mics.wav sample rate가 다릅니다")
            err = np.asarray(audio[:, 0], dtype=np.float64)
            for segment in session["segments"]:
                start = int(segment["start_frame"])
                stop = start + self.segment_samples
                if start < 0 or stop > err.size:
                    raise ValueError(f"{session['session_id']}: segment 범위가 raw 밖입니다")
                ratios = target_d_density_ratios(
                    err[start:stop], sample_rate=self.sample_rate, bands_hz=self.bands
                )
                stored = tuple(float(value) for value in segment["density_ratios"])
                valid = tuple(bool(value) for value in segment["valid_bands"])
                expected_valid = tuple(value >= MIN_TARGET_D_DENSITY_RATIO for value in ratios)
                if len(stored) != len(self.bands) or not np.allclose(
                    stored, ratios, rtol=1e-10, atol=1e-12
                ):
                    raise ValueError(
                        f"{session['session_id']}: ERR density 재계산이 receipt와 다릅니다"
                    )
                if valid != expected_valid:
                    raise ValueError(
                        f"{session['session_id']}: ERR valid mask가 density와 다릅니다"
                    )

    @staticmethod
    def _hierarchical_pick(
        candidates: Sequence[QualifiedSegment],
        *,
        rng: np.random.Generator,
        used: set[tuple[str, int]],
    ) -> QualifiedSegment:
        available = [row for row in candidates if (row.session_id, row.start_frame) not in used]
        if not available:
            raise ValueError(
                "광대역 batch에 필요한 unique eligible ERR segment가 부족합니다"
            )
        components = sorted({row.component_id for row in available})
        component = components[int(rng.integers(len(components)))]
        component_rows = [row for row in available if row.component_id == component]
        sessions = sorted({row.session_id for row in component_rows})
        session = sessions[int(rng.integers(len(sessions)))]
        session_rows = [row for row in component_rows if row.session_id == session]
        return session_rows[int(rng.integers(len(session_rows)))]

    def batch(self, batch_index: int, *, seed: int) -> tuple[QualifiedSegment, ...]:
        if int(batch_index) < 0:
            raise ValueError("batch_index는 0 이상이어야 합니다")
        quota = self.batch_size // len(REQUIRED_FAMILIES)
        selected: list[QualifiedSegment] | None = None
        # 자연 speech/music 한 segment에 일곱 대역 동시 density를 강제하지 않는다.
        # 각 시도는 아직 부족한 band를 가장 많이 덮는 segment를 family quota 안에서
        # greedy set-cover로 고른다. global batch index와 attempt만 RNG 입력이라 resume/
        # worker 수에 무관하다. receipt가 실제로 batch를 구성하지 못하면 bounded하게
        # 예외로 닫힌다.
        for plan_attempt in range(128):
            candidate_rows: list[QualifiedSegment] = []
            used: set[tuple[str, int]] = set()
            family_counts = {family: 0 for family in REQUIRED_FAMILIES}
            valid_counts = [0] * len(self.bands)
            for slot in range(self.batch_size):
                rng = indexed_rng(
                    seed,
                    0x425242,
                    int(batch_index),
                    plan_attempt * self.batch_size + slot,
                )
                eligible: list[QualifiedSegment] = []
                best_score = -1
                for family in REQUIRED_FAMILIES:
                    if family_counts[family] >= quota:
                        continue
                    for row in self._by_family[family]:
                        if (row.session_id, row.start_frame) in used:
                            continue
                        score = sum(
                            count < MIN_VALID_ITEMS_PER_BAND and row.valid_bands[index]
                            for index, count in enumerate(valid_counts)
                        )
                        if score > best_score:
                            best_score = int(score)
                            eligible = [row]
                        elif score == best_score:
                            eligible.append(row)
                if not eligible:
                    break
                # 같은 component의 여러 segment가 후보 수로 가중되지 않도록 기존
                # hierarchy와 같은 component→session→segment 순서로 고른다.
                row = self._hierarchical_pick(eligible, rng=rng, used=used)
                candidate_rows.append(row)
                used.add((row.session_id, row.start_frame))
                family_counts[row.source_family] += 1
                for index, valid in enumerate(row.valid_bands):
                    valid_counts[index] += int(valid)
            if (
                len(candidate_rows) == self.batch_size
                and set(family_counts.values()) == {quota}
                and all(count >= MIN_VALID_ITEMS_PER_BAND for count in valid_counts)
            ):
                selected = candidate_rows
                break
        if selected is None:
            raise ValueError(
                "광대역 receipt에서 family-balanced band별 valid item>=4 batch를 "
                "128회 deterministic plan으로 구성할 수 없습니다"
            )

        # 증강 후 지켜야 할 band reservation만 각 item에 표시한다. band마다 valid
        # item 네 개를 고르되 한 item이 여러 band를 맡을 수 있고 all-seven은 요구하지 않는다.
        assigned: list[set[int]] = [set() for _ in selected]
        for band_index in range(len(self.bands)):
            valid_indices = [
                index
                for index, row in enumerate(selected)
                if row.valid_bands[band_index]
            ]
            order_rng = indexed_rng(
                seed, 0x425241, int(batch_index), band_index
            )
            ordered_valid = [
                valid_indices[index]
                for index in order_rng.permutation(len(valid_indices))
            ]
            for index in ordered_valid[:MIN_VALID_ITEMS_PER_BAND]:
                assigned[index].add(band_index)
        selected = [
            replace(row, required_post_augment_bands=tuple(sorted(assigned[index])))
            for index, row in enumerate(selected)
        ]
        # 순서도 batch index에 결속하되 구성 내용은 바꾸지 않는다.
        order_rng = indexed_rng(seed, 0x42524F, int(batch_index))
        ordered = tuple(selected[index] for index in order_rng.permutation(len(selected)))
        valid_counts = tuple(
            sum(row.valid_bands[index] for row in ordered)
            for index in range(len(self.bands))
        )
        if len(ordered) != self.batch_size or set(family_counts.values()) != {quota}:
            raise RuntimeError("광대역 batch family quota 구성 결함")
        if any(count < MIN_VALID_ITEMS_PER_BAND for count in valid_counts):
            raise RuntimeError(f"광대역 batch band별 valid item 구성 결함: {valid_counts}")
        assigned_counts = tuple(
            sum(index in row.required_post_augment_bands for row in ordered)
            for index in range(len(self.bands))
        )
        if assigned_counts != (MIN_VALID_ITEMS_PER_BAND,) * len(self.bands):
            raise RuntimeError(f"광대역 batch band reservation 결함: {assigned_counts}")
        return ordered

    def item(self, global_item_index: int, *, seed: int) -> QualifiedSegment:
        index = int(global_item_index)
        if index < 0:
            raise ValueError("global_item_index는 0 이상이어야 합니다")
        batch_index, offset = divmod(index, self.batch_size)
        return self.batch(batch_index, seed=seed)[offset]


__all__ = [
    "BROADBAND_BATCH_RECEIPT_SCHEMA",
    "BroadbandQualifiedBatchPlanner",
    "MIN_COMPONENTS_PER_FAMILY_BAND",
    "MIN_TARGET_D_DENSITY_RATIO",
    "MIN_VALID_ITEMS_PER_BAND",
    "QUALIFIED_SAMPLING_MODE",
    "REQUIRED_FAMILIES",
    "QualifiedSegment",
    "build_broadband_batch_receipt",
    "seal_broadband_batch_receipt",
    "target_d_density_ratios",
]
