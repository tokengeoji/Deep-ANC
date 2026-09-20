#!/usr/bin/env python3
"""기본은 validation 탐색 계획 검사만 한다. 실제 탐색은 별도 명시 옵션이 필요하다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from deepanc.calibration import DEFAULT_RIR, load_secondary_path
from deep_anc.eval.baseline_selection import _candidate, select_validation_baselines
from deep_anc.eval.high_frequency_metrics import _real, _int, validate_regions
from deep_anc.eval.high_frequency_readiness import _json, _sha


PLAN_KEYS = {
    "schema", "secondary_path", "secondary_estimate_path", "validation_case_manifest", "measurement_packet", "candidates",
    "additional_delay_samples", "estimate_delay_samples", "sample_rate", "control_limit",
    "power_floor", "max_limited_fraction", "max_low_amplification_db", "max_case_runs", "runtime_conditions",
}
CASE_KEYS = {"split", "session_id", "source_id", "source_kind", "reference_path", "disturbance_path", "regions"}
# HANDOFF: MIMII/machine은 train 보조 전용이며 validation/test에서 제외한다.
SOURCE_KINDS = {"noise", "speech", "music", "mixed", "environment"}
NUMERIC_KEYS = ("additional_delay_samples", "estimate_delay_samples", "control_limit",
                "power_floor", "max_limited_fraction", "max_low_amplification_db", "max_case_runs")


def plan_template():
    return {
        "schema": "classical_validation_search_plan.v1",
        "secondary_path": str(DEFAULT_RIR.relative_to(ROOT)),
        "secondary_estimate_path": str(DEFAULT_RIR.relative_to(ROOT)),
        "sample_rate": 16000, "validation_case_manifest": None, "measurement_packet": None,
        "candidates": [{"id": name + "_candidate_001", "algorithm": name, "mu": None,
                        "control_length": None, "block_samples": None,
                        "normalization_epsilon": None, "weight_norm_limit": None}
                       for name in ("fxlms", "fxnlms")],
        **{name: None for name in NUMERIC_KEYS}, "runtime_conditions": None,
    }


def _resolve(value, parent):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _new_output(value):
    path = Path(value).absolute()
    if path.exists() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("기존 출력/출력 부모 symlink를 사용할 수 없습니다")
    return path


def inspect_plan(path):
    plan = _json(_metadata_path(path, ".json"))
    if set(plan) != PLAN_KEYS or plan["schema"] != "classical_validation_search_plan.v1":
        raise ValueError("알 수 없는 탐색 계약 또는 key 누락")
    if type(plan["sample_rate"]) is not int or plan["sample_rate"] != 16000:
        raise ValueError("16000 Hz 계약이 필요합니다")
    # 현재 CLI는 OMAP 정본만 허용한다. 임의 NPZ·48 kHz S를 조용히 받아들이지 않는다.
    for key in ("secondary_path", "secondary_estimate_path"):
        if not isinstance(plan[key], str) or _resolve(plan[key], ROOT) != DEFAULT_RIR.resolve():
            raise ValueError("S/S_hat은 원본 rir.txt를 명시해야 합니다")
    secondary = load_secondary_path(sample_rate=16000)
    missing = [name for name in (*NUMERIC_KEYS, "validation_case_manifest", "measurement_packet", "runtime_conditions")
               if plan[name] is None]
    for name in NUMERIC_KEYS:
        value = plan[name]
        if value is None:
            continue
        if name in ("additional_delay_samples", "estimate_delay_samples"):
            _int(value, name)
        elif name == "max_case_runs":
            _int(value, name, 1, 1_000_000)
        else:
            _real(value, name, strict=name in ("control_limit", "power_floor"))
    if plan["control_limit"] is not None and plan["control_limit"] > 1:
        raise ValueError("공통 정규화 DAC 한도는 1 이하여야 합니다")
    if plan["max_limited_fraction"] is not None and plan["max_limited_fraction"] > 1:
        raise ValueError("clip 비율은 0~1 범위여야 합니다")
    if plan["runtime_conditions"] is not None and (
            not isinstance(plan["runtime_conditions"], dict) or not plan["runtime_conditions"]):
        raise ValueError("공통 runtime_conditions는 비어 있지 않은 JSON 객체여야 합니다")
    if plan["runtime_conditions"] is not None and "validation_input_binding" in plan["runtime_conditions"]:
        raise ValueError("validation_input_binding은 QA 결합 결과의 예약 key입니다")
    for name in ("validation_case_manifest", "measurement_packet"):
        if plan[name] is not None and (not isinstance(plan[name], str) or not plan[name].strip()):
            raise ValueError(f"{name}: 비어 있지 않은 경로 문자열 또는 null 필요")
    candidates = plan["candidates"]
    candidate_keys = {"id", "algorithm", "mu", "control_length", "block_samples",
                      "normalization_epsilon", "weight_norm_limit"}
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("두 알고리즘의 명시적인 candidate 목록이 필요합니다")
    ids = set()
    algorithms = set()
    for index, item in enumerate(candidates):
        if (not isinstance(item, dict) or set(item) != candidate_keys
                or item["algorithm"] not in ("fxlms", "fxnlms")
                or not isinstance(item["id"], str) or not item["id"].strip() or item["id"] in ids):
            raise ValueError("candidate 형식/ID/algorithm 오류")
        ids.add(item["id"])
        algorithms.add(item["algorithm"])
        absent = [key for key, value in item.items() if value is None]
        missing.extend(f"candidates[{index}].{key}" for key in absent)
        if not absent:
            _candidate(item)
    if algorithms != {"fxlms", "fxnlms"}:
        raise ValueError("FxLMS와 FxNLMS 두 기준선을 모두 명시해야 합니다")
    return plan, missing, secondary


def inspect_cases(path):
    manifest = _json(_metadata_path(path, ".json"))
    if set(manifest) != {"schema", "cases"} or manifest["schema"] != "classical_validation_cases.v1":
        raise ValueError("validation case manifest 계약 오류")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("비어 있지 않은 validation cases 필요")
    # 모든 split을 먼저 검사한다. 뒤쪽에 test가 있어도 앞쪽 WAV부터 열지 않는다.
    for case in cases:
        if not isinstance(case, dict) or case.get("split") != "validation":
            raise ValueError("validation만 허용하며 train/test 파형을 열지 않습니다")
    identifiers = set()
    for case in cases:
        if set(case) != CASE_KEYS or case["source_kind"] not in SOURCE_KINDS:
            raise ValueError("case key/source_kind 오류")
        for key in ("session_id", "source_id", "reference_path", "disturbance_path"):
            if not isinstance(case[key], str) or not case[key].strip():
                raise ValueError(f"{key}: 비어 있지 않은 문자열 필요")
        identifier = (case["session_id"], case["source_id"])
        if identifier in identifiers:
            raise ValueError("validation case 중복")
        identifiers.add(identifier)
        validate_regions(case["regions"], 1_048_576)
    return cases


def _plain_path(path):
    """resolve 전에 symlink/..를 검사한다. 잠긴 WAV를 따라가서 검사하지 않는다."""
    path = Path(path).absolute()
    if ".." in path.parts or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("입력 경로의 symlink/부모 symlink/상위 이동은 금지합니다")
    return path


def _relative_input(value, parent):
    path = Path(value)
    return _plain_path(path if path.is_absolute() else parent / path)


def _metadata_path(path, suffix):
    path = _plain_path(path)
    if path.suffix != suffix or not path.is_file():
        raise ValueError(f"일반 {suffix} 메타데이터 파일만 허용합니다")
    return path


def inspect_measurement_packet(packet, cases, manifest_path):
    """QA/lock/capture/manifest를 메타데이터만으로 결합한다. train/test WAV 0개."""
    from deepanc.measurement_ready import _metadata_errors

    packet = _plain_path(packet)
    if not packet.is_dir():
        raise ValueError("측정 packet 디렉터리 필요")
    snapshots = {}
    def snapshot(path):
        path = _metadata_path(path, ".json")
        value = _json(path)
        snapshots[str(path)] = _sha(path)
        return value

    qa = snapshot(packet / "report.json")
    lock = snapshot(packet / "final_test_lock.json")
    secondary_sha = _sha(DEFAULT_RIR)
    if (qa.get("schema") != "omap_capture_intake_report.v1" or qa.get("intake_passed") is not True
            or qa.get("prepared_train_valid") is not True or qa.get("final_test_isolated") is not True
            or qa.get("sample_rate") != 16000 or qa.get("pcm_bits") != 16
            or qa.get("secondary_source_sha256") != secondary_sha
            or qa.get("training_manifest") != "train_valid/manifest.jsonl"
            or qa.get("final_test_lock") != "final_test_lock.json"):
        raise ValueError("측정 packet QA/정본 S/준비 경로 계약 불일치")
    if (lock.get("schema") != "omap_final_test_lock.v1" or lock.get("split") != "test"
            or lock.get("training_and_tuning_allowed") is not False
            or lock.get("contains_prepared_training_pairs") is not False
            or lock.get("input_root") != qa.get("input_root")):
        raise ValueError("final test 잠금 계약 불일치")
    rows, locked = qa.get("recordings"), lock.get("records")
    if (not isinstance(rows, list) or not rows or not isinstance(locked, list) or not locked
            or not all(isinstance(row, dict) for row in rows + locked)
            or {row.get("split") for row in rows} != {"train", "valid", "test"}
            or locked != [row for row in rows if row["split"] == "test"]):
        raise ValueError("QA recordings와 final test lock 불일치")
    if any(row.get("prepared_reference") is not None or row.get("prepared_disturbance") is not None for row in locked):
        raise ValueError("test에는 prepared pair를 만들 수 없습니다")
    capture_items = qa.get("capture_metadata")
    if not isinstance(capture_items, list) or not capture_items:
        raise ValueError("capture 원본 메타데이터 필요")
    captures, gain_profile = {}, None
    for item in capture_items:
        path = _plain_path(item["path"])
        metadata = snapshot(path)
        split, session = metadata.get("split"), metadata.get("session_id")
        if (path.name != "capture.json" or snapshots[str(path)] != item.get("sha256")
                or metadata != item.get("metadata")
                or _metadata_errors(metadata, split, session, secondary_sha)
                or split not in ("train", "valid", "test")
                or not isinstance(session, str) or Path(session).name != session
                or path != _plain_path(Path(qa["input_root"]) / split / session / "capture.json")):
            raise ValueError("capture 원본 경로/메타데이터/SHA 계약 불일치")
        profile = (metadata["gain_profile_id"], json.dumps(metadata["gains"], sort_keys=True))
        if gain_profile is not None and profile != gain_profile:
            raise ValueError("세션 간 gain profile 불일치")
        gain_profile = profile
        if (split, session) in captures:
            raise ValueError("capture 세션 중복")
        captures[(split, session)] = metadata
    if (set(captures) != {(row["split"], row["session"]) for row in rows}
            or lock.get("capture_metadata") != [item for item in capture_items if item["metadata"]["split"] == "test"]):
        raise ValueError("QA/capture/lock 세션 집합 불일치")
    identities, expected, pcm_splits = {}, {}, {}
    def identity(namespace, value, split):
        if not isinstance(value, str) or not value.strip() or value.strip().casefold() == "unknown":
            raise ValueError("원녹음/세션/그룹 identity 필요")
        key = (namespace, value.strip().casefold())
        if key in identities and identities[key] != split:
            raise ValueError("train/valid/test 원녹음·세션·그룹 누출")
        identities[key] = split
    for row in rows:
        split, session = row["split"], row["session"]
        metadata = captures[(split, session)]
        if any(row.get(field) != metadata.get(field) for field in
               ("source_family", "source_recording_id", "source_recording_ids", "source_group_ids")):
            raise ValueError("QA와 capture source identity 불일치")
        if (row.get("session_id") != session or row.get("source_id") != row["source_recording_id"]
                or row.get("source_kind") != row["source_family"]):
            raise ValueError("QA source/session alias 불일치")
        identity("session", session, split)
        identity("source", row["source_recording_id"], split)
        for field, namespace in (("source_recording_ids", "source"), ("source_group_ids", "group")):
            for value in row[field]:
                identity(namespace, value, split)
        source = Path(row["path"])
        if (source.is_absolute() or len(source.parts) != 3 or source.parts[:2] != (split, session)
                or ".." in source.parts or source.suffix.lower() != ".wav"):
            raise ValueError("QA 원천 split/session 경로 불일치")
        _int(row["frames"], "QA frames", 1, 2**53-1)
        for channel in ("reference", "disturbance"):
            digest = row[channel + "_pcm_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("QA PCM SHA 형식 오류")
            if digest in pcm_splits and pcm_splits[digest] != split:
                raise ValueError("train/valid/test PCM SHA 중복")
            pcm_splits[digest] = split
        if split != "test":
            reference = (source.parent / (source.stem + "_reference.wav")).as_posix()
            if reference in expected:
                raise ValueError("QA 원천 중복")
            expected[reference] = row
    base = _plain_path(packet / "train_valid")
    train_manifest = _metadata_path(base / "manifest.jsonl", ".jsonl")
    if train_manifest.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("manifest 크기 제한 위반")
    snapshots[str(train_manifest)] = _sha(train_manifest)
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("manifest JSON 중복 key")
        return result
    prepared = [json.loads(line, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("비유한 manifest")))
        for line in train_manifest.read_text().splitlines() if line.strip()]
    if (not prepared or any(not isinstance(row, dict) or row.get("split") not in ("train", "valid") for row in prepared)
            or len(prepared) != len(expected) or {row.get("reference") for row in prepared} != set(expected)):
        raise ValueError("QA와 train_valid manifest 집합/split 불일치; WAV 미열기")
    preparation = snapshot(base / "preparation.json")
    if (preparation.get("sample_rate") != 16000 or preparation.get("pcm_bits") != 16
            or preparation.get("attested_anc_off") is not True or preparation.get("attested_raw_pcm") is not True
            or preparation.get("reference_channel") != "left" or preparation.get("disturbance_channel") != "right"):
        raise ValueError("raw preparation 계약 불일치")
    provenance_rows = preparation.get("recordings")
    if not isinstance(provenance_rows, list) or not all(isinstance(row, dict) for row in provenance_rows):
        raise ValueError("preparation 원천 목록 오류")
    provenance = {row["source"]: row for row in provenance_rows}
    if len(provenance) != len(provenance_rows) or set(provenance) != {row["path"] for row in expected.values()}:
        raise ValueError("preparation 원천 집합 불일치")
    approved = {}
    for row in prepared:
        origin = expected[row["reference"]]
        if (row.get("anc_enabled") is not False or row.get("signal_domain") != "raw_pcm"
                or row.get("split") != origin["split"] or row.get("session") != origin["session"]):
            raise ValueError("prepared raw ANC-OFF/split/session 불일치")
        paths = {}
        for channel in ("reference", "disturbance"):
            relative = Path(row[channel])
            source = Path(origin["path"])
            wanted = source.parent / (source.stem + f"_{channel}.wav")
            if (relative != wanted or origin.get("prepared_" + channel) != (Path("train_valid") / wanted).as_posix()
                    or provenance[origin["path"]].get(channel + "_pcm_sha256") != origin[channel + "_pcm_sha256"]
                    or provenance[origin["path"]].get("frames") != origin["frames"]):
                raise ValueError("prepared 경로/PCM SHA/길이의 QA 결합 불일치")
            paths[channel + "_path"] = str(_plain_path(base / relative))
        if row["split"] == "valid":
            approved[paths["reference_path"]] = {**paths, "session_id": row["session"],
                "source_id": origin["source_recording_id"], "source_kind": origin["source_family"],
                "source_recording_ids": origin["source_recording_ids"], "source_group_ids": origin["source_group_ids"],
                "frames": origin["frames"], **{channel + "_pcm_sha256": origin[channel + "_pcm_sha256"]
                    for channel in ("reference", "disturbance")}}
    bound = []
    for case in cases or []:
        paths = {channel + "_path": str(_relative_input(case[channel + "_path"], manifest_path.parent))
                 for channel in ("reference", "disturbance")}
        source = approved.get(paths["reference_path"])
        if source is None or any(source[key] != value for key, value in paths.items()) or any(
                source[key] != case[key] for key in ("session_id", "source_id", "source_kind")):
            raise ValueError("case는 QA의 valid prepared REF/ERR와 session/source identity가 일치해야 합니다")
        if source["source_kind"] not in SOURCE_KINDS:
            raise ValueError("MIMII/machine은 validation에서 금지합니다")
        _int(source["frames"], "case frames", 1, 1_048_576)
        validate_regions(case["regions"], source["frames"])
        bound.append(source)
    return {"schema": "classical_validation_input_binding.v1", "measurement_packet": str(packet),
            "metadata_sha256": snapshots, "cases": bound, "train_wav_opened": False,
            "test_wav_opened": False, "physical_provenance_certified": False}


def _verify_binding(binding):
    for name, digest in binding["metadata_sha256"].items():
        if _sha(_plain_path(name)) != digest:
            raise ValueError("QA 결합 뒤 메타데이터 변경")


def _read_pcm(path, *, expected_pcm_sha256=None, expected_frames=None):
    path = _plain_path(path)
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("준비된 16 kHz mono PCM16만 허용합니다; 정규화/리샘플링하지 않습니다")
        count = handle.getnframes()
        if not 1 <= count <= 1_048_576:
            raise ValueError("각 case는 1~1048576 샘플이어야 합니다; 임의 자르기 금지")
        pcm = handle.readframes(count + 1)
        if len(pcm) != count * 2:
            raise ValueError("WAV PCM 길이 불일치")
        if ((expected_frames is not None and count != expected_frames)
                or (expected_pcm_sha256 is not None and hashlib.sha256(pcm).hexdigest() != expected_pcm_sha256)):
            raise ValueError("QA 이후 valid PCM SHA/길이 변경")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0


def execute_plan(plan_path, out, *, run_validation_search=False):
    destination = _new_output(out)
    plan_path = _metadata_path(plan_path, ".json")
    plan, missing, secondary = inspect_plan(plan_path)
    cases, manifest_path = None, None
    if plan["validation_case_manifest"] is not None:
        manifest_path = _metadata_path(_relative_input(plan["validation_case_manifest"], plan_path.parent), ".json")
        cases = inspect_cases(manifest_path)
        if plan["max_case_runs"] is not None and len(cases) * len(plan["candidates"]) > plan["max_case_runs"]:
            raise ValueError("후보×case 예산 초과")
    binding = None
    if plan["measurement_packet"] is not None:
        packet = _relative_input(plan["measurement_packet"], plan_path.parent)
        binding = inspect_measurement_packet(packet, cases, manifest_path)
        binding["metadata_sha256"][str(plan_path)] = _sha(plan_path)
        if manifest_path is not None:
            binding["metadata_sha256"][str(manifest_path)] = _sha(manifest_path)
        _verify_binding(binding)
    if not run_validation_search:
        report = {
            "schema": "classical_validation_search_preparation.v1",
            "plan": plan, "plan_sha256": _sha(plan_path), "missing_fields": missing,
            "validation_case_count": len(cases) if cases is not None else None,
            "manifest_sha256": _sha(manifest_path) if manifest_path else None,
            "validation_input_binding": binding,
            "static_plan_complete": not missing, "pcm_read": False, "search_executed": False,
            "model_training_executed": False, "physical_claim_allowed": False,
        }
        _new_output(destination)
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "preparation.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        return report
    if missing:
        raise ValueError("미확정 탐색 계약: " + ", ".join(missing))
    array_cases = []
    for case, origin in zip(cases, binding["cases"], strict=True):
        row = {k: v for k, v in case.items() if k not in ("reference_path", "disturbance_path")}
        for source, target in (("reference_path", "reference"), ("disturbance_path", "disturbance")):
            row[target] = _read_pcm(origin[source], expected_pcm_sha256=origin[target + "_pcm_sha256"],
                                    expected_frames=origin["frames"])
        array_cases.append(row)
    _verify_binding(binding)
    arguments = {key: plan[key] for key in (*NUMERIC_KEYS, "sample_rate", "runtime_conditions")}
    arguments["runtime_conditions"] = {**plan["runtime_conditions"], "validation_input_binding": binding}
    return select_validation_baselines(array_cases, plan["candidates"],
        secondary=secondary, secondary_estimate=secondary.copy(), out=destination, **arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-validation-search", action="store_true",
                        help="추후 실행용: 명시된 validation에서만 적응 기준선 탐색")
    args = parser.parse_args(argv)
    try:
        report = execute_plan(args.plan, args.out, run_validation_search=args.run_validation_search)
    except (ValueError, OSError, KeyError, TypeError, wave.Error) as error:
        print(f"[비교 준비 오류] {error}", file=sys.stderr)
        return 1
    if not args.run_validation_search:
        print(f"[계획 검사] 미정 {len(report['missing_fields'])}개; 파형/탐색/모델학습 미실행")
        return 0 if report["static_plan_complete"] else 2
    print(f"[validation 기준선 탐색] 결과: {args.out}; 모델 학습·최종 test 평가 아님")
    return 0 if report["frozen_selection"]["payload"]["selection_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
