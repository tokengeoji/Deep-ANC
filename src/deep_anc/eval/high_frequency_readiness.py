"""고역 비교의 정적 준비 감사. 모델·optimizer·학습·필터 fitting을 호출하지 않는다."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from deepanc.calibration import DEFAULT_RIR, load_secondary_path


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "high_frequency_comparison_preparation.v1"
MAX_METADATA_BYTES = 16 * 1024 * 1024
CANDIDATES = ("sfanc", "hybrid_ancnet", "causal_controller")
DECISIONS = ("primary_path_kind", "additional_delay_samples", "control_limit",
             "minimum_high_band_advantage_db", "maximum_low_band_degradation_db", "candidate_selection")
KEYS = {"schema", "stage", "reference_mode", "sample_rate", "secondary_path", "polarity",
        "sfanc_manifest", "observed_run_directories", *DECISIONS}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    path = Path(path)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("정적 JSON 자료는 16 MiB 이하여야 합니다")
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 중복 key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"비유한 JSON: {value}")))
    if not isinstance(value, dict):
        raise ValueError("정적 계약/manifest/run JSON은 객체여야 합니다")
    return value


def _path(path, root):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def validate_contract(config):
    """null은 미확정으로 보존한다. 누락/오타/암묵적 기본값은 허용하지 않는다."""
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError("준비 계약의 필수 key 누락/알 수 없는 key; 빈 계약은 허용하지 않습니다")
    expected = {"schema": SCHEMA, "stage": "prepare_only", "reference_mode": "acoustic",
                "sample_rate": 16000, "secondary_path": "rir.txt", "polarity": "e=d+S*u"}
    for key, value in expected.items():
        if config[key] != value or type(config[key]) is not type(value):
            raise ValueError(f"준비 계약 {key} 불일치: {value}만 허용")
    if config["primary_path_kind"] not in (None, "measured_synchronized_capture", "explicit_synthetic_scenario"):
        raise ValueError("primary_path_kind는 미확정/실측 동기 capture/명시적 합성 시나리오만 허용")
    delay = config["additional_delay_samples"]
    if delay is not None and (type(delay) is not int or delay < 0):
        raise ValueError("additional_delay_samples는 null 또는 비음수 정수")
    for name in ("control_limit", "minimum_high_band_advantage_db", "maximum_low_band_degradation_db"):
        value = config[name]
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}: null 또는 유한 비음수 수치 필요")
            if name == "control_limit" and not 0 < value <= 1:
                raise ValueError("control_limit은 공통 디지털 단위의 0 초과 1 이하")
    selected = config["candidate_selection"]
    if selected is not None and (not isinstance(selected, list) or not selected
            or any(value not in CANDIDATES for value in selected) or len(set(selected)) != len(selected)):
        raise ValueError("candidate_selection은 null 또는 중복 없는 알려진 후보 목록")
    if config["sfanc_manifest"] is not None and (not isinstance(config["sfanc_manifest"], str)
                                                or not config["sfanc_manifest"].strip()):
        raise ValueError("sfanc_manifest는 null 또는 비어 있지 않은 경로")
    observed = config["observed_run_directories"]
    if not isinstance(observed, list) or any(not isinstance(p, str) or not p.strip() for p in observed):
        raise ValueError("observed_run_directories는 경로 문자열 목록")
    if len(set(observed)) != len(observed):
        raise ValueError("관찰된 run 경로 중복")
    return config


def _file_check(path):
    if path is None:
        return {"provided": False, "file_exists": False, "contents_verified": False}
    path = Path(path)
    exists = path.is_file()
    if exists and path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("파일 검사는 16 MiB 이하 메타데이터 전용; 원본 archive/PCM을 지정하지 마십시오")
    return {"provided": True, "path": str(path), "file_exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": _sha(path) if exists else None, "contents_verified": False,
            "scope": "파일 존재/크기/SHA만 검사; capture 동기성·라이선스·Drive 업로드 완전성 인증 아님"}


def inspect_manifest(path, observed_directories):
    """이미 관찰한 test의 source/group을 새 분할과 대조. 음원 decode/모델 실행 없음."""
    histories, observed = [], {key: set() for key in ("source_id", "group_id", "sha256", "speaker", "book")}
    history_complete = bool(observed_directories)
    for directory in observed_directories:
        report_path = Path(directory) / "report.json"
        record = {"report": str(report_path), "exists": report_path.is_file()}
        if report_path.is_file():
            previous = _json(report_path)
            datasets = previous.get("datasets")
            rows = datasets.get("test") if isinstance(datasets, dict) else None
            if not isinstance(rows, list) or not rows:
                history_complete = False
                record["test_history_readable"] = False
            else:
                record.update(test_history_readable=True, report_sha256=_sha(report_path), test_rows=len(rows))
                for row in rows:
                    if not isinstance(row, dict):
                        raise ValueError("관찰된 run의 test 행 형식 오류")
                    if not any(isinstance(row.get(key), str) and row[key] for key in observed):
                        history_complete = False
                        record["test_history_readable"] = False
                    for key in observed:
                        if row.get(key) is not None:
                            observed[key].add(str(row[key]))
        else:
            history_complete = False
        histories.append(record)
    result = {"manifest": _file_check(path), "observed_runs": histories,
              "declared_history_readable": history_complete,
              "exposure_history_exhaustiveness_verified": False,
              "audio_decoded": False, "source_bytes_verified": False,
              "independent_new_test_verified": False, "training_data_ready": False}
    if path is None or not Path(path).is_file():
        result["status"] = "manifest_missing"
        return result
    manifest = _json(path)
    if (manifest.get("schema") != "sfanc_librispeech_sources.v1" or manifest.get("sample_rate") != 16000
            or manifest.get("simulation_pretraining_only") is not True or manifest.get("measured_ref_err") is not False):
        raise ValueError("기존 SFANC 16 kHz 시뮬레이션 원천 manifest 계약이 필요합니다")
    if not isinstance(manifest.get("root"), str) or not manifest["root"]:
        raise ValueError("manifest 원천 root 필요")
    root = _path(manifest["root"], Path(path).resolve().parent)
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("manifest train/validation/test 분할 필요")
    seen, overlaps, missing, reused, counts = {}, [], [], [], {}
    for split, rows in splits.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"비어 있는 manifest 분할: {split}")
        counts[split] = len(rows)
        for row in rows:
            if not isinstance(row, dict) or row.get("split") != split:
                raise ValueError("manifest 행의 split 불일치")
            for field in ("source_id", "group_id", "speaker", "book", "sha256", "path"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"manifest 행의 {field} 문자열 필요")
            if re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None:
                raise ValueError("manifest 원천 SHA-256 형식 오류")
            source_path = Path(row["path"])
            resolved = (root / source_path).resolve()
            if source_path.is_absolute() or not resolved.is_relative_to(root):
                raise ValueError("manifest 원천 경로는 corpus 내부 상대경로여야 합니다")
            if not resolved.is_file():
                missing.append(str(resolved))
            for field in ("source_id", "group_id", "speaker", "book", "sha256"):
                key = (field, row[field])
                if key in seen and seen[key] != split:
                    overlaps.append({"field": field, "value": row[field], "splits": [seen[key], split]})
                seen[key] = split
            fields = [key for key in observed if row[key] in observed[key]]
            if fields:
                reused.append({"split": split, "source_id": row["source_id"], "matching_fields": fields})
    result.update(status="static_manifest_inspected", source_root=str(root), split_file_counts=counts,
                  missing_source_count=len(missing), missing_source_examples=missing[:20],
                  cross_split_overlap_count=len(overlaps), cross_split_overlaps=overlaps,
                  prior_observed_test_overlap_count=len(reused), prior_observed_test_overlaps=reused,
                  prior_observed_test_in_current_test=sum(row["split"] == "test" for row in reused),
                  prior_observed_test_in_training_or_validation=sum(row["split"] != "test" for row in reused),
                  measured_ref_err=False, simulation_only=True,
                  prior_test_reuse_policy="이미 관찰된 test/group은 새 독립 최종 test로 재사용할 수 없음")
    return result


def prepare_high_frequency(config_path, out, *, sfanc_manifest=None, capture=None, inventory=None,
                           repository_root=ROOT):
    """정적 목록만 새 폴더에 저장. 학습 승인용 preflight가 아니며 항상 ready=False다.

    미확정을 임의로 채우지 않고 학습 비교 준비와 실측 성능의 미충족 조건을 분리한다.
    """
    destination = Path(out)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"기존 출력은 덮어쓰지 않습니다: {destination}")
    if any(parent.is_symlink() for parent in destination.absolute().parents):
        raise ValueError("출력 부모의 심볼릭 링크는 허용하지 않습니다")
    for name, value in (("sfanc_manifest", sfanc_manifest), ("capture", capture), ("inventory", inventory)):
        if value is not None and not str(value).strip():
            raise ValueError(f"{name}: 빈 입력 경로는 허용하지 않습니다")
    root = Path(repository_root).resolve()
    config = validate_contract(_json(config_path))
    coefficients = load_secondary_path(sample_rate=config["sample_rate"])
    if len(coefficients) != 500:
        raise ValueError("권위 있는 OMAP S 전체 500탭이 필요합니다")
    manifest_arg = sfanc_manifest if sfanc_manifest is not None else config["sfanc_manifest"]
    data = inspect_manifest(_path(manifest_arg, root) if manifest_arg is not None else None,
                            [_path(path, root) for path in config["observed_run_directories"]])
    blockers = [{"code": "decision_pending", "field": key} for key in DECISIONS if config[key] is None]
    if data["status"] == "manifest_missing":
        blockers.append({"code": "source_manifest_missing"})
    for key, code in (("missing_source_count", "source_files_missing"),
                      ("cross_split_overlap_count", "source_split_leakage"),
                      ("prior_observed_test_in_current_test", "prior_test_cannot_be_new_test"),
                      ("prior_observed_test_in_training_or_validation", "prior_test_group_moved_into_training_or_validation")):
        if data.get(key, 0):
            blockers.append({"code": code, "count": data[key]})
    if not data["declared_history_readable"]:
        blockers.append({"code": "prior_test_history_incomplete"})
    blockers.extend({"code": code} for code in (
        "source_pcm_license_and_coverage_qa_pending", "new_independent_test_not_verified",
        "comparison_information_and_validation_policy_pending", "baseline_search_policy_and_execution_pending"))
    paths = {
        "sfanc": ["scripts/train/train_sfanc.py", "src/deep_anc/train/sfanc_experiment.py",
                  "scripts/train/prepare_sfanc_paired.py", "src/deep_anc/train/sfanc_paired.py"],
        "hybrid_ancnet": ["src/deep_anc/models/hybrid_anc.py", "src/deep_anc/train/trainer.py"],
        "causal_controller": ["deepanc/model.py", "deepanc/train.py"],
    }
    candidates = {name: {"source_files": {p: (root / p).is_file() for p in files},
                        "training_entrypoint_ready_for_this_comparison": False,
                        "architecture_selected": False, "optimizer_settings_selected": False}
                  for name, files in paths.items()}
    paired_bridge = all((root / path).is_file() for path in (
        "scripts/train/prepare_sfanc_paired.py", "src/deep_anc/train/sfanc_paired.py"))
    candidates["sfanc"].update(adapter_required=False, comparison_contract_adapter_required=not paired_bridge,
        paired_training_bridge_exists=paired_bridge,
        role="current_research_candidate_not_final_selection", existing_training_entrypoint_available=True,
        native_sample_rate=16000,
        existing_entrypoint="직접 실행 시 FIR fitting과 CNN 학습을 시작하므로 준비 CLI에서 호출 금지",
        pending="실측 QA packet·명시 recipe·공통 비교/지연 정책·승인 후 실제 학습 검증 필요; 기존 P=8 ms·가중치3 자동 승계 금지")
    candidates["hybrid_ancnet"].update(adapter_required=True, legacy_training_sample_rate=48000,
        role="optional_unimplemented_omap_comparison_candidate", required_for_sfanc_training=False,
        omap_16khz_training_harness_exists=False,
        pending="모델 구조만 재사용 후보; 원본 S용 별도 하네스·REF/ERR 접근·상태/burn-in·동조건 validation 필요")
    candidates["causal_controller"].update(adapter_required=True, native_sample_rate=16000,
        role="separate_optional_small_baseline", required_for_sfanc_training=False,
        is_hybrid_ancnet=False,
        pending="별도 작은 REF-only 기준선; 기존 raw train/valid loader는 새 test·고역 비교 계약을 대신하지 않음")
    capture_check = _file_check(_path(capture, root) if capture else None)
    inventory_check = _file_check(_path(inventory, root) if inventory else None)
    for name, check in (("capture", capture_check), ("inventory", inventory_check)):
        if check["provided"] and not check["file_exists"]:
            blockers.append({"code": f"provided_{name}_file_missing"})
    for blocker in blockers:
        blocker["scope"] = "training_comparison_preparation"
    report = {
        "schema": "high_frequency_readiness_report.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_before_training", "static_preparation_completed": True,
        "training_ready": False, "training_executed": False, "optimizer_created": False,
        "model_instantiated": False, "forward_backward_executed": False, "bank_fitting_executed": False,
        "audio_devices_opened": False, "deployment_allowed": False, "physical_performance_claim_allowed": False,
        "measured_ref_err_verified": False,
        "config": config, "config_sha256": _sha(config_path), "provisional_contract": True,
        "secondary": {"sample_rate": 16000, "taps": len(coefficients), "source": str(DEFAULT_RIR),
            "source_sha256": _sha(DEFAULT_RIR), "all_taps_gain_sign_delay_preserved": True,
            "target_band_measurement_validated_by_this_tool": False},
        "protocol": {"objective": "1 kHz 이상에서 DL의 FxLMS/FxNLMS 대비 우위를 독립 검증; 현재 미판정",
            "polarity": "e=d+S*u", "reference_mode": "acoustic", "sample_rate": 16000,
            "bands_hz": {"low": [0, 1000], "priority_high": [1000, 1600],
                         "remaining_high": [1600, 8000], "all_high": [1000, 8000], "fullband": [0, 8000]},
            "band_edges": "좌측 포함/우측 제외; Nyquist=8000인 우측 끝만 포함",
            "metrics_required": ["paired_attenuation_advantage_db", "per_source_family", "worst_10_percent",
                                 "amplification", "clipping", "startup_transition_steady", "end_to_end_latency"],
            "matching_required": ["same_REF_and_disturbance", "same_S_and_extra_delay", "same_output_units_and_limit",
                                  "explicit_ERR_and_path_information", "validation_only_model_and_baseline_selection"],
            "test_tuning_allowed": False, "unknowns_are_not_zero": True,
            "required_source_families": ["noise", "speech", "music"],
            "mimii_usage": "train_only_auxiliary; validation/test 금지",
            "mimii_inventory_policy_verified": False},
        "data": data, "capture_file_check": capture_check,
        "inventory_file_check": inventory_check,
        "candidates": candidates,
        "classical_baselines": {"plain_and_normalized_module_exists": (root / "src/deep_anc/eval/classical_anc.py").is_file(),
            "validation_search_runner_exists": all((root / path).is_file() for path in (
                "src/deep_anc/eval/baseline_selection.py", "scripts/eval/tune_classical_baselines.py")),
            "band_and_session_metrics_exists": (root / "src/deep_anc/eval/high_frequency_metrics.py").is_file(),
            "existing_cold_fxnlms_exists": (root / "src/deep_anc/eval/sfanc_fxnlms.py").is_file(),
            "module_execution_performed": False, "strong_tuning_ready": False,
            "scope": "파일 존재만 확인; 구현 정확성·수렴·최적 baseline 검증이 아님"},
        "blockers": blockers,
        "physical_claim_blockers": [
            {"code": "measured_ref_err_not_verified", "scope": "physical_performance_claim"},
            {"code": "target_band_secondary_and_end_to_end_latency_unverified", "scope": "physical_performance_claim"},
            {"code": "same_condition_independent_measured_comparison_missing", "scope": "physical_performance_claim"}],
        "scope": "정적 파일/계약 감사; 학습·모델 연산·오디오·다운로드·시스템 변경 없음",
    }
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report
