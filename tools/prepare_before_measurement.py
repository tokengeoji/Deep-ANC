#!/usr/bin/env python3
"""무학습 실측 전 패킷: 수집 양식·명시 설정 템플릿·후속 명령만 준비한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from deepanc.calibration import DEFAULT_RIR, load_secondary_path
from deepanc.measurement_ready import create_packet
from scripts.eval.tune_classical_baselines import plan_template
from deep_anc.train.sfanc_paired import recipe_template


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_before_measurement(out, *, source_inventory=None):
    destination = Path(out).absolute()
    if destination.exists() or destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError("새 출력 폴더만 허용하며 symlink 부모는 사용할 수 없습니다")
    load_secondary_path(sample_rate=16000)
    recipe_source = ROOT / "configs/sfanc_paired_recipe.template.json"
    recipe = recipe_template()
    if recipe["secondary_source_sha256"] != _sha(DEFAULT_RIR):
        raise ValueError("SFANC 템플릿의 정본 S SHA 불일치")
    inventory = {"provided": False, "content_certified": False}
    if source_inventory is not None:
        source_inventory = Path(source_inventory).resolve()
        if not source_inventory.is_file() or source_inventory.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("기존 자료 inventory JSON 파일(16 MiB 이하)이 필요합니다")
        metadata = json.loads(source_inventory.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("자료 inventory는 JSON 객체여야 합니다")
        inventory.update(provided=True, path=str(source_inventory), sha256=_sha(source_inventory),
                         note="회수/출처 참고 기록만 연결; PCM/라이선스/실측 동기성 자동 인증 아님")
    destination.mkdir(parents=True, exist_ok=False)
    create_packet(destination / "capture")
    _write(destination / "sfanc_recipe.template.json", recipe)
    _write(destination / "baseline_search.template.json", plan_template())
    _write(destination / "validation_cases.template.json", {
        "schema": "classical_validation_cases.v1",
        "cases": [{"split": "validation", "session_id": None, "source_id": None, "source_kind": None,
                   "reference_path": None, "disturbance_path": None,
                   "regions": {"onset": None, "transition": None, "steady": None}}],
    })
    raw = destination / "capture/raw"
    accepted = destination / "accepted_recordings"
    paired = destination / "sfanc_prepared"
    commands = {
        "executed": False,
        "after_recording_intake_only": [
            ".venv/bin/python", "tools/prepare_measurement_session.py",
            "--input", str(raw), "--out", str(destination / "intake_audit"),
        ],
        "after_passed_intake_prepare_train_valid": [
            ".venv/bin/python", "tools/prepare_measurement_session.py",
            "--input", str(raw), "--out", str(accepted), "--prepare",
        ],
        "after_explicit_recipe_prepare_sfanc_without_training": [
            ".venv/bin/python", "scripts/train/prepare_sfanc_paired.py",
            "--packet", str(accepted), "--recipe", str(destination / "sfanc_recipe.template.json"),
            "--out", str(paired),
        ],
        "after_explicit_plan_check_baseline_without_search": [
            ".venv/bin/python", "scripts/eval/tune_classical_baselines.py",
            "--plan", str(destination / "baseline_search.template.json"),
            "--out", str(destination / "baseline_plan_check"),
        ],
        "actual_training_or_validation_search_authorized": False,
        "instructions": "각 명령은 Docker 안에서 수동 실행. null/unknown은 실제 근거로 채우며 일괄 실행 금지.",
    }
    _write(destination / "commands.json", commands)
    inventory_path = destination / "source_inventory_reference.json"
    _write(inventory_path, inventory)
    report = {
        "schema": "pre_measurement_packet.v1", "status": "templates_and_tools_prepared_not_capture",
        "packet_created": True, "sample_rate": 16000, "secondary_source_sha256": _sha(DEFAULT_RIR),
        "secondary_taps": 500, "sfanc_template_source_sha256": _sha(recipe_source),
        "training_executed": False, "bank_fitting_executed": False,
        "validation_search_executed": False, "audio_devices_opened": False, "firmware_changed": False,
        "raw_audio_downloaded": False, "capture_ready": False, "training_ready": False,
        "physical_performance_claim_allowed": False,
        "deferred": [
            "보드 식별/메모리 배치는 사용자가 나중에 제공: 녹음 전용 프로젝트 구현·로드는 보류",
            "손실 없는 raw ADC 수집 방식과 실제 clock/gain/채널 확인",
            "독립 train/valid/test REF/ANC-OFF ERR 수집 및 intake 검사",
            "현재 출력 경로의 고역 S·추가 지연·REF 선행·피드백 실측",
            "recipe/grid/초기·전이·정상상태 구간 확정 후 새 학습/탐색 실행 허가",
        ],
        "public_source_policy": {
            "canonical_storage": "Google Drive",
            "reuse_existing_receipts_first": True, "source_inventory": inventory,
            "public_audio_is_not_measured_ref_err": True,
            "public_pretraining_optional_not_a_substitute_for_capture": True,
            "restore_full_archive_when_needed_only_with_space_and_checksum_verification": True,
            "mimii": "train_only_auxiliary; valid/test 제외",
        },
        "instructions": "docs/20_measurement_runbook.md",
    }
    report["generated_files"] = {
        path.relative_to(destination).as_posix(): _sha(path)
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    _write(destination / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-inventory", help="선택적 기존 Drive/공개 원천 감사 JSON")
    args = parser.parse_args(argv)
    try:
        report = prepare_before_measurement(args.out, source_inventory=args.source_inventory)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[실측 전 준비 오류] {error}", file=sys.stderr)
        return 1
    print(f"[실측 전 패킷 생성] {args.out}; {len(report['generated_files'])}파일; 학습·측정·튜닝 미실행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
