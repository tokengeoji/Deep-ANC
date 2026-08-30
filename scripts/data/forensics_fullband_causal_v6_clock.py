#!/usr/bin/env python3
"""보존된 v6 raw의 비-affine clock 현상을 diagnostic-only로 재검산한다."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from deep_anc.data.repository_fd import (  # noqa: E402
    RepositoryFileGuard,
    canonical_relative_path,
    publish_repository_bytes_noreplace,
    repository_execution_identity,
)
from deep_anc.dsp.fullband_causal_v6_forensics import (  # noqa: E402
    diagnose_short_time_clock_v6,
    replay_affine_clock_admission_v6,
    validate_failure_binding_v6,
)
from deep_anc.dsp.fullband_live_authority_v6 import (  # noqa: E402
    SEALED_HARDWARE_RELATIVE_PATH,
    SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH,
    SEALED_PLAN_ENVELOPE_RELATIVE_PATH,
    SEALED_RAW_RELATIVE_PATH,
)
from deep_anc.dsp.fullband_live_post_v6 import (  # noqa: E402
    external_post_receipt_relative_path,
    _load_external_post_capture_receipt_v6_archival_forensics,
)


SCRIPT_RELATIVE_PATH = "scripts/data/forensics_fullband_causal_v6_clock.py"
FORENSICS_DIRECTORY = "results/fullband_causal_v6/forensics"
DEPENDENCY_RELATIVE_PATHS = (
    SCRIPT_RELATIVE_PATH,
    "requirements.txt",
    "requirements-jetson.txt",
    "src/deep_anc/data/repository_fd.py",
    "src/deep_anc/dsp/fullband_causal_v6.py",
    "src/deep_anc/dsp/fullband_causal_v6_forensics.py",
    "src/deep_anc/dsp/fullband_live_authority_v6.py",
    "src/deep_anc/dsp/fullband_live_post_v6.py",
    "src/deep_anc/dsp/fullband_live_raw_v5.py",
    "src/deep_anc/dsp/fullband_live_raw_v6.py",
)


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
        return (json.dumps(value, **options) + "\n").encode("utf-8")
    options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label}가 lowercase SHA-256이 아닙니다")
    return value


def _json_from_guard(guard: RepositoryFileGuard, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(guard.bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON을 읽을 수 없습니다") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root가 object가 아닙니다")
    return value


def _binding_path(bindings: Mapping[str, Any], key: str, *, label: str) -> str:
    value = bindings.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"post receipt {label} binding이 mapping이 아닙니다")
    return canonical_relative_path(
        value.get("path"), label=f"post receipt {label} path"
    )


def _forensics_relative_path(capture_id: Any) -> str:
    if (
        type(capture_id) is not str
        or len(capture_id) != 32
        or any(character not in "0123456789abcdef" for character in capture_id)
    ):
        raise ValueError("capture_id는 32-char lowercase hex여야 합니다")
    return f"{FORENSICS_DIRECTORY}/clock_{capture_id}.json"


def _validate_failure_canonical_bytes(
    guard: RepositoryFileGuard, value: Mapping[str, Any]
) -> None:
    if guard.bytes != _canonical_json_bytes(value) + b"\n":
        raise ValueError("v6 failure JSON이 canonical publisher bytes가 아닙니다")


def _dependency_receipts(
    stack: ExitStack,
) -> tuple[list[RepositoryFileGuard], list[dict[str, Any]]]:
    guards: list[RepositoryFileGuard] = []
    receipts: list[dict[str, Any]] = []
    for relative in DEPENDENCY_RELATIVE_PATHS:
        guard = stack.enter_context(
            RepositoryFileGuard(ROOT, relative, label="forensics code dependency")
        )
        guards.append(guard)
        receipts.append(
            {
                "path": relative,
                "file_sha256": guard.sha256,
                "size_bytes": guard.size,
            }
        )
    return guards, receipts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=SEALED_RAW_RELATIVE_PATH)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument(
        "--post-receipt",
        default=external_post_receipt_relative_path(SEALED_RAW_RELATIVE_PATH),
    )
    parser.add_argument("--expected-post-receipt-sha256", required=True)
    parser.add_argument("--failure", required=True)
    parser.add_argument("--expected-failure-sha256", required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="동일 검증·수학을 실행하되 결과 파일을 절대 발행하지 않습니다",
    )
    args = parser.parse_args(argv)

    try:
        raw_relative = canonical_relative_path(args.raw, label="raw path")
        if raw_relative != SEALED_RAW_RELATIVE_PATH:
            raise ValueError("forensics raw가 exact sealed v6 path가 아닙니다")
        receipt_relative = canonical_relative_path(
            args.post_receipt, label="post receipt path"
        )
        if receipt_relative != external_post_receipt_relative_path(raw_relative):
            raise ValueError("post receipt가 sealed raw의 canonical sibling이 아닙니다")
        failure_relative = canonical_relative_path(args.failure, label="failure path")
        if (
            not failure_relative.startswith("results/fullband_causal_v6/failure_")
            or not failure_relative.endswith(".json")
        ):
            raise ValueError("failure가 v6 전용 failure namespace가 아닙니다")
        expected_raw_sha = _require_sha256(
            args.expected_raw_sha256, label="expected raw SHA"
        )
        expected_receipt_sha = _require_sha256(
            args.expected_post_receipt_sha256,
            label="expected post receipt SHA",
        )
        expected_failure_sha = _require_sha256(
            args.expected_failure_sha256, label="expected failure SHA"
        )

        # 현재 분석 코드 전체가 clean committed checkout일 때만 artifact를 만든다.
        execution = repository_execution_identity(ROOT, SCRIPT_RELATIVE_PATH)
        with ExitStack() as stack:
            raw_guard = stack.enter_context(
                RepositoryFileGuard(ROOT, raw_relative, label="v6 forensic raw")
            )
            receipt_guard = stack.enter_context(
                RepositoryFileGuard(ROOT, receipt_relative, label="v6 post receipt")
            )
            failure_guard = stack.enter_context(
                RepositoryFileGuard(ROOT, failure_relative, label="v6 failure")
            )
            if raw_guard.sha256 != expected_raw_sha:
                raise ValueError("raw bytes SHA가 expected와 다릅니다")
            if receipt_guard.sha256 != expected_receipt_sha:
                raise ValueError("post receipt bytes SHA가 expected와 다릅니다")
            if failure_guard.sha256 != expected_failure_sha:
                raise ValueError("failure bytes SHA가 expected와 다릅니다")

            receipt_hint = _json_from_guard(receipt_guard, label="post receipt")
            bindings = receipt_hint.get("external_bindings")
            if not isinstance(bindings, Mapping):
                raise ValueError("post receipt external_bindings가 mapping이 아닙니다")
            plan_relative = _binding_path(bindings, "signal_plan", label="signal plan")
            live_authority_relative = _binding_path(
                bindings, "live_capture_authority", label="live authority"
            )
            meter_relative = _binding_path(bindings, "meter", label="meter")
            level_evidence_relative = _binding_path(
                bindings, "level_evidence", label="level evidence"
            )
            hardware_relative = _binding_path(bindings, "hardware", label="hardware")
            if (
                plan_relative != SEALED_PLAN_ENVELOPE_RELATIVE_PATH
                or live_authority_relative
                != SEALED_LIVE_CAPTURE_AUTHORITY_RELATIVE_PATH
                or hardware_relative != SEALED_HARDWARE_RELATIVE_PATH
            ):
                raise ValueError("post receipt의 sealed plan/authority/hardware path가 다릅니다")

            admitted = _load_external_post_capture_receipt_v6_archival_forensics(
                repository_root=ROOT,
                receipt_relative_path=receipt_relative,
                expected_receipt_file_sha256=expected_receipt_sha,
                plan_envelope_path=plan_relative,
                live_authority_path=live_authority_relative,
                meter_raw_path=meter_relative,
                level_evidence_path=level_evidence_relative,
                hardware_path=hardware_relative,
            )
            if (
                admitted.get("schema")
                != "fullband_causal_v6_archival_forensics_source_v1"
                or admitted.get("source_receipt_evidence", {}).get("file_sha256")
                != expected_receipt_sha
                or admitted.get("analysis_admission_eligible") is not False
                or admitted.get("canonical_training_eligible") is not False
                or admitted.get("scope")
                != "archival_forensics_only_no_analysis_no_plant_no_training_authority"
            ):
                raise ValueError("archival loader가 금지된 analysis/training authority를 반환했습니다")
            raw = admitted["forensic_raw_snapshot"]
            if raw["raw_file_sha256"] != expected_raw_sha:
                raise ValueError("official raw loader SHA가 expected와 다릅니다")
            arrays = raw["arrays"]
            metadata = raw["metadata"]
            capture_id = metadata["session"]["capture_id"]
            if failure_relative != f"results/fullband_causal_v6/failure_{capture_id}.json":
                raise ValueError("failure filename이 admitted raw capture-id와 다릅니다")

            failure_value = _json_from_guard(failure_guard, label="v6 failure")
            _validate_failure_canonical_bytes(failure_guard, failure_value)
            validated_failure = validate_failure_binding_v6(
                failure_value,
                raw_relative_path=raw_relative,
                raw_file_sha256=expected_raw_sha,
                receipt_relative_path=receipt_relative,
                receipt_file_sha256=expected_receipt_sha,
            )

            plan_guard = stack.enter_context(
                RepositoryFileGuard(ROOT, plan_relative, label="v6 signal plan envelope")
            )
            if plan_guard.sha256 != bindings["signal_plan"]["file_sha256"]:
                raise ValueError("signal plan guard SHA가 post receipt와 다릅니다")
            plan_envelope = _json_from_guard(plan_guard, label="signal plan envelope")
            plan = plan_envelope.get("signal_plan")
            if not isinstance(plan, Mapping):
                raise ValueError("signal plan envelope에 signal_plan mapping이 없습니다")

            dependency_guards, dependency_receipts = _dependency_receipts(stack)
            script_receipt = next(
                item
                for item in dependency_receipts
                if item["path"] == SCRIPT_RELATIVE_PATH
            )
            if script_receipt["file_sha256"] != execution["script_file_sha256"]:
                raise ValueError("execution identity와 held forensics script SHA가 다릅니다")
            short_time = diagnose_short_time_clock_v6(
                plan=plan,
                submitted_pcm=arrays["actual_submitted_pcm"],
                captured_pcm=arrays["captured_pcm"],
            )
            affine_replay = replay_affine_clock_admission_v6(
                plan=plan,
                submitted_pcm=arrays["actual_submitted_pcm"],
                captured_pcm=arrays["captured_pcm"],
            )
            if (
                affine_replay["passed"] is not False
                or affine_replay["failure_stage"] != "global_grid_basin_search"
                or affine_replay["available_receipt"].get("global_search", {}).get(
                    "unique_basin_passed"
                )
                is not False
            ):
                raise ValueError("현재 코드가 보존 raw의 global ambiguity를 재현하지 못했습니다")

            core = {
                "schema": "fullband_causal_v6_clock_forensics_artifact_v2",
                "authority": "diagnostic_only_no_clock_no_plant_no_training_authority",
                "analysis_admission_eligible": False,
                "clock_estimate_authority": False,
                "canonical_training_eligible": False,
                "deployment_eligible": False,
                "attenuation_assessed": False,
                "plant_identification_assessed": False,
                "raw": {
                    "path": raw_relative,
                    "file_sha256": expected_raw_sha,
                    "capture_id": capture_id,
                    "capture_repository_commit": metadata["session"][
                        "repository_commit"
                    ],
                    "capture_repository_branch": metadata["session"][
                        "repository_branch"
                    ],
                },
                "external_post_receipt": {
                    "path": receipt_relative,
                    "file_sha256": expected_receipt_sha,
                },
                "failure": {
                    "path": failure_relative,
                    "file_sha256": expected_failure_sha,
                    "failure_payload_sha256": validated_failure[
                        "failure_payload_sha256"
                    ],
                    "failure_stage": validated_failure["failure_stage"],
                },
                "analysis_execution": {
                    **execution,
                    "python_version": platform.python_version(),
                    "python_implementation": platform.python_implementation(),
                    "numpy_version": np.__version__,
                    "scipy_version": scipy.__version__,
                    "platform": platform.platform(),
                    "dependencies": dependency_receipts,
                },
                "short_time_diagnostic": short_time,
                "affine_admission_replay": affine_replay,
            }
            artifact = {**core, "artifact_payload_sha256": _payload_sha256(core)}
            payload = _canonical_json_bytes(artifact, pretty=True)
            output_relative = _forensics_relative_path(capture_id)

            # 분석 도중 input/code pathname이나 inode가 바뀌지 않았음을 publication 직전에
            # 다시 확인한다. 실제 write는 전용 namespace에서 staging+fsync+hard-link로 한다.
            for guard in (
                raw_guard,
                receipt_guard,
                failure_guard,
                plan_guard,
                *dependency_guards,
            ):
                guard.verify()
            execution_after = repository_execution_identity(ROOT, SCRIPT_RELATIVE_PATH)
            if execution_after != execution:
                raise ValueError("forensics 실행 중 repository identity가 변경됐습니다")
            if args.verify_only:
                print(
                    f"[VERIFY_ONLY] {output_relative} | expected SHA256 "
                    f"{hashlib.sha256(payload).hexdigest()} | 파일 발행 없음 | "
                    "P/S·ANC·학습 권한 없음"
                )
                return 0
            published = publish_repository_bytes_noreplace(
                ROOT,
                output_relative,
                payload,
                mode=0o644,
                recovery_tag="v6_forensics",
            )
            for guard in (
                raw_guard,
                receipt_guard,
                failure_guard,
                plan_guard,
                *dependency_guards,
            ):
                guard.verify()
    except (
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"[실패] v6 clock forensics 거부: {error}", file=sys.stderr)
        return 2
    print(
        f"[DIAGNOSTIC_ONLY] {published['path']} | SHA256 {published['sha256']} | "
        "P/S·ANC·학습 권한 없음"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
