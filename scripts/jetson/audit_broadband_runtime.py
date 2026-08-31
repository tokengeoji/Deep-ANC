#!/usr/bin/env python3
"""실제 Jetson runtime raw를 checkpoint/export/strict P/S와 결속해 latency를 판정한다.

스피커나 마이크를 열지 않는 offline consumer다. runtime이 저장한
모든 inference step wall-time, PortAudio callback raw, zero-discontinuity counter,
strict ``PlantDelays.lead()``를 다시 계산하고 no-replace JSON을 발행한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (REPO_ROOT / "src").resolve()
sys.path.insert(0, str(SOURCE_ROOT))

from deep_anc.config import load_runtime_config  # noqa: E402
from deep_anc.dsp.control_band_contract import ControlBandContract  # noqa: E402
from deep_anc.eval.broadband_runtime import (  # noqa: E402
    audit_broadband_runtime_evidence,
    build_broadband_runtime_evidence_from_artifacts,
)
from deep_anc.realtime.plant_contract import validate_runtime_plant_contract  # noqa: E402


def _exact_source_origin() -> None:
    import deep_anc

    origin = Path(deep_anc.__file__).resolve()
    if not origin.is_relative_to(SOURCE_ROOT):
        raise RuntimeError(
            f"deep_anc import가 exact repository source가 아닙니다: {origin}"
        )


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()


def _require_clean_expected_commit(expected: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("--expected-commit은 lowercase 40-hex여야 합니다")
    actual = _git_output("rev-parse", "HEAD")
    if actual != expected:
        raise RuntimeError(f"Jetson HEAD가 expected commit과 다릅니다: {actual}")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("runtime audit는 clean exact commit에서만 발행합니다")
    if _git_output("replace", "-l"):
        raise RuntimeError("git replace ref가 있어 exact source를 증명할 수 없습니다")
    return actual


def _power_mode() -> tuple[str, str]:
    completed = subprocess.run(
        ["nvpmodel", "-q"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    raw = completed.stdout.strip()
    match = re.search(r"^NV Power Mode:\s*(\S+)\s*$", raw, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"nvpmodel read-only query를 해석할 수 없습니다: {raw!r}")
    return match.group(1), raw


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--clock-receipt", required=True)
    parser.add_argument("--physical-witness-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    _exact_source_origin()
    commit = _require_clean_expected_commit(args.expected_commit)
    cfg = load_runtime_config(args.config, [])
    power_mode, power_mode_raw = _power_mode()
    contract = ControlBandContract.broadband_point_control()
    evidence = build_broadband_runtime_evidence_from_artifacts(
        contract=contract,
        runtime_cfg=cfg,
        session_npz_path=args.session,
        clock_receipt_path=args.clock_receipt,
        physical_witness_receipt_path=args.physical_witness_receipt,
        power_mode=power_mode,
        repo_root=REPO_ROOT,
    )
    plant = validate_runtime_plant_contract(cfg)
    if plant is None:
        raise RuntimeError("strict runtime plant contract가 누락됐습니다")
    audit = audit_broadband_runtime_evidence(
        contract,
        evidence,
        expected_plant_lead_samples=int(
            plant.timing.digital_reference_lead_samples
        ),
    )
    payload = {
        "schema_version": "broadband_runtime_evidence_bundle_v1",
        "status": audit.status,
        "git_commit": commit,
        "runtime_config": str(Path(args.config)),
        "session_npz": str(Path(args.session)),
        "clock_receipt": str(Path(args.clock_receipt)),
        "physical_witness_receipt": str(
            Path(args.physical_witness_receipt)
        ),
        "nvpmodel_query": power_mode_raw,
        "evidence": evidence.model_dump(mode="json"),
        "audit": audit.model_dump(mode="json"),
    }
    _write_exclusive(Path(args.output), payload)
    label = "PASS" if audit.ok else "BLOCKED"
    print(
        f"[{label}] broadband runtime: P99={evidence.inference_p99_ms:.3f} ms, "
        f"max={evidence.inference_max_ms:.3f} ms, "
        f"lead={evidence.runtime_lead_samples} samples"
    )
    if not audit.ok:
        for reason in audit.reasons:
            print(f"- {reason}", file=sys.stderr)
    print(f"저장: {Path(args.output)}")
    return 0 if audit.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
