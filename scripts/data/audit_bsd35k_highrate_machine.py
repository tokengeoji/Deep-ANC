#!/usr/bin/env python3
"""BSD35k ``fx-m`` native high-rate machine source evidence를 발행/재검증한다.

이 명령은 스피커·마이크·네트워크를 사용하지 않는다. 이미 내려받은 official
``audio.zip``과 선택 raw WAV, complete decoder audit을 read-only로 대조한 뒤 native
PSD evidence JSON을 O_EXCL로 발행한다. 발행 후 archive를 보관할지 삭제할지는 별도
retention 정책이며, 재검증은 selected raw bytes와 evidence receipt를 사용한다.

예시 (Elice, archive와 selected raw가 모두 있을 때):

    .venv/bin/python scripts/data/audit_bsd35k_highrate_machine.py issue \
      --selection-plan data/provenance/bsd35k_machine_selection_v1.json \
      --metadata-csv data/raw/bsd35k/BSD35k-CS_metadata.csv \
      --selected-raw-root data/raw/bsd35k_fx_m \
      --audio-archive data/raw/bsd35k/BSD35k-CS_audio.zip \
      --decoder-audit results/provenance/decoder_audit.json \
      --output results/provenance/bsd35k_fx_m_highrate_source.json

``verify``는 full-octave bootstrap이 호출하는 read-only gate다. PASS는 source-stage
조건만 뜻하며 causal P/S·ERR density·population authority·training을 열지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.data.bsd35k_highrate_evidence import (  # noqa: E402
    BSD35kHighRateEvidenceError,
    build_bsd35k_highrate_machine_evidence,
    load_and_validate_bsd35k_highrate_machine_evidence,
    write_bsd35k_highrate_machine_evidence_exclusive,
)


def _inside_repo(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    target = REPO_ROOT / path
    try:
        target.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}가 repository 밖을 가리킵니다") from exc
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue = subparsers.add_parser("issue", help="actual raw/archive에서 no-replace evidence 발행")
    issue.add_argument("--selection-plan", required=True)
    issue.add_argument("--metadata-csv", required=True)
    issue.add_argument("--selected-raw-root", required=True)
    issue.add_argument("--audio-archive", required=True)
    issue.add_argument("--decoder-audit", required=True)
    issue.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify", help="persisted evidence와 selected raw를 재검증")
    verify.add_argument("--evidence", required=True)
    verify.add_argument(
        "--expected-file-sha256",
        required=True,
        help="외부에서 고정한 evidence file bytes SHA-256",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "issue":
            selection = _inside_repo(args.selection_plan, label="--selection-plan")
            metadata = _inside_repo(args.metadata_csv, label="--metadata-csv")
            raw_root = _inside_repo(args.selected_raw_root, label="--selected-raw-root")
            archive = _inside_repo(args.audio_archive, label="--audio-archive")
            decoder = _inside_repo(args.decoder_audit, label="--decoder-audit")
            output = _inside_repo(args.output, label="--output")
            evidence = build_bsd35k_highrate_machine_evidence(
                repository_root=REPO_ROOT,
                selection_plan_path=selection,
                metadata_csv_path=metadata,
                selected_raw_root=raw_root,
                audio_archive_path=archive,
                decoder_audit_path=decoder,
            )
            path, file_sha = write_bsd35k_highrate_machine_evidence_exclusive(output, evidence)
            result = {
                "status": evidence["status"],
                "evidence_sha256": evidence["evidence_sha256"],
                "file_sha256": file_sha,
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "canonical_training_eligible": False,
                "blockers": evidence["authority"]["blockers"],
            }
        else:
            evidence = _inside_repo(args.evidence, label="--evidence")
            result = load_and_validate_bsd35k_highrate_machine_evidence(
                evidence,
                repository_root=REPO_ROOT,
                expected_sha256=args.expected_file_sha256,
            )
    except (BSD35kHighRateEvidenceError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
