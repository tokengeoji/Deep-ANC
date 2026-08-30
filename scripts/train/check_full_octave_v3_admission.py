#!/usr/bin/env python3
"""125 Hz--8 kHz v3 학습 admission을 읽기 전용으로 감사한다.

이 명령은 Trainer, GPU, Audio/ALSA/sounddevice, run directory를 사용하지 않는다.
현재의 정상 결과는 ``BLOCKED``이며, 이는 기존 150--1600 Hz artifact를 full-octave
학습에 잘못 쓰지 않도록 하는 안전한 결과다.

기본값은 stdout만 쓴다. ``--out``을 줄 때만 JSON을 O_EXCL no-replace로 기록한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.train.evaluation_contract import write_json_exclusive  # noqa: E402
from deep_anc.train.full_octave_v3_admission import (  # noqa: E402
    audit_full_octave_v3_admission,
    render_full_octave_v3_admission_markdown,
)


def _inside_repo_relative_path(value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label}는 repository 내부 상대경로여야 합니다")
    target = REPO_ROOT / candidate
    try:
        target.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}가 repository 밖을 가리킵니다") from exc
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/full_octave_v3_admission.yaml", help="admission-only YAML"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="선택적 no-replace JSON 출력 (repository 내부 상대경로)",
    )
    parser.add_argument("--markdown", action="store_true", help="JSON 대신 markdown summary 출력")
    args = parser.parse_args(argv)
    try:
        config_path = _inside_repo_relative_path(args.config, label="config")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("admission config 최상위는 mapping이어야 합니다")
        report = audit_full_octave_v3_admission(loaded, repo_root=REPO_ROOT)
        if args.out is not None:
            output = _inside_repo_relative_path(args.out, label="out")
            write_json_exclusive(output, report)
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    if args.markdown:
        print(render_full_octave_v3_admission_markdown(report), end="")
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
