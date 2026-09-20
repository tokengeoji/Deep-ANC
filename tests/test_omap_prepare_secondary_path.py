"""2차경로 파생 파일을 만들 때 측정 원본과 기존 산출물을 먼저 보호한다."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_secondary_path", ROOT / "tools/prepare_secondary_path.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
OUTPUT_NAMES = ("secondary_path.npy", "secondary_path.csv", "secondary_path.h", "report.json")
COEFFICIENTS = b"0.125\n0.0\n-0.0625\n"


def run_prepare(monkeypatch, source, output, *arguments):
    monkeypatch.setattr(
        sys, "argv",
        [str(SPEC.origin), "--source", str(source), "--output-dir", str(output), *arguments],
    )
    return PREPARE.main()


@pytest.mark.parametrize("name", OUTPUT_NAMES)
@pytest.mark.parametrize("alias", ("direct", "symlink", "hardlink"))
def test_source_collision_fails_before_any_artifact_write(tmp_path, monkeypatch, name, alias):
    output = tmp_path / "artifacts"
    output.mkdir()
    for filename in OUTPUT_NAMES:
        (output / filename).write_bytes(b"existing artifact to preserve")
    target = output / name
    if alias == "direct":
        source = target
        source.write_bytes(COEFFICIENTS)
    else:
        source = tmp_path / "measured.txt"
        source.write_bytes(COEFFICIENTS)
        target.unlink()
        if alias == "symlink":
            target.symlink_to(source)
        else:
            target.hardlink_to(source)
    before = {filename: (output / filename).read_bytes() for filename in OUTPUT_NAMES}

    with pytest.raises(ValueError, match="overwrite measured source"):
        run_prepare(monkeypatch, source, output)

    assert source.read_bytes() == COEFFICIENTS
    assert {filename: (output / filename).read_bytes() for filename in OUTPUT_NAMES} == before


def test_output_directory_alias_cannot_overwrite_source(tmp_path, monkeypatch):
    output = tmp_path / "artifacts"
    output.mkdir()
    source = output / "secondary_path.csv"
    source.write_bytes(COEFFICIENTS)
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)

    with pytest.raises(ValueError, match="overwrite measured source"):
        run_prepare(monkeypatch, source, alias)

    assert source.read_bytes() == COEFFICIENTS
    assert list(output.iterdir()) == [source]


def test_verify_only_does_not_write_even_when_output_would_alias_source(tmp_path, monkeypatch):
    source = tmp_path / "secondary_path.csv"
    source.write_bytes(COEFFICIENTS)

    run_prepare(monkeypatch, source, tmp_path, "--verify-only")

    assert source.read_bytes() == COEFFICIENTS
    assert list(tmp_path.iterdir()) == [source]


def test_disjoint_export_preserves_source_and_writes_all_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "measured.txt"
    source.write_bytes(COEFFICIENTS)
    output = tmp_path / "new" / "artifacts"

    run_prepare(monkeypatch, source, output)

    assert source.read_bytes() == COEFFICIENTS
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    np.testing.assert_array_equal(
        np.load(output / "secondary_path.npy", allow_pickle=False),
        np.array([0.125, 0.0, -0.0625], dtype=np.float32),
    )
    report = json.loads((output / "report.json").read_text())
    assert report["source_sha256"] == hashlib.sha256(COEFFICIENTS).hexdigest()
