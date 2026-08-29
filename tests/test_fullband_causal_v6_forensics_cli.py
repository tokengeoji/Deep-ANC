from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest


def _load_cli() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/data/forensics_fullband_causal_v6_clock.py"
    )
    spec = importlib.util.spec_from_file_location("v6_forensics_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forensics_output_is_fixed_to_capture_id_namespace() -> None:
    module = _load_cli()
    capture_id = "a" * 32
    assert module._forensics_relative_path(capture_id) == (
        "results/fullband_causal_v6/forensics/clock_" + capture_id + ".json"
    )
    for invalid in ("", "A" * 32, "a" * 31, "../" + "a" * 32):
        with pytest.raises(ValueError, match="capture_id"):
            module._forensics_relative_path(invalid)


def test_forensics_main_rejects_nonsealed_raw_before_repository_or_publication(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    result = module.main(
        [
            "--raw",
            "results/not_the_sealed_raw.npz",
            "--expected-raw-sha256",
            "1" * 64,
            "--expected-post-receipt-sha256",
            "2" * 64,
            "--failure",
            "results/fullband_causal_v6/failure_" + "a" * 32 + ".json",
            "--expected-failure-sha256",
            "3" * 64,
        ]
    )
    assert result == 2
    assert "exact sealed v6 path" in capsys.readouterr().err


def test_failure_canonical_bytes_rejects_reformatted_json() -> None:
    module = _load_cli()
    value = {"b": 2, "a": 1}

    class Guard:
        bytes = module._canonical_json_bytes(value) + b"\n"

    module._validate_failure_canonical_bytes(Guard(), value)
    Guard.bytes = b'{"a": 1, "b": 2}\n'
    with pytest.raises(ValueError, match="canonical publisher bytes"):
        module._validate_failure_canonical_bytes(Guard(), value)


@pytest.mark.skipif(
    os.environ.get("DEEP_ANC_RUN_LOCAL_FORENSICS_INTEGRATION") != "1",
    reason="보존된 local v6 raw가 있는 clean checkout에서만 명시적으로 실행",
)
def test_actual_v6_verify_only_runs_full_core_without_publication(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_cli()
    output_parent = module.ROOT / module.FORENSICS_DIRECTORY
    before = (
        sorted(path.name for path in output_parent.iterdir())
        if output_parent.is_dir()
        else []
    )
    result = module.main(
        [
            "--expected-raw-sha256",
            "f153c8664106b0c341b67db940fb2fb1d76cb7e58c2fa9a6e49558e1dba50a63",
            "--expected-post-receipt-sha256",
            "6372cfdec4ce15013f7bdc958f47c25fa1055f1e368adaeaa1a8d5627608dbda",
            "--failure",
            "results/fullband_causal_v6/"
            "failure_232a4e53a4eaa024d54b740a01c95fe1.json",
            "--expected-failure-sha256",
            "10856999254a8dc70c3696b02aed239db1b80f217a3dfd771442cedb2aacc75d",
            "--verify-only",
        ]
    )
    after = (
        sorted(path.name for path in output_parent.iterdir())
        if output_parent.is_dir()
        else []
    )
    assert result == 0
    assert before == after
    output = capsys.readouterr().out
    assert "[VERIFY_ONLY]" in output
    assert "파일 발행 없음" in output
