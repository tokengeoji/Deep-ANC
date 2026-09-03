"""Elice 부트스트랩/학습 시작 셸의 안전 불변식."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from deep_anc.data.holdout_contract import (
    EXPECTED_HISTORICAL_BUILDERS,
    EXPECTED_INVOCATIONS,
    EXPECTED_SELECTION_COUNTS,
    RECORDED_CONTENT_INTEGRITY_BOUNDARY,
    RECORDED_TREE_CONTENT_SNAPSHOT_ENCODING,
    RECORDED_TREE_SNAPSHOT_ENCODING,
    snapshot_regular_tree_metadata,
)
from deep_anc.data.public_lineage import canonical_json_sha256
from deep_anc.data.source_trust import (
    SourceTrustError,
    validate_environment_freeze_source_commit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ELICE_TRAINING_DOCUMENT = REPO_ROOT / "docs/05_training_elice.md"
ELICE_STAGING_DOCUMENT = REPO_ROOT / "docs/14_elice_external_data_staging.md"
ELICE_SCRIPTS = (
    REPO_ROOT / "scripts/elice/bootstrap_all.sh",
    REPO_ROOT / "scripts/elice/setup_env.sh",
    REPO_ROOT / "scripts/elice/run_parallel_models.sh",
    REPO_ROOT / "scripts/elice/run_pretrain.sh",
    REPO_ROOT / "scripts/elice/run_structure_search.sh",
    REPO_ROOT / "scripts/elice/bootstrap.sh",
)
STATIC_REFERENCE_CHECKER = (
    REPO_ROOT / "scripts/ci/check_static_contract_references.py"
)
FIXTURE_FMA_TRACKS = b",artist,album\ntrack_id,id,id\n1,artist-10,album-20\n"
FIXTURE_FMA_TRACKS_SHA256 = hashlib.sha256(FIXTURE_FMA_TRACKS).hexdigest()


@pytest.mark.parametrize("script", ELICE_SCRIPTS)
def test_elice_shell_scripts_parse(script: Path):
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_bootstrap_is_permanently_fail_closed():
    script = REPO_ROOT / "scripts/elice/bootstrap.sh"
    for arguments in ([], ["--train"], ["--anything"]):
        result = subprocess.run(
            ["bash", str(script), *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "폐기된 legacy 진입점" in result.stderr
        assert "bootstrap_all.sh" in result.stderr
        assert "train.py" not in result.stderr


def test_elice_docs_preserve_single_cold_start_authority_and_raw_only_restore():
    training = ELICE_TRAINING_DOCUMENT.read_text(encoding="utf-8")
    staging = ELICE_STAGING_DOCUMENT.read_text(encoding="utf-8")

    for text in (training, staging):
        for token in (
            "유일한 권위 진입점",
            "scripts/elice/bootstrap_all.sh",
            "--expected-commit",
            "--expected-holdout-sha256",
            "--expected-transfer-manifest-sha256",
            "--no-update",
            "scripts/elice/bootstrap.sh",
            "--train",
            "exit 2",
        ):
            assert token in text, f"Elice cold-start 문서의 필수 경계가 빠졌습니다: {token}"
        assert "bash scripts/elice/bootstrap.sh" not in text

    for token in (
        "--recorded-generation",
        "schema v2",
        "combined 101",
        "--rotate-existing-transfer-sha256",
        "data/manifests/elice_transfer_history/",
        "자동 overwrite하지 않는다",
        "transfer bytes나 하드웨어",
    ):
        assert token in training, f"transfer schema-v2 회전 규칙이 빠졌습니다: {token}"

    transfer_section = training.split("## 2.", 1)[1].split("## 3.", 1)[0]
    cursor = 0
    for token in (
        "clean exact commit 고정",
        "DNS/DEMAND selector와 source plan",
        "combined 101",
        "transfer schema v2 발행",
    ):
        position = transfer_section.find(token, cursor)
        assert position >= 0, (
            "Elice data artifact는 exact commit 뒤에 순서대로 발행해야 합니다: "
            f"{token}"
        )
        cursor = position + len(token)

    for token in (
        "raw-only cache",
        "Drive 검증 receipt",
        "forensic backup/cache",
        "data/manifests/canonical_v4/",
        "data/manifests/elice_bootstrap_receipt.json",
        "public 재다운로드",
        "`schema_version`은 2",
        "bootstrap은 학습을 자동 시작하지 않으므로",
    ):
        assert token in staging, f"Drive raw-only 복원 경계가 빠졌습니다: {token}"

    assert "현재 Elice에는" not in staging
    assert "tar -xf" not in staging
    for destructive_command in ("rclone delete", "rclone move", "rclone purge"):
        assert destructive_command not in staging


def test_bootstrap_has_explicit_completeness_and_empty_array_guards():
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")

    assert '"${pids[@]:-}"' not in text
    assert 'for p in "${pids[@]}"' in text
    assert "-eq 2000" in text
    assert "meta/esc50.csv" in text
    assert "-eq 8000" in text
    assert "-eq 16" in text
    assert "-eq 3600" in text
    assert "unzip -tq" in text
    assert 'flock -n 8' in text
    assert "active_train=$(pgrep -af '[t]rain\\.py' || true)" in text
    assert 'for p in "${extract_pids[@]}"' in text
    assert "file_list_complete" in text
    assert "dns_marker_complete" in text
    assert "구버전에서 이미 해제된 데이터 보존" not in text
    assert "expected_shape = (300, 8192)" in text
    assert "np.isfinite(value).all()" in text
    assert "scripts/data/build_rir_bank.py" not in text
    assert "transferred RIR bank" in text
    assert "--expected-commit" in text
    assert "--expected-holdout-sha256" in text
    assert "--no-update" in text
    assert 'HOLDOUT_MANIFEST="$REPO/data/manifests/recorded_holdout.json"' in text
    assert "환경 설치나 데이터 다운로드를 시작하지 않습니다" in text
    assert "현재 체크아웃으로 부트스트랩을 계속합니다" not in text
    assert "git pull" not in text
    assert "fma_metadata.zip" in text
    assert "d9527a5297a65da31c5676484d5047c3e2b8a8060ce72a46e26158be736bf265" in text
    assert "f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b" in text
    assert "FMA_TRACK_COUNT=106574" in text
    assert "raw_wav_tree_exact dns_fullband 16000" in text
    assert "raw_wav_tree_exact speech 8065" in text
    assert "duplicate raw WAV path/inode" in text
    assert text.count("verify_exact_checkout") >= 3
    assert '["git", "ls-tree", "-r", "-z", "--full-tree", commit]' in text
    assert "prepare_noise_pool.py" in text
    assert "--reuse-decoder-audit" in text
    assert "--expected-decoder-audit-sha256" in text
    assert "--expected-decoder-audit-file-sha256" in text
    assert "verify_decoder_audit_reuse.py" in text
    assert 'RECORDED_SUBBAND_COVERAGE_REPORT_DIR="results/data_audit/recorded_subband_coverage"' in text
    recorded_qa_call = text.index(
        '"$VENV_PYTHON" scripts/data/validate_recorded_sessions.py'
    )
    coverage_call = text.index(
        '"$VENV_PYTHON" scripts/data/audit_recorded_subband_coverage.py'
    )
    assert "export PYTHONDONTWRITEBYTECODE=1" in text
    final_pytest_stage = text.index('echo "=== [5/6] 검증 (pytest) ==="')
    pytest_call = text.index(
        '"$VENV_PYTHON" -B -m pytest -q', final_pytest_stage
    )
    assert recorded_qa_call < coverage_call < pytest_call
    assert '--canonical-out-dir "$RECORDED_SUBBAND_COVERAGE_REPORT_DIR"' in text
    assert 'if [ "$coverage_status" -gt 1 ]' in text
    assert 'if [ "$coverage_status" -eq 1 ]' in text
    assert "--raw-hash-workers" in text
    assert "--full-octave" in text
    assert "--full-octave-highrate-machine-evidence" in text
    assert "--expected-full-octave-highrate-machine-evidence-sha256" in text
    assert "audit_bsd35k_highrate_machine.py verify" in text
    assert "check_full_octave_v3_admission.py" in text
    assert '--expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256"' in text
    assert "--expected-transfer-manifest-sha256" in text
    assert 'TRANSFER_MANIFEST="$REPO/data/manifests/elice_transfer_manifest.json"' in text
    assert "--query-gpu=name,memory.total" in text
    assert "minimum_mib = 79 * 1024" in text
    assert "minimum_total_bytes=$((128 * 1024 * 1024 * 1024 - 128 * 1024 * 1024))" in text
    assert "minimum_available_bytes=$((96 * 1024 * 1024 * 1024))" in text
    assert "minimum_available_bytes=$((72 * 1024 * 1024 * 1024))" in text
    assert "public corpus가 이미 완전하므로 재개 시 ${minimum_available_gib}GiB 초기 예산 검사를 건너뜁니다" in text
    assert 'df -B1 --output=size,avail "$REPO"' in text
    assert 'str(torch.__version__) != "2.5.1+cu121"' in text
    assert 'str(torch.version.cuda) != "12.1"' in text
    assert 'ENVIRONMENT_RECEIPT="$REPO/.venv/environment-freeze.txt"' in text
    assert "pip freeze --all" in text
    assert "grep -Fxq 'torch==2.5.1+cu121'" in text
    assert "validate_environment_freeze_source_commit" in text
    assert "기존 exact 환경을 재사용하고 freeze를 expected commit에 갱신했습니다" in text
    assert 'BOOTSTRAP_RECEIPT="$REPO/data/manifests/elice_bootstrap_receipt.json"' in text
    assert '"recorded_aggregate_sha256": summary["recorded_aggregate_sha256"]' in text
    assert '"schema_version": 3' in text
    assert '"archive_cache_consumption": archive_cache_binding' in text
    assert '"recorded_subband_coverage": {' in text
    assert '"coverage_contract_sha256": coverage_payload[' in text
    assert '"freeze_receipt_sha256": environment.sha256' in text
    assert "data.bootstrap_receipt" in text
    exact_checkout_gate = text.index("if ! verify_exact_checkout; then")
    static_reference_gate = text.index(
        'python3 -I -B "$STATIC_REFERENCE_CHECKER" --repo-root "$REPO"'
    )
    canonical_bundle_gate = text.index(
        "if ! verify_canonical_bundle; then", static_reference_gate
    )
    preflight_exit = text.index(
        'if [ "$PREFLIGHT_ONLY" -eq 1 ]', canonical_bundle_gate
    )
    anchor_preflight = text.index(
        "if ! verify_transfer_manifest_anchor || ! hardware_storage_preflight"
    )
    environment_stage = text.index('echo "=== [1/6] 환경 (venv + torch cu121 + 패키지) ==="')
    freeze_refresh = text.index("if environment_probe; then", environment_stage)
    full_transfer_gate = text.index('! verify_transfer_bundle "$VENV_PYTHON"; then')
    early_collect_gate = text.index(
        '"$VENV_PYTHON" -B -m pytest -qq -p no:cacheprovider --collect-only'
    )
    early_focused_gate = text.index(
        "tests/test_elice_scripts.py", early_collect_gate
    )
    download_stage = text.index('echo "=== [2/6] 데이터 다운로드 (병렬) ==="')
    assert (
        exact_checkout_gate
        < static_reference_gate
        < canonical_bundle_gate
        < preflight_exit
        < anchor_preflight
        < environment_stage
        < freeze_refresh
        < full_transfer_gate
        < early_collect_gate
        < early_focused_gate
        < download_stage
    )
    for node in (
        "tests/test_realtime_start.py::test_runtime_artifact_cohort_ignores_training_only_runs_directory",
        "tests/test_realtime_start.py::test_engine_preflight_accepts_the_shipped_runtime_configs",
        "tests/test_public_decode_audit.py::test_reuse_cli_requires_external_file_and_semantic_sha_then_rehashes_raw",
        "tests/test_prepare_noise_pool.py::test_decoder_audit_path_index_is_cached_once_per_raw_root_context",
        "tests/test_prepare_noise_pool.py::test_failed_raw_inventory_verification_does_not_leave_a_path_index",
    ):
        assert node in text
    assert "verify_transfer_manifest_anchor()" in text
    assert "full transfer validator에는 완성된 venv Python이 필요합니다" in text
    assert "GIT_NO_REPLACE_OBJECTS=1" in text

    runner = ELICE_SCRIPTS[2].read_text(encoding="utf-8")
    assert "rollback_startup" in runner
    assert "startup_committed=1" in runner

    ddp_runner = ELICE_SCRIPTS[3].read_text(encoding="utf-8")
    assert '[ ! -x "$VENV_TORCHRUN" ]' in ddp_runner
    assert 'kill -0 "$PID"' in ddp_runner

    search_runner = ELICE_SCRIPTS[4].read_text(encoding="utf-8")
    assert "validate_tiny_completion" in search_runner
    assert 'EXPECTED_TINY_MODEL="hybrid_anc_tiny"' in search_runner
    assert "tiny_pid_is_expected" in search_runner
    assert "wait_for_gpu1_free" in search_runner
    assert "terminate_active_child" in search_runner
    assert "configs/model_tiny.yaml" in search_runner
    assert '--set run_until_step="$PILOT_STEPS"' in search_runner
    assert 'eval_pilot_${artifact}' in search_runner
    assert "eta_probe.txt" in search_runner


def test_bootstrap_uses_transfer_validated_recorded_manifest_for_all_evidence():
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    transfer_gate = text[
        text.index("verify_transfer_bundle() {") : text.index(
            "hardware_storage_preflight() {"
        )
    ]
    receipt = text[text.index('BOOTSTRAP_RECEIPT="$REPO/') :]

    assert 'RECORDED_MANIFEST=""' in text
    assert 'RECORDED_GENERATION=""' in text
    assert 'RECORDED_GENERATION_SHA256=""' in text
    assert 'summary.get("_validated_recorded_manifest_snapshot")' in transfer_gate
    assert 'summary.get("_validated_recorded_generation_snapshot")' in transfer_gate
    assert "EXPECTED_RECORDED_SESSIONS" in transfer_gate
    assert "COMBINED_SESSION_COUNT" in transfer_gate
    assert "recorded schema/session 수가 일치하지 않습니다" in transfer_gate
    assert "generation_relative = generation.path.relative_to(root).as_posix()" in transfer_gate
    prepare_call = text.index('"$VENV_PYTHON" scripts/data/prepare_noise_pool.py')
    validate_call = text.index('"$VENV_PYTHON" scripts/data/validate_noise_pool.py')
    prepare_block = text[prepare_call - 500 : validate_call]
    assert 'if [[ "$RECORDED_TRANSFER_SCHEMA" =~ ^(2|3)$ ]]; then' in prepare_block
    assert '--recorded-generation "$RECORDED_GENERATION"' in prepare_block
    assert (
        '--expected-recorded-generation-sha256 "$RECORDED_GENERATION_SHA256"'
        in prepare_block
    )
    assert '"${recorded_generation_prepare_args[@]}"' in prepare_block
    assert text.count('--manifest "$RECORDED_MANIFEST"') == 2
    assert "--manifest data/manifests/recorded_regrouped.jsonl" not in text
    assert (
        '"$RECORDED_SUBBAND_COVERAGE_REPORT_DIR" "$RECORDED_MANIFEST"'
        in receipt
    )
    assert "selected_recorded_manifest = sys.argv[9]" in receipt
    assert "validated_recorded_relative != selected_recorded_manifest" in receipt
    assert "coverage_manifest = validated_recorded_manifest" in receipt
    assert 'root / "data/manifests/recorded_regrouped.jsonl"' not in receipt


def test_cache_preflight_verifies_existing_cache_and_exits_before_generation():
    """cache 점검은 download/canonical generation/full suite/receipt를 열지 않는다."""

    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    environment_branch_start = text.index(
        'if [ "$CACHE_PREFLIGHT_ONLY" -eq 1 ]; then\n'
        '  if ! environment_complete; then'
    )
    environment_else = text.index("\nelse\n", environment_branch_start)
    environment_cache_branch = text[environment_branch_start:environment_else]
    assert "environment_complete" in environment_cache_branch
    for forbidden in (
        "write_environment_receipt",
        "bash scripts/elice/setup_env.sh",
        'touch "$SETUP_MARKER"',
    ):
        assert forbidden not in environment_cache_branch

    full_transfer_gate = text.index('! verify_transfer_bundle "$VENV_PYTHON"; then')
    early_gate = text.index(
        '"$VENV_PYTHON" -B -m pytest -qq -p no:cacheprovider --collect-only'
    )
    branch_start = text.index(
        'if [ "$CACHE_PREFLIGHT_ONLY" -eq 1 ]; then\n'
        '  begin_status_stage cache_verification\n'
        '  echo "=== [cache preflight] existing public raw + decoder audit reuse ==="'
    )
    download_stage = text.index(
        'echo "=== [2/6] 데이터 다운로드 (병렬) ==="', branch_start
    )
    branch = text[branch_start:download_stage]

    assert full_transfer_gate < early_gate < branch_start < download_stage
    assert "verify_public_raw_cache" in branch
    assert "verify_decoder_audit_reuse.py" in branch
    assert '--expected-audit-sha256 "$EXPECTED_DECODER_AUDIT_SHA256"' in branch
    assert (
        '--expected-file-sha256 "$EXPECTED_DECODER_AUDIT_FILE_SHA256"'
        in branch
    )
    assert '--hash-workers "$RAW_HASH_WORKERS"' in branch
    assert "verify_exact_checkout" in branch
    assert "verify_canonical_bundle" in branch
    assert "exit 0" in branch
    for forbidden in (
        "mkdir -p data/raw/noise",
        "prepare_noise_pool.py",
        "validate_noise_pool.py",
        "validate_recorded_sessions.py",
        'echo "=== [5/6] 검증 (pytest) ==="',
        "BOOTSTRAP_RECEIPT=",
    ):
        assert forbidden not in branch

    raw_cache = text[
        text.index("verify_public_raw_cache() {") : text.index(
            "zip_valid() {", text.index("verify_public_raw_cache() {")
        )
    ]
    for expected in (
        'dns_fullband" 16000',
        'noise/speech" 8065',
        'ESC-50-master/audio" 2000',
        'noise/machine" 3600',
        'fma_small" 8000',
        '"${DEMAND_ENVIRONMENTS[@]}"',
        "fma_metadata_complete",
        "fma_audio_metadata_match",
    ):
        assert expected in raw_cache


def test_early_pytest_gate_stops_when_collection_fails(tmp_path: Path):
    """함수가 ``if !`` 안에서 호출돼도 collect 실패를 focused PASS가 숨기지 못한다."""

    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    function_start = text.index("run_early_pytest_gate() {")
    function_end = text.index("\n}\n", function_start) + 2
    function = text[function_start:function_end]
    fake_python = tmp_path / "python"
    call_log = tmp_path / "calls.log"
    fake_python.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$EARLY_GATE_CALL_LOG"\n'
        'if [[ " $* " == *" --collect-only "* ]]; then exit 42; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["EARLY_GATE_CALL_LOG"] = str(call_log)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail\nVENV_PYTHON="$1"\n{function}\nrun_early_pytest_gate',
            "bootstrap-early-gate",
            str(fake_python),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "--collect-only" in calls[0]


@pytest.mark.parametrize(
    ("mode", "expected_schema", "expected_manifest", "should_pass"),
    [
        (
            "v1",
            "1",
            "data/manifests/recorded_regrouped.jsonl",
            True,
        ),
        (
            "v2",
            "2",
            "data/manifests/recorded_generations/generation-99/recorded.jsonl",
            True,
        ),
        (
            "v3",
            "3",
            "data/manifests/recorded_generations/generation-99/recorded.jsonl",
            True,
        ),
        (
            "v2-wrong-count",
            "",
            "",
            False,
        ),
    ],
)
def test_bootstrap_transfer_selector_handles_v1_v2_and_count_mismatch(
    tmp_path: Path,
    mode: str,
    expected_schema: str,
    expected_manifest: str,
    should_pass: bool,
):
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    gate = text[
        text.index("verify_transfer_bundle() {") : text.index(
            "hardware_storage_preflight() {"
        )
    ]
    program_start = gate.index("<<'PY'\n") + len("<<'PY'\n")
    program_end = gate.index("\nPY\n", program_start)
    program = gate[program_start:program_end]

    stub_root = tmp_path / "stubs"
    data_package = stub_root / "deep_anc/data"
    data_package.mkdir(parents=True)
    (stub_root / "deep_anc/__init__.py").write_text("", encoding="utf-8")
    (data_package / "__init__.py").write_text("", encoding="utf-8")
    (data_package / "holdout_contract.py").write_text(
        """\
class FileSnapshot:
    def __init__(self, path, data=b"manifest", sha256="b" * 64):
        self.path = path
        self.data = data
        self.sha256 = sha256
""",
        encoding="utf-8",
    )
    (data_package / "recorded_generation.py").write_text(
        "COMBINED_SESSION_COUNT = 99\n",
        encoding="utf-8",
    )
    (data_package / "transfer_contract.py").write_text(
        """\
from pathlib import Path
from .holdout_contract import FileSnapshot

EXPECTED_RECORDED_SESSIONS = 82

class TransferContractError(ValueError):
    pass

def validate_transfer_manifest(path, *, repo_root, expected_sha256):
    mode = Path(path).name
    if mode == "v1":
        relative = "data/manifests/recorded_regrouped.jsonl"
        generation = None
        sessions = 82
    else:
        relative = "data/manifests/recorded_generations/generation-99/recorded.jsonl"
        generation = FileSnapshot(
            Path(repo_root)
            / "data/manifests/recorded_generations/generation-99/generation.json"
        )
        sessions = 82 if mode == "v2-wrong-count" else 99
    return {
        "schema_version": 1 if mode == "v1" else (3 if mode == "v3" else 2),
        "_validated_recorded_manifest_snapshot": FileSnapshot(
            Path(repo_root) / relative
        ),
        "_validated_recorded_generation_snapshot": generation,
        "recorded_session_count": sessions,
        "manifest_sha256": expected_sha256,
        "file_count": 1,
    }
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(stub_root)
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, mode, str(tmp_path), "a" * 64],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    if should_pass:
        assert result.returncode == 0, result.stderr
        fields = result.stdout.strip().split("\t")
        expected_sessions = "82" if mode == "v1" else "99"
        assert fields[:4] == [
            expected_schema,
            expected_manifest,
            "b" * 64,
            expected_sessions,
        ]
        assert fields[6:] == (
            ["-", "-"]
            if mode == "v1"
            else [
                "data/manifests/recorded_generations/generation-99/generation.json",
                "b" * 64,
            ]
        )
    else:
        assert result.returncode == 1
        assert "recorded schema/session 수가 일치하지 않습니다" in result.stderr


def test_bootstrap_transfer_anchor_runs_without_site_packages(tmp_path: Path):
    """새 Elice system Python은 optional audio/ML wheel 없이 SHA anchor만 검증한다."""

    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    anchor = text[
        text.index("verify_transfer_manifest_anchor() {") : text.index(
            "verify_transfer_bundle() {"
        )
    ]
    program_start = anchor.index("<<'PY'\n") + len("<<'PY'\n")
    program_end = anchor.index("\nPY\n", program_start)
    program = anchor[program_start:program_end]

    # fixture package가 아니라 현재 tracked source를 직접 import해야, 향후
    # ``deep_anc`` package initializer에 optional dependency가 새로 생겨도 이
    # pre-venv 경계가 즉시 깨진다.
    root = tmp_path / "repo"
    manifest = root / "data/manifests/elice_transfer_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b'{"schema_version":1,"files":[]}\n')
    expected_sha256 = _sha256(manifest)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-B",
            "-c",
            program,
            str(manifest),
            str(root),
            expected_sha256,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{expected_sha256}\t{manifest.stat().st_size}"


def test_bootstrap_binds_full_decoder_audit_to_canonical_v4_manifest_generation():
    """기존 manifest가 있어도 canonical 학습이 audit 결속 세대만 읽어야 한다."""

    bootstrap = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    data_config = (REPO_ROOT / "configs/data_sim.yaml").read_text(encoding="utf-8")

    assert "noise_manifest_dir: data/manifests/canonical_v4" in data_config
    assert 'CANONICAL_MANIFEST_DIR="data/manifests/canonical_v4"' in bootstrap
    assert 'DECODER_AUDIT_REPORT="results/provenance/decoder_audit.json"' in bootstrap

    audit_call = bootstrap.index('"$VENV_PYTHON" scripts/data/audit_decoder_eligibility.py')
    prepare_call = bootstrap.index('"$VENV_PYTHON" scripts/data/prepare_noise_pool.py')
    validate_call = bootstrap.index('"$VENV_PYTHON" scripts/data/validate_noise_pool.py')
    assert audit_call < prepare_call < validate_call
    assert "--root ." in bootstrap[audit_call:prepare_call]
    assert "--scan-root data/raw" in bootstrap[audit_call:prepare_call]
    assert '--out "$DECODER_AUDIT_REPORT"' in bootstrap[audit_call:prepare_call]
    assert "--allow-rejections" in bootstrap[audit_call:prepare_call]
    assert '--out "$CANONICAL_MANIFEST_DIR"' in bootstrap[prepare_call:validate_call]
    assert '--decoder-audit "$DECODER_AUDIT_REPORT"' in bootstrap[prepare_call:validate_call]
    assert '--manifest-dir "$CANONICAL_MANIFEST_DIR"' in bootstrap[validate_call:]
    assert '--out "$CANONICAL_MANIFEST_DIR/dataset_qa.md"' in bootstrap[validate_call:]
    assert 'if [ "$REUSE_DECODER_AUDIT" -eq 1 ]; then' in bootstrap[audit_call - 1800:prepare_call]
    assert "재사용 실패 시 새 audit으로 자동 fallback하지 않는다" in bootstrap


def test_setup_env_requires_exact_torch_cuda_and_writes_freeze_receipt():
    text = (REPO_ROOT / "scripts/elice/setup_env.sh").read_text(encoding="utf-8")

    assert 'str(torch.__version__) != "2.5.1+cu121"' in text
    assert 'str(torch.version.cuda) != "12.1"' in text
    assert 'ENVIRONMENT_RECEIPT="$PWD/.venv/environment-freeze.txt"' in text
    assert "pip freeze --all" in text
    assert "validate_environment_freeze_source_commit" in text
    assert text.index('mv -f "${ENVIRONMENT_RECEIPT}.building"') < text.index(
        'touch "$SETUP_MARKER"'
    )


def _freeze_with_source(commit: str) -> bytes:
    return (
        "numpy==2.1.0\n"
        "-e git+https://github.com/Roka-jsj/Deep-ANC.git@"
        f"{commit}#egg=deep_anc\n"
        "torch==2.5.1+cu121\n"
    ).encode("utf-8")


def test_environment_freeze_source_commit_parser_rejects_stale_and_ambiguous_lines():
    expected = "a" * 40
    stale = "b" * 40

    assert (
        validate_environment_freeze_source_commit(
            b"-e git+https://example.invalid/other.git@"
            + b"c" * 40
            + b"#egg=other_package\n"
            + _freeze_with_source(expected),
            expected_commit=expected,
        )
        == "-e git+https://github.com/Roka-jsj/Deep-ANC.git@"
        f"{expected}#egg=deep_anc"
    )
    with pytest.raises(SourceTrustError, match="expected checkout과 다릅니다"):
        validate_environment_freeze_source_commit(
            _freeze_with_source(stale), expected_commit=expected
        )
    with pytest.raises(SourceTrustError, match="정확히 하나"):
        validate_environment_freeze_source_commit(
            b"torch==2.5.1+cu121\n", expected_commit=expected
        )
    with pytest.raises(SourceTrustError, match="정확히 하나"):
        validate_environment_freeze_source_commit(
            _freeze_with_source(expected) + _freeze_with_source(expected),
            expected_commit=expected,
        )
    with pytest.raises(SourceTrustError, match="전체 40자리 revision"):
        validate_environment_freeze_source_commit(
            _freeze_with_source(expected).replace(expected.encode(), b"deadbeef"),
            expected_commit=expected,
        )


def _make_bootstrap_git_repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "Deep-ANC"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "bootstrap-test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Bootstrap Test"], cwd=root, check=True
    )
    (root / ".gitignore").write_text(
        "/data/\n/results/\n/.venv/\n", encoding="utf-8"
    )
    (root / "marker.txt").write_text("committed\n", encoding="utf-8")
    validator = root / "src/deep_anc/data/holdout_contract.py"
    validator.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "src/deep_anc/data/holdout_contract.py", validator)
    lineage_validator = validator.parent / "public_lineage.py"
    shutil.copy2(REPO_ROOT / "src/deep_anc/data/public_lineage.py", lineage_validator)
    static_reference_checker = (
        root / "scripts/ci/check_static_contract_references.py"
    )
    static_reference_checker.parent.mkdir(parents=True)
    shutil.copy2(STATIC_REFERENCE_CHECKER, static_reference_checker)
    (root / "configs").mkdir()
    static_registry = root / "src/bootstrap_static_registry.py"
    static_registry.write_text(
        'POSITIVE = "tests/test_bootstrap_positive.py::test_bootstrap_positive"\n',
        encoding="utf-8",
    )
    static_target = root / "tests/test_bootstrap_positive.py"
    static_target.parent.mkdir(parents=True)
    static_target.write_text(
        "def test_bootstrap_positive():\n    pass\n",
        encoding="utf-8",
    )
    archive_cache_cli = root / "scripts/elice/public_archive_cache.py"
    archive_cache_cli.parent.mkdir(parents=True, exist_ok=True)
    archive_cache_cli.write_text(
        "import json, sys\n"
        "print(json.dumps({'authority':'fixture','command':sys.argv[1]}))\n",
        encoding="utf-8",
    )
    pget = root / "scripts/elice/pget.py"
    pget.write_text("# fixture pget\n", encoding="utf-8")
    validator.write_text(
        validator.read_text(encoding="utf-8").replace(
            "f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b",
            FIXTURE_FMA_TRACKS_SHA256,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "add",
            ".gitignore",
            "marker.txt",
            str(validator.relative_to(root)),
            str(lineage_validator.relative_to(root)),
            str(static_reference_checker.relative_to(root)),
            str(static_registry.relative_to(root)),
            str(static_target.relative_to(root)),
            str(archive_cache_cli.relative_to(root)),
            str(pget.relative_to(root)),
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "bootstrap test fixture"],
        cwd=root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert len(commit) == 40
    return root, commit


def _run_bootstrap_gate(
    root: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: int = 10,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DEEP_ANC_BOOTSTRAP_REPO"] = str(root)
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(ELICE_SCRIPTS[0]), *args],
        cwd=root.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _make_cache_preflight_runtime(
    root: Path, commit: str
) -> tuple[dict[str, str], str, Path, Path, Path, tuple[str, ...]]:
    """실제 GPU/37k raw 없이 cache-only shell 경계를 끝까지 실행하는 fixture."""

    fake_bin = root.parent / "cache-preflight-fake-bin"
    fake_bin.mkdir()
    call_log = root.parent / "cache-preflight-calls.log"
    transfer = root / "data/manifests/elice_transfer_manifest.json"
    transfer.parent.mkdir(parents=True, exist_ok=True)
    transfer.write_bytes(b'{"schema_version":1,"files":[]}\n')
    transfer_sha = _sha256(transfer)
    audit = root / "results/provenance/decoder_audit.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text("{}\n", encoding="utf-8")
    cache_root = root.parent / "archive-cache"
    cache_root.mkdir()
    cache_manifest = cache_root / "manifest.json"
    cache_manifest.write_bytes(b"{}\n")
    cache_args = (
        "--archive-cache-root",
        str(cache_root),
        "--archive-cache-manifest",
        str(cache_manifest),
        "--expected-archive-cache-manifest-sha256",
        _sha256(cache_manifest),
    )

    venv_python = root / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text(
        "#!/bin/bash\n"
        "set -u\n"
        'printf "%s\\n" "$*" >> "$FAKE_CALL_LOG"\n'
        'if [[ " $* " == *" -m pytest "* ]]; then\n'
        '  if [[ " $* " == *" --collect-only "* ]]; then\n'
        '    exit "${FAKE_COLLECT_EXIT:-0}"\n'
        "  fi\n"
        '  exit "${FAKE_FOCUSED_EXIT:-0}"\n'
        "fi\n"
        'if [[ " $* " == *" -m pip freeze "* ]]; then exit 91; fi\n'
        'if [[ " $* " == *" scripts/data/verify_decoder_audit_reuse.py "* ]]; then\n'
        '  exit "${FAKE_AUDIT_EXIT:-0}"\n'
        "fi\n"
        'if [[ " $* " == *"public_archive_cache.py verify-consumed-raw"* ]]; then\n'
        '  printf \'{"completion_path":"fixture/complete.json","completion_sha256":"%064d","inventory_path":"fixture/inventory.json","inventory_sha256":"%064d","current_output_projection_sha256":"%064d","decoder_audit_path":"results/provenance/decoder_audit.json","decoder_audit_file_sha256":"%064d","decoder_audit_semantic_sha256":"%064d","decoder_cache_projection_sha256":"%064d"}\\n\' 0 0 0 0 0 0\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "${1:-}" == "-B" && "${2:-}" == "-" && "${3:-}" == *"elice_transfer_manifest.json" ]]; then\n'
        '  printf "1\\tdata/manifests/recorded_regrouped.jsonl\\t%s\\t82\\t%s\\t1\\t-\\t-\\n" '
        '"$FAKE_RECORDED_MANIFEST_SHA" "$FAKE_TRANSFER_SHA"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    venv_python.chmod(0o755)
    marker = root / ".venv/.setup-complete"
    marker.write_text("complete\n", encoding="utf-8")
    freeze = root / ".venv/environment-freeze.txt"
    freeze.write_bytes(_freeze_with_source(commit))

    command_stubs = {
        "pgrep": "#!/bin/bash\nexit 1\n",
        "nvidia-smi": (
            "#!/bin/bash\n"
            'printf "NVIDIA A100 80GB PCIe, 81920\\n"\n'
        ),
        "df": (
            "#!/bin/bash\n"
            'printf "Size Avail\\n137438953472 103079215104\\n"\n'
        ),
        "stat": (
            "#!/bin/bash\n"
            'if [[ " $* " == *"tracks.csv "* ]]; then printf "260414445\\n"; '
            "else exec /usr/bin/stat \"$@\"; fi\n"
        ),
        "sha256sum": (
            "#!/bin/bash\n"
            'if [[ " $* " == *"tracks.csv "* ]]; then '
            'printf "f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b  %s\\n" "${@: -1}"; '
            "else exec /usr/bin/sha256sum \"$@\"; fi\n"
        ),
    }
    for name, payload in command_stubs.items():
        path = fake_bin / name
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o755)

    environment = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_CALL_LOG": str(call_log),
        "FAKE_TRANSFER_SHA": transfer_sha,
        "FAKE_RECORDED_MANIFEST_SHA": "b" * 64,
    }
    return environment, transfer_sha, call_log, freeze, marker, cache_args


def test_bootstrap_derives_default_repo_from_its_own_path(tmp_path: Path):
    """Elice clone 이름이 Deep-ANC가 아니어도 하드코딩 경로로 실패하지 않는다."""

    root, _commit = _make_bootstrap_git_repo(tmp_path)
    underscore_root = tmp_path / "Deep_ANC"
    root.rename(underscore_root)
    root = underscore_root
    script = root / "scripts/elice/bootstrap_all.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ELICE_SCRIPTS[0], script)
    subprocess.run(["git", "add", str(script.relative_to(root))], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "bootstrap script fixture"],
        cwd=root,
        check=True,
    )
    env = os.environ.copy()
    env.pop("DEEP_ANC_BOOTSTRAP_REPO", None)
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--expected-commit",
            "0" * 40,
            "--expected-holdout-sha256",
            "1" * 64,
            "--no-update",
            "--preflight-only",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1
    assert str(root) not in result.stderr
    assert "저장소에 들어갈 수 없습니다" not in result.stderr
    assert "expected commit" in result.stderr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_holdout_bundle(root: Path) -> tuple[Path, str]:
    csv_paths = {
        "source_pool": root / "data/source_pool/sources.csv",
        "source_pool_v2": root / "data/source_pool_v2/sources.csv",
    }
    families = ("environment", "machine", "music", "speech")
    csv_audit_rows = {}
    pcm_audit_rows = {}
    for name, path in csv_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "source_family,session_index,group_id,path,seconds,sample_rate_hz,clip_count,clips"
        ]
        csv_rows = []
        pcm_rows = []
        builder = "v1" if name == "source_pool" else "v2"
        for family in families:
            for index in range(20):
                row_path = f"data/{name}/{family}/{family}_{index:03d}.wav"
                if name == "source_pool" and family == "music" and index == 0:
                    clip = "000001.mp3"
                elif name == "source_pool" and family == "speech" and index == 0:
                    clip = "1-2-0001.flac"
                else:
                    clip = f"{name}-{family}-{index:03d}.wav"
                clips_field = json.dumps([clip]).replace('"', '""')
                lines.append(
                    f'{family},{index},{family}-g{index},{row_path},70.0,48000,1,"{clips_field}"'
                )
                identity = {
                    "family": family,
                    "session_index": index,
                    "path": row_path,
                }
                csv_rows.append(
                    {
                        **identity,
                        "declared_clips": 1,
                        "reconstructed_clips": 1,
                        "missing_clips": 0,
                        "prefix_pass": True,
                    }
                )
                pcm_rows.append(
                    {
                        **identity,
                        "builder_commit": EXPECTED_HISTORICAL_BUILDERS[builder]["commit"],
                        "status": "PASS",
                        "sample_rate_hz": 48000,
                        "channels": 1,
                        "frames": 3_360_000,
                        "wav_sha256": hashlib.sha256(row_path.encode()).hexdigest(),
                        "pcm": {
                            "status": "PASS",
                            "shape_match": True,
                            "expected_shape": [3_360_000],
                            "actual_shape": [3_360_000],
                        },
                    }
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        csv_audit_rows[name] = csv_rows
        pcm_audit_rows[name] = pcm_rows
    csv_hashes = {name: _sha256(path) for name, path in csv_paths.items()}

    tracks_path = root / "data/raw/music/fma_metadata/tracks.csv"
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    tracks_path.write_bytes(FIXTURE_FMA_TRACKS)
    chapters_path = root / "data/raw/speech/LibriSpeech/CHAPTERS.TXT"
    chapters_path.parent.mkdir(parents=True, exist_ok=True)
    chapters_path.write_text(
        "2 | 1 | 1.0 | train | fixture | 100\n", encoding="utf-8"
    )
    esc50_path = root / "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv"
    esc50_path.parent.mkdir(parents=True, exist_ok=True)
    esc50_path.write_text(
        "filename,fold,target,category,esc10,src_file,take\n"
        "source_pool-environment-000.wav,1,1,fixture,True,env-source,1\n"
        "source_pool-machine-000.wav,1,2,fixture,True,machine-source,1\n",
        encoding="utf-8",
    )
    clip_lineage_rows = [
        {
            "family": "environment",
            "clip": "source_pool-environment-000.wav",
            "content_sha256": "1" * 64,
            "lineage_keys": ["esc50_src:env-source"],
        },
        {
            "family": "machine",
            "clip": "source_pool-machine-000.wav",
            "content_sha256": "2" * 64,
            "lineage_keys": ["esc50_src:machine-source"],
        },
        {
            "family": "music",
            "clip": "000001.mp3",
            "content_sha256": "3" * 64,
            "lineage_keys": ["fma_album:album-20", "fma_artist:artist-10"],
        },
        {
            "family": "speech",
            "clip": "1-2-0001.flac",
            "content_sha256": "4" * 64,
            "lineage_keys": ["gutenberg_book:100", "librivox_reader:1"],
        },
    ]
    clip_lineage_metadata = {
        "librispeech_chapters": {
            "path": "data/raw/speech/LibriSpeech/CHAPTERS.TXT",
            "sha256": _sha256(chapters_path),
            "size": chapters_path.stat().st_size,
        },
        "fma_tracks": {
            "path": "data/raw/music/fma_metadata/tracks.csv",
            "sha256": _sha256(tracks_path),
            "size": tracks_path.stat().st_size,
        },
        "esc50": {
            "path": "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
            "sha256": _sha256(esc50_path),
            "size": esc50_path.stat().st_size,
        },
    }
    clip_lineage_sha = canonical_json_sha256(clip_lineage_rows)
    components = {
        (
            f"{family}-lineage-"
            + hashlib.sha256(f"{family}-session".encode()).hexdigest()[:12]
        ): [f"{family}-session"]
        for family in families
    }
    membership_sha = hashlib.sha256(
        json.dumps(
            components,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    regrouped_path = root / "data/manifests/recorded_regrouped.jsonl"
    regrouped_path.parent.mkdir(parents=True, exist_ok=True)
    regrouped_rows = [
        {
            "session_id": f"{family}-session",
            "source_family": family,
            "group_id": component,
            "source_pool_group_id": f"{family}-source-pool",
            "split": "train",
            "lineage_schema": (
                "shared_clip+music_artist_album+speech_reader_gutenberg_book/v2"
            ),
            "path": f"../recorded/{family}-session",
        }
        for family, component in (
            (family, next(name for name in components if name.startswith(f"{family}-lineage-")))
            for family in families
        )
    ]
    regrouped_path.write_text(
        "".join(json.dumps(row) + "\n" for row in regrouped_rows),
        encoding="utf-8",
    )
    groups_by_family_split = {
        family: {"train": 1, "val": 0, "test": 0} for family in families
    }
    for family in families:
        session_dir = root / "data/recorded" / f"{family}-session"
        session_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = session_dir / "session.json"
        metadata_bytes = json.dumps(
            {
                "session_id": f"{family}-session",
                "source_family": family,
                "program": {
                    "type": "file",
                    "file": f"data/source_pool/{family}/{family}_000.wav",
                },
            }
        ).encode("utf-8")
        if not metadata_path.exists() or metadata_path.read_bytes() != metadata_bytes:
            metadata_path.write_bytes(metadata_bytes)

    recorded_snapshot = snapshot_regular_tree_metadata(
        root / "data/recorded",
        repo_root=root,
        label="fixture recorded tree",
    )

    report = {
        "schema_version": 1,
        "authority": "historical_builder_reproduction_plus_pcm_validation",
        "status": "PASS",
        "mode": "repair",
        "recorded_tree_protection": {
            "schema_version": 1,
            "status": "PASS",
            "root": "data/recorded",
            "file_count": recorded_snapshot.file_count,
            "snapshot_encoding": RECORDED_TREE_SNAPSHOT_ENCODING,
            "before_sha256": recorded_snapshot.sha256,
            "after_sha256": recorded_snapshot.sha256,
            "content_snapshot_encoding": RECORDED_TREE_CONTENT_SNAPSHOT_ENCODING,
            "before_content_sha256": recorded_snapshot.content_sha256,
            "after_content_sha256": recorded_snapshot.content_sha256,
            "unchanged": True,
            "content_integrity_boundary": RECORDED_CONTENT_INTEGRITY_BOUNDARY,
        },
        "historical_builders": EXPECTED_HISTORICAL_BUILDERS,
        "selection": {
            "seed": 20260804,
            **EXPECTED_INVOCATIONS,
            "v2_exclusion_semantics": "historical v1 full plan의 used[:12] unique set",
            "counts": {**EXPECTED_SELECTION_COUNTS, "v2_unique": 400},
            "expected": EXPECTED_SELECTION_COUNTS,
        },
        "pools": {
            name: {
                "status": "PASS",
                "csv": {
                    "status": "PASS",
                    "row_count": 80,
                    "expected_row_count": 80,
                    "issues": [],
                    "rows": csv_audit_rows[name],
                },
                "pcm": {
                    "status": "PASS",
                    "passed_rows": 80,
                    "expected_rows": 80,
                    "rows": pcm_audit_rows[name],
                },
            }
            for name in csv_paths
        },
        "repair": {
            "requested": True,
            "performed": True,
            "files": {
                name: {"after_sha256": csv_hashes[name]} for name in csv_paths
            },
        },
        "post_repair_csv_sha256": csv_hashes,
        "downstream_gates": {
            "active_holdout": {
                "status": "PASS",
                "active_session_count": 4,
                "active_source_row_count": 4,
                "total_clips": 4,
                "clip_lineage_sha256": clip_lineage_sha,
                "clip_lineage_metadata": clip_lineage_metadata,
            }
        },
        "lineage_contract": {
            "schema_version": 2,
            "tracks_csv": "data/raw/music/fma_metadata/tracks.csv",
            "tracks_csv_sha256": FIXTURE_FMA_TRACKS_SHA256,
            "librispeech_chapters_path": "data/raw/speech/LibriSpeech/CHAPTERS.TXT",
            "librispeech_chapters_sha256": _sha256(chapters_path),
            "esc50_metadata_path": "data/raw/noise/esc50/ESC-50-master/meta/esc50.csv",
            "esc50_metadata_sha256": _sha256(esc50_path),
            "holdout_clip_lineage_sha256": clip_lineage_sha,
            "active_session_count": 4,
            "component_count": 4,
            "component_count_by_family": {family: 1 for family in families},
            "components": components,
            "component_membership_sha256": membership_sha,
            "regrouped_manifest": "data/manifests/recorded_regrouped.jsonl",
            "regrouped_manifest_sha256": _sha256(regrouped_path),
            "regrouped_row_count": 4,
            "regrouped_component_count": 4,
            "groups_by_family_split": groups_by_family_split,
            "lineage_cross_split_count": 0,
            "source_clip_cross_split_count": 0,
        },
    }
    report_bytes = json.dumps(report).encode("utf-8")
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    report_relative = (
        "results/provenance/"
        f"source_pool_provenance_report.{report_sha}.json"
    )
    report_path = root / report_relative
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(report_bytes)

    holdout = {
        "purpose": "canonical active-session provenance",
        "scope": "active_sessions_only",
        "active_session_count": 4,
        "active_source_row_count": 4,
        "sources_csv": [
            "data/source_pool/sources.csv",
            "data/source_pool_v2/sources.csv",
        ],
        "sources_csv_sha256": csv_hashes,
        "provenance_report": report_relative,
        "provenance_report_sha256": report_sha,
        "source_rows": [
            "data/source_pool/environment/environment_000.wav",
            "data/source_pool/machine/machine_000.wav",
            "data/source_pool/music/music_000.wav",
            "data/source_pool/speech/speech_000.wav",
        ],
        "families": {
            family: [
                (
                    "000001.mp3"
                    if family == "music"
                    else "1-2-0001.flac"
                    if family == "speech"
                    else f"source_pool-{family}-000.wav"
                )
            ]
            for family in families
        },
        "clip_lineage": {
            "schema_version": 1,
            "metadata": clip_lineage_metadata,
            "clips": clip_lineage_rows,
            "clips_sha256": clip_lineage_sha,
        },
        "total_clips": 4,
    }
    holdout_path = root / "data/manifests/recorded_holdout.json"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text(json.dumps(holdout), encoding="utf-8")
    return holdout_path, _sha256(holdout_path)


def test_bootstrap_requires_full_expected_commit_before_repo_side_effects(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)

    missing = _run_bootstrap_gate(root, "--no-update")
    short = _run_bootstrap_gate(root, "--expected-commit", "deadbeef", "--no-update")
    duplicate = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        f"--expected-commit={commit}",
        "--no-update",
    )

    assert missing.returncode == 2
    assert short.returncode == 2
    assert duplicate.returncode == 2
    assert "전체 40자리" in missing.stderr
    assert "전체 40자리" in short.stderr
    assert "한 번만 지정" in duplicate.stderr
    assert not (root / "data").exists()


def test_bootstrap_requires_full_expected_holdout_sha_before_repo_side_effects(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)

    missing = _run_bootstrap_gate(
        root, "--expected-commit", commit, "--no-update"
    )
    short = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "deadbeef",
        "--no-update",
    )
    duplicate = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        f"--expected-holdout-sha256={'b' * 64}",
        "--no-update",
    )

    assert missing.returncode == 2
    assert short.returncode == 2
    assert duplicate.returncode == 2
    assert "64자리 SHA-256" in missing.stderr
    assert "64자리 SHA-256" in short.stderr
    assert "한 번만 지정" in duplicate.stderr
    assert not (root / "data").exists()


def test_bootstrap_no_update_rejects_commit_mismatch_before_setup(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    wrong = ("0" if commit[0] != "0" else "1") + commit[1:]

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        wrong,
        "--expected-holdout-sha256",
        "a" * 64,
        "--no-update",
    )

    assert result.returncode == 1
    assert "--no-update 상태에서 HEAD가 expected commit과 다릅니다" in result.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_bootstrap_requires_explicit_no_update_before_setup(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
    )

    assert result.returncode == 2
    assert "--no-update는 필수" in result.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_bootstrap_rejects_invalid_or_duplicate_raw_hash_worker_setting_before_setup(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    invalid = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--raw-hash-workers",
        "0",
        "--no-update",
    )
    duplicate = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--raw-hash-workers",
        "8",
        "--raw-hash-workers=8",
        "--no-update",
    )

    assert invalid.returncode == 2
    assert duplicate.returncode == 2
    assert "--raw-hash-workers" in invalid.stderr
    assert "한 번만 지정" in duplicate.stderr
    assert not (root / "data").exists()


def test_bootstrap_archive_cache_arguments_are_all_or_nothing_before_setup(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    common = (
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
    )
    orphan = _run_bootstrap_gate(
        root,
        *common,
        "--archive-cache-root",
        str(tmp_path / "incoming"),
        "--no-update",
    )
    invalid_sha = _run_bootstrap_gate(
        root,
        *common,
        "--archive-cache-root",
        str(tmp_path / "incoming"),
        "--archive-cache-manifest",
        str(tmp_path / "incoming/manifest.json"),
        "--expected-archive-cache-manifest-sha256",
        "deadbeef",
        "--no-update",
    )
    duplicate = _run_bootstrap_gate(
        root,
        *common,
        "--archive-cache-root",
        str(tmp_path / "incoming"),
        f"--archive-cache-root={tmp_path / 'incoming'}",
        "--no-update",
    )
    missing_only_contract = _run_bootstrap_gate(
        root, *common, "--archive-cache-only", "--no-update"
    )

    assert orphan.returncode == 2
    assert invalid_sha.returncode == 2
    assert duplicate.returncode == 2
    assert missing_only_contract.returncode == 2
    assert "모두 함께" in orphan.stderr
    assert "64자리" in invalid_sha.stderr
    assert "한 번만" in duplicate.stderr
    assert "모두 필수" in missing_only_contract.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_bootstrap_archive_cache_rejects_modes_that_would_ignore_or_overclaim_it(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    cache_args = (
        "--archive-cache-root",
        str(tmp_path / "incoming"),
        "--archive-cache-manifest",
        str(tmp_path / "incoming/manifest.json"),
        "--expected-archive-cache-manifest-sha256",
        "b" * 64,
    )
    common = (
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
    )
    preflight = _run_bootstrap_gate(
        root, *common, *cache_args, "--preflight-only", "--no-update"
    )
    cache_preflight = _run_bootstrap_gate(
        root, *common, *cache_args, "--cache-preflight-only", "--no-update"
    )
    full_octave = _run_bootstrap_gate(
        root,
        *common,
        *cache_args,
        "--archive-cache-only",
        "--full-octave",
        "--no-update",
    )
    duplicate_only = _run_bootstrap_gate(
        root,
        *common,
        *cache_args,
        "--archive-cache-only",
        "--archive-cache-only",
        "--no-update",
    )

    assert preflight.returncode == 2
    assert cache_preflight.returncode == 2
    assert full_octave.returncode == 2
    assert duplicate_only.returncode == 2
    assert "--preflight-only" in preflight.stderr
    assert "--reuse-decoder-audit" in cache_preflight.stderr
    assert "--full-octave" in full_octave.stderr
    assert "한 번만" in duplicate_only.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_archive_cache_only_stops_after_manifest_verify_and_restore_without_authority(
    tmp_path: Path,
):
    root, _initial_commit = _make_bootstrap_git_repo(tmp_path)
    fake_cli = root / "scripts/elice/public_archive_cache.py"
    fake_cli.parent.mkdir(parents=True, exist_ok=True)
    fake_cli.write_text(
        "from __future__ import annotations\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "command = sys.argv[1]\n"
        "if command not in {'verify-manifest', 'restore', 'consume'}:\n"
        "    raise SystemExit(91)\n"
        "with Path(os.environ['FAKE_ARCHIVE_CACHE_LOG']).open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(json.dumps({'authority': 'transport_acceleration_only_not_raw_or_training_authority', 'command': command}))\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", str(fake_cli.relative_to(root))], cwd=root, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "add archive cache fixture"],
        cwd=root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    manifest = incoming / "manifest.json"
    manifest.write_bytes(b"{}\n")
    log = tmp_path / "archive-cache-calls.jsonl"

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--archive-cache-root",
        str(incoming),
        "--archive-cache-manifest",
        str(manifest),
        "--expected-archive-cache-manifest-sha256",
        _sha256(manifest),
        "--archive-cache-only",
        "--no-update",
        extra_env={"FAKE_ARCHIVE_CACHE_LOG": str(log)},
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [call[0] for call in calls] == ["verify-manifest", "restore"]
    assert "raw/training authority가 아닙니다" in result.stdout
    assert not (root / ".venv").exists()
    assert not (root / "data/manifests/elice_bootstrap_receipt.json").exists()

    full = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--archive-cache-root",
        str(incoming),
        "--archive-cache-manifest",
        str(manifest),
        "--expected-archive-cache-manifest-sha256",
        _sha256(manifest),
        "--no-update",
        extra_env={"FAKE_ARCHIVE_CACHE_LOG": str(log)},
    )
    assert full.returncode == 2
    assert "expected-transfer-manifest-sha256" in full.stderr
    assert "held-fd extractor handoff" not in full.stderr
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [call[0] for call in calls] == ["verify-manifest", "restore", "verify-manifest"]
    assert not (root / ".venv").exists()


def test_plain_bootstrap_rejects_cache_restored_archive_and_forged_local_marker(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    restored = root / "data/raw/noise/shard000.tar.bz2"
    restored.parent.mkdir(parents=True, exist_ok=True)
    restored.write_bytes(b"cache-restored-but-unanchored")
    origin_dir = root / "data/raw/noise/.archive_cache_origins"
    origin_dir.mkdir()
    forged = origin_dir / f"archive_cache_origin.{'b' * 64}.{commit}.json"
    forged.write_text(
        json.dumps(
            {
                "kind": "deep_anc_archive_cache_origin_receipt",
                "manifest_sha256": "b" * 64,
                "publisher_commit": commit,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = restored.read_bytes()

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
    )

    assert result.returncode == 1
    assert "origin laundering" in result.stderr
    assert "matching archive-cache external manifest anchor" in result.stderr
    assert restored.read_bytes() == before
    assert not (root / ".venv").exists()
    assert not (root / "data/manifests/elice_bootstrap_receipt.json").exists()


def test_plain_bootstrap_rejects_even_empty_cache_consume_intent_directory(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    marker_directory = root / "data/raw/noise/.archive_cache_consumptions"
    marker_directory.mkdir(parents=True)

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
    )

    assert result.returncode == 1
    assert "plain bootstrap raw 재사용을 금지" in result.stderr
    assert "matching archive-cache external anchors" in result.stderr
    assert list(marker_directory.iterdir()) == []
    assert not (root / ".venv").exists()
    assert not (root / "data/manifests/elice_bootstrap_receipt.json").exists()


def test_plain_archive_helpers_reject_any_existing_final_before_network(
    tmp_path: Path,
):
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    wget_function = text[
        text.index("ensure_wget_zip() {") : text.index("ensure_pget_zip() {")
    ]
    pget_function = text[
        text.index("ensure_pget_zip() {") : text.index("download_dns_archive() {")
    ]
    dns_function = text[
        text.index("download_dns_archive() {") : text.index("download_esc50() {")
    ]
    network_log = tmp_path / "network.log"
    cases = [
        (
            wget_function,
            "archive=invalid-demand.zip\n"
            "wget() { echo wget >> \"$NETWORK_LOG\"; return 0; }\n"
            "zip_valid() { return 1; }\n"
            "ensure_wget_zip https://example.invalid \"$archive\" 1\n",
        ),
        (
            pget_function,
            "archive=invalid-mimii.zip\n"
            "fake_pget() { echo pget >> \"$NETWORK_LOG\"; return 0; }\n"
            "PGET=(fake_pget)\n"
            "zip_valid() { return 1; }\n"
            "ensure_pget_zip https://example.invalid \"$archive\" 4 1\n",
        ),
        (
            dns_function,
            "archive=invalid-dns.tar.bz2\n"
            "fake_pget() { echo dns >> \"$NETWORK_LOG\"; return 0; }\n"
            "PGET=(fake_pget)\n"
            "download_dns_archive https://example.invalid \"$archive\"\n",
        ),
    ]
    for index, (function, invocation) in enumerate(cases):
        case_dir = tmp_path / f"case-{index}"
        case_dir.mkdir()
        archive = case_dir / invocation.splitlines()[0].split("=", 1)[1]
        archive.write_bytes(b"injected-invalid-final")
        result = subprocess.run(
            [
                "bash",
                "-c",
                "set -u\n"
                f"NETWORK_LOG={str(network_log)!r}\n"
                + function
                + "\n"
                + invocation,
            ],
            cwd=case_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "덮어쓰거나 재사용하지 않습니다" in result.stderr
        assert archive.read_bytes() == b"injected-invalid-final"
    assert not network_log.exists()


def test_plain_pget_final_injection_is_no_replace_and_resume_path_is_stable(
    tmp_path: Path,
):
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    prepare_function = text[
        text.index("prepare_pget_download_stage() {") : text.index(
            "# ZIP은 live corpus", text.index("prepare_pget_download_stage() {")
        )
    ]
    pget_function = text[
        text.index("ensure_pget_zip() {") : text.index("download_dns_archive() {")
    ]
    seed = tmp_path / "seed.zip"
    import zipfile

    with zipfile.ZipFile(seed, "w") as archive:
        archive.writestr("root/a.wav", b"fixture")
    target = tmp_path / "mimii.zip"
    output_log = tmp_path / "pget-outputs.log"
    first = subprocess.run(
        [
            "bash",
            "-c",
            "set -u\n"
            f"VENV_PYTHON={sys.executable!r}\n"
            f"SEED={str(seed)!r}\nTARGET={str(target)!r}\n"
            "zip_valid() { [ -f \"$1\" ] && unzip -tq \"$1\" >/dev/null 2>&1; }\n"
            "fake_pget() { cp \"$SEED\" \"$2\"; printf injected > \"$TARGET\"; }\n"
            "PGET=(fake_pget)\n"
            + prepare_function
            + pget_function
            + "\nensure_pget_zip https://example.invalid \"$TARGET\" 4 1\n",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode != 0
    assert "no-replace 경합" in first.stderr
    assert target.read_bytes() == b"injected"

    resume_target = tmp_path / "resume-mimii.zip"
    resume_script = (
        "set -u\n"
        f"VENV_PYTHON={sys.executable!r}\nOUTPUT_LOG={str(output_log)!r}\n"
        "zip_valid() { return 1; }\n"
        "fake_pget() { echo \"$2\" >> \"$OUTPUT_LOG\"; "
        "touch \"$2.part\" \"$2.part.lock\" \"$2.part.state.json\"; return 1; }\n"
        "PGET=(fake_pget)\n"
        + prepare_function
        + pget_function
        + f"\nensure_pget_zip https://example.invalid {str(resume_target)!r} 4 1 || true\n"
        + "stage=$(dirname \"$(tail -n 1 \"$OUTPUT_LOG\")\")\n"
        + "quarantine=\"$stage/.resume-mimii.zip.part.quarantine.abc123_\"\n"
        + "mkdir -m 700 \"$quarantine\"\n"
        + "printf old > \"$quarantine/resume-mimii.zip.part\"\n"
        + f"ensure_pget_zip https://example.invalid {str(resume_target)!r} 4 1 || true\n"
    )
    resumed = subprocess.run(
        ["bash", "-c", resume_script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    outputs = output_log.read_text(encoding="utf-8").splitlines()
    assert len(outputs) == 2
    assert outputs[0] == outputs[1]


def test_plain_bootstrap_rejects_intermediate_archive_parent_symlink_before_setup(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    outside = tmp_path / "outside-demand"
    outside.mkdir()
    noise = root / "data/raw/noise"
    noise.mkdir(parents=True, exist_ok=True)
    (noise / "demand").symlink_to(outside, target_is_directory=True)

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
    )

    assert result.returncode == 1
    assert "fixed archive parent" in result.stderr
    assert list(outside.iterdir()) == []
    assert not (root / ".venv").exists()


def test_archive_cache_gate_order_precedes_setup_and_held_consume_precedes_download():
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    exact = text.index("if ! verify_exact_checkout; then")
    static = text.index('python3 -I -B "$STATIC_REFERENCE_CHECKER"')
    holdout = text.index("if ! verify_canonical_bundle; then", static)
    anchor = text.index("begin_status_stage archive_cache_anchor")
    plain_intent_guard = text.index(
        "archive-cache consumption intent/completion directory가 있어"
    )
    transfer = text.index("if ! verify_transfer_manifest_anchor || ! hardware_storage_preflight")
    early = text.index("begin_status_stage early_pytest")
    consume = text.index(
        'echo "=== [archive cache] fixed DNS3 + DEMAND6 + MIMII1 held-fd consume ==="'
    )
    download = text.index('echo "=== [2/6] 데이터 다운로드 (병렬) ==="')

    assert exact < static < holdout < plain_intent_guard < anchor < transfer < early < consume < download
    assert '"$ARCHIVE_CACHE_CLI" consume' in text[consume:download]
    assert "archive-cache full consumption은 held-fd extractor handoff가 없어" not in text
    raw_start = text.index('echo "=== [3/6] DNS 샤드 무결성 검사 + 해제 ==="')
    raw_end = text.index("if ! raw_wav_tree_exact dns_fullband 16000", raw_start)
    raw_section = text[raw_start:raw_end]
    assert raw_section.index('archive_cache_argument_count" -eq 3') < raw_section.index(
        'bzip2 -t "$f"'
    )
    assert "legacy DNS pathname bzip2/tar extractor를 건너뜁니다" in raw_section


def test_archive_cache_active_never_falls_back_for_dns_demand_or_mimii():
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    dns_loop = text[text.index('for f in "${!DL[@]}"; do') :]
    dns_guard = dns_loop.index('if [ "$archive_cache_argument_count" -eq 3 ]; then')
    dns_network = dns_loop.index('start_download "DNS $f"')
    demand_function = text[
        text.index("download_demand() {") : text.index("download_mimii() {")
    ]
    mimii_start = text.index("download_mimii() {")
    mimii_function = text[
        mimii_start : text.index(
            'if [ "$CACHE_PREFLIGHT_ONLY" -eq 1 ]; then', mimii_start
        )
    ]

    assert dns_guard < dns_network
    assert "official network fallback을 금지" in dns_loop[dns_guard:dns_network]
    assert demand_function.index('archive_cache_argument_count" -eq 3') < demand_function.index(
        "ensure_wget_zip"
    )
    assert mimii_function.index('archive_cache_argument_count" -eq 3') < mimii_function.index(
        "ensure_wget_zip"
    )
    assert "official network fallback" in demand_function
    assert "official network fallback" in mimii_function
    assert "legacy DEMAND pathname downloader/extractor 호출을 금지" in demand_function
    assert "legacy MIMII pathname downloader/extractor 호출을 금지" in mimii_function


def test_archive_cache_active_corruption_calls_zero_network_helpers(tmp_path: Path):
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")
    log = tmp_path / "network.log"
    demand_function = text[
        text.index("download_demand() {") : text.index("download_mimii() {")
    ]
    mimii_start = text.index("download_mimii() {")
    mimii_function = text[
        mimii_start : text.index(
            'if [ "$CACHE_PREFLIGHT_ONLY" -eq 1 ]; then', mimii_start
        )
    ]
    dns_loop = text[
        text.index('for f in "${!DL[@]}"; do') : text.index(
            "if esc50_complete; then", text.index('for f in "${!DL[@]}"; do')
        )
    ]
    common = (
        "set -u\n"
        "archive_cache_argument_count=3\n"
        f"NETWORK_LOG={str(log)!r}\n"
        "zip_valid() { return 1; }\n"
        "ensure_wget_zip() { echo wget >> \"$NETWORK_LOG\"; }\n"
        "ensure_pget_zip() { echo pget >> \"$NETWORK_LOG\"; }\n"
        "demand_environment_complete() { return 1; }\n"
        "file_count() { echo 0; }\n"
        "safe_extract_zip() { return 91; }\n"
        "publish_staged_directory() { return 92; }\n"
        "ZEN=https://example.invalid\n"
    )
    demand = subprocess.run(
        [
            "bash",
            "-c",
            common
            + "DEMAND_ENVIRONMENTS=(DKITCHEN)\n"
            + demand_function
            + "\ndownload_demand\n",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    mimii = subprocess.run(
        ["bash", "-c", common + mimii_function + "\ndownload_mimii\n"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    dns = subprocess.run(
        [
            "bash",
            "-c",
            "set -u\n"
            "archive_cache_argument_count=3\n"
            f"NETWORK_LOG={str(log)!r}\n"
            "declare -A DL=([shard000.tar.bz2]=https://example.invalid/dns)\n"
            "declare -A DEST=([shard000.tar.bz2]=dns_fullband)\n"
            "dns_marker_complete() { return 1; }\n"
            "bzip2() { return 1; }\n"
            "start_download() { echo dns >> \"$NETWORK_LOG\"; }\n"
            + dns_loop,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert demand.returncode != 0
    assert mimii.returncode != 0
    assert dns.returncode != 0
    assert not log.exists()


def test_bootstrap_full_octave_requires_external_highrate_source_evidence_before_setup(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    missing = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--full-octave",
        "--no-update",
    )
    orphan = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--full-octave-highrate-machine-evidence",
        "results/provenance/bsd35k.json",
        "--expected-full-octave-highrate-machine-evidence-sha256",
        "b" * 64,
        "--no-update",
    )
    preflight = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--full-octave",
        "--full-octave-highrate-machine-evidence",
        "results/provenance/bsd35k.json",
        "--expected-full-octave-highrate-machine-evidence-sha256",
        "b" * 64,
        "--no-update",
        "--preflight-only",
    )

    assert missing.returncode == 2
    assert orphan.returncode == 2
    assert preflight.returncode == 2
    assert "high-rate machine evidence" in missing.stderr
    assert "--full-octave와 함께만" in orphan.stderr
    assert "--preflight-only" in preflight.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_bootstrap_reuse_decoder_audit_requires_both_external_sha_anchors_before_setup(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)

    missing = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--reuse-decoder-audit",
        "--no-update",
    )
    orphan = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--expected-decoder-audit-sha256",
        "b" * 64,
        "--no-update",
    )

    assert missing.returncode == 2
    assert orphan.returncode == 2
    assert "--expected-decoder-audit-sha256" in missing.stderr
    assert "--reuse-decoder-audit" in orphan.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_cache_preflight_requires_reuse_anchors_and_rejects_other_modes_before_setup(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    common = (
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
    )
    reuse = (
        "--reuse-decoder-audit",
        "--expected-decoder-audit-sha256",
        "b" * 64,
        "--expected-decoder-audit-file-sha256",
        "c" * 64,
    )

    missing_reuse = _run_bootstrap_gate(
        root, *common, "--cache-preflight-only", "--no-update"
    )
    missing_cache = _run_bootstrap_gate(
        root, *common, *reuse, "--cache-preflight-only", "--no-update"
    )
    preflight_conflict = _run_bootstrap_gate(
        root,
        *common,
        *reuse,
        "--cache-preflight-only",
        "--preflight-only",
        "--no-update",
    )
    full_octave_conflict = _run_bootstrap_gate(
        root,
        *common,
        *reuse,
        "--cache-preflight-only",
        "--full-octave",
        "--no-update",
    )
    duplicate = _run_bootstrap_gate(
        root,
        *common,
        *reuse,
        "--cache-preflight-only",
        "--cache-preflight-only",
        "--no-update",
    )

    assert missing_reuse.returncode == 2
    assert missing_cache.returncode == 2
    assert preflight_conflict.returncode == 2
    assert full_octave_conflict.returncode == 2
    assert duplicate.returncode == 2
    assert "--reuse-decoder-audit" in missing_reuse.stderr
    assert "archive cache root/manifest" in missing_cache.stderr
    assert "--preflight-only" in preflight_conflict.stderr
    assert "--full-octave" in full_octave_conflict.stderr
    assert "한 번만 지정" in duplicate.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_cache_preflight_runs_full_read_only_cache_verification_in_order(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    environment, transfer_sha, call_log, freeze, marker, cache_args = (
        _make_cache_preflight_runtime(root, commit)
    )
    freeze_before = freeze.read_bytes()
    freeze_mtime_before = freeze.stat().st_mtime_ns
    marker_inode_before = marker.stat().st_ino
    marker_mtime_before = marker.stat().st_mtime_ns
    status_root = tmp_path / "bootstrap-status"
    status_root.mkdir()

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--expected-transfer-manifest-sha256",
        transfer_sha,
        "--reuse-decoder-audit",
        "--expected-decoder-audit-sha256",
        "d" * 64,
        "--expected-decoder-audit-file-sha256",
        "e" * 64,
        "--raw-hash-workers",
        "8",
        *cache_args,
        "--cache-preflight-only",
        "--status-root",
        str(status_root),
        "--no-update",
        extra_env=environment,
        timeout_seconds=30,
    )

    assert result.returncode == 0, result.stderr
    assert "[cache preflight] PASS" in result.stdout
    assert "=== [2/6]" not in result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    transfer_calls = [
        index
        for index, call in enumerate(calls)
        if "elice_transfer_manifest.json" in call
    ]
    collect = next(
        index for index, call in enumerate(calls) if "--collect-only" in call
    )
    focused = next(
        index
        for index, call in enumerate(calls)
        if " -m pytest " in f" {call} " and "--collect-only" not in call
    )
    first_raw = next(
        index
        for index, call in enumerate(calls)
        if "data/raw/noise/dns_fullband 16000" in call
    )
    audit = next(
        index
        for index, call in enumerate(calls)
        if "verify_decoder_audit_reuse.py" in call
    )
    held_raw = next(
        index
        for index, call in enumerate(calls)
        if "public_archive_cache.py verify-consumed-raw" in call
    )
    assert len(transfer_calls) == 2
    assert (
        transfer_calls[0]
        < collect
        < focused
        < first_raw
        < audit
        < held_raw
        < transfer_calls[1]
    )
    assert not any("public_archive_cache.py consume" in call for call in calls)
    assert not any("--decoder-projection-only" in call for call in calls)
    assert not any("-m pip freeze" in call for call in calls)
    assert freeze.read_bytes() == freeze_before
    assert freeze.stat().st_mtime_ns == freeze_mtime_before
    assert marker.stat().st_ino == marker_inode_before
    assert marker.stat().st_mtime_ns == marker_mtime_before
    status_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(status_root.glob("*.json"))
    ]
    assert {payload["stage"] for payload in status_payloads} == {
        "source_preflight",
        "archive_cache_anchor",
        "environment",
        "early_pytest",
        "cache_verification",
    }
    for payload in status_payloads:
        assert payload["state"] == "complete"
        assert payload["exit_code"] == 0
        assert payload["ended_at_epoch"] >= payload["started_at_epoch"]
        assert payload["elapsed_seconds"] >= 0
        assert payload["expected_commit"] == commit
    for forbidden in (
        root / "data/manifests/canonical_v4",
        root / "data/manifests/recorded_qa.json",
        root / "data/manifests/elice_bootstrap_receipt.json",
    ):
        assert not forbidden.exists()
    assert not list(root.rglob("*.part"))


def test_cache_preflight_missing_existing_venv_never_runs_setup_or_creates_it(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    environment, transfer_sha, call_log, _freeze, _marker, cache_args = (
        _make_cache_preflight_runtime(root, commit)
    )
    shutil.rmtree(root / ".venv")
    status_root = tmp_path / "bootstrap-status"
    status_root.mkdir()

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--expected-transfer-manifest-sha256",
        transfer_sha,
        "--reuse-decoder-audit",
        "--expected-decoder-audit-sha256",
        "d" * 64,
        "--expected-decoder-audit-file-sha256",
        "e" * 64,
        *cache_args,
        "--cache-preflight-only",
        "--status-root",
        str(status_root),
        "--no-update",
        extra_env=environment,
        timeout_seconds=30,
    )

    assert result.returncode == 1
    assert "이미 완성된 exact venv" in result.stderr
    assert not (root / ".venv").exists()
    assert not call_log.exists()
    assert "=== [early gate]" not in result.stdout
    assert "=== [2/6]" not in result.stdout
    states = {
        payload["stage"]: payload
        for payload in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in status_root.glob("*.json")
        )
    }
    assert states["source_preflight"]["state"] == "complete"
    assert states["environment"]["state"] == "failed"
    assert states["environment"]["exit_code"] == 1


def test_early_focused_failure_stops_before_raw_and_decoder_audit(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    environment, transfer_sha, call_log, _freeze, _marker, cache_args = (
        _make_cache_preflight_runtime(root, commit)
    )
    environment["FAKE_FOCUSED_EXIT"] = "43"

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--expected-transfer-manifest-sha256",
        transfer_sha,
        "--reuse-decoder-audit",
        "--expected-decoder-audit-sha256",
        "d" * 64,
        "--expected-decoder-audit-file-sha256",
        "e" * 64,
        *cache_args,
        "--cache-preflight-only",
        "--no-update",
        extra_env=environment,
        timeout_seconds=30,
    )

    assert result.returncode == 1
    assert "조기 pytest collection/Elice 핵심 회귀 실패" in result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert any("--collect-only" in call for call in calls)
    assert any(
        " -m pytest " in f" {call} " and "--collect-only" not in call
        for call in calls
    )
    assert not any("data/raw/noise/dns_fullband" in call for call in calls)
    assert not any("verify_decoder_audit_reuse.py" in call for call in calls)
    assert "=== [2/6]" not in result.stdout
    assert not (root / "data/manifests/canonical_v4").exists()


def test_bootstrap_requires_nonempty_holdout_before_setup(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--no-update",
    )

    assert result.returncode == 1
    assert "held-out manifest가 없습니다" in result.stderr
    assert "canonical holdout" in result.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_bootstrap_rejects_dirty_tree_even_at_expected_commit(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    (root / "marker.txt").write_text("dirty\n", encoding="utf-8")

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        "a" * 64,
        "--no-update",
    )

    assert result.returncode == 1
    assert "작업 트리에 로컬 변경" in result.stderr
    assert not (root / "data").exists()
    assert not (root / ".venv").exists()


def test_bootstrap_rejects_truncated_holdout_before_setup(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout = root / "data/manifests/recorded_holdout.json"
    holdout.parent.mkdir(parents=True)
    holdout.write_text('{"families": {"speech": ["cut.flac"]}', encoding="utf-8")

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        _sha256(holdout),
        "--no-update",
    )

    assert result.returncode == 1
    assert "JSON이 잘렸거나 손상" in result.stderr
    assert "환경 설치나 데이터 다운로드를 시작하지 않습니다" in result.stderr
    assert not (root / ".venv").exists()


def test_bootstrap_rejects_holdout_hash_mismatch_before_schema_or_setup(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    wrong = ("0" if holdout_sha[0] != "0" else "1") + holdout_sha[1:]

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        wrong,
        "--no-update",
    )

    assert result.returncode == 1
    assert "holdout SHA-256 불일치" in result.stderr
    assert holdout.is_file()
    assert not (root / ".venv").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["families"].pop("machine"), "네 계열"),
        (lambda data: data["families"].__setitem__("music", []), "비어 있지 않은"),
            (
                lambda data: data["families"]["music"].append(
                    "1-2-0001.flac"
                ),
            "서로 다른 family",
        ),
        (lambda data: data.__setitem__("total_clips", 999), "total_clips 불일치"),
        (lambda data: data.pop("provenance_report"), "provenance_report"),
    ],
)
def test_bootstrap_rejects_malformed_holdout_schema_before_setup(
    tmp_path: Path, mutation, message: str
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, _ = _write_canonical_holdout_bundle(root)
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    mutation(payload)
    holdout.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        _sha256(holdout),
        "--no-update",
    )

    assert result.returncode == 1
    assert message in result.stderr
    assert not (root / ".venv").exists()


def test_bootstrap_rejects_provenance_or_source_csv_tampering(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    (root / "data/source_pool/sources.csv").write_text("tampered\n", encoding="utf-8")

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
    )

    assert result.returncode == 1
    assert "sources.csv SHA-256 불일치" in result.stderr
    assert holdout.is_file()
    assert not (root / ".venv").exists()


def test_bootstrap_rejects_noncanonical_historical_builder_provenance(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, _holdout_sha = _write_canonical_holdout_bundle(root)
    holdout_payload = json.loads(holdout.read_text(encoding="utf-8"))
    report_path = root / holdout_payload["provenance_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["historical_builders"]["v1"]["commit"] = "a" * 40
    forged = json.dumps(report).encode("utf-8")
    forged_sha = hashlib.sha256(forged).hexdigest()
    forged_relative = (
        "results/provenance/"
        f"source_pool_provenance_report.{forged_sha}.json"
    )
    (root / forged_relative).write_bytes(forged)
    holdout_payload["provenance_report"] = forged_relative
    holdout_payload["provenance_report_sha256"] = forged_sha
    holdout.write_text(json.dumps(holdout_payload), encoding="utf-8")
    holdout_sha = _sha256(holdout)

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
    )

    assert result.returncode == 1
    assert "고정된 canonical 값과 다릅니다" in result.stderr
    assert not (root / ".venv").exists()


def test_bootstrap_rejects_forged_recorded_tree_before_after_digest(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, _ = _write_canonical_holdout_bundle(root)
    holdout_payload = json.loads(holdout.read_text(encoding="utf-8"))
    report = json.loads(
        (root / holdout_payload["provenance_report"]).read_text(encoding="utf-8")
    )
    report["recorded_tree_protection"]["after_sha256"] = "0" * 64
    report_bytes = json.dumps(report).encode("utf-8")
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    report_relative = (
        "results/provenance/"
        f"source_pool_provenance_report.{report_sha}.json"
    )
    (root / report_relative).write_bytes(report_bytes)
    holdout_payload["provenance_report"] = report_relative
    holdout_payload["provenance_report_sha256"] = report_sha
    holdout.write_text(json.dumps(holdout_payload), encoding="utf-8")

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        _sha256(holdout),
        "--no-update",
        "--preflight-only",
    )

    assert result.returncode == 1
    assert "before/after metadata digest" in result.stderr


@pytest.mark.parametrize(
    ("missing_relative", "message"),
    [
        ("data/source_pool/sources.csv", "sources_csv[0] 파일이 없습니다"),
        (
            "__REPORT__",
            "provenance_report 파일이 없습니다",
        ),
    ],
)
def test_bootstrap_requires_full_canonical_provenance_bundle_before_setup(
    tmp_path: Path, missing_relative: str, message: str
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    relative = missing_relative
    if relative == "__REPORT__":
        relative = json.loads(holdout.read_text(encoding="utf-8"))["provenance_report"]
    (root / relative).unlink()

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
    )

    assert result.returncode == 1
    assert message in result.stderr
    assert not (root / ".venv").exists()


def test_bootstrap_valid_preflight_never_starts_training_or_setup(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    assert (root / "src/deep_anc/data/holdout_contract.py").is_file(), (
        "positive preflight fixture는 nested tracked blob 재귀 열거를 반드시 포함한다"
    )
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
        "--preflight-only",
    )

    assert result.returncode == 0, result.stderr
    assert "환경/데이터는 변경하지 않았습니다" in result.stdout
    assert not (root / ".venv").exists()
    assert not (root / ".git/bootstrap_all.lock").exists()


def test_bootstrap_rejects_stale_static_node_before_bundle_or_raw_scan(
    tmp_path: Path,
):
    root, _old_commit = _make_bootstrap_git_repo(tmp_path)
    registry = root / "src/example_registry.py"
    registry.write_text(
        'NEGATIVE = "tests/test_example.py::test_renamed"\n',
        encoding="utf-8",
    )
    target = root / "tests/test_example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def test_current_name():\n    pass\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", str(registry.relative_to(root)), str(target.relative_to(root))],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "stale static node fixture"],
        cwd=root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
    )

    assert result.returncode == 1
    assert "[FAIL] static pytest node reference audit" in result.stderr
    assert "test function not found: test_renamed" in result.stderr
    assert "[holdout] canonical 계약 확인" not in result.stdout
    assert "[transfer anchor]" not in result.stdout
    assert "[hardware]" not in result.stdout
    assert "raw scan" not in result.stdout
    assert "=== [1/6]" not in result.stdout
    assert not (root / ".venv").exists()


def test_normal_bootstrap_requires_transfer_manifest_before_hardware_or_setup(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
    )
    assert result.returncode == 2
    assert "--expected-transfer-manifest-sha256" in result.stderr
    assert not (root / ".venv").exists()


def test_normal_bootstrap_rejects_transfer_anchor_before_hardware_or_setup(
    tmp_path: Path,
):
    """fresh system Python은 full audio stack 없이 bad anchor를 먼저 막아야 한다."""

    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    transfer = root / "data/manifests/elice_transfer_manifest.json"
    transfer.write_bytes(b'{"schema_version":1,"files":[]}\n')
    actual_sha = _sha256(transfer)
    wrong_sha = ("0" if actual_sha[0] != "0" else "1") + actual_sha[1:]

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--expected-transfer-manifest-sha256",
        wrong_sha,
        "--no-update",
    )

    assert result.returncode == 1
    assert "transfer manifest 외부 SHA-256 anchor 불일치" in result.stderr
    assert "nvidia-smi" not in result.stderr
    assert not (root / ".venv").exists()


def test_content_addressed_bundle_remains_valid_across_reaudit_and_audit_artifact(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    first_holdout, first_sha = _write_canonical_holdout_bundle(root)
    first_holdout_bytes = first_holdout.read_bytes()
    first_payload = json.loads(first_holdout.read_text(encoding="utf-8"))
    canonical_path = root / first_payload["provenance_report"]
    canonical_bytes = canonical_path.read_bytes()

    # audit-only/FAIL은 별도 content-addressed evidence로 생겨도 canonical pointer를
    # 바꾸지 않는다. 같은 입력의 실제 재감사도 같은 bundle bytes로 수렴한다.
    audit_bytes = b'{"mode":"audit_only","status":"FAIL"}\n'
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    (root / "results/provenance" / f"source_pool_provenance_audit.{audit_sha}.json").write_bytes(
        audit_bytes
    )
    second_holdout, second_sha = _write_canonical_holdout_bundle(root)
    assert second_sha == first_sha
    assert second_holdout.read_bytes() == first_holdout_bytes
    assert canonical_path.read_bytes() == canonical_bytes

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        second_sha,
        "--no-update",
        "--preflight-only",
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_rejects_symlinked_content_addressed_report(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    report = root / payload["provenance_report"]
    saved = report.parent / "saved-report-copy.json"
    saved.write_bytes(report.read_bytes())
    report.unlink()
    report.symlink_to(saved)

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
        "--preflight-only",
    )
    assert result.returncode == 1
    assert "symlink" in result.stderr


def test_bootstrap_rejects_assume_unchanged_hidden_tracked_mutation(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
    (root / "marker.txt").write_text("hidden mutation\n", encoding="utf-8")
    subprocess.run(
        ["git", "update-index", "--assume-unchanged", "marker.txt"],
        cwd=root,
        check=True,
    )

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        holdout_sha,
        "--no-update",
        "--preflight-only",
    )

    assert result.returncode == 1
    assert "assume-unchanged/skip-worktree" in result.stderr
    assert not (root / ".venv").exists()


def test_bootstrap_rejects_replace_refs_and_legacy_grafts(tmp_path: Path):
    for escape in ("replace", "grafts"):
        case_root = tmp_path / escape
        case_root.mkdir()
        root, commit = _make_bootstrap_git_repo(case_root)
        _holdout, holdout_sha = _write_canonical_holdout_bundle(root)
        if escape == "replace":
            tree = subprocess.run(
                ["git", "rev-parse", f"{commit}^{{tree}}"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            replacement = subprocess.run(
                ["git", "commit-tree", tree],
                cwd=root,
                input="replacement\n",
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            subprocess.run(["git", "replace", commit, replacement], cwd=root, check=True)
            expected_message = "git replace ref"
        else:
            grafts = root / ".git/info/grafts"
            grafts.parent.mkdir(parents=True, exist_ok=True)
            grafts.write_text(commit + "\n", encoding="utf-8")
            expected_message = "legacy git grafts"

        result = _run_bootstrap_gate(
            root,
            "--expected-commit",
            commit,
            "--expected-holdout-sha256",
            holdout_sha,
            "--no-update",
            "--preflight-only",
        )
        assert result.returncode == 1
        assert expected_message in result.stderr


def test_bootstrap_rejects_minimal_forged_pass_report_even_when_rebound(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, _ = _write_canonical_holdout_bundle(root)
    forged = json.dumps(
        {
            "schema_version": 1,
            "authority": "historical_builder_reproduction_plus_pcm_validation",
            "status": "PASS",
        }
    ).encode("utf-8")
    forged_sha = hashlib.sha256(forged).hexdigest()
    forged_relative = (
        "results/provenance/"
        f"source_pool_provenance_report.{forged_sha}.json"
    )
    (root / forged_relative).write_bytes(forged)
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    payload["provenance_report"] = forged_relative
    payload["provenance_report_sha256"] = forged_sha
    holdout.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        _sha256(holdout),
        "--no-update",
        "--preflight-only",
    )

    assert result.returncode == 1
    assert "selection 증거가 없습니다" in result.stderr


def test_bootstrap_rejects_holdout_family_not_equal_to_selected_csv_union(tmp_path: Path):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, _ = _write_canonical_holdout_bundle(root)
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    payload["families"]["music"] = ["forged-music.wav"]
    holdout.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        _sha256(holdout),
        "--no-update",
        "--preflight-only",
    )

    assert result.returncode == 1
    assert "source_rows→clips exact 합집합" in result.stderr


def test_bootstrap_recomputes_lineage_and_rejects_consistently_forged_group_ids(
    tmp_path: Path,
):
    root, commit = _make_bootstrap_git_repo(tmp_path)
    holdout, _ = _write_canonical_holdout_bundle(root)
    holdout_payload = json.loads(holdout.read_text(encoding="utf-8"))
    report = json.loads(
        (root / holdout_payload["provenance_report"]).read_text(encoding="utf-8")
    )
    lineage = report["lineage_contract"]
    old_component = next(
        name for name in lineage["components"] if name.startswith("machine-lineage-")
    )
    members = lineage["components"].pop(old_component)
    forged_component = "machine-lineage-forged000000"
    lineage["components"][forged_component] = members
    lineage["component_membership_sha256"] = hashlib.sha256(
        json.dumps(
            lineage["components"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    regrouped = root / "data/manifests/recorded_regrouped.jsonl"
    rows = [json.loads(line) for line in regrouped.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["session_id"] == "machine-session":
            row["group_id"] = forged_component
    regrouped.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    lineage["regrouped_manifest_sha256"] = _sha256(regrouped)

    report_bytes = json.dumps(report).encode()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    report_relative = (
        "results/provenance/"
        f"source_pool_provenance_report.{report_sha}.json"
    )
    (root / report_relative).write_bytes(report_bytes)
    holdout_payload["provenance_report"] = report_relative
    holdout_payload["provenance_report_sha256"] = report_sha
    holdout.write_text(json.dumps(holdout_payload), encoding="utf-8")

    result = _run_bootstrap_gate(
        root,
        "--expected-commit",
        commit,
        "--expected-holdout-sha256",
        _sha256(holdout),
        "--no-update",
        "--preflight-only",
    )
    assert result.returncode == 1
    assert "재계산과 다릅니다" in result.stderr


def test_bootstrap_default_is_environment_data_only_without_legacy_runner():
    text = ELICE_SCRIPTS[0].read_text(encoding="utf-8")

    assert "START_TRAINING" not in text
    assert "run_parallel_models.sh" not in text
    assert "학습은 시작하지 않음" in text


def test_legacy_launchers_require_an_explicit_acknowledgement():
    pretrain = subprocess.run(
        ["bash", str(ELICE_SCRIPTS[3])],
        capture_output=True,
        text=True,
        check=False,
    )
    parallel = subprocess.run(
        ["bash", str(ELICE_SCRIPTS[2])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert pretrain.returncode == 2
    assert parallel.returncode == 2
    assert "canonical tiny" in pretrain.stderr
    assert "legacy base/tiny" in parallel.stderr


def _make_fake_runner(
    tmp_path: Path,
    *,
    gpu_count: int,
    train_exit: int,
    keep_gpu0_alive: bool = False,
) -> Path:
    root = tmp_path / "Deep-ANC"
    scripts = root / "scripts/elice"
    scripts.mkdir(parents=True)
    shutil.copy2(ELICE_SCRIPTS[2], scripts / "run_parallel_models.sh")

    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    keep_alive_script = (
        'if [ "${CUDA_VISIBLE_DEVICES:-}" = "0" ]; then\n'
        "  trap 'exit 0' TERM INT\n"
        "  (trap '' TERM INT; sleep 30) &\n"
        "  child=$!\n"
        "  printf '%s\\n' \"$child\" > runs/fake_orphan_child.pid\n"
        "  wait \"$child\"\n"
        "  exit 0\n"
        "fi\n"
        if keep_gpu0_alive
        else ""
    )
    python.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        f"  echo {gpu_count}\n"
        "  exit 0\n"
        "fi\n"
        f"{keep_alive_script}"
        'echo "모의 학습 프로세스 종료" >&2\n'
        f"exit {train_exit}\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return root


def test_parallel_runner_does_not_overwrite_existing_log(tmp_path: Path):
    root = _make_fake_runner(tmp_path, gpu_count=1, train_exit=0)
    log = root / "runs/train_base_corrected.log"
    log.parent.mkdir()
    log.write_text("보존할 로그\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", "scripts/elice/run_parallel_models.sh"],
        cwd=root,
        env={**os.environ, "DEEP_ANC_ALLOW_LEGACY_DIAGNOSTIC": "1"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "자동 덮어쓰지 않습니다" in result.stderr
    assert log.read_text(encoding="utf-8") == "보존할 로그\n"
    assert not (root / "runs/train_base_corrected.pid").exists()


def test_parallel_runner_reports_immediate_process_exit(tmp_path: Path):
    root = _make_fake_runner(tmp_path, gpu_count=2, train_exit=42)

    result = subprocess.run(
        ["bash", "scripts/elice/run_parallel_models.sh"],
        cwd=root,
        env={**os.environ, "DEEP_ANC_ALLOW_LEGACY_DIAGNOSTIC": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode != 0
    assert "base 학습 프로세스" in result.stderr
    assert "tiny 학습 프로세스" in result.stderr
    assert "모의 학습 프로세스 종료" in result.stderr
    rollback_dirs = list((root / "runs").glob("failed_start_*"))
    assert len(rollback_dirs) == 1
    assert (rollback_dirs[0] / "train_base_corrected.pid").is_file()
    assert (rollback_dirs[0] / "train_tiny_corrected.pid").is_file()
    assert not (root / "runs/train_base_corrected.pid").exists()
    assert not (root / "runs/train_tiny_corrected.pid").exists()


def test_parallel_runner_rolls_back_survivor_on_partial_start_failure(tmp_path: Path):
    root = _make_fake_runner(
        tmp_path,
        gpu_count=2,
        train_exit=42,
        keep_gpu0_alive=True,
    )

    result = subprocess.run(
        ["bash", "scripts/elice/run_parallel_models.sh"],
        cwd=root,
        env={**os.environ, "DEEP_ANC_ALLOW_LEGACY_DIAGNOSTIC": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode != 0
    assert "tiny 학습 프로세스" in result.stderr
    assert "시작 작업을 되돌립니다" in result.stderr
    rollback_dirs = list((root / "runs").glob("failed_start_*"))
    assert len(rollback_dirs) == 1
    base_pid = int((rollback_dirs[0] / "train_base_corrected.pid").read_text())
    orphan_pid = int((root / "runs/fake_orphan_child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(base_pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(orphan_pid, 0)
    assert not (root / "runs/train_base_corrected.log").exists()
    assert not (root / "runs/pretrain_base_corrected").exists()
