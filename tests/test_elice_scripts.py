"""Elice 부트스트랩/학습 시작 셸의 안전 불변식."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
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


REPO_ROOT = Path(__file__).resolve().parents[1]
ELICE_SCRIPTS = (
    REPO_ROOT / "scripts/elice/bootstrap_all.sh",
    REPO_ROOT / "scripts/elice/setup_env.sh",
    REPO_ROOT / "scripts/elice/run_parallel_models.sh",
    REPO_ROOT / "scripts/elice/run_pretrain.sh",
    REPO_ROOT / "scripts/elice/run_structure_search.sh",
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
    assert "--raw-hash-workers" in text
    assert '--expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256"' in text
    assert "--expected-transfer-manifest-sha256" in text
    assert 'TRANSFER_MANIFEST="$REPO/data/manifests/elice_transfer_manifest.json"' in text
    assert "--query-gpu=name,memory.total" in text
    assert "minimum_mib = 79 * 1024" in text
    assert "minimum_total_bytes=$((128 * 1024 * 1024 * 1024 - 128 * 1024 * 1024))" in text
    assert "minimum_available_bytes=$((96 * 1024 * 1024 * 1024))" in text
    assert "public corpus가 이미 완전하므로 재개 시 96GiB staging 예산 검사를 건너뜁니다" in text
    assert 'df -B1 --output=size,avail "$REPO"' in text
    assert 'str(torch.__version__) != "2.5.1+cu121"' in text
    assert 'str(torch.version.cuda) != "12.1"' in text
    assert 'ENVIRONMENT_RECEIPT="$REPO/.venv/environment-freeze.txt"' in text
    assert "pip freeze --all" in text
    assert "grep -Fxq 'torch==2.5.1+cu121'" in text
    assert 'BOOTSTRAP_RECEIPT="$REPO/data/manifests/elice_bootstrap_receipt.json"' in text
    assert '"recorded_aggregate_sha256": summary["recorded_aggregate_sha256"]' in text
    assert '"freeze_receipt_sha256": environment.sha256' in text
    assert "data.bootstrap_receipt" in text
    preflight_exit = text.index('if [ "$PREFLIGHT_ONLY" -eq 1 ]')
    elice_hardware_call = text.index("if ! verify_transfer_bundle || ! hardware_storage_preflight")
    assert preflight_exit < elice_hardware_call
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
    assert text.index('mv -f "${ENVIRONMENT_RECEIPT}.building"') < text.index(
        'touch "$SETUP_MARKER"'
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


def _run_bootstrap_gate(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DEEP_ANC_BOOTSTRAP_REPO"] = str(root)
    return subprocess.run(
        ["bash", str(ELICE_SCRIPTS[0]), *args],
        cwd=root.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_bootstrap_derives_default_repo_from_its_own_path(tmp_path: Path):
    """Elice clone 이름이 Deep-ANC가 아니어도 하드코딩 경로로 실패하지 않는다."""

    root, _commit = _make_bootstrap_git_repo(tmp_path)
    underscore_root = tmp_path / "Deep_ANC"
    root.rename(underscore_root)
    root = underscore_root
    script = root / "scripts/elice/bootstrap_all.sh"
    script.parent.mkdir(parents=True)
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
