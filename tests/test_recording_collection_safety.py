"""실측 수집 오케스트레이터의 무출력 안전 계약."""

from __future__ import annotations

import builtins
import contextlib
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deep_anc.config import REPO_ROOT
from deep_anc.realtime.noise_gen import NoiseProgram, render_recording_file_window


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BATCH = _load("scripts/data/record_session_batch.py", "record_session_batch_safety_test")
RECORD = _load("scripts/data/record_duct.py", "record_duct_safety_test")


def _fake_campaign_args(module, monkeypatch, *, root: Path) -> list[str]:
    receipt = "results/recording_level_campaigns/recording-level-" + "a" * 64 + "/campaign.json"
    digest = "b" * 64
    hardware = "configs/hardware_jetson.yaml"
    monkeypatch.setattr(
        module,
        "validate_recording_level_campaign",
        lambda **_kwargs: {
            "campaign_id": "recording-level-" + "a" * 64,
            "receipt_path": receipt,
            "receipt_size": 123,
            "receipt_sha256": digest,
            "payload": {"hardware": {"config": {"path": hardware}}},
        },
    )
    return [
        "--hardware",
        str(root / hardware),
        "--recording-level-campaign",
        receipt,
        "--recording-level-campaign-sha256",
        digest,
        "--confirm-same-amplifier-setting",
    ]


def _source_plan(tmp_path: Path, *, split: str | None = "train", seconds: float = 15.0):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(2_000, dtype=np.float32), 100, subtype="FLOAT")
    plan = tmp_path / "sources.csv"
    fields = ["path", "seconds", "source_family", "group_id", "lineage_key"]
    if split is not None:
        fields.append("split")
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        row = {
            "path": str(source),
            "seconds": str(seconds),
            "source_family": "speech",
            "group_id": "speaker-book-001",
            "lineage_key": "speaker-book-001",
        }
        if split is not None:
            row["split"] = split
        writer.writerow(row)
    return source, plan


def test_batch_retry_is_explicit_opt_in():
    args = BATCH.build_parser().parse_args([])
    assert args.retry_once is False
    assert args.no_retry is False
    assert args.amplitude == BATCH.CANONICAL_RECORDING_AMPLITUDE == 0.06
    assert BATCH.build_parser().parse_args(["--retry-once"]).retry_once is True


def test_batch_dry_run_does_not_create_files_or_spawn_child(tmp_path, monkeypatch, capsys):
    _, plan = _source_plan(tmp_path)
    out_root = tmp_path / "must_not_exist"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run이 자식 process를 실행했습니다")

    monkeypatch.setattr(BATCH.subprocess, "Popen", forbidden)
    result = BATCH.main(
        ["--sources", str(plan), "--out-root", str(out_root), "--dry-run"]
    )
    assert result == 0
    assert not out_root.exists()
    output = capsys.readouterr().out
    assert "예상 audible: 15.0초" in output
    assert "예상 connected 상한" in output
    assert "재생 amplitude: 0.06" in output
    assert "공용 peak 안전 상한 0.15" in output
    assert "자동 재시도: 없음" in output
    assert "source-list SHA256" in output
    assert "lineage=speaker-book-001" in output
    assert "split=train" in output


def test_noncanonical_diagnostic_batch_keeps_explicit_level_override(
    tmp_path, capsys
):
    _, plan = _source_plan(tmp_path)
    assert BATCH.main(
        ["--sources", str(plan), "--amplitude", "0.15", "--dry-run"]
    ) == 0
    assert "재생 amplitude: 0.15" in capsys.readouterr().out


def test_batch_requires_preassigned_split_before_any_output(tmp_path, monkeypatch):
    _, plan = _source_plan(tmp_path, split=None)
    monkeypatch.setattr(
        BATCH.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("자식 실행 금지")),
    )
    assert BATCH.main(["--sources", str(plan), "--dry-run"]) == 2
    assert BATCH.main(
        ["--sources", str(plan), "--preassigned-split", "test", "--dry-run"]
    ) == 0


def test_batch_live_requires_all_three_confirmations_before_writes(tmp_path):
    _, plan = _source_plan(tmp_path)
    out_root = tmp_path / "recorded"
    assert BATCH.main(["--sources", str(plan), "--out-root", str(out_root)]) == 2
    assert not out_root.exists()


def test_child_stdout_stderr_are_streamed_and_logged(tmp_path, capsys):
    log = tmp_path / "child.log"
    result = BATCH.run_child_live(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('live-out'); print('live-err', file=sys.stderr)",
        ],
        timeout_seconds=5.0,
        log_path=log,
    )
    assert result.returncode == 0
    assert not result.timed_out
    assert "live-out" in result.stdout
    assert "live-err" in result.stderr
    captured = capsys.readouterr()
    assert "live-out" in captured.out
    assert "live-err" in captured.err
    log_text = log.read_text(encoding="utf-8")
    assert "live-out" in log_text and "live-err" in log_text


def test_child_has_hard_timeout(tmp_path):
    result = BATCH.run_child_live(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_seconds=0.05,
        log_path=tmp_path / "timeout.log",
    )
    assert result.returncode == 124
    assert result.timed_out


def test_record_duct_dry_run_never_imports_sounddevice_or_writes(tmp_path, monkeypatch):
    out_root = tmp_path / "recorded"
    failed_root = tmp_path / "failed"
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("dry-run이 sounddevice를 import했습니다")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert RECORD.main(
        [
            "--program", "tone", "--seconds", "15", "--dry-run",
            "--out-root", str(out_root), "--failed-root", str(failed_root),
        ]
    ) == 0
    assert not out_root.exists()
    assert not failed_root.exists()


def test_record_duct_requires_three_confirmations_before_audio_or_writes(tmp_path, monkeypatch):
    out_root = tmp_path / "recorded"
    failed_root = tmp_path / "failed"
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("확인 누락인데 sounddevice를 import했습니다")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert RECORD.main(
        [
            "--program", "tone", "--seconds", "15",
            "--out-root", str(out_root), "--failed-root", str(failed_root),
        ]
    ) == 2
    assert not out_root.exists()
    assert not failed_root.exists()


def test_exact_collection_plan_is_bound_in_dry_run(tmp_path, capsys):
    source, plan = _source_plan(tmp_path)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    result = RECORD.main(
        [
            "--program", "file", "--file", str(source), "--seconds", "15",
            "--source-family", "speech", "--group-id", "speaker-book-001",
            "--source-list", str(plan), "--source-list-sha256", digest,
            "--source-row-number", "2", "--lineage-key", "speaker-book-001",
            "--preassigned-split", "train", "--dry-run",
        ]
    )
    assert result == 0
    assert "collection provenance: exact" in capsys.readouterr().out


def test_record_duct_canonical_dry_run_allows_campaign_to_be_issued_afterward(
    tmp_path, capsys
):
    source, plan = _source_plan(tmp_path)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    assert RECORD.main(
        [
            "--program", "file", "--file", str(source), "--seconds", "15",
            "--amplitude", "0.06", "--source-family", "speech",
            "--group-id", "speaker-book-001", "--source-list", str(plan),
            "--source-list-sha256", digest, "--source-row-number", "2",
            "--lineage-key", "speaker-book-001", "--preassigned-split", "train",
            "--require-recording-level-campaign", "--dry-run",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "live canonical 실행에는 fresh recording-level" in output
    assert "[DRY-RUN PASS]" in output


def test_record_duct_canonical_live_refuses_missing_campaign_before_audio(
    tmp_path,
):
    source, plan = _source_plan(tmp_path)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    with pytest.raises(SystemExit) as excinfo:
        RECORD.main(
            [
                "--program", "file", "--file", str(source), "--seconds", "15",
                "--amplitude", "0.06", "--source-family", "speech",
                "--group-id", "speaker-book-001", "--source-list", str(plan),
                "--source-list-sha256", digest, "--source-row-number", "2",
                "--lineage-key", "speaker-book-001", "--preassigned-split", "train",
                "--require-recording-level-campaign",
            ]
        )
    assert excinfo.value.code == 2


def test_record_duct_direct_canonical_path_cannot_omit_v2_source_gain(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(RECORD, "REPO_ROOT", tmp_path)
    source, ordinary_plan = _source_plan(tmp_path)
    canonical_plan = (
        tmp_path / RECORD.SOURCE_PLAN_ROOT / "direct-bypass-fixture.csv"
    )
    canonical_plan.parent.mkdir(parents=True)
    canonical_plan.write_bytes(ordinary_plan.read_bytes())
    digest = hashlib.sha256(canonical_plan.read_bytes()).hexdigest()
    monkeypatch.setattr(
        RECORD,
        "load_yaml",
        lambda _path: {"audio": {"sample_rate": 48_000, "block_size": 256}},
    )
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise AssertionError("v2 authority 거절 전에 sounddevice를 import했습니다")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as excinfo:
        RECORD.main(
            [
                "--program", "file", "--file", str(source), "--seconds", "15",
                "--amplitude", "0.06", "--source-family", "speech",
                "--group-id", "speaker-book-001", "--source-list", str(canonical_plan),
                "--source-list-sha256", digest, "--source-row-number", "2",
                "--lineage-key", "speaker-book-001", "--preassigned-split", "train",
                "--out-root", str(tmp_path / RECORD.ADDITIONS_ROOT / "direct-bypass-fixture"),
                "--confirm-user-present", "--confirm-volume-minimum",
                "--confirm-routing-and-geometry",
            ]
        )
    assert excinfo.value.code == 2


def test_stream_constructor_mutation_is_caught_before_callback_start(tmp_path):
    authority = tmp_path / "authority.json"
    authority.write_text('{"version":1}\n', encoding="utf-8")
    state = {"closed": False, "started": False}

    class FakeStream:
        def __init__(self, **_kwargs):
            authority.write_text('{"version":2}\n', encoding="utf-8")

        def close(self):
            state["closed"] = True

        def start(self):
            state["started"] = True

    class FakeSoundDevice:
        Stream = FakeStream

    with contextlib.ExitStack() as stack:
        guard = stack.enter_context(
            RECORD.RepositoryFileGuard(tmp_path, "authority.json", label="fixture authority")
        )
        with pytest.raises(RuntimeError, match="변경"):
            RECORD._construct_stream_after_authority_check(
                FakeSoundDevice,
                pre_open_check=guard.verify,
                stream_kwargs={},
            )
    assert state == {"closed": True, "started": False}


def test_collection_plan_rejects_wrong_sha(tmp_path):
    source, plan = _source_plan(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        RECORD.main(
            [
                "--program", "file", "--file", str(source), "--seconds", "15",
                "--source-family", "speech", "--group-id", "speaker-book-001",
                "--source-list", str(plan), "--source-list-sha256", "0" * 64,
                "--source-row-number", "2", "--lineage-key", "speaker-book-001",
                "--preassigned-split", "train", "--dry-run",
            ]
        )
    assert excinfo.value.code == 2


def test_declared_source_file_sha_is_checked_by_batch_and_record_duct(tmp_path):
    source, plan = _source_plan(tmp_path)
    rows = list(csv.DictReader(plan.open(encoding="utf-8")))
    rows[0]["source_file_sha256"] = "0" * 64
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert BATCH.main(["--sources", str(plan), "--dry-run"]) == 2
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    with pytest.raises(SystemExit) as excinfo:
        RECORD.main(
            [
                "--program", "file", "--file", str(source), "--seconds", "15",
                "--source-family", "speech", "--group-id", "speaker-book-001",
                "--source-list", str(plan), "--source-list-sha256", plan_sha,
                "--source-row-number", "2", "--lineage-key", "speaker-book-001",
                "--preassigned-split", "train", "--dry-run",
            ]
        )
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "stale_generation_id",
    ("stage1-coverage-v2", "stage1-coverage-v3-gain012"),
)
def test_canonical_batch_rejects_stale_generation_before_audio_open(
    tmp_path, monkeypatch, capsys, stale_generation_id
):
    monkeypatch.setattr(BATCH, "REPO_ROOT", tmp_path)
    _source, plan = _source_plan(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        BATCH.main(
            [
                "--sources",
                str(plan),
                "--canonical-additions-generation",
                stale_generation_id,
                "--dry-run",
            ]
        )
    assert excinfo.value.code == 2
    assert "현행 exact source plan generation-id" in capsys.readouterr().err


def test_canonical_additions_mode_requires_exact_generation_paths_and_header(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(BATCH, "REPO_ROOT", tmp_path)
    generation_id = BATCH.CANONICAL_GENERATION_ID
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(2_000, dtype=np.float32), 100, subtype="FLOAT")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = (
        tmp_path
        / BATCH.SOURCE_PLAN_ROOT
        / f"{generation_id}.csv"
    )
    plan.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(BATCH.ADDITION_SESSION_COUNT):
        rows.append(
            {
                "source_kind": "source_pool_row",
                "path": str(source),
                "seconds": "15.0",
                "start_seconds": "0.0",
                "source_family": "speech",
                "group_id": f"source-group-{index:02d}",
                "lineage_key": f"source-lineage-{index:02d}",
                "split": ("train", "val", "test")[index % 3],
                "source_file_sha256": source_sha,
                "raw_member_path": "",
                "raw_member_sha256": "",
                "raw_member_lineage_key": "",
                "authority_metadata_sha256": "",
                "inventory_path": "",
                "inventory_sha256": "",
                "transform": "identity",
                "transform_repeat_count": "1",
            }
        )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH.SOURCE_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    authoritative_calls = []

    def validate_authoritative_plan(*, repo_root, relative, require_source_files):
        authoritative_calls.append((repo_root, relative, require_source_files))
        snapshot = type(
            "Snapshot",
            (),
            {"sha256": hashlib.sha256(plan.read_bytes()).hexdigest()},
        )()
        canonical_rows = [
            {**row, "source_row_number": index}
            for index, row in enumerate(rows, start=2)
        ]
        return snapshot, canonical_rows, "e" * 64, {"evidence_sha256": "s" * 64}

    monkeypatch.setattr(BATCH, "_read_source_plan", validate_authoritative_plan)
    out_root = tmp_path / BATCH.ADDITIONS_ROOT / generation_id
    canonical_dry_run = [
        "--sources", str(plan),
        "--out-root", str(out_root),
        "--canonical-additions-generation", generation_id,
        "--dry-run",
    ]
    assert BATCH.main(canonical_dry_run) == 0
    with pytest.raises(SystemExit) as excinfo:
        BATCH.main(
            [
                *canonical_dry_run[:-1],
                "--confirm-user-present",
                "--confirm-volume-minimum",
                "--confirm-routing-and-geometry",
            ]
        )
    assert excinfo.value.code == 2
    common = [
        *canonical_dry_run[:-1],
        *_fake_campaign_args(BATCH, monkeypatch, root=tmp_path),
        "--dry-run",
    ]
    assert BATCH.main(common) == 0
    assert authoritative_calls == [
        (
            tmp_path,
            f"{BATCH.SOURCE_PLAN_ROOT}/{generation_id}.csv",
            True,
        )
    ] * 3

    with pytest.raises(SystemExit) as excinfo:
        BATCH.main([*common, "--amplitude", "0.15"])
    assert excinfo.value.code == 2

    outside = tmp_path / "outside-recorded-additions"
    outside.mkdir()
    out_root.parent.mkdir(parents=True, exist_ok=True)
    out_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SystemExit) as excinfo:
        BATCH.main(common)
    assert excinfo.value.code == 2
    out_root.unlink()

    bad_session = out_root / "partial-session"
    bad_session.mkdir(parents=True)
    (bad_session / "session.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        BATCH.main(common)
    assert excinfo.value.code == 2
    (bad_session / "session.json").unlink()
    bad_session.rmdir()
    out_root.rmdir()

    with pytest.raises(SystemExit) as excinfo:
        BATCH.main([*common, "--out-root", str(tmp_path / "wrong")])
    assert excinfo.value.code == 2

    # 같은 19행이어도 exact header에서 authority 열 하나를 빼면 canonical 수집을
    # 시작하지 않는다. 일반 diagnostic CSV의 유연성은 유지한다.
    wrong_fields = [
        field for field in BATCH.SOURCE_PLAN_FIELDS
        if field != "authority_metadata_sha256"
    ]
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=wrong_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit) as excinfo:
        BATCH.main(common)
    assert excinfo.value.code == 2


def test_canonical_batch_qa_failure_is_quarantined_and_exits_nonzero(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(BATCH, "REPO_ROOT", tmp_path)
    generation_id = BATCH.CANONICAL_GENERATION_ID
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(2_000, dtype=np.float32), 100, subtype="FLOAT")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = tmp_path / BATCH.SOURCE_PLAN_ROOT / f"{generation_id}.csv"
    plan.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(BATCH.ADDITION_SESSION_COUNT):
        rows.append(
            {
                "source_kind": "source_pool_row",
                "path": str(source),
                "seconds": "15.0",
                "start_seconds": "0.0",
                "source_family": "speech",
                "group_id": f"source-group-{index:02d}",
                "lineage_key": f"source-lineage-{index:02d}",
                "split": ("train", "val", "test")[index % 3],
                "source_file_sha256": source_sha,
                "raw_member_path": "",
                "raw_member_sha256": "",
                "raw_member_lineage_key": "",
                "authority_metadata_sha256": "",
                "inventory_path": "",
                "inventory_sha256": "",
                "transform": "identity",
                "transform_repeat_count": "1",
            }
        )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH.SOURCE_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    canonical_rows = [
        {**row, "source_row_number": index, "seconds": 15.0, "start_seconds": 0.0}
        for index, row in enumerate(rows, start=2)
    ]
    monkeypatch.setattr(
        BATCH,
        "_read_source_plan",
        lambda **_kwargs: (
            type("Snapshot", (), {"sha256": plan_sha})(),
            canonical_rows,
            "e" * 64,
            {"evidence_sha256": "s" * 64},
        ),
    )
    # 이 테스트의 책임은 이미 열린 live child가 QA 실패했을 때 raw를 격리하는
    # downstream 동작이다. source-gain v1 자체가 canonical live를 차단하는지는
    # test_recording_source_gain.py에서 별도로 실검증한다.
    monkeypatch.setattr(
        BATCH,
        "_validate_batch_source_gain_plan",
        lambda *_args, **_kwargs: {
            "plan_sha256": "g" * 64,
            "payload": {"contract": {"reference_amplitude_millionths": 6_000}},
        },
    )
    monkeypatch.setattr(
        BATCH,
        "_canonical_source_gain_by_row",
        lambda _summary, current_entries: {
            int(entry["source_row_number"]): 0.006 for entry in current_entries
        },
    )

    def fake_child(command, *, timeout_seconds, log_path):
        del timeout_seconds, log_path
        child_root = Path(command[command.index("--out-root") + 1])
        session = child_root / "qa-failed-session"
        session.mkdir()
        (session / "session.json").write_text("{}\n", encoding="utf-8")
        sf.write(
            session / "mics.wav",
            np.zeros((720_000, 2), dtype=np.float32),
            48_000,
            subtype="PCM_32",
        )
        return BATCH.ChildResult(0, "", "", False)

    monkeypatch.setattr(BATCH, "run_child_live", fake_child)
    out_root = tmp_path / BATCH.ADDITIONS_ROOT / generation_id
    result = BATCH.main(
        [
            "--sources",
            str(plan),
            "--out-root",
            str(out_root),
            "--canonical-additions-generation",
            generation_id,
            "--limit",
            "1",
            "--settle-seconds",
            "0",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
            *_fake_campaign_args(BATCH, monkeypatch, root=tmp_path),
        ]
    )
    assert result == 1
    assert not list(out_root.glob("*/session.json"))
    progress = list(
        csv.DictReader((out_root / "batch_progress.csv").open(encoding="utf-8"))
    )
    assert progress[0]["verdict"] == "qa_failed"
    assert progress[0]["seconds"] == "15.0"
    failures = list(
        (tmp_path / "results/recording_failures/record_duct/batch_qa").glob(
            "**/qa-failed-session/session.json"
        )
    )
    assert len(failures) == 1


def test_canonical_resume_rejects_progress_duration_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(BATCH, "REPO_ROOT", tmp_path)
    source_list = tmp_path / "plan.csv"
    source_list.write_text("fixture\n", encoding="utf-8")
    source_sha = "a" * 64
    row = {
        "source_row_number": 2,
        "path": "source.wav",
        "seconds": 15.0,
        "start_seconds": 0.0,
        "source_family": "speech",
        "group_id": "speech-source",
        "lineage_key": "speech-lineage",
        "split": "train",
        "source_file_sha256": source_sha,
    }
    out_root = tmp_path / "recorded"
    session = out_root / "session-001"
    session.mkdir(parents=True)
    (session / "session.json").write_text("{}\n", encoding="utf-8")
    metadata = {
        "session_id": session.name,
        "collection_plan": {
            "status": "exact",
            "source_list": str(source_list),
            "source_list_sha256": "b" * 64,
            "source_row_number": 2,
            "lineage_key": "speech-lineage",
            "preassigned_split": "train",
            "split_source": "csv",
            "source_file_sha256": source_sha,
            "start_seconds": 0.0,
        },
        "program": {
            "type": "file",
            "file": "source.wav",
            "file_start_seconds": 0.0,
        },
        "source_family": "speech",
        "group_id": "speech-source",
        "preassigned_split": "train",
    }
    monkeypatch.setattr(BATCH, "_read_session_metadata", lambda *_a, **_k: metadata)
    monkeypatch.setattr(BATCH, "_validate_session_artifacts", lambda **_kwargs: None)
    with (out_root / "batch_progress.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "verdict",
                "source_row_number",
                "session_id",
                "seconds",
                "source_path",
                "source_file_sha256",
                "source_list_sha256",
                "lineage_key",
                "preassigned_split",
                "start_seconds",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "verdict": "ok",
                "source_row_number": "2",
                "session_id": session.name,
                "seconds": "14.0",
                "source_path": "source.wav",
                "source_file_sha256": source_sha,
                "source_list_sha256": "b" * 64,
                "lineage_key": "speech-lineage",
                "preassigned_split": "train",
                "start_seconds": "0.0",
            }
        )
    with pytest.raises(ValueError, match="exact source plan"):
        BATCH._canonical_already_recorded(
            out_root=out_root,
            rows=[row],
            source_list=source_list,
            source_list_sha256="b" * 64,
        )


def test_failed_raw_and_metadata_are_no_replace(tmp_path):
    failure_dir = RECORD._preserve_failed_capture(
        failed_root=tmp_path / "failed",
        stage="fixture_gate",
        reason="fixture",
        sample_rate=48_000,
        metadata={"collection_plan": {"status": "exact"}},
        mics_raw=np.zeros((256, 2), dtype=np.float32),
        source_raw=np.zeros(256, dtype=np.float32),
    )
    assert (failure_dir / "mics_raw.wav").is_file()
    assert (failure_dir / "source_raw.wav").is_file()
    payload = json.loads((failure_dir / "failure.json").read_text(encoding="utf-8"))
    assert payload["raw_available"] is True
    assert payload["failure_stage"] == "fixture_gate"
    assert {item["path"] for item in payload["artifacts"]} == {
        "mics_raw.wav", "source_raw.wav"
    }
    assert all(len(item["sha256"]) == 64 and item["size_bytes"] > 0 for item in payload["artifacts"])


def test_record_duct_emits_machine_readable_durable_failure_pointer(
    tmp_path, capsys
):
    assert BATCH.FAILURE_RECEIPT_MARKER == RECORD.FAILURE_RECEIPT_MARKER
    failure_dir = RECORD._preserve_failed_capture(
        failed_root=tmp_path / "failed",
        stage="timeline_gate",
        reason="high_band_coherence=0.812345 < 0.90",
        sample_rate=48_000,
        metadata={"collection_plan": {"status": "exact"}},
    )
    stderr = capsys.readouterr().err
    marker_line = next(
        line
        for line in stderr.splitlines()
        if line.startswith(RECORD.FAILURE_RECEIPT_MARKER)
    )
    marker = json.loads(marker_line[len(RECORD.FAILURE_RECEIPT_MARKER) :])
    assert marker == {
        "schema_version": 1,
        "failure_stage": "timeline_gate",
        "failure_reason": "high_band_coherence=0.812345 < 0.90",
        "failure_artifact": str(failure_dir.resolve()),
        "failure_receipt": str((failure_dir / "failure.json").resolve()),
    }


def test_failed_capture_reserved_metadata_cannot_override_receipt_contract(
    tmp_path, capsys
):
    failure_dir = RECORD._preserve_failed_capture(
        failed_root=tmp_path / "failed",
        stage="timeline_gate",
        reason="authoritative reason",
        sample_rate=48_000,
        metadata={
            "schema_version": 999,
            "status": "passed",
            "failure_stage": "forged_stage",
            "failure_reason": "forged reason",
            "raw_available": True,
            "artifacts": [{"path": "forged"}],
        },
    )
    capsys.readouterr()
    payload = json.loads((failure_dir / "failure.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "failed_capture"
    assert payload["failure_stage"] == "timeline_gate"
    assert payload["failure_reason"] == "authoritative reason"
    assert payload["raw_available"] is False
    assert payload["artifacts"] == []


def test_staging_failure_reserved_metadata_cannot_override_receipt_contract(
    tmp_path, capsys
):
    staging = tmp_path / "staging" / ".staging_fixture"
    staging.mkdir(parents=True)
    failure_dir = RECORD._seal_staging_failure(
        staging_dir=staging,
        failed_root=tmp_path / "failed",
        stage="canonical_publish",
        reason="authoritative publish reason",
        metadata={
            "schema_version": 999,
            "status": "passed",
            "failure_stage": "forged_stage",
            "failure_reason": "forged reason",
            "raw_available": True,
            "artifacts": [{"path": "forged"}],
        },
    )
    capsys.readouterr()
    payload = json.loads((failure_dir / "failure.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["status"] == "failed_capture"
    assert payload["failure_stage"] == "canonical_publish"
    assert payload["failure_reason"] == "authoritative publish reason"
    assert payload["raw_available"] is False
    assert payload["artifacts"] == []


def test_batch_progress_uses_structured_child_failure_receipt(
    tmp_path, monkeypatch
):
    _, plan = _source_plan(tmp_path)
    out_root = tmp_path / "recorded"
    failed_root = tmp_path / "failed"
    seen_amplitudes: list[str] = []

    def fake_child(command, *, timeout_seconds, log_path):
        del timeout_seconds, log_path
        seen_amplitudes.append(command[command.index("--amplitude") + 1])
        artifact = failed_root / "capture_timeline_gate"
        artifact.mkdir(parents=True)
        receipt = artifact / "failure.json"
        reason = "high_band_coherence=0.812345 < 0.90"
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed_capture",
                    "failure_stage": "timeline_gate",
                    "failure_reason": reason,
                    "raw_available": True,
                    "artifacts": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        marker = {
            "schema_version": 1,
            "failure_stage": "timeline_gate",
            "failure_reason": reason,
            "failure_artifact": str(artifact.resolve()),
            "failure_receipt": str(receipt.resolve()),
        }
        stderr = (
            BATCH.FAILURE_RECEIPT_MARKER
            + json.dumps(marker, separators=(",", ":"))
            + "\n[중단] 사람이 읽는 후속 안내\n"
        )
        return BATCH.ChildResult(
            1,
            "출력 종료 — 지금 스피커를 분리하세요.\n",
            stderr,
            False,
        )

    monkeypatch.setattr(BATCH, "run_child_live", fake_child)
    result = BATCH.main(
        [
            "--sources", str(plan),
            "--out-root", str(out_root),
            "--failed-root", str(failed_root),
            "--settle-seconds", "0",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
        ]
    )
    assert result == 1
    assert seen_amplitudes == ["0.06"]
    progress = list(
        csv.DictReader((out_root / "batch_progress.csv").open(encoding="utf-8"))
    )
    assert len(progress) == 1
    row = progress[0]
    assert row["verdict"] == "record_failed"
    assert row["failure_stage"] == "timeline_gate"
    assert row["detail"] == "high_band_coherence=0.812345 < 0.90"
    assert row["failure_artifact"] == str(
        (failed_root / "capture_timeline_gate").resolve()
    )
    receipt = failed_root / "capture_timeline_gate/failure.json"
    assert row["failure_receipt"] == str(receipt.resolve())
    assert row["failure_receipt_sha256"] == hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()


def test_batch_progress_migrates_existing_narrow_failure_rows_without_loss(
    tmp_path, monkeypatch
):
    _, plan = _source_plan(tmp_path)
    out_root = tmp_path / "recorded"
    out_root.mkdir()
    legacy_row = {
        "source_family": "speech",
        "group_id": "speaker-book-001",
        "lineage_key": "speaker-book-001",
        "preassigned_split": "train",
        "start_seconds": "0.0",
        "seconds": "15.0",
        "source_path": str(tmp_path / "source.wav"),
        "source_file_sha256": "0" * 64,
        "source_list_sha256": "1" * 64,
        "source_row_number": "2",
        "returncode": "1",
        "timed_out": "0",
        "session_id": "",
        "verdict": "record_failed",
        "detail": "legacy 마지막 stdout 줄",
    }
    with (out_root / "batch_progress.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(legacy_row))
        writer.writeheader()
        writer.writerow(legacy_row)

    failed_root = tmp_path / "failed"

    def fake_child(command, *, timeout_seconds, log_path):
        del command, timeout_seconds, log_path
        artifact = failed_root / "capture_timeline_gate"
        artifact.mkdir(parents=True)
        receipt = artifact / "failure.json"
        reason = "coh²(source_aligned→ERR,600-1600Hz) 0.590000 < 0.600000"
        receipt.write_text(
            json.dumps(
                    {
                        "schema_version": 1,
                        "status": "failed_capture",
                        "failure_stage": "timeline_gate",
                        "failure_reason": reason,
                        "raw_available": False,
                    }
            ),
            encoding="utf-8",
        )
        marker = {
            "schema_version": 1,
            "failure_stage": "timeline_gate",
            "failure_reason": reason,
            "failure_artifact": str(artifact.absolute()),
            "failure_receipt": str(receipt.absolute()),
        }
        return BATCH.ChildResult(
            1,
            "",
            BATCH.FAILURE_RECEIPT_MARKER + json.dumps(marker) + "\n",
            False,
        )

    monkeypatch.setattr(BATCH, "run_child_live", fake_child)
    assert BATCH.main(
        [
            "--sources",
            str(plan),
            "--out-root",
            str(out_root),
            "--failed-root",
            str(failed_root),
            "--settle-seconds",
            "0",
            "--confirm-user-present",
            "--confirm-volume-minimum",
            "--confirm-routing-and-geometry",
        ]
    ) == 1

    with (out_root / "batch_progress.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        progress = list(reader)
        fields = tuple(reader.fieldnames or ())
    assert len(progress) == 2
    assert progress[0]["detail"] == legacy_row["detail"]
    assert progress[0]["failure_stage"] == ""
    assert progress[0]["failure_receipt_sha256"] == ""
    assert progress[1]["failure_stage"] == "timeline_gate"
    assert len(progress[1]["failure_receipt_sha256"]) == 64
    assert fields[: len(legacy_row)] == tuple(legacy_row)
    assert fields[-5:] == (
        "raw_available",
        "failure_stage",
        "failure_artifact",
        "failure_receipt",
        "failure_receipt_sha256",
    )


def test_batch_rejects_failure_marker_outside_declared_failure_root(tmp_path):
    failed_root = tmp_path / "failed"
    outside = tmp_path / "outside"
    outside.mkdir()
    receipt = outside / "failure.json"
    payload = {
        "schema_version": 1,
        "status": "failed_capture",
        "failure_stage": "timeline_gate",
        "failure_reason": "fixture",
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    marker = {
        "schema_version": 1,
        "failure_stage": "timeline_gate",
        "failure_reason": "fixture",
        "failure_artifact": str(outside.resolve()),
        "failure_receipt": str(receipt.resolve()),
    }
    result = BATCH.ChildResult(
        1,
        "legacy final line\n",
        BATCH.FAILURE_RECEIPT_MARKER + json.dumps(marker) + "\n",
        False,
    )
    assert BATCH._read_child_failure_evidence(result, failed_root=failed_root) is None
    assert BATCH._fallback_child_failure_detail(result) == "legacy final line"


@pytest.mark.parametrize("symlink_kind", ["root", "intermediate", "artifact", "receipt"])
def test_batch_rejects_failure_receipt_through_any_symlink_component(
    tmp_path, symlink_kind
):
    failed_root = tmp_path / "failed"
    outside = tmp_path / "outside"
    outside.mkdir()

    if symlink_kind == "root":
        real_root = tmp_path / "real_failed"
        real_root.mkdir()
        failed_root.symlink_to(real_root, target_is_directory=True)
        artifact = real_root / "capture"
        marker_artifact = failed_root / "capture"
    elif symlink_kind == "intermediate":
        failed_root.mkdir()
        redirect = failed_root / "redirect"
        redirect.symlink_to(outside, target_is_directory=True)
        artifact = outside / "capture"
        marker_artifact = redirect / "capture"
    elif symlink_kind == "artifact":
        failed_root.mkdir()
        artifact = outside / "capture"
        marker_artifact = failed_root / "capture"
    else:
        failed_root.mkdir()
        artifact = failed_root / "capture"
        marker_artifact = artifact

    artifact.mkdir()
    real_receipt = artifact / "real_failure.json"
    payload = {
        "schema_version": 1,
        "status": "failed_capture",
        "failure_stage": "timeline_gate",
        "failure_reason": "fixture",
    }
    real_receipt.write_text(json.dumps(payload), encoding="utf-8")

    marker_receipt = marker_artifact / "failure.json"
    if symlink_kind == "artifact":
        marker_artifact.symlink_to(artifact, target_is_directory=True)
        (artifact / "failure.json").write_text(json.dumps(payload), encoding="utf-8")
    elif symlink_kind == "receipt":
        marker_receipt.symlink_to(real_receipt)
    else:
        (artifact / "failure.json").write_text(json.dumps(payload), encoding="utf-8")

    marker = {
        "schema_version": 1,
        "failure_stage": "timeline_gate",
        "failure_reason": "fixture",
        "failure_artifact": str(marker_artifact.absolute()),
        "failure_receipt": str(marker_receipt.absolute()),
    }
    result = BATCH.ChildResult(
        1,
        "legacy final line\n",
        BATCH.FAILURE_RECEIPT_MARKER + json.dumps(marker) + "\n",
        False,
    )
    assert BATCH._read_child_failure_evidence(result, failed_root=failed_root) is None
    assert BATCH._fallback_child_failure_detail(result) == "legacy final line"


def test_force_cannot_bypass_input_safety_gate():
    with pytest.raises(SystemExit) as excinfo:
        RECORD.main(["--force", "--dry-run"])
    assert excinfo.value.code == 2


def test_collection_plan_binds_start_seconds(tmp_path):
    source, plan = _source_plan(tmp_path)
    rows = list(csv.DictReader(plan.open(encoding="utf-8")))
    rows[0]["start_seconds"] = "2.0"
    fields = list(rows[0])
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    common = [
        "--program", "file", "--file", str(source), "--seconds", "15",
        "--source-family", "speech", "--group-id", "speaker-book-001",
        "--source-list", str(plan), "--source-list-sha256", digest,
        "--source-row-number", "2", "--lineage-key", "speaker-book-001",
        "--preassigned-split", "train", "--dry-run",
    ]
    with pytest.raises(SystemExit) as excinfo:
        RECORD.main([*common, "--file-start-seconds", "1.0"])
    assert excinfo.value.code == 2
    assert RECORD.main([*common, "--file-start-seconds", "2.0"]) == 0


def test_partial_publish_never_enters_active_recorded_tree(tmp_path, monkeypatch):
    real_rename = RECORD._atomic_rename_noreplace
    out_root = tmp_path / "recorded"
    failed_root = tmp_path / "failed"

    def fail_active_publish(source, destination):
        if destination.parent == out_root:
            raise OSError("fixture publish failure")
        return real_rename(source, destination)

    monkeypatch.setattr(RECORD, "_atomic_rename_noreplace", fail_active_publish)
    with pytest.raises(OSError, match="fixture publish failure"):
        RECORD._publish_session(
            out_root=out_root,
            staging_root=tmp_path / "staging",
            failed_root=failed_root,
            session_name="fixture_file_deadbeef",
            sample_rate=48_000,
            mics=np.zeros((256, 2), dtype=np.float32),
            source=np.zeros(256, dtype=np.float32),
            aligned=np.zeros(256, dtype=np.float32),
            metadata={"session_id": "fixture_file_deadbeef"},
        )
    assert not list(out_root.glob("*/session.json"))
    failures = list(failed_root.glob("*/failure.json"))
    assert len(failures) == 1
    payload = json.loads(failures[0].read_text(encoding="utf-8"))
    assert payload["failure_stage"] == "canonical_publish"
    assert {item["path"] for item in payload["artifacts"]} >= {
        "mics.wav", "source.wav", "source_aligned.wav", "session.json"
    }


def test_noise_program_file_window_starts_at_planned_offset(tmp_path):
    source = tmp_path / "offset.wav"
    samples = np.linspace(-0.5, 0.5, 30, dtype=np.float32)
    samples[0] = 1.0  # offset 밖 peak도 gain 기준에 남아야 한다.
    sf.write(source, samples, 10, subtype="FLOAT")
    program = NoiseProgram(
        {
            "type": "file",
            "file": str(source),
            "file_start_seconds": 1.0,
            "amplitude": 0.1,
        },
        10,
    )
    observed = program.generate(3)
    expected = samples[10:13] / np.max(np.abs(samples)) * 0.1
    np.testing.assert_allclose(observed, expected, atol=1e-6)


def test_recording_file_window_does_not_consume_settle_and_binds_fade(tmp_path):
    source = tmp_path / "nonperiodic.wav"
    samples = np.linspace(-0.9, 0.7, 80, dtype=np.float32)
    sf.write(source, samples, 10, subtype="FLOAT")
    program = NoiseProgram(
        {
            "type": "file",
            "file": str(source),
            "file_start_seconds": 2.0,
            "amplitude": 0.15,
        },
        10,
    )

    observed = render_recording_file_window(
        program,
        20,
        sample_rate=10,
        fade_seconds=0.2,
    )
    peak = np.max(np.abs(samples)) + 1e-9
    expected = (samples[20:40] / peak * 0.15).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, 2, dtype=np.float32)
    expected[:2] *= ramp
    expected[-2:] *= ramp[::-1]

    np.testing.assert_array_equal(observed, expected)
    # settle 1초를 소비한 옛 동작이면 첫 audible 원본 index는 30이 된다.
    shifted = (samples[30:50] / peak * 0.15).astype(np.float32)
    shifted[:2] *= ramp
    shifted[-2:] *= ramp[::-1]
    assert not np.array_equal(observed, shifted)


def test_record_duct_file_timeline_keeps_settle_exact_zero(tmp_path):
    source = tmp_path / "timeline.wav"
    samples = np.linspace(-1.0, 0.5, 80, dtype=np.float32)
    sf.write(source, samples, 10, subtype="FLOAT")
    program = NoiseProgram(
        {
            "type": "file",
            "file": str(source),
            "file_start_seconds": 2.0,
            "amplitude": 0.15,
        },
        10,
    )
    timeline = RECORD._prepare_file_source_timeline(
        program,
        settle_frames=10,
        keep_frames=20,
        sample_rate=10,
    )
    assert timeline.shape == (30,)
    np.testing.assert_array_equal(timeline[:10], np.zeros(10, dtype=np.float32))

    fresh_program = NoiseProgram(
        {
            "type": "file",
            "file": str(source),
            "file_start_seconds": 2.0,
            "amplitude": 0.15,
        },
        10,
    )
    expected = render_recording_file_window(
        fresh_program,
        20,
        sample_rate=10,
    )
    np.testing.assert_array_equal(timeline[10:], expected)
