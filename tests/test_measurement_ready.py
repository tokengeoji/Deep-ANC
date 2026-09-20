"""스피커/보드/학습 없이 원본 보존형 수집 packet과 intake를 검증한다."""

import ast
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import wave

import pytest

from deepanc import measurement_ready as ready


def write_stereo(path, values, rate=16000):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<" + "h" * len(values), *values))


@pytest.fixture
def capture(tmp_path):
    packet = tmp_path / "packet"
    ready.create_packet(packet)
    root = packet / "raw"
    for index, split in enumerate(ready.SPLITS):
        folder = root / split / f"session_{split}_001"
        sidecar = folder / "capture.json"
        meta = json.loads(sidecar.read_text())
        for name in ("operator", "capture_utc", "board_model", "board_revision", "firmware_revision",
                     "capture_method", "clock_source"):
            meta[name] = "fixture-observation"
        meta.update(source_family="speech", source_corpus="fixture", source_recording_id=f"source-{index}",
                    source_recording_ids=[f"parent-{index}"], source_group_ids=[f"speaker:{index}", f"book:{index}"],
                    gain_profile_id="profile-A")
        # REF/ERR의 gain은 서로 다를 수 있으며 세션 간 각 값만 같아야 한다.
        meta["gains"] = {name: {"value": i, "unit": "dB"} for i, name in enumerate(ready.GAIN_NAMES)}
        meta["attestations"] = {name: True for name in ready.ATTESTATIONS}
        meta["recording_loss"] = {"dropped_frames": 0, "duplicate_frames": 0,
                                  "sequence_check_method": "fixture sequence counters"}
        if split == "test":
            meta["final_test_not_used_for_tuning"] = True
        sidecar.write_text(json.dumps(meta))
        values = [index * 100 + n * 3 + c + 1 for n in range(16) for c in (0, 1)]
        write_stereo(folder / "noise.wav", values)
    return root


def change_meta(root, split, update):
    path = root / split / f"session_{split}_001" / "capture.json"
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


def test_packet_has_only_unknown_templates_and_never_creates_audio(tmp_path):
    out = tmp_path / "packet"
    report = ready.create_packet(out)
    assert report["status"] == "empty_plan_not_capture"
    assert not report["capture_ready"] and not report["recording_executed"]
    assert report["hardware_capture_implementation_pending_board_identity"] is True
    assert len(list(out.rglob("capture.json"))) == 3
    assert not list(out.rglob("*.wav"))
    result = ready.audit_capture(out / "raw")
    assert not result["intake_passed"]
    assert any("unknown" in issue for issue in result["errors"])
    assert any("WAV 없음" in issue for issue in result["errors"])


def test_success_prepares_legacy_train_valid_and_seals_test_without_changing_bytes(capture, tmp_path):
    originals = {str(p): p.read_bytes() for p in capture.rglob("*") if p.is_file()}
    out = tmp_path / "prepared"
    report = ready.prepare_measurement_session(capture, out, prepare=True)
    assert report["intake_passed"] and report["prepared_train_valid"]
    assert not report["physical_synchronization_certified"]
    assert not report["anc_off_automatically_verified"]
    assert report["operator_attestations_only"]
    assert not report["source_coverage"]["all_sound_readiness_claim_allowed"]
    assert report["source_coverage"]["recordings_by_split_and_family"]["valid"]["speech"] == 1
    for row in report["recordings"]:
        assert row["source_kind"] == row["source_family"]
        assert row["session_id"] == row["session"]
        assert row["source_id"] == row["source_recording_id"]
        if row["split"] == "test":
            assert row["prepared_reference"] is None and row["prepared_disturbance"] is None
        else:
            assert (out / row["prepared_reference"]).is_file()
            assert (out / row["prepared_disturbance"]).is_file()
    rows = [json.loads(line) for line in (out / report["training_manifest"]).read_text().splitlines()]
    assert {row["split"] for row in rows} == {"train", "valid"}
    for row in rows:
        source = capture / row["split"] / row["session"] / "noise.wav"
        with wave.open(str(source)) as handle:
            payload = handle.readframes(handle.getnframes())
        for channel, key in enumerate(("reference", "disturbance")):
            with wave.open(str(out / "train_valid" / row[key])) as handle:
                actual = handle.readframes(handle.getnframes())
            expected = b"".join(payload[i+2*channel:i+2*channel+2] for i in range(0, len(payload), 4))
            assert actual == expected
    lock = json.loads((out / report["final_test_lock"]).read_text())
    assert len(lock["records"]) == 1 and lock["records"][0]["split"] == "test"
    assert lock["training_and_tuning_allowed"] is False
    assert not (out / "train_valid" / "test").exists()
    assert {name: Path(name).read_bytes() for name in originals} == originals


def test_audit_only_never_calls_prepare(capture, tmp_path, monkeypatch):
    monkeypatch.setattr(ready, "prepare_recordings", lambda *a, **k: pytest.fail("audit-only에서 분리 금지"))
    out = tmp_path / "audit"
    result = ready.prepare_measurement_session(capture, out)
    assert result["intake_passed"] and not result["prepared_train_valid"]
    assert not list(out.rglob("*.wav"))
    assert {p.name for p in out.iterdir()} == {"report.json", "final_test_lock.json"}


@pytest.mark.parametrize("field", ready.ATTESTATIONS)
def test_every_operator_attestation_is_required(capture, tmp_path, field):
    change_meta(capture, "train", lambda m: m["attestations"].update({field: "unknown"}))
    report = ready.prepare_measurement_session(capture, tmp_path / "failed", prepare=True)
    assert not report["intake_passed"] and not report["prepared_train_valid"]
    assert not report["final_test_isolated"]
    assert not (tmp_path / "failed" / "train_valid").exists()
    assert any(field in error for error in report["errors"])


@pytest.mark.parametrize("kind", ["unknown_output_gain", "different_gain", "different_profile", "nan_gain",
                                    "drop", "duplicate", "unknown_clock", "source_overlap", "test_exposed", "mimii"])
def test_fail_closed_on_metadata_and_test_boundaries(capture, kind):
    def mutate(meta):
        if kind == "unknown_output_gain": meta["gains"]["output_amplifier"]["value"] = "unknown"
        elif kind == "different_gain": meta["gains"]["adc_reference"]["value"] = 42
        elif kind == "different_profile": meta["gain_profile_id"] = "profile-B"
        elif kind == "nan_gain": meta["gains"]["dac"]["value"] = float("nan")
        elif kind == "drop": meta["recording_loss"]["dropped_frames"] = 1
        elif kind == "duplicate": meta["recording_loss"]["duplicate_frames"] = 1
        elif kind == "unknown_clock": meta["clock_source"] = "unknown"
        elif kind == "source_overlap": meta["source_recording_id"] = "source-0"
        elif kind == "test_exposed": meta["final_test_not_used_for_tuning"] = False
        elif kind == "mimii": meta["source_corpus"] = "MIMII_DG_fan"
    change_meta(capture, "test", mutate)
    assert not ready.audit_capture(capture)["intake_passed"]


@pytest.mark.parametrize("kind", ["session", "channel", "stereo", "rate", "truncated", "rail_low", "rail_high"])
def test_fail_closed_on_pcm_format_clipping_and_leakage(capture, kind):
    source = capture / "train/session_train_001/noise.wav"
    target = capture / "test/session_test_001/noise.wav"
    if kind == "session":
        folder = target.parent
        folder.rename(folder.parent / "session_train_001")
    elif kind == "stereo": shutil.copyfile(source, target)
    elif kind == "truncated": target.write_bytes(target.read_bytes()[:-4])
    else:
        values = [201+n*3+c for n in range(16) for c in (0, 1)]
        if kind == "channel": values[::2] = [1+n*3 for n in range(16)]
        elif kind == "rail_low": values[0] = -32768
        elif kind == "rail_high": values[0] = 32767
        write_stereo(target, values, rate=48000 if kind == "rate" else 16000)
    assert not ready.audit_capture(capture)["intake_passed"]


def test_near_rail_is_diagnostic_not_automatic_failure(capture):
    path = capture / "train/session_train_001/noise.wav"
    write_stereo(path, [32500, -32500, 1, 2])
    report = ready.audit_capture(capture)
    assert report["intake_passed"]
    row = next(r for r in report["recordings"] if r["split"] == "train")
    assert row["rail_samples_by_channel"] == [0, 0]
    assert row["near_rail_samples_by_channel"] == [1, 1]


@pytest.mark.parametrize("kind", ["existing", "output_parent_link", "input_link", "nested_output"])
def test_path_protection(capture, tmp_path, kind):
    output = tmp_path / "out"
    raw = capture
    if kind == "existing": output.mkdir()
    elif kind == "output_parent_link":
        parent = tmp_path / "alias"
        parent.symlink_to(tmp_path, target_is_directory=True)
        output = parent / "out"
    elif kind == "input_link":
        raw = tmp_path / "alias"
        raw.symlink_to(capture, target_is_directory=True)
    else: output = capture / "prepared"
    with pytest.raises(ValueError):
        ready.prepare_measurement_session(raw, output, prepare=True)


def test_changed_source_during_prepare_never_promotes_pending_manifest(capture, tmp_path, monkeypatch):
    original_prepare = ready.prepare_recordings
    def changing(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        path = capture / "test/session_test_001/noise.wav"
        write_stereo(path, [100, 200, 101, 201])
        return result
    monkeypatch.setattr(ready, "prepare_recordings", changing)
    with pytest.raises(ValueError, match="변경"):
        ready.prepare_measurement_session(capture, tmp_path / "out", prepare=True)
    assert not (tmp_path / "out/train_valid").exists()
    assert not (tmp_path / "out/report.json").exists()
    assert (tmp_path / "out/train_valid.pending").is_dir()


@pytest.mark.parametrize("kind", ["wav", "session", "symlink"])
def test_inventory_change_during_prepare_never_promotes_output(capture, tmp_path, monkeypatch, kind):
    original_prepare = ready.prepare_recordings
    def changing(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        folder = capture / "train/session_train_001"
        if kind == "wav": write_stereo(folder / "added.wav", [111, 222, 333, 444])
        elif kind == "session": (capture / "test/new_session").mkdir()
        else:
            saved = tmp_path / "saved.wav"
            (folder / "noise.wav").rename(saved)
            (folder / "noise.wav").symlink_to(saved)
        return result
    monkeypatch.setattr(ready, "prepare_recordings", changing)
    with pytest.raises(ValueError, match="변경"):
        ready.prepare_measurement_session(capture, tmp_path / "out", prepare=True)
    assert not (tmp_path / "out/train_valid").exists()
    assert not (tmp_path / "out/report.json").exists()


@pytest.mark.parametrize("field", ["source_recording_ids", "source_group_ids"])
@pytest.mark.parametrize("identities", [[], ["unknown"], ["parent", " PARENT "], None, "parent"])
def test_source_identity_lists_required_and_unique(capture, field, identities):
    change_meta(capture, "test", lambda m: m.update({field: identities}))
    assert not ready.audit_capture(capture)["intake_passed"]


@pytest.mark.parametrize("kind", ["mixed_parent", "group", "top_id_matches_parent"])
def test_parent_and_group_leaks_cannot_hide_behind_distinct_mix_ids(capture, kind):
    def mutate(meta):
        meta.update(source_family="mixed", source_recording_id="new-mix",
                    source_recording_ids=["independent-a", "independent-b"])
        if kind == "mixed_parent": meta["source_recording_ids"][1] = "parent-0"
        elif kind == "group": meta["source_group_ids"] = ["speaker:0"]
        else: meta["source_recording_id"] = "parent-0"
    change_meta(capture, "test", mutate)
    assert not ready.audit_capture(capture)["intake_passed"]


def test_mixed_requires_multiple_parents_and_preserves_known_identity_lists(capture):
    change_meta(capture, "test", lambda m: m.update(source_family="mixed"))
    assert not ready.audit_capture(capture)["intake_passed"]
    change_meta(capture, "test", lambda m: m.update(source_recording_ids=["parent-2", "parent-3"]))
    report = ready.audit_capture(capture)
    assert report["intake_passed"]
    assert report["source_coverage"]["recordings_by_split_and_family"]["test"]["mixed"] == 1
    assert report["recordings"][-1]["source_recording_ids"] == ["parent-2", "parent-3"]


def test_module_has_no_training_or_audio_imports():
    tree = ast.parse(Path(ready.__file__).read_text())
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(any(t in name for t in ("torch", "train", "sounddevice", "scipy")) for name in names)


def test_cli_defaults_to_packet_and_rejects_prepare_without_input(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/prepare_measurement_session.py"
    result = subprocess.run([sys.executable, str(script), "--out", str(tmp_path / "packet")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list((tmp_path / "packet").rglob("*.wav"))
    failed = subprocess.run([sys.executable, str(script), "--out", str(tmp_path / "bad"), "--prepare"],
                            capture_output=True, text=True)
    assert failed.returncode == 2 and not (tmp_path / "bad").exists()
