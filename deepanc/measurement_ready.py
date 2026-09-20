"""현장 수집 계획과 수집 후 raw PCM intake. 보드·오디오·학습을 실행하지 않는다."""

from array import array
import hashlib
import json
import math
from pathlib import Path
import sys
import wave

from .calibration import DEFAULT_RIR, load_secondary_path
from tools.prepare_recordings import _stereo_info, prepare_recordings, READ_FRAMES


SPLITS = ("train", "valid", "test")
SOURCE_FAMILIES = ("noise", "speech", "music", "environment", "machine", "mixed")
GAIN_NAMES = ("adc_reference", "adc_error", "dac", "output_amplifier")
ATTESTATIONS = ("anc_off", "test_output_off", "raw_pcm", "same_adc_clock",
                "contiguous_stream", "no_normalization_resampling_or_alignment",
                "matches_secondary_path_conditions")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_directory(path):
    value = Path(path).absolute()
    if value.exists() or value.is_symlink():
        raise ValueError(f"기존 출력 덮어쓰기 금지: {value}")
    if any(parent.is_symlink() for parent in value.parents):
        raise ValueError("출력 부모 symlink 금지")
    return value


def _json(path):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("capture 메타데이터는 1 MiB 이하")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"중복 JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("비유한 JSON")))
    if not isinstance(value, dict):
        raise ValueError("capture.json 객체 필요")
    json.dumps(value, allow_nan=False)  # 1e999처럼 parse_constant를 통과한 overflow도 거부.
    return value


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def capture_template(split, session):
    """unknown은 현장에서 확인해야 한다. 고정 format은 수집 요구 규격이다."""
    if split not in SPLITS:
        raise ValueError("train/valid/test split 필요")
    return {
        "schema": "omap_capture_intake.v1", "split": split, "session_id": session,
        "source_family": "unknown", "source_recording_id": "unknown", "source_corpus": "unknown",
        "source_recording_ids": ["unknown"], "source_group_ids": ["unknown"],
        "operator": "unknown", "capture_utc": "unknown", "board_model": "unknown",
        "board_revision": "unknown", "firmware_revision": "unknown", "capture_method": "unknown",
        "clock_source": "unknown", "sample_rate": 16000, "pcm_bits": 16,
        "channels": {"left": "reference", "right": "disturbance"},
        "gain_profile_id": "unknown",
        "gains": {name: {"value": "unknown", "unit": "unknown"} for name in GAIN_NAMES},
        "secondary_source_sha256": _sha(DEFAULT_RIR),
        "attestations": {name: "unknown" for name in ATTESTATIONS},
        "recording_loss": {"dropped_frames": "unknown", "duplicate_frames": "unknown",
                           "sequence_check_method": "unknown"},
        "final_test_not_used_for_tuning": "unknown" if split == "test" else None,
        "regions": {"onset": None, "transition": None, "steady": None},
    }


def create_packet(out):
    """새 빈 계획 패킷만 생성한다. 실제 WAV/수집 성공 선언은 만들지 않는다."""
    destination = _new_directory(out)
    load_secondary_path(sample_rate=16000)
    destination.mkdir(parents=True, exist_ok=False)
    sessions = []
    for split in SPLITS:
        session = f"session_{split}_001"
        directory = destination / "raw" / split / session
        directory.mkdir(parents=True)
        _write_json(directory / "capture.json", capture_template(split, session))
        sessions.append({"split": split, "session_id": session,
                         "directory": directory.relative_to(destination).as_posix()})
    report = {"schema": "omap_measurement_packet.v1", "status": "empty_plan_not_capture",
              "sessions": sessions, "sample_rate": 16000, "pcm_bits": 16, "channels": 2,
              "recording_executed": False, "audio_devices_opened": False,
              "training_executed": False, "capture_ready": False,
              "hardware_capture_implementation_pending_board_identity": True,
              "instructions": "docs/20_measurement_runbook.md; unknown을 실측/운영자 확인 값으로 채운 뒤 intake"}
    _write_json(destination / "session_plan.json", report)
    return report


def _known(value):
    return isinstance(value, str) and bool(value.strip()) and value.strip().lower() not in {
        "unknown", "none", "null", "n/a", "tbd"}


def _identity_list(value):
    if not isinstance(value, list) or not value or not all(_known(item) for item in value):
        return None
    keys = [item.strip().casefold() for item in value]
    return keys if len(keys) == len(set(keys)) else None


def _metadata_errors(meta, split, session, secondary_sha):
    errors = []
    if meta.get("schema") != "omap_capture_intake.v1":
        errors.append("capture sidecar schema 필요")
    if meta.get("split") != split or meta.get("session_id") != session:
        errors.append("capture split/session_id와 폴더 불일치")
    if meta.get("sample_rate") != 16000 or meta.get("pcm_bits") != 16:
        errors.append("capture 16000 Hz PCM16 선언 필요")
    if meta.get("channels") != {"left": "reference", "right": "disturbance"}:
        errors.append("Left=REF/Right=ANC-OFF disturbance 매핑 필요")
    for field in ("source_recording_id", "source_corpus", "operator", "capture_utc", "board_model",
                  "board_revision", "firmware_revision", "capture_method", "clock_source", "gain_profile_id"):
        if not _known(meta.get(field)):
            errors.append(f"{field}: unknown/빈값은 인증할 수 없음")
    if meta.get("source_family") not in SOURCE_FAMILIES:
        errors.append("source_family: noise/speech/music/environment/machine/mixed 필요")
    for field in ("source_recording_ids", "source_group_ids"):
        identities = _identity_list(meta.get(field))
        if identities is None:
            errors.append(f"{field}: 중복/unknown 없는 원천 식별자 목록 필요")
        elif field == "source_recording_ids" and meta.get("source_family") == "mixed" and len(identities) < 2:
            errors.append("mixed에는 부모 source_recording_ids 2개 이상 필요")
    if split != "train" and (meta.get("source_family") == "machine"
            or "mimii" in str(meta.get("source_corpus", "")).lower()):
        errors.append("MIMII/machine은 학습 보조 전용; valid/test 금지")
    gains = meta.get("gains")
    if not isinstance(gains, dict) or set(gains) != set(GAIN_NAMES):
        errors.append("ADC REF/ERR·DAC·출력 amp gain 항목 필요")
    else:
        for name, setting in gains.items():
            if (not isinstance(setting, dict) or set(setting) != {"value", "unit"}
                    or type(setting["value"]) not in (int, float)
                    or not math.isfinite(setting["value"]) or not _known(setting["unit"])):
                errors.append(f"gains.{name}: 유한 수치와 실제 단위 필요")
    if meta.get("secondary_source_sha256") != secondary_sha:
        errors.append("원본 OMAP S SHA 불일치")
    attestations = meta.get("attestations")
    for name in ATTESTATIONS:
        if not isinstance(attestations, dict) or attestations.get(name) is not True:
            errors.append(f"attestations.{name}: 운영자 true 확인 필요")
    loss = meta.get("recording_loss")
    if not isinstance(loss, dict):
        errors.append("recording_loss 기록 필요")
    else:
        for name in ("dropped_frames", "duplicate_frames"):
            if type(loss.get(name)) is not int or loss[name] != 0:
                errors.append(f"recording_loss.{name}: 명시적 0 필요")
        if not _known(loss.get("sequence_check_method")):
            errors.append("샘플 순서·누락 확인 방법 기록 필요")
    if split == "test" and meta.get("final_test_not_used_for_tuning") is not True:
        errors.append("최종 test를 학습/튜닝에 사용하지 않았다는 운영자 확인 필요")
    return errors


def _pcm_stats(path):
    """raw 정수 레일은 거부 사유, near rail은 문턱 승격 없는 관측 통계다."""
    rail, near, peak, frames = [0, 0], [0, 0], [0, 0], 0
    with wave.open(str(path), "rb") as source:
        for payload in iter(lambda: source.readframes(READ_FRAMES), b""):
            values = array("h")
            values.frombytes(payload)
            if sys.byteorder != "little":
                values.byteswap()
            for channel in (0, 1):
                current = values[channel::2]
                rail[channel] += sum(value in (-32768, 32767) for value in current)
                near[channel] += sum(abs(value) >= 32440 for value in current)
                peak[channel] = max(peak[channel], max(map(abs, current), default=0))
            frames += len(values) // 2
    return {"frames": frames, "rail_samples_by_channel": rail,
            "near_rail_samples_by_channel": near, "near_rail_abs_pcm_threshold": 32440,
            "peak_abs_pcm_by_channel": peak}


def audit_capture(input_dir):
    """기존 raw/train|valid/session 구조 + 별도 raw/test를 읽기 전용 검사."""
    requested = Path(input_dir).absolute()
    if requested.is_symlink() or any(p.is_symlink() for p in requested.parents):
        raise ValueError("입력 symlink 금지")
    root = requested.resolve()
    if not root.is_dir():
        raise ValueError("raw 입력 폴더 없음")
    load_secondary_path(sample_rate=16000)
    secondary_sha = _sha(DEFAULT_RIR)
    errors, records, metadata_records = [], [], []
    sessions, sources, groups, pcm_splits, stereo_seen = {}, {}, {}, {}, set()
    gain_profile = None
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir() or split_root.is_symlink():
            errors.append(f"{split}: 독립 split 폴더 없음/symlink")
            continue
        count = 0
        for folder in sorted(split_root.iterdir()):
            if not folder.is_dir() or folder.is_symlink():
                errors.append(f"세션 폴더만 허용: {folder}")
                continue
            key = folder.name.casefold()
            if key in sessions:
                errors.append(f"session 중복: {folder.name}")
            sessions[key] = split
            sidecar = folder / "capture.json"
            if not sidecar.is_file() or sidecar.is_symlink():
                errors.append(f"capture.json 없음/symlink: {folder}")
                continue
            try:
                meta = _json(sidecar)
                failures = _metadata_errors(meta, split, folder.name, secondary_sha)
            except (ValueError, OSError, TypeError, OverflowError) as exc:
                errors.append(f"{sidecar}: {exc}")
                continue
            errors.extend(f"{split}/{folder.name}: {value}" for value in failures)
            metadata_records.append({"path": str(sidecar), "sha256": _sha(sidecar), "metadata": meta})
            source = meta.get("source_recording_id")
            source_keys = set(_identity_list(meta.get("source_recording_ids")) or [])
            if _known(source):
                source_keys.add(source.strip().casefold())
            for kind, keys, seen in (("source_recording_id", source_keys, sources),
                                     ("source_group_id", _identity_list(meta.get("source_group_ids")) or [], groups)):
                for source_key in keys:
                    if source_key in seen and seen[source_key] != split:
                        errors.append(f"{kind} cross-split 중복: {source_key}")
                    seen[source_key] = split
            if not failures:
                profile = (meta["gain_profile_id"], json.dumps(meta["gains"], sort_keys=True))
                if gain_profile is not None and profile != gain_profile:
                    errors.append(f"세션 간 common gain profile 불일치: {folder}")
                gain_profile = profile
            wav_count = 0
            for path in sorted(folder.iterdir()):
                if path.is_symlink() or path.is_dir():
                    errors.append(f"세션 내부 symlink/중첩 폴더 금지: {path}")
                    continue
                if path.suffix.lower() != ".wav":
                    continue
                if not path.is_file():
                    errors.append(f"일반 WAV 파일만 허용: {path}")
                    continue
                try:
                    info = _stereo_info(path)
                    stats = _pcm_stats(path)
                    if stats["frames"] != info["frames"]:
                        raise ValueError("PCM 통계 중 frame 수 변경")
                except (ValueError, OSError, wave.Error, EOFError) as exc:
                    errors.append(f"{path}: {exc}")
                    continue
                if any(stats["rail_samples_by_channel"]):
                    errors.append(f"PCM16 rail 포화 감지: {path}")
                if info["stereo_pcm_sha256"] in stereo_seen:
                    errors.append(f"동일 stereo PCM 중복: {path}")
                stereo_seen.add(info["stereo_pcm_sha256"])
                for name in ("reference_pcm_sha256", "disturbance_pcm_sha256"):
                    digest = info[name]
                    if digest in pcm_splits and pcm_splits[digest] != split:
                        errors.append(f"채널 PCM cross-split 중복: {path}")
                    pcm_splits[digest] = split
                records.append({"path": path.relative_to(root).as_posix(), "split": split,
                                "session": folder.name, "source_family": meta.get("source_family"),
                                "source_recording_id": meta.get("source_recording_id"),
                                "session_id": folder.name, "source_kind": meta.get("source_family"),
                                "source_id": meta.get("source_recording_id"),
                                "source_recording_ids": meta.get("source_recording_ids"),
                                "source_group_ids": meta.get("source_group_ids"),
                                "prepared_reference": None, "prepared_disturbance": None,
                                **info, **stats})
                count += 1
                wav_count += 1
            if not wav_count:
                errors.append(f"세션 WAV 없음: {folder}")
        if not count:
            errors.append(f"{split}: 검사된 WAV 없음")
    coverage = {split: {family: sum(row["split"] == split and row["source_family"] == family
                                   for row in records) for family in SOURCE_FAMILIES} for split in SPLITS}
    return {"schema": "omap_capture_intake_report.v1", "input_root": str(root),
            "intake_passed": not errors, "errors": errors, "recordings": records,
            "capture_metadata": metadata_records, "secondary_source_sha256": secondary_sha,
            "source_coverage": {"recordings_by_split_and_family": coverage,
                                "all_sound_readiness_claim_allowed": False,
                                "identity_groups_automatically_verified": False},
            "sample_rate": 16000, "pcm_bits": 16, "physical_synchronization_certified": False,
            "anc_off_automatically_verified": False, "operator_attestations_only": True,
            "physical_performance_claim_allowed": False, "audio_devices_opened": False,
            "training_executed": False, "resampling_or_alignment_performed": False,
            "limitations": ["시간 이동·부분 복제·재인코딩 누수는 완전 검출하지 않음",
                            "gain/배선/clock/ANC OFF/누락 0은 운영자 관측 선언이며 자동 물리 인증 아님"]}


def _verify_unchanged(report):
    root = Path(report["input_root"])
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("검사 후 입력 경로 변경")
    observed_wavs, observed_metadata = set(), set()
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir() or split_root.is_symlink():
            raise ValueError("검사 후 split 경로 변경")
        for folder in split_root.iterdir():
            if not folder.is_dir() or folder.is_symlink():
                raise ValueError("검사 후 세션 경로 변경")
            observed_metadata.add(str(folder / "capture.json"))
            for path in folder.iterdir():
                if path.is_symlink() or not path.is_file():
                    raise ValueError("검사 후 세션 파일 경로 변경")
                if path.suffix.lower() == ".wav":
                    observed_wavs.add(path.relative_to(root).as_posix())
    if (observed_wavs != {item["path"] for item in report["recordings"]}
            or observed_metadata != {item["path"] for item in report["capture_metadata"]}):
        raise ValueError("검사 후 raw 파일/세션 목록 변경")
    for item in report["capture_metadata"]:
        if _sha(item["path"]) != item["sha256"]:
            raise ValueError("검사 후 capture 메타데이터 변경")
    for item in report["recordings"]:
        info = _stereo_info(root / item["path"])
        if any(info[key] != item[key] for key in info):
            raise ValueError("검사 후 raw PCM 변경")


def prepare_measurement_session(input_dir, out, *, prepare=False):
    """audit-only가 기본. 통과 + prepare=True일 때만 기존 train/valid 변환 호출."""
    if type(prepare) is not bool:
        raise ValueError("prepare는 bool")
    destination = _new_directory(out)
    root = Path(input_dir).resolve()
    resolved_out = destination.resolve()
    if root == resolved_out or root in resolved_out.parents or resolved_out in root.parents:
        raise ValueError("입력과 출력은 서로 포함되지 않는 별도 경로여야 합니다")
    report = audit_capture(input_dir)
    report.update(prepared_train_valid=False, final_test_isolated=False,
                  training_manifest=None, final_test_lock=None)
    destination.mkdir(parents=True, exist_ok=False)
    if report["intake_passed"]:
        _verify_unchanged(report)
        if prepare:
            pending = destination / "train_valid.pending"
            prepare_recordings(root, pending, anc_off=True, raw_pcm=True)
            _verify_unchanged(report)
            pending.rename(destination / "train_valid")
            report.update(prepared_train_valid=True, training_manifest="train_valid/manifest.jsonl")
            for row in report["recordings"]:
                if row["split"] in ("train", "valid"):
                    source = Path(row["path"])
                    prefix = Path("train_valid") / source.parent
                    row["prepared_reference"] = (prefix / f"{source.stem}_reference.wav").as_posix()
                    row["prepared_disturbance"] = (prefix / f"{source.stem}_disturbance.wav").as_posix()
        lock = {"schema": "omap_final_test_lock.v1", "input_root": str(root), "split": "test",
                "training_and_tuning_allowed": False, "contains_prepared_training_pairs": False,
                "records": [row for row in report["recordings"] if row["split"] == "test"],
                "capture_metadata": [row for row in report["capture_metadata"]
                                     if row["metadata"]["split"] == "test"],
                "scope": "원본 최종 test의 경로/PCM hash 봉인; 학습기는 test WAV를 열지 않음"}
        _write_json(destination / "final_test_lock.json", lock)
        report["final_test_lock"] = "final_test_lock.json"
        report["final_test_isolated"] = True
    _write_json(destination / "report.json", report)
    return report
