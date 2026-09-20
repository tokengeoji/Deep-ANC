"""실측 REF/ANC-OFF d → SFANC 준비/라벨/명시 승인 후 학습 연결.

준비 CLI는 fit·신경망·optimizer를 실행하지 않는다. train_paired는 별도 명시
승인을 요구하는 미래 학습 진입점이며 test WAV를 받거나 열지 않는다.
S·raw gain·부호·시간축을 보존하고 합성 P를 만들지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import wave

import numpy as np

from deepanc.calibration import DEFAULT_RIR, load_secondary_path


_RECIPE_KEYS = {"schema", "sample_rate", "secondary_source_sha256", "selector_architecture", "label_policy",
                "window_samples", "stride_samples", "n_fft", "control_length", "additional_delay_samples",
                "control_limit", "maximum_windows_per_split", "cost", "bank", "training"}


def _sha(path):
    _regular_file(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    _regular_file(path)
    if Path(path).stat().st_size > 16 * 1024 * 1024:
        raise ValueError("JSON은 16 MiB 이하만 허용합니다")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"JSON 중복 key: {key}")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("비유한 JSON")))
    if not isinstance(value, dict):
        raise ValueError("JSON 객체가 필요합니다")
    json.dumps(value, allow_nan=False)
    return value


def recipe_template():
    """단일 원본의 새 사본. null은 실측/설계 결정 전 실행을 차단한다."""
    return _json(Path(__file__).resolve().parents[3] / "configs/sfanc_paired_recipe.template.json")


def _snapshot_json(path, snapshots):
    _regular_file(path)
    if Path(path).suffix != ".json" or Path(path).stat().st_size > 16 * 1024 * 1024:
        raise ValueError("메타데이터 JSON 파일만 허용합니다")
    snapshots[str(path)] = _sha(path)
    return _json(path)


def _regular_file(path):
    value = Path(path)
    if value.is_symlink() or any(parent.is_symlink() for parent in value.parents) or not value.is_file():
        raise ValueError(f"일반 파일만 허용하며 symlink는 열지 않습니다: {value}")


def _keys(value, required, name):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"{name}: 필수 key 누락/알 수 없는 key")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: {low}~{high} 정수를 명시해야 합니다(null 미확정 거부)")


def _number(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name}: 유한 {'양수' if positive else '비음수'}를 명시해야 합니다")


def validate_recipe(recipe):
    """기존 합성 실험 숫자를 기본값으로 가져오지 않는 명시적 학습 recipe."""
    _keys(recipe, _RECIPE_KEYS, "recipe")
    for key, expected in {"schema": "sfanc_paired_recipe.v1", "sample_rate": 16000,
                         "selector_architecture": "ref_spectrum_v1",
                         "label_policy": "past_ref_next_window_zero_start_hard_clip"}.items():
        if recipe[key] != expected or type(recipe[key]) is not type(expected):
            raise ValueError(f"{key} 계약 불일치")
    if recipe["secondary_source_sha256"] != _sha(DEFAULT_RIR):
        raise ValueError("recipe 원본 S SHA 불일치")
    for key, low, high in (("window_samples", 256, 32768), ("stride_samples", 256, 65536),
                            ("n_fft", 32, 16384), ("control_length", 1, 512),
                            ("additional_delay_samples", 0, 32768), ("maximum_windows_per_split", 1, 100000)):
        _integer(recipe[key], key, low, high)
    window, length = recipe["window_samples"], recipe["control_length"]
    if window % 256 or recipe["stride_samples"] % 256 or recipe["n_fft"] > window:
        raise ValueError("창/stride는 256배수이며 FFT는 관측창 이하여야 합니다")
    memory = 499 + length - 1 + recipe["additional_delay_samples"]
    if window - memory < 256:
        raise ValueError("다음 창에서 S/제어 FIR settling 뒤 최소 256샘플이 필요합니다")
    _number(recipe["control_limit"], "control_limit", True)
    if recipe["control_limit"] > 1:
        raise ValueError("control_limit은 공통 디지털 단위에서 1 이하여야 합니다")
    cost = recipe["cost"]
    _keys(cost, {"band_weights", "effort_penalty", "clipping_penalty", "power_floor"}, "cost")
    _keys(cost["band_weights"], {"low", "priority_high", "remaining_high"}, "band_weights")
    for name, value in cost["band_weights"].items():
        _number(value, name, positive=name != "low")
    for name in ("effort_penalty", "clipping_penalty", "power_floor"):
        _number(cost[name], name, positive=name == "power_floor")
    bank = recipe["bank"]
    _keys(bank, {"regularization", "effort_penalty", "candidates"}, "bank")
    _number(bank["regularization"], "regularization", True)
    _number(bank["effort_penalty"], "bank effort_penalty")
    candidates = bank["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 255:
        raise ValueError("명시적 train segment 후보 1~255개가 필요합니다")
    names = set()
    for item in candidates:
        _keys(item, {"name", "reference", "start_sample", "sample_count"}, "bank candidate")
        for name in ("name", "reference"):
            if not isinstance(item[name], str) or not item[name].strip():
                raise ValueError(f"candidate {name}를 명시해야 합니다")
        if item["name"] in names or item["name"] == "zero":
            raise ValueError("후보 이름 중복/예약 zero 금지")
        names.add(item["name"])
        _integer(item["start_sample"], "start_sample", 0, 2**53-1)
        _integer(item["sample_count"], "sample_count", 1, 131072)
        if item["sample_count"] - memory < length or item["sample_count"] * length > 16777216:
            raise ValueError("후보 fit 구간의 settling/설계 행렬 크기 오류")
    training = recipe["training"]
    _keys(training, {"epochs", "batch_size", "learning_rate", "seed", "temperature", "risk_weight"}, "training")
    for name, low, high in (("epochs", 1, 10000), ("batch_size", 1, 65536), ("seed", 0, 2**32-1)):
        _integer(training[name], name, low, high)
    for name in ("learning_rate", "temperature", "risk_weight"):
        _number(training[name], name, positive=name != "risk_weight")
    cap = recipe["maximum_windows_per_split"]
    if max(cap * 2 * (recipe["n_fft"]//2+1), cap * (len(candidates)+1)) > 64000000:
        raise ValueError("선택기 준비 배열 크기 제한 위반")
    return recipe


def _new_directory(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"기존 출력 덮어쓰기 금지: {path}")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("출력 부모 symlink 금지")
    return path


def _publish_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    # 같은 디렉터리의 완성 파일을 원자적으로 공개하며 동시 생성도 덮어쓰지 않는다.
    path.hardlink_to(temporary)
    temporary.unlink()


def _pcm_info(path):
    """QA와 같은 raw channel byte SHA. rate/width prefix를 붙이지 않는다."""
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("QA 준비 결과는 16 kHz mono PCM16이어야 합니다")
        frames, observed = stream.getnframes(), 0
        for data in iter(lambda: stream.readframes(65536), b""):
            digest.update(data)
            observed += len(data)
    if frames < 1 or observed != frames * 2:
        raise ValueError("비어 있거나 잘린 PCM")
    return frames, digest.hexdigest()


def _read_pcm(path, start, count):
    with wave.open(str(path), "rb") as stream:
        stream.setpos(start)
        payload = stream.readframes(count)
    if len(payload) != count * 2:
        raise ValueError("PCM 구간이 잘렸습니다")
    return np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0


def _build_plan(packet_directory, recipe_path):
    packet = Path(packet_directory).resolve()
    recipe_path = Path(recipe_path).absolute()
    snapshots = {}
    recipe = validate_recipe(_snapshot_json(recipe_path, snapshots))
    secondary = load_secondary_path(sample_rate=16000)
    if len(secondary) != 500:
        raise ValueError("원본 S 500탭 필요")
    qa_path, lock_path = packet / "report.json", packet / "final_test_lock.json"
    qa, lock = _snapshot_json(qa_path, snapshots), _snapshot_json(lock_path, snapshots)
    if (qa.get("schema") != "omap_capture_intake_report.v1" or qa.get("intake_passed") is not True
            or qa.get("prepared_train_valid") is not True or qa.get("sample_rate") != 16000 or qa.get("pcm_bits") != 16
            or qa.get("secondary_source_sha256") != _sha(DEFAULT_RIR)
            or qa.get("training_manifest") != "train_valid/manifest.jsonl"
            or qa.get("final_test_lock") != "final_test_lock.json"):
        raise ValueError("측정 packet QA/원본 S/준비 경로 계약 불일치")
    if (lock.get("schema") != "omap_final_test_lock.v1" or lock.get("split") != "test"
            or lock.get("training_and_tuning_allowed") is not False
            or lock.get("contains_prepared_training_pairs") is not False
            or lock.get("input_root") != qa.get("input_root") or qa.get("final_test_isolated") is not True):
        raise ValueError("잠긴 final test 메타데이터 계약 불일치")
    locked = lock.get("records")
    qa_rows = qa.get("recordings")
    if (not isinstance(locked, list) or not locked or not isinstance(qa_rows, list)
            or not all(isinstance(row, dict) for row in locked + qa_rows)):
        raise ValueError("분리된 locked test와 QA recording 목록 필요")
    if locked != [row for row in qa_rows if row.get("split") == "test"]:
        raise ValueError("QA와 final test lock 불일치")
    if any(row.get("prepared_reference") is not None or row.get("prepared_disturbance") is not None for row in locked):
        raise ValueError("final test는 prepared training pair를 가질 수 없습니다")
    locked_sessions = {row["session"].casefold() for row in locked}
    locked_pcm = {row[key] for row in locked for key in ("reference_pcm_sha256", "disturbance_pcm_sha256")}
    identities = {}
    for row in qa_rows:
        if row.get("split") not in ("train", "valid", "test"):
            raise ValueError("QA split 오류")
        for field in ("source_recording_id", "source_recording_ids", "source_group_ids"):
            values = [row.get(field)] if field == "source_recording_id" else row.get(field)
            if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value.strip() or value.casefold() == "unknown" for value in values):
                raise ValueError("원녹음/부모/화자·책·아티스트 그룹 ID를 명시해야 합니다")
            namespace = "group" if field == "source_group_ids" else "source"
            for value in values:
                key = (namespace, value.strip().casefold())
                if key in identities and identities[key] != row["split"]:
                    raise ValueError("원녹음/그룹 cross-split 누출 또는 locked test 유입")
                identities[key] = row["split"]
    capture_metadata = qa.get("capture_metadata")
    if not isinstance(capture_metadata, list) or not capture_metadata:
        raise ValueError("원본 capture 메타데이터 필요")
    from deepanc.measurement_ready import _metadata_errors
    metadata_sessions = {}
    gain_profile = None
    for item in capture_metadata:
        path = Path(item["path"])
        if path.name != "capture.json" or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError("capture 메타데이터는 capture.json만 허용")
        metadata = _snapshot_json(path, snapshots)
        if snapshots[str(path)] != item["sha256"] or metadata != item["metadata"]:
            raise ValueError("QA 뒤 capture 메타데이터 변경")
        split, session = metadata.get("split"), metadata.get("session_id")
        failures = _metadata_errors(metadata, split, session, _sha(DEFAULT_RIR))
        if failures:
            raise ValueError(f"capture 메타데이터 계약: {failures}")
        if path != Path(qa["input_root"]) / split / session / "capture.json":
            raise ValueError("capture 메타데이터 원본 세션 경로 불일치")
        profile = (metadata["gain_profile_id"], json.dumps(metadata["gains"], sort_keys=True))
        if gain_profile is not None and gain_profile != profile:
            raise ValueError("세션 사이 gain profile 불일치")
        gain_profile = profile
        if metadata.get("split") != "train" and "mimii" in str(metadata.get("source_corpus", "")).casefold():
            raise ValueError("MIMII는 train-only이며 validation/final test 사용 금지")
        if (split, session) in metadata_sessions:
            raise ValueError("capture 메타데이터 세션 중복")
        metadata_sessions[(split, session)] = metadata
    if set(metadata_sessions) != {(row["split"], row["session"]) for row in qa_rows}:
        raise ValueError("capture 메타데이터 session 집합 불일치")
    if lock.get("capture_metadata") != [item for item in capture_metadata if item["metadata"]["split"] == "test"]:
        raise ValueError("final test capture 메타데이터 봉인 불일치")
    for row in qa_rows:
        metadata = metadata_sessions[(row["split"], row["session"])]
        for field in ("source_family", "source_recording_id", "source_recording_ids", "source_group_ids"):
            if row.get(field) != metadata.get(field):
                raise ValueError("QA 원녹음 identity와 capture 메타데이터 불일치")
        source = Path(row["path"])
        if len(source.parts) != 3 or source.parts[:2] != (row["split"], row["session"]):
            raise ValueError("QA 원녹음 split/session 경로 불일치")
    base = packet / "train_valid"
    if base.is_symlink():
        raise ValueError("train_valid symlink 금지")
    manifest_path, preparation_path = base / "manifest.jsonl", base / "preparation.json"
    _regular_file(manifest_path)
    if manifest_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("manifest 크기 제한 위반")
    snapshots[str(manifest_path)] = _sha(manifest_path)
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    # manifest 전체를 먼저 검사하므로 뒤쪽 test 행 때문에 앞쪽 WAV를 먼저 열지 않는다.
    if not rows or any(not isinstance(row, dict) or row.get("split") not in ("train", "valid") for row in rows):
        raise ValueError("학습 manifest에 test/unknown 행이 있습니다; WAV 미열기")
    expected = {}
    for row in qa_rows:
        if row.get("split") in ("train", "valid"):
            source = Path(row["path"])
            key = (source.parent / (source.stem + "_reference.wav")).as_posix()
            if key in expected:
                raise ValueError("QA 원천 경로 중복")
            expected[key] = row
    if len(rows) != len(expected) or {row.get("reference") for row in rows} != set(expected):
        raise ValueError("QA와 준비 manifest 원천 집합 불일치")
    for row in rows:
        if (row.get("anc_enabled") is not False or row.get("signal_domain") != "raw_pcm"
                or not isinstance(row.get("session"), str) or row["session"].casefold() in locked_sessions):
            raise ValueError("raw ANC-OFF/session 잠금 불일치")
        for channel in ("reference", "disturbance"):
            relative = Path(row[channel])
            original = base / relative
            resolved = original.resolve()
            if (relative.is_absolute() or not resolved.is_relative_to(base / row["split"] / row["session"])
                    or len(relative.parts) != 3 or relative.parts[:2] != (row["split"], row["session"])
                    or original.is_symlink() or any(parent.is_symlink() for parent in original.parents)):
                raise ValueError("학습 원천은 train_valid 내부의 해당 split 상대경로만 허용")
        source = Path(expected[row["reference"]]["path"])
        if row["disturbance"] != (source.parent / (source.stem + "_disturbance.wav")).as_posix():
            raise ValueError("REF와 ERR의 QA 원천 경로 불일치")
    preparation = _snapshot_json(preparation_path, snapshots)
    if (preparation.get("sample_rate") != 16000 or preparation.get("attested_anc_off") is not True
            or preparation.get("attested_raw_pcm") is not True or preparation.get("pcm_bits") != 16
            or preparation.get("reference_channel") != "left" or preparation.get("disturbance_channel") != "right"):
        raise ValueError("raw preparation 계약 불일치")
    prepared_sources = {row["source"]: row for row in preparation["recordings"]}
    snapshots[str(DEFAULT_RIR)] = _sha(DEFAULT_RIR)
    records, seen_sessions, seen_pcm = [], {}, {}
    for row in rows:
        origin = expected[row["reference"]]
        if origin["session"] != row["session"] or origin["split"] != row["split"]:
            raise ValueError("QA와 manifest 세션/split 불일치")
        if origin["source_recording_id"] in {r["source_recording_id"] for r in locked}:
            raise ValueError("locked test 원녹음 ID의 학습 유입")
        source_meta = prepared_sources.get(origin["path"], {})
        frame_count = None
        for channel in ("reference", "disturbance"):
            path = (base / row[channel]).resolve()
            snapshots[str(path)] = _sha(path)
            count, pcm = _pcm_info(path)
            key = channel + "_pcm_sha256"
            if (count != origin["frames"] or source_meta.get(key) != pcm or origin[key] != pcm
                    or pcm in locked_pcm or (pcm in seen_pcm and seen_pcm[pcm] != row["split"])):
                raise ValueError("QA PCM/SHA/분할 또는 locked test 누출 불일치")
            frame_count = count
            seen_pcm[pcm] = row["split"]
        session = row["session"].casefold()
        if session in seen_sessions and seen_sessions[session] != row["split"]:
            raise ValueError("train/valid session 누출")
        seen_sessions[session] = row["split"]
        if frame_count < 2 * recipe["window_samples"]:
            raise ValueError("각 녹음에 최소 과거창+다음창 길이가 필요합니다")
        records.append({**row, "reference_path": str((base / row["reference"]).resolve()),
                        "disturbance_path": str((base / row["disturbance"]).resolve()),
                        "frames": frame_count, "source_recording_id": origin["source_recording_id"],
                        "source_family": origin["source_family"], "source_recording_ids": origin["source_recording_ids"],
                        "source_group_ids": origin["source_group_ids"]})
    if {row["split"] for row in records} != {"train", "valid"}:
        raise ValueError("서로 다른 train/valid 세션 필요")
    by_reference = {row["reference"]: row for row in records}
    for candidate in recipe["bank"]["candidates"]:
        row = by_reference.get(candidate["reference"])
        if row is None or row["split"] != "train" or candidate["start_sample"] + candidate["sample_count"] > row["frames"]:
            raise ValueError("bank 후보는 범위 내 train segment만 지정할 수 있습니다")
    counts = {split: sum(1 + (row["frames"] - 2*recipe["window_samples"]) // recipe["stride_samples"]
                        for row in records if row["split"] == split) for split in ("train", "valid")}
    if max(counts.values()) > recipe["maximum_windows_per_split"]:
        raise ValueError("창 수가 명시한 상한을 넘습니다; 임의 샘플 누락 없이 중단")
    plan = {"schema": "sfanc_paired_preparation.v1", "recipe": recipe, "recipe_path": str(recipe_path), "records": records,
            "input_sha256": snapshots, "window_counts": counts, "packet_directory": str(packet),
            "secondary_source_sha256": _sha(DEFAULT_RIR), "locked_test_wav_opened": False,
            "training_executed": False, "bank_fitting_executed": False, "model_instantiated": False,
            "labels_materialized": False, "deployment_allowed": False,
            "scope": "사용자 선언+측정 packet QA의 raw REF/d; 실제 음향 감쇠/전송 지연 인증 아님"}
    _verify_inputs(plan)
    return plan


def prepare_paired(packet_directory, recipe_path, out):
    """QA packet train/valid만 읽어 봉인된 계획 생성; fit·특징·NN 실행 없음."""
    destination = _new_directory(out)
    plan = _build_plan(packet_directory, recipe_path)
    destination.mkdir(parents=True, exist_ok=False)
    _publish_json(destination / "plan.json", plan)
    _publish_json(destination / "ready.json", {"schema": "sfanc_paired_ready.v1", "plan_sha256": _sha(destination / "plan.json")})
    return plan


def _verify_inputs(plan):
    for path, expected in plan["input_sha256"].items():
        if _sha(path) != expected:
            raise ValueError(f"준비 뒤 입력 SHA 변경: {path}")


def load_preparation(directory):
    directory = Path(directory)
    ready = _json(directory / "ready.json")
    if ready.get("schema") != "sfanc_paired_ready.v1" or ready.get("plan_sha256") != _sha(directory / "plan.json"):
        raise ValueError("준비 plan SHA/ready 불일치")
    plan = _json(directory / "plan.json")
    if plan.get("schema") != "sfanc_paired_preparation.v1":
        raise ValueError("준비 plan schema 불일치")
    # 수정된 plan/ready 쌍도 임의 WAV 경로를 신뢰하지 않고 QA에서 재구성한다.
    canonical = _build_plan(plan["packet_directory"], plan["recipe_path"])
    if canonical != plan:
        raise ValueError("준비 뒤 입력/계획 SHA 또는 계약 변경")
    return plan


def _features(reference, n_fft):
    from deep_anc.train.sfanc_selector import reference_features
    return reference_features(reference, n_fft=n_fft)[0]


def paired_label(reference, disturbance, coefficients, recipe):
    """한 paired crop의 과거 REF 특징/다음창 실제 제한 명령 비용. 모델 실행 없음."""
    validate_recipe(recipe)
    window = recipe["window_samples"]
    arrays = [np.asarray(value) for value in (reference, disturbance, coefficients)]
    if any(value.dtype.kind not in "fiu" for value in arrays):
        raise ValueError("실수 수치 paired/bank 배열만 허용")
    x, d, bank = arrays
    if (x.shape != (2*window,) or d.shape != x.shape
            or bank.shape != (len(recipe["bank"]["candidates"])+1, recipe["control_length"])
            or not all(np.isfinite(value).all() for value in arrays) or np.any(bank[0])):
        raise ValueError("paired crop/bank shape·유한값·zero 후보 오류")
    x, d, bank = (value.astype(np.float64) for value in arrays)
    if max(float(np.max(np.abs(x))), float(np.max(np.abs(d)))) > 1:
        raise ValueError("REF/ERR은 공통 full-scale PCM 단위 [-1,1]이어야 합니다")
    secondary = load_secondary_path(sample_rate=16000)
    delay = recipe["additional_delay_samples"]
    start = window + len(secondary)-1 + bank.shape[1]-1 + delay
    frequency = np.fft.rfftfreq(2*window-start, 1/16000)
    frequency_weights = np.ones(len(frequency)) * 2
    frequency_weights[0] = 1
    if (2*window-start) % 2 == 0:
        frequency_weights[-1] = 1
    masks = {"low": frequency < 1000, "priority_high": (frequency >= 1000) & (frequency < 1600),
             "remaining_high": frequency >= 1600}
    def power(value):
        spectrum = np.abs(np.fft.rfft(value))**2 * frequency_weights / len(value)**2
        return {name: float(spectrum[mask].sum()) for name, mask in masks.items()}
    baseline = power(d[start:])
    cost_cfg = recipe["cost"]
    weighted_baseline = sum(cost_cfg["band_weights"][key] * value for key, value in baseline.items())
    costs, details = [], []
    for coefficients_row in bank:
        raw = np.convolve(x, coefficients_row)[:len(x)]
        if not np.isfinite(raw).all():
            raise ValueError("hard clip 이전 FIR 명령이 비유한 값입니다")
        raw[:window] = 0
        control = np.clip(raw, -recipe["control_limit"], recipe["control_limit"])
        delayed = np.pad(control, (delay, 0))[:len(x)]
        error = d + np.convolve(delayed, secondary)[:len(x)]
        if not np.isfinite(error).all():
            raise ValueError("원본 S 적용 잔차가 비유한 값입니다")
        residual = power(error[start:])
        weighted_error = sum(cost_cfg["band_weights"][key] * value for key, value in residual.items())
        effort = float(np.mean(control[start:]**2))
        limited = int(np.count_nonzero(np.abs(raw[window:]) > recipe["control_limit"]))
        cost = (weighted_error + cost_cfg["effort_penalty"]*effort) / max(weighted_baseline, cost_cfg["power_floor"])
        cost += cost_cfg["clipping_penalty"] * (limited/window)
        if not np.isfinite(cost) or cost > np.finfo(np.float32).max:
            raise ValueError("paired label 비용 비유한 값")
        costs.append(cost)
        details.append({"baseline_band_power": baseline, "residual_band_power": residual,
                        "raw_peak": float(np.max(np.abs(raw))), "output_peak": float(np.max(np.abs(control))),
                        "limited_samples": limited, "effort_power": effort})
    return {"features": _features(x[:window], recipe["n_fft"]), "costs": np.asarray(costs, dtype=np.float32),
            "details": details, "observed_until_sample": window, "scored_from_sample": start,
            "label_policy": recipe["label_policy"], "continuous_switching_evaluated": False}


def materialize_paired_labels(preparation, coefficients, split):
    """train/valid만 허용한다. 이미 계산한 bank로 라벨만 만들며 fitting/NN 없음."""
    if split not in ("train", "valid"):
        raise ValueError("test 라벨/파일 접근 금지; train/valid만 허용")
    plan = load_preparation(preparation)
    recipe = plan["recipe"]
    if np.asarray(coefficients).shape != (len(recipe["bank"]["candidates"])+1, recipe["control_length"]):
        raise ValueError("recipe와 bank 후보 수/길이 불일치")
    features, costs, rows = [], [], []
    for row in plan["records"]:
        if row["split"] != split:
            continue
        for start in range(0, row["frames"] - 2*recipe["window_samples"] + 1, recipe["stride_samples"]):
            x = _read_pcm(row["reference_path"], start, 2*recipe["window_samples"])
            d = _read_pcm(row["disturbance_path"], start, 2*recipe["window_samples"])
            label = paired_label(x, d, coefficients, recipe)
            features.append(label.pop("features"))
            costs.append(label.pop("costs"))
            rows.append({"session": row["session"], "source_recording_id": row["source_recording_id"],
                         "reference": row["reference"], "crop_start_sample": start, **label})
    _verify_inputs(plan)
    return {"features": np.asarray(features), "costs": np.asarray(costs), "rows": rows}


def _fit_candidate(x, d, secondary, recipe):
    from deep_anc.baselines.sfanc_design import fit_control_fir
    return fit_control_fir(x, d, secondary, control_length=recipe["control_length"],
        secondary_delay_samples=recipe["additional_delay_samples"],
        warmup_samples=len(secondary)-1+recipe["control_length"]-1+recipe["additional_delay_samples"],
        regularization=recipe["bank"]["regularization"], effort_penalty=recipe["bank"]["effort_penalty"])


def _train_selector(train, valid, recipe, device):
    from deep_anc.train.sfanc_selector import train_selector
    return train_selector(train["features"], train["costs"], valid["features"], valid["costs"],
                          device=device, **recipe["training"])


def _save_selector(path, result, recipe, bank_sha):
    import torch
    artifact = {"schema": "sfanc_paired_selector.v1", "bank_sha256": bank_sha, "recipe": recipe,
                "candidate_count": result.model.candidate_count, "deployment_allowed": False,
                "state_dict": {k: v.detach().cpu() for k, v in result.model.state_dict().items()}}
    with path.open("xb") as handle:
        torch.save(artifact, handle)


def train_paired(preparation, out, *, approve_training=False, device):
    """미래 명시 승인 후 호출용. 기본값은 모든 입력/모델 실행 전에 거부한다.

    이 함수의 존재는 학습 실행 승인이나 해당 코드의 실측 학습 검증을 의미하지 않는다.
    이전 합성 model/bank/optimizer를 불러오지 않으며 locked test는 평가하지 않는다.
    """
    if approve_training is not True:
        raise PermissionError("새 학습은 별도 명시 승인이 필요합니다; 준비 CLI는 학습하지 않습니다")
    if device not in ("cpu", "cuda"):
        raise ValueError("device를 cpu/cuda로 명시해야 합니다")
    destination = _new_directory(out)
    plan = load_preparation(preparation)
    recipe = plan["recipe"]
    secondary = load_secondary_path(sample_rate=16000)
    destination.mkdir(parents=True, exist_ok=False)
    bank, fits = [np.zeros(recipe["control_length"])], []
    records = {row["reference"]: row for row in plan["records"]}
    for candidate in recipe["bank"]["candidates"]:
        row = records[candidate["reference"]]
        if row["split"] != "train":
            raise ValueError("FIR fitting에 validation/test를 사용할 수 없습니다")
        x = _read_pcm(row["reference_path"], candidate["start_sample"], candidate["sample_count"])
        d = _read_pcm(row["disturbance_path"], candidate["start_sample"], candidate["sample_count"])
        fit = _fit_candidate(x, d, secondary, recipe)
        bank.append(fit.coefficients)
        fits.append({"candidate": candidate, "diagnostics": fit.diagnostics})
    coefficients = np.asarray(bank, dtype=np.float64)
    train = materialize_paired_labels(preparation, coefficients, "train")
    valid = materialize_paired_labels(preparation, coefficients, "valid")
    _verify_inputs(plan)
    result = _train_selector(train, valid, recipe, device)
    _verify_inputs(plan)
    bank_path = destination / "bank.npz"
    with bank_path.open("xb") as handle:
        np.savez_compressed(handle, coefficients=coefficients, secondary=secondary)
    _save_selector(destination / "selector.pt", result, recipe, _sha(bank_path))
    report = {"schema": "sfanc_paired_training.v1", "recipe": recipe,
        "preparation_plan_sha256": _sha(Path(preparation) / "plan.json"), "input_sha256": plan["input_sha256"],
        "training_executed": True, "bank_fit_split": "train", "checkpoint_selection_split": "valid",
        "fixed_candidate_from_train": int(np.argmin(train["costs"].mean(axis=0))),
        "best_epoch": result.best_epoch, "history": result.history,
        "best_validation_cost": result.best_validation_loss, "fits": fits,
        "train_label_rows": train["rows"], "valid_label_rows": valid["rows"],
        "test_opened": False, "primary_is_synthetic": False, "physical_performance_claim_allowed": False,
        "deployment_allowed": False, "real_time_claim_allowed": False,
        "continuous_switching_evaluated": False, "optimizer_resume_supported": False,
        "bank_fit_objective": "full-band linear ridge proposal; high-band/hard-limit optimization claim false",
        "label_objective": "explicit disjoint low/1k-1.6k/1.6k-Nyquist weights after actual hard clip and S",
        "artifacts": {name: _sha(destination / name) for name in ("bank.npz", "selector.pt")},
        "scope": "raw paired d를 사용하는 오프라인 라벨; 실제 ANC-ON 실험/전송 지연 검증 아님"}
    _publish_json(destination / "report.json", report)
    return report
