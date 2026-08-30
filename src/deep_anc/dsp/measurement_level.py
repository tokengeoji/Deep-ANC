"""P/S 경로 측정과 앰프 레벨 미터가 공유하는 물리 레벨 계약.

레벨 미터와 실제 경로 측정이 서로 다른 프로브 peak를 쓰면 미터가 표시한 dBFS와
측정 SNR/채널 결합 조건이 갈라진다. 숫자를 각 스크립트에 다시 적지 않고 이 불변
계약 하나를 직접 소비한다.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import time
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MeasurementLevelContract:
    """동시 인터리브 P/S 측정의 재생 레벨과 미터 판정 규약."""

    sample_rate: int
    design_band_hz: tuple[float, float]
    meter_band_hz: tuple[float, float]
    period_seconds: float
    warmup_periods: int
    analysis_repeats: int
    input_probe_seconds: float
    meter_seconds: float
    probe_amplitude: float
    meter_target_dbfs: float
    meter_tolerance_db: float
    interleaved_err_noise_bin_dbfs: float
    interleaved_err_noise_bin_tolerance_db: float

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate는 양수여야 합니다")
        nyquist = self.sample_rate / 2.0
        for name, band in (
            ("design_band_hz", self.design_band_hz),
            ("meter_band_hz", self.meter_band_hz),
        ):
            if not (
                len(band) == 2
                and math.isfinite(float(band[0]))
                and math.isfinite(float(band[1]))
                and 0.0 < float(band[0]) < float(band[1]) < nyquist
            ):
                raise ValueError(f"{name}가 유효하지 않습니다: {band!r}")
        if not math.isfinite(self.period_seconds) or self.period_seconds <= 0.0:
            raise ValueError("period_seconds는 양수 finite여야 합니다")
        if self.warmup_periods <= 0 or self.analysis_repeats <= 0:
            raise ValueError("warmup_periods/analysis_repeats는 양수여야 합니다")
        if not math.isfinite(self.input_probe_seconds) or self.input_probe_seconds <= 0.0:
            raise ValueError("input_probe_seconds는 양수 finite여야 합니다")
        if not math.isfinite(self.meter_seconds) or self.meter_seconds <= 0.0:
            raise ValueError("meter_seconds는 양수 finite여야 합니다")
        if not math.isfinite(self.probe_amplitude) or not 0.0 < self.probe_amplitude <= 1.0:
            raise ValueError("probe_amplitude는 (0, 1] finite여야 합니다")
        if not math.isfinite(self.meter_target_dbfs) or self.meter_target_dbfs >= 0.0:
            raise ValueError("meter_target_dbfs는 0 미만 finite여야 합니다")
        if not math.isfinite(self.meter_tolerance_db) or self.meter_tolerance_db <= 0.0:
            raise ValueError("meter_tolerance_db는 양수 finite여야 합니다")
        if not math.isfinite(self.interleaved_err_noise_bin_dbfs):
            raise ValueError("interleaved_err_noise_bin_dbfs는 finite여야 합니다")
        if (
            not math.isfinite(self.interleaved_err_noise_bin_tolerance_db)
            or self.interleaved_err_noise_bin_tolerance_db <= 0.0
        ):
            raise ValueError(
                "interleaved_err_noise_bin_tolerance_db는 양수 finite여야 합니다"
            )

    @property
    def meter_min_dbfs(self) -> float:
        return self.meter_target_dbfs - self.meter_tolerance_db

    @property
    def meter_max_dbfs(self) -> float:
        return self.meter_target_dbfs + self.meter_tolerance_db


OFFICIAL_MEASUREMENT_LEVEL = MeasurementLevelContract(
    sample_rate=48_000,
    design_band_hz=(60.0, 1650.0),
    meter_band_hz=(150.0, 1600.0),
    period_seconds=0.125,
    warmup_periods=32,
    analysis_repeats=64,
    input_probe_seconds=3.0,
    meter_seconds=20.0,
    probe_amplitude=0.003,
    meter_target_dbfs=-50.1,
    meter_tolerance_db=2.0,
    interleaved_err_noise_bin_dbfs=-46.4,
    # 사용자 계획의 "약 -46.4"는 meter 허용창과 같은 ±2 dB로 검증한다.
    # 보존 fresh_rir A..G raw의 nominal-bin 재계산 중앙은 -47.55 dBFS이고,
    # A를 현행 fractional joint-LS로 전수 재분석한 값은 -47.49 dBFS다.
    interleaved_err_noise_bin_tolerance_db=2.0,
)


MEASUREMENT_CPU_IDLE_SAMPLE_SECONDS = 0.25
"""실기 출력 직전 CPU 유휴율을 관측하는 짧은 무출력 구간."""

MIN_MEASUREMENT_CPU_IDLE_FRACTION = 0.50
"""official 측정 중 최소 절반의 aggregate CPU가 유휴여야 한다."""
"""출하 P/S 측정과 ``set_amp_level.py``가 직접 공유하는 유일한 인스턴스."""


OFFICIAL_MEASUREMENT_CHANNEL_MAP = {
    "error_mic": 0,
    "reference_mic": 1,
    "noise_out": 0,
    "cancel_out": 1,
}
"""ERR/REF 입력과 noise/cancel 출력의 출하 채널 규약."""


MEASUREMENT_LEVEL_EVIDENCE_SCHEMA = "measurement_level_evidence_v2_bootstrap_pair"
DEFAULT_MEASUREMENT_LEVEL_EVIDENCE_PATH = Path(
    "assets/measured/measurement_level_evidence.json"
)

BOOTSTRAP_METER_RAW_SCHEMA = "measurement_level_meter_raw_v1"
BOOTSTRAP_METER_RECEIPT_SCHEMA = "measurement_level_meter_raw_receipt_v1"
INTERLEAVED_RAW_CAPTURE_SCHEMA = (
    "interleaved_raw_v4_user_present_observed_pcm_preanalysis"
)
BOOTSTRAP_METER_MAX_AGE_SECONDS = 10 * 60
BOOTSTRAP_CLOCK_FUTURE_TOLERANCE_SECONDS = 5.0
METER_HOP_SECONDS = 0.25
LIVE_WATCHDOG_GRACE_SECONDS = 1.0
ALSA_PHYSICAL_FINGERPRINT_SCHEMA = "alsa_physical_hardware_fingerprint_v1"

_PCM_INFO_STABLE_FIELDS = (
    "device",
    "stream",
    "id",
    "name",
    "subname",
    "class",
    "subclass",
    "subdevices_count",
)
_UEVENT_STABLE_FIELDS = {
    "DRIVER",
    "DEVTYPE",
    "PRODUCT",
    "TYPE",
    "INTERFACE",
    "MODALIAS",
    "OF_NAME",
    "OF_FULLNAME",
    "OF_COMPATIBLE_N",
}
_SYSFS_STABLE_ATTRIBUTES = (
    "serial",
    "manufacturer",
    "product",
    "idVendor",
    "idProduct",
    "bcdDevice",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_publish_noreplace(temporary: str | Path, target: str | Path) -> Path:
    """같은 filesystem의 완성 temp를 race-safe no-replace로 publish한다.

    ``exists()`` 뒤 ``os.replace``는 검사 직후 공격자/다른 프로세스가 만든 파일을
    덮어쓴다. hard-link 생성은 대상 이름이 이미 있으면 원자적으로 ``EEXIST``이며,
    성공한 inode만 temp 이름에서 떼므로 기존 target을 절대 교체하지 않는다.
    """

    source = Path(temporary)
    destination = Path(target)
    if source.parent.resolve() != destination.parent.resolve():
        raise ValueError("atomic no-replace publish는 same-directory temp가 필요합니다")
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"race-safe publish가 기존 target을 발견했습니다: {destination}"
        ) from exc
    try:
        source.unlink()
    except OSError:
        # destination은 이미 같은 inode에 안전하게 결박됐다. 숨은 temp 정리 실패를
        # publish 실패로 오인해 유효 final을 롤백하지 않는다.
        pass
    _fsync_directory(destination.parent)
    return destination


class LiveAudioTermination(BaseException):
    """live audio 구간에 받은 종료 신호. stream cleanup 뒤 exit code로 변환한다."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        self.exit_code = 128 + self.signum
        super().__init__(f"live audio termination signal {self.signum}")


@contextmanager
def scoped_live_audio_signal_handlers():
    """stream start/wait에만 INT/TERM/HUP를 예외로 바꿔 cleanup을 보장한다."""

    watched = tuple(
        item
        for item in (
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGTERM", None),
            getattr(signal, "SIGHUP", None),
        )
        if item is not None
    )
    previous = {item: signal.getsignal(item) for item in watched}

    def terminate(signum, _frame):  # noqa: ANN001
        raise LiveAudioTermination(int(signum))

    try:
        for item in watched:
            signal.signal(item, terminate)
        yield
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


@contextmanager
def repository_audio_lock(repository_root: str | Path, *, purpose: str):
    """같은 저장소·UID의 cooperating audio process를 stream close까지 배제한다."""

    root = Path(repository_root).resolve()
    lock_path = root / "results" / f".live_audio_uid_{os.getuid()}.lock"
    if not lock_path.parent.is_dir():
        raise FileNotFoundError(f"audio lock 상위 디렉터리가 없습니다: {lock_path.parent}")
    descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode) or file_status.st_uid != os.getuid():
            raise RuntimeError("audio lock은 현재 UID 소유의 regular file이어야 합니다")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"같은 저장소/UID의 live audio 작업이 이미 실행 중입니다: {lock_path}"
            ) from exc
        os.ftruncate(descriptor, 0)
        payload = _canonical_json(
            {"pid": os.getpid(), "uid": os.getuid(), "purpose": str(purpose)}
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield {
            "path": str(lock_path.relative_to(root)),
            "pid": os.getpid(),
            "uid": os.getuid(),
            "purpose": str(purpose),
        }
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_npz_snapshot(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _path_in_repository(path: str | Path, *, repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"측정 증거 경로는 저장소 안에 있어야 합니다: {candidate}") from exc
    return candidate


def _parse_utc(value: Any, *, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} UTC timestamp가 필요합니다")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} UTC timestamp가 잘못되었습니다: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field}는 timezone-aware UTC timestamp여야 합니다")
    return parsed.astimezone(dt.timezone.utc)


def _parse_key_value_lines(text: str, *, separator: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if separator not in line:
            continue
        key, value = line.split(separator, 1)
        key = key.strip()
        if key:
            parsed[key] = value.strip()
    return parsed


def _alsa_endpoint_physical_fingerprint(
    endpoint: dict[str, Any],
    *,
    direction: str,
    proc_asound_root: Path,
    sys_class_sound_root: Path,
    sys_root: Path,
) -> dict[str, Any]:
    """ALSA short ID를 실제 proc/sys 장치에 결박한 안정 fingerprint."""

    configured_card_id = str(endpoint["card"]).strip()
    matches: list[tuple[int, Path]] = []
    card_pattern = re.compile(r"^card([0-9]+)$")
    if not proc_asound_root.is_dir():
        raise FileNotFoundError(f"ALSA proc root를 읽을 수 없습니다: {proc_asound_root}")
    for candidate in proc_asound_root.glob("card[0-9]*"):
        match = card_pattern.fullmatch(candidate.name)
        if match is None:
            continue
        id_path = candidate / "id"
        if not id_path.is_file():
            continue
        observed = id_path.read_text(encoding="utf-8", errors="strict").strip()
        if observed == configured_card_id:
            matches.append((int(match.group(1)), candidate))
    if len(matches) != 1:
        raise RuntimeError(
            f"ALSA card id {configured_card_id!r}의 물리 매핑은 정확히 1개여야 "
            f"합니다: {[index for index, _path in matches]}"
        )
    card_index, card_path = matches[0]
    pcm_device = int(endpoint["pcm"])
    suffix = "c" if direction == "input" else "p"
    expected_stream = "CAPTURE" if direction == "input" else "PLAYBACK"
    info_path = card_path / f"pcm{pcm_device}{suffix}" / "info"
    if not info_path.is_file():
        raise FileNotFoundError(f"ALSA PCM info를 읽을 수 없습니다: {info_path}")
    info = _parse_key_value_lines(
        info_path.read_text(encoding="utf-8", errors="strict"), separator=":"
    )
    try:
        info_card = int(info["card"])
        info_device = int(info["device"])
        info_stream = info["stream"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ALSA PCM info 필수 필드가 없습니다: {info_path}") from exc
    if (
        info_card != card_index
        or info_device != pcm_device
        or info_stream != expected_stream
    ):
        raise ValueError(
            "ALSA PCM info가 선택한 card/device/direction과 다릅니다: "
            f"{info_path}"
        )
    stable_pcm_info = {
        key: info[key]
        for key in _PCM_INFO_STABLE_FIELDS
        if key in info
    }
    if set(_PCM_INFO_STABLE_FIELDS) - set(stable_pcm_info):
        raise ValueError(f"ALSA PCM info 안정 필드가 불완전합니다: {info_path}")

    sys_device = sys_class_sound_root / f"card{card_index}" / "device"
    try:
        resolved_device = sys_device.resolve(strict=True)
        resolved_sys_root = sys_root.resolve(strict=True)
        relative_device = resolved_device.relative_to(resolved_sys_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"ALSA sysfs physical device를 안전하게 해석하지 못했습니다: {sys_device}"
        ) from exc
    uevent_path = resolved_device / "uevent"
    if not uevent_path.is_file():
        raise FileNotFoundError(f"ALSA sysfs uevent를 읽을 수 없습니다: {uevent_path}")
    uevent_all = _parse_key_value_lines(
        uevent_path.read_text(encoding="utf-8", errors="strict"), separator="="
    )
    stable_uevent = {
        key: value
        for key, value in uevent_all.items()
        if key in _UEVENT_STABLE_FIELDS or key.startswith("OF_COMPATIBLE_")
    }
    if not stable_uevent:
        raise ValueError(f"ALSA sysfs uevent에 안정 identity 필드가 없습니다: {uevent_path}")

    stable_attributes: list[dict[str, Any]] = []
    current = resolved_device
    # USB serial/product는 sound interface의 한 단계 위에 있다. 플랫폼 codec은
    # device realpath+uevent가 identity이고, 없는 attribute를 임의 생성하지 않는다.
    for _depth in range(6):
        values: dict[str, str] = {}
        for name in _SYSFS_STABLE_ATTRIBUTES:
            attribute = current / name
            if attribute.is_file():
                value = attribute.read_text(encoding="utf-8", errors="strict").strip()
                if value:
                    values[name] = value
        if values:
            stable_attributes.append(
                {
                    "sys_relative_path": str(current.relative_to(resolved_sys_root)),
                    "values": values,
                }
            )
        if current == resolved_sys_root or current.parent == current:
            break
        current = current.parent

    return {
        "configured_card_id": configured_card_id,
        "proc_card_id": (card_path / "id").read_text(
            encoding="utf-8", errors="strict"
        ).strip(),
        "pcm_device": pcm_device,
        "pcm_stream": expected_stream,
        "pcm_info": stable_pcm_info,
        "sys_device_realpath": str(relative_device),
        "sys_device_uevent": stable_uevent,
        "stable_attributes": stable_attributes,
    }


def collect_alsa_physical_fingerprint(
    config: dict[str, Any],
    *,
    proc_asound_root: str | Path = "/proc/asound",
    sys_class_sound_root: str | Path = "/sys/class/sound",
    sys_root: str | Path = "/sys",
) -> dict[str, Any]:
    """현재 ALSA PCM의 물리 codec/DAC fingerprint를 fail-closed로 수집한다."""

    audio, _channel_map = validate_measurement_hardware_contract(config)
    payload = {
        "schema": ALSA_PHYSICAL_FINGERPRINT_SCHEMA,
        "input": _alsa_endpoint_physical_fingerprint(
            audio["input"],
            direction="input",
            proc_asound_root=Path(proc_asound_root),
            sys_class_sound_root=Path(sys_class_sound_root),
            sys_root=Path(sys_root),
        ),
        "output": _alsa_endpoint_physical_fingerprint(
            audio["output"],
            direction="output",
            proc_asound_root=Path(proc_asound_root),
            sys_class_sound_root=Path(sys_class_sound_root),
            sys_root=Path(sys_root),
        ),
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def _validate_physical_fingerprint(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != ALSA_PHYSICAL_FINGERPRINT_SCHEMA:
        raise ValueError("hardware identity에 ALSA physical fingerprint가 필요합니다")
    if not all(isinstance(value.get(key), dict) for key in ("input", "output")):
        raise ValueError("ALSA physical fingerprint input/output mapping이 필요합니다")
    expected_sha = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    actual_sha = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if expected_sha != actual_sha:
        raise ValueError("ALSA physical fingerprint SHA가 내용과 다릅니다")
    for direction in ("input", "output"):
        endpoint = value[direction]
        required = {
            "configured_card_id",
            "proc_card_id",
            "pcm_device",
            "pcm_stream",
            "pcm_info",
            "sys_device_realpath",
            "sys_device_uevent",
            "stable_attributes",
        }
        if set(endpoint) != required:
            raise ValueError(f"ALSA physical fingerprint {direction} 필드가 다릅니다")
        if endpoint["configured_card_id"] != endpoint["proc_card_id"]:
            raise ValueError(f"ALSA physical fingerprint {direction} card id가 다릅니다")
        if not endpoint["sys_device_realpath"] or not endpoint["sys_device_uevent"]:
            raise ValueError(f"ALSA physical fingerprint {direction} sysfs identity가 없습니다")
    return json.loads(_canonical_json(value))


def measurement_hardware_identity(
    config: dict[str, Any],
    *,
    physical_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """서로 다른 두 캡처가 같은 논리·물리 I/O 계약인지 비교할 canonical identity."""

    audio, channel_map = validate_measurement_hardware_contract(config)
    identity = {
        "sample_rate": int(audio["sample_rate"]),
        "block_size": int(audio["block_size"]),
        "latency": str(audio["latency"]),
        "input": {
            "card": str(audio["input"]["card"]),
            "pcm": int(audio["input"]["pcm"]),
            "channels": int(audio["input"]["channels"]),
        },
        "output": {
            "card": str(audio["output"]["card"]),
            "pcm": int(audio["output"]["pcm"]),
            "channels": int(audio["output"]["channels"]),
        },
        "channel_map": dict(channel_map),
    }
    identity["physical_fingerprint"] = _validate_physical_fingerprint(
        physical_fingerprint
    )
    return identity


def require_physical_hardware_identity(identity: Any) -> dict[str, Any]:
    """official live/raw/evidence에서 논리-only legacy identity를 거부한다."""

    if not isinstance(identity, dict):
        raise ValueError("hardware_identity mapping이 필요합니다")
    _validate_physical_fingerprint(identity.get("physical_fingerprint"))
    return identity


def _read_proc_cpu_counters(proc_stat_path: str | Path = "/proc/stat") -> tuple[int, int]:
    """``/proc/stat`` 첫 행에서 ``(idle, total)`` jiffy를 읽는다.

    guest/guest_nice는 user/nice에 이미 포함되므로 중복 합산하지 않는다. iowait는
    측정 callback에 CPU 시간을 제공한다고 보장할 수 없어 idle로 세지 않는다.
    """

    try:
        first = Path(proc_stat_path).read_text(encoding="utf-8").splitlines()[0]
        fields = first.split()
        if not fields or fields[0] != "cpu" or len(fields) < 5:
            raise ValueError("aggregate cpu 행이 없습니다")
        counters = [int(value) for value in fields[1:9]]
    except (IndexError, OSError, ValueError) as exc:
        raise RuntimeError(f"CPU 유휴율 witness를 읽을 수 없습니다: {proc_stat_path}") from exc
    if any(value < 0 for value in counters):
        raise RuntimeError("CPU jiffy counter가 음수입니다")
    return counters[3], sum(counters)


def measurement_cpu_idle_fraction(
    *,
    proc_stat_path: str | Path = "/proc/stat",
    sample_seconds: float = MEASUREMENT_CPU_IDLE_SAMPLE_SECONDS,
    sleep_fn=time.sleep,
) -> float:
    """짧은 두 snapshot 사이 aggregate CPU 유휴 비율을 계산한다."""

    interval = float(sample_seconds)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("CPU idle sample_seconds는 양수 finite여야 합니다")
    idle_before, total_before = _read_proc_cpu_counters(proc_stat_path)
    sleep_fn(interval)
    idle_after, total_after = _read_proc_cpu_counters(proc_stat_path)
    idle_delta = idle_after - idle_before
    total_delta = total_after - total_before
    if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
        raise RuntimeError(
            "CPU 유휴율 counter가 단조 증가하지 않았습니다: "
            f"idle_delta={idle_delta}, total_delta={total_delta}"
        )
    return float(idle_delta / total_delta)


def assert_measurement_cpu_idle(
    *, minimum_idle_fraction: float = MIN_MEASUREMENT_CPU_IDLE_FRACTION
) -> float:
    """출력 직전 시스템 부하가 official 측정을 방해하지 않음을 확인한다."""

    minimum = float(minimum_idle_fraction)
    if not math.isfinite(minimum) or not 0.0 < minimum <= 1.0:
        raise ValueError("minimum_idle_fraction은 (0, 1] finite여야 합니다")
    observed = measurement_cpu_idle_fraction()
    if observed < minimum:
        raise RuntimeError(
            "official 측정 CPU 유휴율이 부족합니다: "
            f"observed={observed:.1%}, required>={minimum:.1%}. "
            "테스트·학습·브라우저의 고부하 작업을 멈춘 뒤 다시 확인하세요."
        )
    return observed


def assert_live_pcm_clock_preconditions(hardware: dict[str, Any]) -> None:
    """출력 open 직전 즉시 수행하는 read-only PCM/CPU/clock gate.

    마이크 생존/rail 판정은 앞선 input-only preflight가 담당한다. 여기서 다시
    입력을 캡처하면 그 1.5초 동안 duplex open과의 race window만 늘어나므로,
    proc PCM status, 짧은 aggregate CPU idle witness와 capture clock을 확인한다.
    """

    from deep_anc.audio_io import (
        assert_capture_clock_undisturbed,
        assert_measurement_pcm_unoccupied,
    )

    assert_measurement_pcm_unoccupied(hardware)
    assert_measurement_cpu_idle()
    input_endpoint = hardware.get("input")
    if not isinstance(input_endpoint, dict) or not str(
        input_endpoint.get("card", "")
    ).strip():
        raise ValueError("read-only live gate에 audio.input.card가 필요합니다")
    assert_capture_clock_undisturbed(str(input_endpoint["card"]))


def band_rms_dbfs(samples: np.ndarray) -> float:
    """공식 meter 대역 RMS. live 표시와 raw 재검증이 같은 함수를 쓴다."""

    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    n = int(values.size)
    if n < 256:
        return -120.0
    window = np.hanning(n)
    spectrum = np.fft.rfft(values * window)
    frequencies = np.fft.rfftfreq(n, 1.0 / OFFICIAL_MEASUREMENT_LEVEL.sample_rate)
    lo, hi = OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz
    selected = (frequencies >= lo) & (frequencies <= hi)
    power = 2.0 * np.sum(np.abs(spectrum[selected]) ** 2) / (
        n * np.sum(window**2)
    )
    return 10.0 * np.log10(max(float(power), 1e-24))


def meter_raw_level_dbfs(input_raw_int32: np.ndarray, *, error_channel: int) -> float:
    """저장 raw에서 live 미터의 마지막 2초 중앙값을 exact 재계산한다."""

    from deep_anc.audio_io import pcm_int32_to_float32

    raw = np.asarray(input_raw_int32)
    expected_frames = int(
        round(
            OFFICIAL_MEASUREMENT_LEVEL.sample_rate
            * OFFICIAL_MEASUREMENT_LEVEL.meter_seconds
        )
    )
    if raw.dtype != np.int32 or raw.shape != (expected_frames, 2):
        raise ValueError(
            "meter input raw shape/dtype 계약 위반: "
            f"{raw.shape}/{raw.dtype}, expected=({expected_frames}, 2)/int32"
        )
    hop = int(round(METER_HOP_SECONDS * OFFICIAL_MEASUREMENT_LEVEL.sample_rate))
    signal = pcm_int32_to_float32(raw[:, int(error_channel)]).astype(np.float64)
    levels = [band_rms_dbfs(signal[start : start + hop]) for start in range(0, expected_frames, hop)]
    if len(levels) < 8 or not np.all(np.isfinite(levels)):
        raise ValueError("meter raw에서 finite한 0.25초 level을 재구성하지 못했습니다")
    return float(np.median(np.asarray(levels[-8:], dtype=np.float64)))


def _expected_meter_output_pcm(*, noise_channel: int) -> np.ndarray:
    from deep_anc.audio_io import float32_to_pcm_int16
    from deep_anc.dsp.interleaved_probe import build_interleaved_probe

    contract = OFFICIAL_MEASUREMENT_LEVEL
    probe = build_interleaved_probe(
        sample_rate=contract.sample_rate,
        period_seconds=contract.period_seconds,
        band_hz=contract.design_band_hz,
        amplitude=contract.probe_amplitude,
        tone_spacing_hz=None,
    )
    total = int(round(contract.sample_rate * contract.meter_seconds))
    repeats = int(np.ceil(total / probe.noise_signal.size))
    noise = np.tile(probe.noise_signal, repeats)[:total].astype(np.float32)
    output = np.zeros((total, 2), dtype=np.int16)
    output[:, int(noise_channel)] = float32_to_pcm_int16(noise)
    return output


def _load_npz_metadata_bytes(
    path: Path, payload: bytes
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise ValueError(f"raw NPZ에 metadata_json이 없습니다: {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "metadata_json"
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"raw NPZ를 검증할 수 없습니다: {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"raw metadata root는 mapping이어야 합니다: {path}")
    return metadata, arrays


def _load_npz_metadata(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    payload, digest = _read_npz_snapshot(path)
    metadata, arrays = _load_npz_metadata_bytes(path, payload)
    return metadata, arrays, digest


def meter_receipt_path(raw_path: str | Path) -> Path:
    raw = Path(raw_path)
    return raw.with_name(f"{raw.stem}.receipt.json")


def write_bootstrap_meter_raw_atomic(
    raw_path: str | Path,
    *,
    repository_root: str | Path,
    metadata: dict[str, Any],
    submitted_output_pcm_int16: np.ndarray,
    input_raw_int32: np.ndarray,
) -> dict[str, Any]:
    """meter raw와 SHA receipt를 새 경로에 durable/immutable하게 저장한다."""

    root = Path(repository_root).resolve()
    raw = _path_in_repository(raw_path, repository_root=root)
    receipt = meter_receipt_path(raw)
    if raw.exists() or receipt.exists():
        raise FileExistsError(f"기존 bootstrap meter raw/receipt는 덮어쓰지 않습니다: {raw}")
    if not raw.parent.is_dir():
        raise FileNotFoundError(f"meter raw 상위 디렉터리가 없습니다: {raw.parent}")
    safe_metadata = json.loads(_canonical_json(metadata))
    token = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    raw_temp = raw.parent / f".{raw.name}.{token}.partial"
    receipt_temp = raw.parent / f".{receipt.name}.{token}.partial"
    try:
        with raw_temp.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(_canonical_json(safe_metadata)),
                submitted_output_pcm_int16=np.asarray(
                    submitted_output_pcm_int16, dtype=np.int16
                ),
                input_raw_int32=np.asarray(input_raw_int32, dtype=np.int32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        atomic_publish_noreplace(raw_temp, raw)
        raw_sha = _sha256_file(raw)
        receipt_payload = {
            "schema": BOOTSTRAP_METER_RECEIPT_SCHEMA,
            "raw_path": str(raw.relative_to(root)),
            "raw_sha256": raw_sha,
        }
        with receipt_temp.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(receipt_payload))
            handle.flush()
            os.fsync(handle.fileno())
        atomic_publish_noreplace(receipt_temp, receipt)
    except BaseException:
        raw_temp.unlink(missing_ok=True)
        receipt_temp.unlink(missing_ok=True)
        # raw가 이미 승격된 뒤 receipt만 실패할 수 있다. 유일한 캡처를 지우지 않는다.
        raise
    return {"raw": raw, "receipt": receipt, "sha256": raw_sha}


def _validate_clean_meter_telemetry(telemetry: Any, *, expected_frames: int) -> None:
    if not isinstance(telemetry, dict):
        raise ValueError("meter raw telemetry mapping이 필요합니다")
    failures = {
        "interrupted": bool(telemetry.get("interrupted")),
        "xrun_count": int(telemetry.get("xrun_count", -1)) != 0,
        "unexpected_status_count": int(telemetry.get("unexpected_status_count", -1)) != 0,
        "callback_error": bool(telemetry.get("callback_error")),
        "stream_abort_error": bool(telemetry.get("stream_abort_error")),
        "stream_close_error": bool(telemetry.get("stream_close_error")),
        "output_stop_unconfirmed": telemetry.get("output_stop_confirmed") is not True,
        "capture_incomplete": (
            telemetry.get("completed") is not True
            or int(telemetry.get("output_frames", -1)) != expected_frames
        ),
        "meter_queue_drop": int(telemetry.get("meter_drop_count", -1)) != 0,
        "termination_signal": telemetry.get("termination_signal") is not None,
    }
    bad = [name for name, failed in failures.items() if failed]
    if bad:
        raise ValueError("bootstrap meter raw safety gate 실패: " + ", ".join(bad))
    nominal = expected_frames / float(OFFICIAL_MEASUREMENT_LEVEL.sample_rate)
    if not math.isclose(
        float(telemetry.get("nominal_output_seconds", float("nan"))),
        nominal,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(telemetry.get("hard_max_output_seconds", float("nan"))),
        nominal + LIVE_WATCHDOG_GRACE_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("meter raw nominal/hard-max watchdog 계약 위반")
    _parse_utc(
        telemetry.get("stream_started_at_utc"),
        field="meter telemetry.stream_started_at_utc",
    )


def validate_bootstrap_meter_raw(
    raw_path: str | Path,
    *,
    repository_root: str | Path,
    expected_hardware_identity: dict[str, Any] | None = None,
    now_utc: dt.datetime | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """strict capture 직전 meter raw의 receipt/SHA/device/recipe/status/target 검증."""

    root = Path(repository_root).resolve()
    raw = _path_in_repository(raw_path, repository_root=root)
    receipt = meter_receipt_path(raw)
    if not raw.is_file() or not receipt.is_file():
        raise FileNotFoundError(f"bootstrap meter raw/receipt 쌍이 없습니다: {raw}")
    try:
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"meter raw receipt를 읽을 수 없습니다: {receipt}: {exc}") from exc
    if receipt_payload.get("schema") != BOOTSTRAP_METER_RECEIPT_SCHEMA:
        raise ValueError("meter raw receipt schema가 다릅니다")
    if receipt_payload.get("raw_path") != str(raw.relative_to(root)):
        raise ValueError("meter raw receipt path가 실제 raw와 다릅니다")
    expected_sha = str(receipt_payload.get("raw_sha256", "")).lower()
    metadata, arrays, actual_sha = _load_npz_metadata(raw)
    if len(expected_sha) != 64 or actual_sha != expected_sha:
        raise ValueError(
            f"bootstrap meter raw SHA 불일치: expected={expected_sha}, actual={actual_sha}"
        )
    if metadata.get("schema") != BOOTSTRAP_METER_RAW_SCHEMA:
        raise ValueError(f"bootstrap meter raw schema가 다릅니다: {metadata.get('schema')!r}")
    if metadata.get("status") != "PASS" or metadata.get("passed") is not True:
        raise ValueError("bootstrap meter raw가 PASS 상태가 아닙니다")
    confirmations = metadata.get("operator_confirmations")
    required_confirmations = {
        "speaker_output": True,
        "user_present": True,
        "volume_minimum_before_start": True,
    }
    if confirmations != required_confirmations:
        raise ValueError(
            "bootstrap meter raw operator confirmation 계약 위반: "
            f"{confirmations!r}"
        )
    identity = metadata.get("hardware_identity")
    if not isinstance(identity, dict):
        raise ValueError("bootstrap meter raw에 hardware_identity가 필요합니다")
    require_physical_hardware_identity(identity)
    if expected_hardware_identity is not None and identity != expected_hardware_identity:
        raise ValueError("bootstrap meter raw device/channel identity가 strict capture와 다릅니다")
    channel_map = identity.get("channel_map")
    if channel_map != OFFICIAL_MEASUREMENT_CHANNEL_MAP:
        raise ValueError("bootstrap meter raw channel map이 official 계약과 다릅니다")

    recipe = metadata.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("bootstrap meter raw recipe mapping이 필요합니다")
    exact_recipe = {
        "sample_rate": OFFICIAL_MEASUREMENT_LEVEL.sample_rate,
        "block_size": 256,
        "latency": "low",
        "seconds": OFFICIAL_MEASUREMENT_LEVEL.meter_seconds,
        "probe_amplitude": OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
        "period_seconds": OFFICIAL_MEASUREMENT_LEVEL.period_seconds,
        "design_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.design_band_hz),
        "meter_band_hz": list(OFFICIAL_MEASUREMENT_LEVEL.meter_band_hz),
        "meter_target_dbfs": OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs,
        "meter_tolerance_db": OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db,
        "noise_output_channel": OFFICIAL_MEASUREMENT_CHANNEL_MAP["noise_out"],
        "cancel_output_silent": True,
    }
    if recipe != exact_recipe:
        raise ValueError(f"bootstrap meter raw recipe가 official 계약과 다릅니다: {recipe!r}")
    expected_frames = int(
        round(OFFICIAL_MEASUREMENT_LEVEL.sample_rate * OFFICIAL_MEASUREMENT_LEVEL.meter_seconds)
    )
    _validate_clean_meter_telemetry(metadata.get("telemetry"), expected_frames=expected_frames)
    submitted = arrays.get("submitted_output_pcm_int16")
    input_raw = arrays.get("input_raw_int32")
    if submitted is None or input_raw is None:
        raise ValueError("bootstrap meter raw에 submitted output/int input 배열이 필요합니다")
    expected_output = _expected_meter_output_pcm(
        noise_channel=OFFICIAL_MEASUREMENT_CHANNEL_MAP["noise_out"]
    )
    if submitted.dtype != np.int16 or submitted.shape != expected_output.shape or not np.array_equal(submitted, expected_output):
        raise ValueError("bootstrap meter submitted PCM이 official peak 0.003 recipe와 다릅니다")
    level = meter_raw_level_dbfs(
        input_raw, error_channel=OFFICIAL_MEASUREMENT_CHANNEL_MAP["error_mic"]
    )
    stored_level = float(metadata.get("meter_ch0_dbfs", float("nan")))
    if not math.isfinite(stored_level) or not math.isclose(
        level, stored_level, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"bootstrap meter raw level 재계산 불일치: stored={stored_level}, actual={level}"
        )
    if abs(level - OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs) > OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db:
        raise ValueError(f"bootstrap meter raw가 목표 범위 밖입니다: {level:+.2f} dBFS")

    completed = _parse_utc(metadata.get("completed_at_utc"), field="completed_at_utc")
    stream_started_at = _parse_utc(
        metadata["telemetry"].get("stream_started_at_utc"),
        field="meter telemetry.stream_started_at_utc",
    )
    if completed < stream_started_at:
        raise ValueError("meter 완료시각이 실제 stream start보다 빠릅니다")
    if require_fresh:
        now = (now_utc or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
        age = (now - completed).total_seconds()
        if age < -BOOTSTRAP_CLOCK_FUTURE_TOLERANCE_SECONDS:
            raise ValueError(f"bootstrap meter raw 완료시각이 미래입니다: age={age:.1f}s")
        if age > BOOTSTRAP_METER_MAX_AGE_SECONDS:
            raise ValueError(
                f"bootstrap meter raw가 {age:.1f}초 지나 freshness {BOOTSTRAP_METER_MAX_AGE_SECONDS}초를 넘었습니다"
            )
    return {
        "path": raw,
        "receipt_path": receipt,
        "sha256": actual_sha,
        "metadata": metadata,
        "meter_ch0_dbfs": level,
        "completed_at_utc": completed,
    }


def interleaved_err_noise_bin_dbfs(
    input_raw_int32: np.ndarray,
    *,
    error_channel: int,
    lead_in_samples: int,
    warmup_periods: int,
    repeats: int,
) -> float:
    """보존 strict raw의 분석 반복에서 noise 소유 nominal-bin RMS 중앙값."""

    from deep_anc.audio_io import pcm_int32_to_float32
    from deep_anc.dsp.interleaved_probe import build_interleaved_probe

    contract = OFFICIAL_MEASUREMENT_LEVEL
    probe = build_interleaved_probe(
        sample_rate=contract.sample_rate,
        period_seconds=contract.period_seconds,
        band_hz=contract.design_band_hz,
        amplitude=contract.probe_amplitude,
        tone_spacing_hz=None,
    )
    raw = np.asarray(input_raw_int32)
    if raw.dtype != np.int32 or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("interleaved input raw는 int32 [frames,2]여야 합니다")
    signal = pcm_int32_to_float32(raw[:, int(error_channel)]).astype(np.float64)
    values: list[float] = []
    for index in range(int(repeats)):
        start = int(lead_in_samples) + (int(warmup_periods) + index) * probe.period_samples
        stop = start + probe.period_samples
        segment = signal[start:stop]
        if segment.size != probe.period_samples:
            raise ValueError("interleaved raw가 official 분석 반복보다 짧습니다")
        spectrum = np.fft.rfft(segment)
        power = 2.0 * np.sum(np.abs(spectrum[probe.noise_bins]) ** 2) / (probe.period_samples**2)
        values.append(10.0 * np.log10(max(float(power), 1e-24)))
    if len(values) != contract.analysis_repeats or not np.all(np.isfinite(values)):
        raise ValueError("interleaved ERR noise-bin level을 official 64반복에서 계산하지 못했습니다")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def validate_measurement_hardware_contract(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """레벨 미터와 strict P/S가 공유하는 exact hardware/channel 규약."""

    if not isinstance(config, dict):
        raise ValueError("hardware YAML root는 mapping이어야 합니다")
    audio = config.get("audio")
    channel_map = config.get("channels")
    if not isinstance(audio, dict) or not isinstance(channel_map, dict):
        raise ValueError("hardware YAML에 audio/channels mapping이 모두 필요합니다")
    if int(audio.get("sample_rate", -1)) != OFFICIAL_MEASUREMENT_LEVEL.sample_rate:
        raise ValueError(
            "official hardware sample_rate는 "
            f"{OFFICIAL_MEASUREMENT_LEVEL.sample_rate}여야 합니다: "
            f"{audio.get('sample_rate')!r}"
        )
    if int(audio.get("block_size", -1)) != 256:
        raise ValueError(
            f"official hardware block_size는 256이어야 합니다: {audio.get('block_size')!r}"
        )
    if str(audio.get("latency")) != "low":
        raise ValueError(
            f"official hardware latency는 'low'여야 합니다: {audio.get('latency')!r}"
        )
    for direction in ("input", "output"):
        endpoint = audio.get(direction)
        if not isinstance(endpoint, dict) or int(endpoint.get("channels", -1)) != 2:
            raise ValueError(f"hardware audio.{direction}.channels는 정확히 2여야 합니다")
        if not str(endpoint.get("card", "")).strip():
            raise ValueError(f"hardware audio.{direction}.card가 필요합니다")
        try:
            pcm = int(endpoint["pcm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"hardware audio.{direction}.pcm이 필요합니다") from exc
        if pcm < 0:
            raise ValueError(f"hardware audio.{direction}.pcm은 0 이상이어야 합니다")
    normalized = {
        name: int(channel_map.get(name, -1))
        for name in OFFICIAL_MEASUREMENT_CHANNEL_MAP
    }
    if normalized != OFFICIAL_MEASUREMENT_CHANNEL_MAP or set(channel_map) != set(
        OFFICIAL_MEASUREMENT_CHANNEL_MAP
    ):
        raise ValueError(
            f"hardware channels는 exact {OFFICIAL_MEASUREMENT_CHANNEL_MAP}이어야 합니다: "
            f"{channel_map!r}"
        )
    return audio, normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_interleaved_level_raw(
    raw_path: str | Path,
    *,
    repository_root: str | Path,
    expected_hardware_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """paired evidence에 쓸 strict raw의 device/recipe/status/PCM/level 검증."""

    from deep_anc.audio_io import analyze_int32_input_probe, float32_to_pcm_int16
    from deep_anc.dsp.interleaved_probe import build_interleaved_probe

    root = Path(repository_root).resolve()
    raw = _path_in_repository(raw_path, repository_root=root)
    if not raw.is_file():
        raise FileNotFoundError(f"strict interleaved raw가 없습니다: {raw}")
    metadata, arrays, raw_sha256 = _load_npz_metadata(raw)
    if metadata.get("raw_capture_schema") != INTERLEAVED_RAW_CAPTURE_SCHEMA:
        raise ValueError(
            "strict interleaved raw schema가 다릅니다: "
            f"{metadata.get('raw_capture_schema')!r}"
        )
    identity = metadata.get("hardware_identity")
    if not isinstance(identity, dict):
        raise ValueError("strict interleaved raw에 hardware_identity가 필요합니다")
    require_physical_hardware_identity(identity)
    if expected_hardware_identity is not None and identity != expected_hardware_identity:
        raise ValueError("strict interleaved raw device/channel identity가 meter와 다릅니다")
    if identity.get("channel_map") != OFFICIAL_MEASUREMENT_CHANNEL_MAP:
        raise ValueError("strict interleaved raw channel map이 official 계약과 다릅니다")
    resolved_devices = metadata.get("resolved_devices")
    if (
        not isinstance(resolved_devices, dict)
        or not isinstance(resolved_devices.get("input"), int)
        or not isinstance(resolved_devices.get("output"), int)
    ):
        raise ValueError("strict interleaved raw에 resolved input/output device가 필요합니다")

    contract = OFFICIAL_MEASUREMENT_LEVEL
    exact = {
        "sample_rate": contract.sample_rate,
        "block_size": 256,
        "latency": "low",
        "amplitude": contract.probe_amplitude,
        "period_seconds": contract.period_seconds,
        "warmup_periods": contract.warmup_periods,
        "repeats": contract.analysis_repeats,
        "lead_in_samples": int(round(0.5 * contract.sample_rate)),
    }
    for key, expected in exact.items():
        observed = metadata.get(key)
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"strict interleaved raw {key} 계약 위반: {observed!r} != {expected!r}"
            )
    if metadata.get("invalid_reasons") != []:
        raise ValueError(
            f"strict interleaved raw safety gate 실패: {metadata.get('invalid_reasons')!r}"
        )
    confirmations = metadata.get("operator_confirmations")
    if (
        not isinstance(confirmations, dict)
        or confirmations.get("user_present") is not True
        or confirmations.get("volume_minimum") is not True
        or confirmations.get("routing_and_geometry") is not True
    ):
        raise ValueError("strict interleaved raw operator confirmation이 완전하지 않습니다")
    bootstrap = metadata.get("measurement_level_bootstrap")
    if not isinstance(bootstrap, dict) or bootstrap.get("enabled") is not True or bootstrap.get("same_amplifier_setting_confirmed") is not True:
        raise ValueError("strict interleaved raw가 명시적 same-amplifier bootstrap 캡처가 아닙니다")
    telemetry = metadata.get("telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError("strict interleaved raw telemetry mapping이 필요합니다")
    strict_failures = []
    for name in (
        "xrun_count",
        "unexpected_status_count",
    ):
        if int(telemetry.get(name, -1)) != 0:
            strict_failures.append(name)
    for name in ("callback_error", "stream_abort_error", "stream_close_error"):
        if telemetry.get(name):
            strict_failures.append(name)
    if telemetry.get("completed") is not True:
        strict_failures.append("capture_incomplete")
    if telemetry.get("output_stop_confirmed") is not True:
        strict_failures.append("output_stop_unconfirmed")
    if telemetry.get("termination_signal") is not None:
        strict_failures.append("termination_signal")
    probe = build_interleaved_probe(
        sample_rate=contract.sample_rate,
        period_seconds=contract.period_seconds,
        band_hz=contract.design_band_hz,
        amplitude=contract.probe_amplitude,
        tone_spacing_hz=None,
    )
    total = exact["lead_in_samples"] + (
        contract.warmup_periods + contract.analysis_repeats
    ) * probe.period_samples
    if int(telemetry.get("captured_frames", -1)) != total:
        strict_failures.append("captured_frames")
    nominal_seconds = total / float(contract.sample_rate)
    if not math.isclose(
        float(telemetry.get("nominal_output_seconds", float("nan"))),
        nominal_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(telemetry.get("hard_max_output_seconds", float("nan"))),
        nominal_seconds + LIVE_WATCHDOG_GRACE_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        strict_failures.append("nominal_hard_max_watchdog")
    if strict_failures:
        raise ValueError(
            "strict interleaved raw telemetry gate 실패: " + ", ".join(strict_failures)
        )

    submitted = arrays.get("output_pcm_int16")
    input_raw = arrays.get("input_raw_int32")
    if submitted is None or input_raw is None:
        raise ValueError("strict raw에 observed submitted PCM/int input 배열이 필요합니다")
    expected_float = np.zeros((total, 2), dtype=np.float32)
    repeated = contract.warmup_periods + contract.analysis_repeats
    expected_float[exact["lead_in_samples"] :, OFFICIAL_MEASUREMENT_CHANNEL_MAP["noise_out"]] = np.tile(
        probe.noise_signal, repeated
    )
    expected_float[exact["lead_in_samples"] :, OFFICIAL_MEASUREMENT_CHANNEL_MAP["cancel_out"]] = np.tile(
        probe.cancel_signal, repeated
    )
    expected_pcm = float32_to_pcm_int16(expected_float)
    if submitted.dtype != np.int16 or submitted.shape != expected_pcm.shape or not np.array_equal(submitted, expected_pcm):
        raise ValueError("strict raw observed submitted PCM이 official interleaved recipe와 다릅니다")
    if input_raw.dtype != np.int32 or input_raw.shape != (total, 2):
        raise ValueError(
            f"strict raw input shape/dtype 계약 위반: {input_raw.shape}/{input_raw.dtype}"
        )
    input_report = analyze_int32_input_probe(input_raw)
    channels = input_report.get("channels", [])
    if len(channels) < 2 or not all(bool(item.get("valid")) for item in channels[:2]):
        raise ValueError("strict raw ERR/REF 입력 safety gate가 PASS가 아닙니다")
    if any(float(item.get("clip_ratio", 1.0)) > 0.0 for item in channels[:2]):
        raise ValueError("strict raw ERR/REF 입력에 clipping sample이 있습니다")
    level = interleaved_err_noise_bin_dbfs(
        input_raw,
        error_channel=OFFICIAL_MEASUREMENT_CHANNEL_MAP["error_mic"],
        lead_in_samples=exact["lead_in_samples"],
        warmup_periods=contract.warmup_periods,
        repeats=contract.analysis_repeats,
    )
    if abs(level - contract.interleaved_err_noise_bin_dbfs) > contract.interleaved_err_noise_bin_tolerance_db:
        raise ValueError(
            f"strict interleaved ERR noise-bin이 목표 범위 밖입니다: {level:+.2f} dBFS"
        )
    started = _parse_utc(
        telemetry.get("stream_started_at_utc"),
        field="telemetry.stream_started_at_utc",
    )
    return {
        "path": raw,
        "sha256": raw_sha256,
        "metadata": metadata,
        "interleaved_err_noise_bin_dbfs": level,
        "started_at_utc": started,
        "resolved_devices": dict(resolved_devices),
    }


def _measurement_level_evidence_payload(
    *,
    root: Path,
    meter: dict[str, Any],
    interleaved: dict[str, Any],
    hardware_identity: dict[str, Any],
) -> dict[str, Any]:
    gap = (interleaved["started_at_utc"] - meter["completed_at_utc"]).total_seconds()
    if gap < -BOOTSTRAP_CLOCK_FUTURE_TOLERANCE_SECONDS:
        raise ValueError(f"strict capture가 meter 완료보다 먼저 시작됐습니다: gap={gap:.1f}s")
    if gap > BOOTSTRAP_METER_MAX_AGE_SECONDS:
        raise ValueError(
            f"meter→strict 간격 {gap:.1f}초가 freshness {BOOTSTRAP_METER_MAX_AGE_SECONDS}초를 넘었습니다"
        )
    meter_devices = meter["metadata"].get("resolved_devices")
    if meter_devices != interleaved["resolved_devices"]:
        raise ValueError(
            "meter와 strict capture의 resolved PortAudio device가 다릅니다: "
            f"{meter_devices!r} != {interleaved['resolved_devices']!r}"
        )
    contract = OFFICIAL_MEASUREMENT_LEVEL
    return {
        "schema": MEASUREMENT_LEVEL_EVIDENCE_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_rate": contract.sample_rate,
        "probe_amplitude": contract.probe_amplitude,
        "meter_target_dbfs": contract.meter_target_dbfs,
        "meter_tolerance_db": contract.meter_tolerance_db,
        "interleaved_err_noise_bin_target_dbfs": contract.interleaved_err_noise_bin_dbfs,
        "interleaved_err_noise_bin_tolerance_db": contract.interleaved_err_noise_bin_tolerance_db,
        "meter_ch0_dbfs": meter["meter_ch0_dbfs"],
        "interleaved_err_noise_bin_dbfs": interleaved["interleaved_err_noise_bin_dbfs"],
        "same_amplifier_setting": True,
        "capture_gap_seconds": gap,
        "max_capture_gap_seconds": BOOTSTRAP_METER_MAX_AGE_SECONDS,
        "hardware_identity": hardware_identity,
        "passed": True,
        "meter_raw": {
            "path": str(meter["path"].relative_to(root)),
            "sha256": meter["sha256"],
            "status": "PASS",
            "completed_at_utc": meter["completed_at_utc"].isoformat(),
        },
        "interleaved_raw": {
            "path": str(interleaved["path"].relative_to(root)),
            "sha256": interleaved["sha256"],
            "status": "PASS",
            "started_at_utc": interleaved["started_at_utc"].isoformat(),
        },
    }


def create_measurement_level_evidence_atomic(
    evidence_path: str | Path,
    *,
    repository_root: str | Path,
    meter_raw_path: str | Path,
    interleaved_raw_path: str | Path,
    hardware_identity: dict[str, Any],
) -> dict[str, Any]:
    """fresh paired raw에서 evidence를 temp 검증한 뒤 한 번만 원자 승격한다."""

    root = Path(repository_root).resolve()
    require_physical_hardware_identity(hardware_identity)
    evidence = _path_in_repository(evidence_path, repository_root=root)
    if evidence.exists():
        raise FileExistsError(f"기존 canonical level evidence는 덮어쓰지 않습니다: {evidence}")
    if not evidence.parent.is_dir():
        raise FileNotFoundError(f"level evidence 상위 디렉터리가 없습니다: {evidence.parent}")
    meter = validate_bootstrap_meter_raw(
        meter_raw_path,
        repository_root=root,
        expected_hardware_identity=hardware_identity,
        # live strict entry already checked wall-clock freshness before opening
        # output. 여기서는 raw에 결박된 meter-complete→strict-start 간격을 아래에서
        # 다시 계산한다. 12.5초 캡처 동안 now-age가 경계를 넘었다고 유효 쌍을 버리지 않는다.
        require_fresh=False,
    )
    interleaved = validate_interleaved_level_raw(
        interleaved_raw_path,
        repository_root=root,
        expected_hardware_identity=hardware_identity,
    )
    payload = _measurement_level_evidence_payload(
        root=root,
        meter=meter,
        interleaved=interleaved,
        hardware_identity=hardware_identity,
    )
    token = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    temporary = evidence.parent / f".{evidence.name}.{token}.partial"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        # final 이름을 노출하기 전에 JSON/raw SHA/수치 재계산까지 전부 통과시킨다.
        load_measurement_level_evidence(temporary, repository_root=root)
        atomic_publish_noreplace(temporary, evidence)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return load_measurement_level_evidence(evidence, repository_root=root)


def load_measurement_level_evidence(
    evidence_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """보존된 paired raw에 결박된 레벨 교정 증거를 fail-closed로 검증한다.

    ``-50.1 dBFS``는 예전 probe에서 얻은 메모만으로 새 ``0.003`` probe에
    이식할 수 없다. 이 gate는 ch0 단독 미터 raw와 같은 노브 상태의 interleaved
    raw가 둘 다 남아 있고 SHA가 일치하는 경우에만 실기 진입을 허용한다. JSON의
    수치만 맞춰 쓰는 것은 증거가 아니다.
    """

    root = Path(repository_root).resolve()
    evidence = Path(evidence_path)
    if not evidence.is_absolute():
        evidence = root / evidence
    evidence = evidence.resolve()
    try:
        evidence.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"레벨 증거는 저장소 안에 있어야 합니다: {evidence}") from exc
    if not evidence.is_file():
        raise FileNotFoundError(
            "probe peak 0.003↔meter -50.1 dBFS paired raw 증거가 없습니다: "
            f"{evidence}. 보존 raw 없이 실기 레벨 교정/P/S 측정을 시작하지 않습니다."
        )
    try:
        evidence_bytes = evidence.read_bytes()
        evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
        payload = json.loads(evidence_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"레벨 증거 JSON을 읽을 수 없습니다: {evidence}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("레벨 증거 root는 mapping이어야 합니다")
    if payload.get("schema") != MEASUREMENT_LEVEL_EVIDENCE_SCHEMA:
        raise ValueError(f"알 수 없는 레벨 증거 schema: {payload.get('schema')!r}")
    exact_values = {
        "sample_rate": OFFICIAL_MEASUREMENT_LEVEL.sample_rate,
        "probe_amplitude": OFFICIAL_MEASUREMENT_LEVEL.probe_amplitude,
        "meter_target_dbfs": OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs,
        "meter_tolerance_db": OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db,
        "interleaved_err_noise_bin_target_dbfs": (
            OFFICIAL_MEASUREMENT_LEVEL.interleaved_err_noise_bin_dbfs
        ),
        "interleaved_err_noise_bin_tolerance_db": (
            OFFICIAL_MEASUREMENT_LEVEL.interleaved_err_noise_bin_tolerance_db
        ),
    }
    for name, expected in exact_values.items():
        try:
            observed = float(payload[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"레벨 증거에 {name}가 필요합니다") from exc
        if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"레벨 증거 {name} 계약 위반: {observed!r} != {expected!r}"
            )
    if payload.get("same_amplifier_setting") is not True:
        raise ValueError("두 raw가 같은 앰프 노브 상태라는 확인이 필요합니다")
    if payload.get("passed") is not True:
        raise ValueError("레벨 증거가 PASS로 확정되지 않았습니다")
    try:
        meter_dbfs = float(payload["meter_ch0_dbfs"])
        interleaved_dbfs = float(payload["interleaved_err_noise_bin_dbfs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("레벨 증거의 두 dBFS 값이 필요합니다") from exc
    if not math.isfinite(meter_dbfs) or not math.isfinite(interleaved_dbfs):
        raise ValueError("레벨 증거 dBFS는 finite여야 합니다")
    if abs(meter_dbfs - OFFICIAL_MEASUREMENT_LEVEL.meter_target_dbfs) > (
        OFFICIAL_MEASUREMENT_LEVEL.meter_tolerance_db
    ):
        raise ValueError(f"paired meter ch0가 목표 범위 밖입니다: {meter_dbfs:+.2f} dBFS")
    if abs(
        interleaved_dbfs
        - OFFICIAL_MEASUREMENT_LEVEL.interleaved_err_noise_bin_dbfs
    ) > OFFICIAL_MEASUREMENT_LEVEL.interleaved_err_noise_bin_tolerance_db:
        raise ValueError(
            "paired interleaved ERR noise-bin이 보존 기준과 맞지 않습니다: "
            f"{interleaved_dbfs:+.2f} dBFS"
        )

    raw_entries: list[tuple[str, Path]] = []
    for key in ("meter_raw", "interleaved_raw"):
        item = payload.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"레벨 증거에 {key} mapping이 필요합니다")
        relative = item.get("path")
        expected_sha = str(item.get("sha256", "")).lower()
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{key}.path가 필요합니다")
        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise ValueError(f"{key}.sha256은 64자리 lowercase hex여야 합니다")
        raw_path = (root / relative).resolve()
        try:
            raw_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{key}는 저장소 안에 있어야 합니다: {raw_path}") from exc
        if not raw_path.is_file():
            raise FileNotFoundError(f"보존 raw가 없습니다: {raw_path}")
        actual_sha = _sha256_file(raw_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{key} SHA 불일치: expected={expected_sha}, actual={actual_sha}"
            )
        raw_entries.append((key, raw_path))
    if raw_entries[0][1] == raw_entries[1][1]:
        raise ValueError("meter/interleaved paired evidence는 서로 다른 보존 raw여야 합니다")

    hardware_identity = payload.get("hardware_identity")
    if not isinstance(hardware_identity, dict):
        raise ValueError("레벨 증거에 hardware_identity가 필요합니다")
    require_physical_hardware_identity(hardware_identity)
    meter_verified = validate_bootstrap_meter_raw(
        raw_entries[0][1],
        repository_root=root,
        expected_hardware_identity=hardware_identity,
        require_fresh=False,
    )
    interleaved_verified = validate_interleaved_level_raw(
        raw_entries[1][1],
        repository_root=root,
        expected_hardware_identity=hardware_identity,
    )
    if meter_verified["sha256"] != payload["meter_raw"]["sha256"]:
        raise ValueError("meter raw가 SHA 검사와 NPZ load 사이에 변경됐습니다")
    if interleaved_verified["sha256"] != payload["interleaved_raw"]["sha256"]:
        raise ValueError("interleaved raw가 SHA 검사와 NPZ load 사이에 변경됐습니다")
    if not math.isclose(
        meter_dbfs,
        float(meter_verified["meter_ch0_dbfs"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("레벨 증거 meter dBFS가 raw 재계산과 다릅니다")
    if not math.isclose(
        interleaved_dbfs,
        float(interleaved_verified["interleaved_err_noise_bin_dbfs"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("레벨 증거 interleaved dBFS가 raw 재계산과 다릅니다")
    gap = (
        interleaved_verified["started_at_utc"]
        - meter_verified["completed_at_utc"]
    ).total_seconds()
    try:
        stored_gap = float(payload["capture_gap_seconds"])
        stored_max_gap = float(payload["max_capture_gap_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("레벨 증거에 capture gap 계약이 필요합니다") from exc
    if not math.isclose(stored_gap, gap, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"레벨 증거 capture gap 불일치: {stored_gap} != {gap}")
    if stored_max_gap != float(BOOTSTRAP_METER_MAX_AGE_SECONDS):
        raise ValueError("레벨 증거 max capture gap 계약이 다릅니다")
    if gap < -BOOTSTRAP_CLOCK_FUTURE_TOLERANCE_SECONDS or gap > stored_max_gap:
        raise ValueError(f"레벨 증거 meter→strict freshness 위반: {gap:.1f}s")
    for key in ("meter_raw", "interleaved_raw"):
        if payload[key].get("status") != "PASS":
            raise ValueError(f"레벨 증거 {key} status가 PASS가 아닙니다")
    if payload["meter_raw"].get("completed_at_utc") != meter_verified[
        "completed_at_utc"
    ].isoformat():
        raise ValueError("레벨 증거 meter 완료시각이 raw metadata와 다릅니다")
    if payload["interleaved_raw"].get("started_at_utc") != interleaved_verified[
        "started_at_utc"
    ].isoformat():
        raise ValueError("레벨 증거 strict 시작시각이 raw metadata와 다릅니다")
    _parse_utc(payload.get("created_at_utc"), field="created_at_utc")
    verified = dict(payload)
    verified["_evidence_path"] = str(evidence)
    verified["_evidence_sha256"] = evidence_sha256
    return verified
