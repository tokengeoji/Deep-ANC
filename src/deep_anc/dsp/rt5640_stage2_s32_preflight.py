"""RT5640/J511 Stage-2 S32의 읽기 전용 physical preflight.

이 모듈은 이후의 actual Stage-2 P/S capture adapter가 stream을 열기 *전*에
통과해야 하는 한정된 하드웨어 점검이다. ALSA PCM, PortAudio, Pulse profile,
mixer, pinmux를 변경하거나 열지 않으며 speaker PCM을 만들지 않는다.

특히 USB AB13X output + APE input의 split-clock 경로를 Stage-2 RT5640/J511
common-clock 후보로 잘못 승격하지 않도록, sealed Stage-2 S32 contract와 현재
APE mux/J511/PCM 점유를 한 receipt에 묶는다. 이 preflight는 구형 fallback static
receipt가 아니라 actual P/S 전용 sealed config의 SHA·Stage-2 contract·금지 계보를
반드시 결속한다. J511 plug가 물리적으로 감지되지 않으면 ``HP``/``HS`` 3회 안정
관측 전 단계에서 fail-closed한다.

이 receipt의 PASS도 실제 output voltage, amp 반대편 연결, shared-frame identity,
P/S, lead, ANC attenuation 또는 학습 권한을 뜻하지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import MappingProxyType
from typing import Any

from deep_anc.realtime.rt5640_jack import JACK_CONTROL_NAME, parse_jack_state_amixer

from .stage2_2khz_actual_ps_plan import (
    DEFAULT_CONFIG_RELATIVE_PATH as ACTUAL_PS_CONFIG_RELATIVE_PATH,
    STATIC_RECEIPT_SCHEMA as ACTUAL_PS_STATIC_RECEIPT_SCHEMA,
    load_stage2_actual_ps_static_config,
)
from .stage2_2khz_contract import Stage2TwoKilohertzContract


SCHEMA = "rt5640_stage2_s32_read_only_preflight_v1"
PASS_STATUS = "PASS_READ_ONLY_RT5640_STAGE2_S32_PREFLIGHT"
J511_DISCONNECTED_STATUS = "BLOCKED_J511_NOT_CONNECTED"
J511_INVALID_STATUS = "BLOCKED_J511_NOT_HP_OR_HS_STABLE"
PCM_OCCUPIED_STATUS = "BLOCKED_PCM_STREAM_OCCUPIED"
ROUTE_MISMATCH_STATUS = "BLOCKED_APE_ROUTE_OR_TRANSPORT_MISMATCH"
ACTUAL_CONFIG_PROVENANCE_STATUS = "BLOCKED_STAGE2_ACTUAL_PS_CONFIG_PROVENANCE"
READ_ONLY_FAILURE_STATUS = "FAIL_READ_ONLY_PREFLIGHT"

J511_ALLOWED_STATES = frozenset({"HP", "HS"})
J511_REQUIRED_SAMPLES = 3
EXPECTED_ROUTES = {
    "I2S1 Mux": "ADMAIF1",
    "ADMAIF1 Mux": "I2S1",
    "ADMAIF2 Mux": "I2S2",
    "I2S2 Mux": "ADMAIF2",
}
EXPECTED_HARDWARE_AUDIO = {
    "sample_rate_hz": 48_000,
    "block_size": 256,
    "latency": "low",
    "clock_domain": "APE_PLL_A_SHARED",
    "input": {
        "card": "APE",
        "pcm": 1,
        "channels": 2,
        "format": "S32_LE",
        "route": "I2S2_ADMAIF2_ERR_REF",
    },
    "output": {
        "card": "APE",
        "pcm": 0,
        "channels": 2,
        "format": "S32_LE",
        "route": "ADMAIF1_I2S1_RT5640_J511",
    },
}
_ITEM_RE = re.compile(r"^\s*;\s*Item\s+#(?P<index>[0-9]+)\s+'(?P<name>[^']+)'\s*$")
_VALUE_RE = re.compile(r"^\s*:\s*values=(?P<value>[0-9]+)\s*$")
_CARD_RE = re.compile(r"^\s*(?P<index>[0-9]+)\s+\[(?P<card>[^\]]+)\]\s*:")


@dataclass(frozen=True)
class ReadOnlyCommandResult:
    """shell 없이 실행한 진단 명령의 immutable 결과."""

    returncode: int
    stdout: str
    stderr: str


ReadOnlyRunner = Callable[[tuple[str, ...]], ReadOnlyCommandResult]
ActualPsConfigLoader = Callable[[], Mapping[str, Any]]

_ACTUAL_PS_CONFIG_STATUS = "PLAN_ONLY_ACTUAL_PS_CAPTURE_NOT_AUTHORIZED"
_LEGACY_FALLBACK_STATIC_RECEIPT_SCHEMA = "stage2_2khz_rt5640_s32_static_receipt_v1"
_REQUIRED_FORBIDDEN_LINEAGE = {
    "usb_ab13x": True,
    "output_master_split_clock": True,
    "bandlimited_fallback": True,
    "s16_transport": True,
    "legacy_relabel_or_promotion": True,
}
_REQUIRED_PLAN_ONLY_AUTHORITY = {
    "plan_preparation_only": True,
    "j511_connection_observed": False,
    "same_card_s32_transport_pass": False,
    "post_start_hw_params_route_receipt_pass": False,
    "raw_first_capture_published_no_replace": False,
    "shared_clock_or_fixed_lti_condition_pass": False,
    "stage2_ps_identification_pass": False,
    "physical_ps_authority": False,
    "canonical_training_eligible": False,
    "deployment_eligible": False,
}


def _default_runner(command: tuple[str, ...]) -> ReadOnlyCommandResult:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"read-only command 실행 실패: {list(command)!r}: {error}") from error
    return ReadOnlyCommandResult(
        returncode=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        receipt_to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    """receipt 반환 뒤 caller mutation을 차단한다."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def receipt_to_jsonable(value: Any) -> Any:
    """immutable receipt를 stdout JSON에만 직렬화 가능한 값으로 복원한다."""

    if isinstance(value, Mapping):
        return {str(key): receipt_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [receipt_to_jsonable(item) for item in value]
    return value


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"read-only 경로를 읽지 못했습니다: {path}: {error}") from error


def _run_exact(runner: ReadOnlyRunner, command: tuple[str, ...]) -> ReadOnlyCommandResult:
    result = runner(command)
    if not isinstance(result, ReadOnlyCommandResult):
        raise TypeError("read-only runner는 ReadOnlyCommandResult를 반환해야 합니다")
    if type(result.returncode) is not int or not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
        raise TypeError("read-only runner 반환 형식이 다릅니다")
    return result


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise RuntimeError(f"{label}는 64자리 SHA-256이어야 합니다")
    try:
        int(value, 16)
    except ValueError as error:
        raise RuntimeError(f"{label}는 hexadecimal SHA-256이어야 합니다") from error
    return value


def _require_actual_ps_same_card_s32(config: Mapping[str, Any]) -> dict[str, Any]:
    """actual P/S sealed config만 허용하고 구형 fallback receipt를 거절한다."""

    schema = config.get("schema")
    if schema == _LEGACY_FALLBACK_STATIC_RECEIPT_SCHEMA:
        raise RuntimeError("구형 fallback Stage-2 RT5640 static receipt는 actual P/S preflight에 사용할 수 없습니다")
    if schema != ACTUAL_PS_STATIC_RECEIPT_SCHEMA:
        raise RuntimeError(f"actual P/S static receipt schema가 다릅니다: {schema!r}")
    if config.get("status") != _ACTUAL_PS_CONFIG_STATUS:
        raise RuntimeError("actual P/S static receipt status가 plan-only sealed 상태가 아닙니다")
    for key in ("audio_opened", "speaker_output", "results_written"):
        if config.get(key) is not False:
            raise RuntimeError(f"actual P/S static receipt.{key}는 false여야 합니다")

    hardware = config.get("hardware_audio")
    if not isinstance(hardware, Mapping):
        raise RuntimeError("actual P/S static config에 hardware_audio가 없습니다")
    for key, expected in EXPECTED_HARDWARE_AUDIO.items():
        if hardware.get(key) != expected:
            raise RuntimeError(
                "USB AB13X/output-master split-clock/S16 경로 또는 다른 APE route를 거절합니다: "
                f"hardware_audio.{key}={hardware.get(key)!r}"
            )
    forbidden = config.get("forbidden_source_or_receipt_origins")
    if forbidden != _REQUIRED_FORBIDDEN_LINEAGE:
        raise RuntimeError("actual P/S config의 USB/split-clock/fallback/S16 금지 계보가 다릅니다")
    authority = config.get("authority")
    if authority != _REQUIRED_PLAN_ONLY_AUTHORITY:
        raise RuntimeError("actual P/S static config authority가 plan-only 상태와 다릅니다")

    expected_contract_sha256 = Stage2TwoKilohertzContract.canonical().digest()
    if config.get("stage2_contract_sha256") != expected_contract_sha256:
        raise RuntimeError("actual P/S static config의 Stage-2 contract SHA가 다릅니다")
    config_payload_sha256 = _require_sha256(
        config.get("config_payload_sha256"), label="actual P/S config_payload_sha256"
    )
    file_binding = config.get("config")
    if not isinstance(file_binding, Mapping) or set(file_binding) != {"path", "file_sha256"}:
        raise RuntimeError("actual P/S static config file binding이 exact하지 않습니다")
    config_path = file_binding.get("path")
    if type(config_path) is not str or not config_path.replace("\\", "/").endswith(ACTUAL_PS_CONFIG_RELATIVE_PATH):
        raise RuntimeError("actual P/S static config path가 sealed default와 다릅니다")
    config_file_sha256 = _require_sha256(
        file_binding.get("file_sha256"), label="actual P/S config file_sha256"
    )
    return {
        "schema": schema,
        "status": _ACTUAL_PS_CONFIG_STATUS,
        "audio_opened": False,
        "speaker_output": False,
        "results_written": False,
        "hardware_audio": receipt_to_jsonable(hardware),
        "config_path": config_path,
        "config_file_sha256": config_file_sha256,
        "config_payload_sha256": config_payload_sha256,
        "stage2_contract_sha256": expected_contract_sha256,
        "forbidden_source_or_receipt_origins": receipt_to_jsonable(forbidden),
        "authority": receipt_to_jsonable(authority),
        "prohibited_transports": {
            "usb_ab13x_selected": False,
            "output_master_split_clock_selected": False,
            "bandlimited_fallback_selected": False,
            "s16_selected": False,
            "contract_forbids_usb_ab13x": True,
            "contract_forbids_output_master_split_clock": True,
            "contract_forbids_bandlimited_fallback": True,
            "contract_forbids_s16": True,
        },
    }


def _read_cards(proc_asound_root: Path) -> tuple[dict[str, Any], ...]:
    raw = _read_text(proc_asound_root / "cards")
    cards: list[dict[str, Any]] = []
    for line in raw.splitlines():
        match = _CARD_RE.match(line)
        if match is None:
            continue
        cards.append({"index": int(match.group("index")), "id": match.group("card").strip()})
    if not cards:
        raise RuntimeError("/proc/asound/cards에서 ALSA card를 읽지 못했습니다")
    if sum(card["id"] == "APE" for card in cards) != 1:
        raise RuntimeError("ALSA APE card가 exact 하나가 아닙니다")
    return tuple(cards)


def _read_j511_samples(runner: ReadOnlyRunner) -> tuple[str, ...]:
    command = ("amixer", "-c", "APE", "cget", f"name={JACK_CONTROL_NAME}")
    states: list[str] = []
    for _ in range(J511_REQUIRED_SAMPLES):
        result = _run_exact(runner, command)
        if result.returncode != 0:
            raise RuntimeError(
                f"RT5640 J511 상태 조회가 실패했습니다 (exit={result.returncode}): {result.stderr.strip()}"
            )
        states.append(parse_jack_state_amixer(result.stdout))
    return tuple(states)


def _read_ape_routes(runner: ReadOnlyRunner) -> Mapping[str, Any]:
    controls: dict[str, Any] = {}
    for control, expected in EXPECTED_ROUTES.items():
        result = _run_exact(runner, ("amixer", "-c", "APE", "cget", f"name={control}"))
        if result.returncode != 0:
            raise RuntimeError(f"APE route {control!r} 조회 실패 (exit={result.returncode})")
        items: dict[int, str] = {}
        values: list[int] = []
        for line in result.stdout.splitlines():
            item = _ITEM_RE.match(line)
            if item is not None:
                index = int(item.group("index"))
                if index in items:
                    raise RuntimeError(f"APE route {control!r} enum index가 중복됩니다")
                items[index] = item.group("name")
                continue
            value = _VALUE_RE.match(line)
            if value is not None:
                values.append(int(value.group("value")))
        if len(values) != 1 or values[0] not in items:
            raise RuntimeError(f"APE route {control!r} current enum 값을 해석하지 못했습니다")
        observed = items[values[0]]
        if observed != expected:
            raise RuntimeError(
                f"APE route {control!r}={observed!r}, expected={expected!r}; 자동 변경하지 않습니다"
            )
        controls[control] = {
            "observed": observed,
            "expected": expected,
            "enum_index": values[0],
            "raw_sha256": _sha256_bytes(result.stdout.encode("utf-8")),
            "read_only_command": "amixer cget",
        }
    return MappingProxyType(controls)


def _read_pcm_occupancy(
    *, proc_asound_root: Path, dev_snd_root: Path, runner: ReadOnlyRunner
) -> Mapping[str, Any]:
    statuses = sorted(proc_asound_root.glob("card*/pcm*/sub*/status"))
    if not statuses:
        raise RuntimeError("/proc/asound 아래 PCM substream status가 없습니다")
    rows: list[dict[str, Any]] = []
    busy: list[str] = []
    for path in statuses:
        raw = _read_text(path)
        state = raw.strip()
        relative = str(path.relative_to(proc_asound_root))
        if state != "closed":
            busy.append(f"{relative}={state!r}")
        rows.append(
            {
                "path": relative,
                "state": state,
                "raw_sha256": _sha256_bytes(raw.encode("utf-8")),
            }
        )

    pcm_nodes = tuple(sorted(path for path in dev_snd_root.glob("pcm*") if path.is_file() or path.exists()))
    if not pcm_nodes:
        raise RuntimeError("/dev/snd 아래 PCM device node가 없습니다")
    fuser = _run_exact(runner, ("fuser", "-v", *(str(path) for path in pcm_nodes)))
    if fuser.returncode == 0:
        busy.append("fuser reports an owner for one or more /dev/snd/pcm* nodes")
    elif fuser.returncode != 1:
        raise RuntimeError(f"fuser PCM ownership 조회 실패 (exit={fuser.returncode}): {fuser.stderr.strip()}")
    if busy:
        raise RuntimeError("PCM stream이 점유되어 있습니다: " + "; ".join(busy))
    return MappingProxyType(
        {
            "all_pcm_substreams_closed": True,
            "status_rows": tuple(rows),
            "fuser_pcm_nodes": tuple(str(path) for path in pcm_nodes),
            "fuser_returncode": 1,
            "fuser_stdout_sha256": _sha256_bytes(fuser.stdout.encode("utf-8")),
            "fuser_stderr_sha256": _sha256_bytes(fuser.stderr.encode("utf-8")),
            "owners": tuple(),
        }
    )


def _failure_status(
    *, actual_config_error: str | None, j511_states: tuple[str, ...] | None, errors: list[str]
) -> str:
    if actual_config_error is not None:
        return ACTUAL_CONFIG_PROVENANCE_STATUS
    if j511_states is not None and all(state == "None" for state in j511_states):
        return J511_DISCONNECTED_STATUS
    if j511_states is not None and (
        len(j511_states) != J511_REQUIRED_SAMPLES
        or any(state not in J511_ALLOWED_STATES for state in j511_states)
        or len(set(j511_states)) != 1
    ):
        return J511_INVALID_STATUS
    if any("PCM stream" in error or "PCM substream" in error or "/dev/snd" in error for error in errors):
        return PCM_OCCUPIED_STATUS
    if any("APE route" in error for error in errors):
        return ROUTE_MISMATCH_STATUS
    return READ_ONLY_FAILURE_STATUS


def collect_rt5640_stage2_s32_preflight(
    *,
    proc_asound_root: Path = Path("/proc/asound"),
    dev_snd_root: Path = Path("/dev/snd"),
    runner: ReadOnlyRunner = _default_runner,
    actual_ps_config_loader: ActualPsConfigLoader = load_stage2_actual_ps_static_config,
) -> Mapping[str, Any]:
    """실제 stream open 없이 Stage-2 RT5640 S32 physical preflight를 수집한다.

    반환 receipt는 성공·실패 모두 immutable이다. 실패도 raw-first 원칙에 맞춰 현재
    관측값과 read-only 오류를 보존하되, 파일에는 쓰지 않는다.
    """

    errors: list[str] = []
    actual_config_summary: Mapping[str, Any] | None = None
    j511_states: tuple[str, ...] | None = None
    cards: tuple[dict[str, Any], ...] | None = None
    routes: Mapping[str, Any] | None = None
    occupancy: Mapping[str, Any] | None = None

    try:
        actual_config_summary = _require_actual_ps_same_card_s32(actual_ps_config_loader())
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError) as error:
        errors.append(f"actual_ps_config: {type(error).__name__}: {error}")
    try:
        cards = _read_cards(proc_asound_root)
    except RuntimeError as error:
        errors.append(f"alsa_cards: {error}")
    try:
        j511_states = _read_j511_samples(runner)
        if not (
            len(j511_states) == J511_REQUIRED_SAMPLES
            and len(set(j511_states)) == 1
            and j511_states[0] in J511_ALLOWED_STATES
        ):
            errors.append(
                "j511: 3회 모두 같은 HP 또는 HS여야 합니다; observed=" + repr(list(j511_states))
            )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        errors.append(f"j511: {type(error).__name__}: {error}")
    try:
        routes = _read_ape_routes(runner)
    except (OSError, RuntimeError, TypeError) as error:
        errors.append(f"routes: {type(error).__name__}: {error}")
    try:
        occupancy = _read_pcm_occupancy(
            proc_asound_root=proc_asound_root, dev_snd_root=dev_snd_root, runner=runner
        )
    except (OSError, RuntimeError, TypeError) as error:
        errors.append(f"occupancy: {type(error).__name__}: {error}")

    actual_config_error = next((item for item in errors if item.startswith("actual_ps_config:")), None)
    passed = not errors
    status = (
        PASS_STATUS
        if passed
        else _failure_status(
            actual_config_error=actual_config_error, j511_states=j511_states, errors=errors
        )
    )
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "passed": passed,
        "dry_run": True,
        "audio_backend_imported": False,
        "alsa_pcm_opened": False,
        "speaker_output": False,
        "filesystem_write_performed": False,
        "pulse_profile_mutation_performed": False,
        "mixer_mutation_performed": False,
        "pinmux_or_device_tree_mutation_performed": False,
        "required_j511_samples": J511_REQUIRED_SAMPLES,
        "allowed_j511_states": tuple(sorted(J511_ALLOWED_STATES)),
        "j511": {
            "control": JACK_CONTROL_NAME,
            "observed_states": tuple() if j511_states is None else j511_states,
            "three_identical_connected_samples": bool(
                j511_states is not None
                and len(j511_states) == J511_REQUIRED_SAMPLES
                and len(set(j511_states)) == 1
                and j511_states[0] in J511_ALLOWED_STATES
            ),
            "amplifier_end_connected": False,
            "electrical_output_witness": False,
            "acoustic_output_witness": False,
        },
        "alsa_cards": tuple() if cards is None else cards,
        "actual_ps_config": None if actual_config_summary is None else actual_config_summary,
        "ape_routes": None if routes is None else routes,
        "pcm_occupancy": None if occupancy is None else occupancy,
        "errors": tuple(errors),
        "authority": {
            "read_only_connection_and_idle_gate_pass": passed,
            "same_card_s32_actual_config_provenance_pass": actual_config_summary is not None,
            "j511_physical_plug_detected": bool(
                j511_states is not None
                and len(j511_states) == J511_REQUIRED_SAMPLES
                and len(set(j511_states)) == 1
                and j511_states[0] in J511_ALLOWED_STATES
            ),
            "usb_ab13x_or_output_master_path_allowed": False,
            "s16_transport_allowed": False,
            "actual_s32_stream_opened": False,
            "same_hardware_frame_identity_pass": False,
            "clock_or_fixed_lti_witness_pass": False,
            "stage2_ps_identification_pass": False,
            "canonical_training_eligible": False,
            "deployment_eligible": False,
        },
    }
    payload = _canonical_json_bytes(core)
    core["receipt_sha256"] = _sha256_bytes(payload)
    return _freeze(core)


def assert_rt5640_stage2_s32_preflight(**kwargs: Any) -> Mapping[str, Any]:
    """PASS receipt만 반환하고, 나머지는 capture admission 전에 예외로 막는다."""

    receipt = collect_rt5640_stage2_s32_preflight(**kwargs)
    if receipt["passed"] is not True:
        raise RuntimeError(
            f"RT5640 Stage-2 S32 read-only preflight가 통과하지 않았습니다: "
            f"{receipt['status']}; errors={list(receipt['errors'])}"
        )
    return receipt


__all__ = [
    "ACTUAL_CONFIG_PROVENANCE_STATUS",
    "EXPECTED_HARDWARE_AUDIO",
    "EXPECTED_ROUTES",
    "J511_ALLOWED_STATES",
    "J511_REQUIRED_SAMPLES",
    "PASS_STATUS",
    "READ_ONLY_FAILURE_STATUS",
    "ReadOnlyCommandResult",
    "SCHEMA",
    "assert_rt5640_stage2_s32_preflight",
    "collect_rt5640_stage2_s32_preflight",
    "receipt_to_jsonable",
]
