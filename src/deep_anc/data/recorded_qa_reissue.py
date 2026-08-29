"""현행 recorded QA의 manifest·시간축 authority를 재발행/검증한다.

``recorded_qa.json`` 자체는 파일 품질 QA의 결과일 뿐이다. 과거에는 그 파일에
manifest 바이트 지문도, strict P/S에서 유도한 timing contract도 없어서, 예전
``recorded_train.jsonl``/lead=116 결과를 현재
``recorded_regrouped.jsonl``/lead=115 evidence처럼 읽을 수 있었다.

이 모듈은 그 두 authority를 QA 결과와 함께 봉인한다. 이는 P/S 또는 raw 녹음을
바꾸지 않는 읽기 전용 data gate다. 새 strict P/S 또는 canonical manifest가 바뀌면
기존 reissue report는 의도적으로 검증에 실패해야 한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, load_yaml
from ..dsp.secondary_path import load_secondary_path
from ..dsp.timing import PlantDelays, TrainingTimingContract
from .manifest import VALID_SPLITS, read_manifest
from .recorded_qa import (
    render_recorded_qa_markdown,
    settings_from_data_config,
    validate_recorded_sessions,
)


RECORDED_QA_REISSUE_SCHEMA = "recorded_qa_reissue_v1"
"""새 QA report의 ``provenance.schema`` exact 값."""

CANONICAL_RECORDED_MANIFEST = "data/manifests/recorded_regrouped.jsonl"
"""현재 82-session lineage regrouping authority의 유일한 경로."""


class RecordedQAReissueError(ValueError):
    """QA report가 현재 canonical manifest/timing evidence가 아닐 때 발생한다."""


def sha256_file(path: str | Path) -> str:
    """regular file의 SHA-256을 반환한다. 디렉터리/누락 파일은 fail-closed다."""

    value = Path(path)
    if not value.is_file():
        raise RecordedQAReissueError(f"regular file이 아닙니다: {value}")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within_root(path: Path, root: Path, *, label: str) -> Path:
    """root 내부의 regular file만 authority 입력으로 허용한다."""

    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RecordedQAReissueError(
            f"{label}가 repository 밖입니다: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise RecordedQAReissueError(f"{label}가 regular file이 아닙니다: {resolved}")
    return resolved


def _relative_file(path: str | Path, root: Path, *, label: str) -> tuple[Path, str]:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = root / value
    resolved = _within_root(value, root, label=label)
    return resolved, resolved.relative_to(root).as_posix()


def _strict_measured_timing_contract(
    *,
    root: Path,
    data_cfg: dict[str, Any],
    duct_cfg: dict[str, Any],
) -> tuple[TrainingTimingContract, dict[str, object]]:
    """strict measured P/S에서 QA의 lead를 유도한다.

    ``data_sim.yaml``의 기본 primary mode는 surrogate pretrain용이다. recorded
    fine-tune QA의 길이/시간축은 그 표현용 기본값이 아니라 현행 strict measured
    P/S가 결정하므로 여기서는 mode를 *선택*하지 않고 명시적으로 ``measured``로
    고정한다. lead 숫자는 절대 전달받지 않으며 ``PlantDelays.lead()``에서만 나온다.
    """

    if str(data_cfg.get("reference_mode", "digital")) != "digital":
        raise RecordedQAReissueError("canonical recorded QA는 digital reference만 허용합니다")
    sample_rate = data_cfg.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise RecordedQAReissueError("data_config.sample_rate는 양의 exact int여야 합니다")

    secondary_cfg = duct_cfg.get("secondary_path")
    digital_cfg = duct_cfg.get("digital_reference")
    if not isinstance(secondary_cfg, dict) or not isinstance(digital_cfg, dict):
        raise RecordedQAReissueError("duct config의 secondary_path/digital_reference가 필요합니다")
    secondary_value = secondary_cfg.get("npz")
    primary_value = digital_cfg.get("primary_path_npz")
    if not isinstance(secondary_value, str) or not secondary_value:
        raise RecordedQAReissueError("strict secondary_path.npz가 필요합니다")
    if not isinstance(primary_value, str) or not primary_value:
        raise RecordedQAReissueError("strict digital_reference.primary_path_npz가 필요합니다")

    secondary_path, secondary_relative = _relative_file(
        secondary_value, root, label="strict secondary P/S"
    )
    primary_path, primary_relative = _relative_file(
        primary_value, root, label="strict primary P/S"
    )
    secondary = load_secondary_path(secondary_path)
    primary = load_secondary_path(primary_path)
    if int(secondary.sample_rate) != sample_rate or int(primary.sample_rate) != sample_rate:
        raise RecordedQAReissueError(
            "strict P/S sample_rate와 data_config.sample_rate가 모두 같아야 합니다: "
            f"P={primary.sample_rate}, S={secondary.sample_rate}, data={sample_rate}"
        )

    delays = PlantDelays.from_config(
        duct_cfg=duct_cfg,
        primary_delay_samples=int(primary.delay_samples),
        secondary_delay_samples=int(secondary.delay_samples),
        sample_rate=sample_rate,
    )
    contract = TrainingTimingContract.derive(
        primary_fir=primary.fir,
        plant_delays=delays,
    )
    lead = delays.lead()
    if int(contract.digital_reference_lead_samples) != int(lead.samples):
        raise RuntimeError("TrainingTimingContract와 PlantDelays.lead()가 다릅니다")
    return contract, {
        "primary_path": {
            "path": primary_relative,
            "sha256": sha256_file(primary_path),
            "delay_samples": int(primary.delay_samples),
        },
        "secondary_path": {
            "path": secondary_relative,
            "sha256": sha256_file(secondary_path),
            "delay_samples": int(secondary.delay_samples),
        },
        "handoff_samples": int(delays.handoff_samples),
        "plant_delays_lead_samples": int(lead.samples),
    }


def build_current_recorded_qa_provenance(
    *,
    repo_root: str | Path = REPO_ROOT,
    manifest_path: str | Path = CANONICAL_RECORDED_MANIFEST,
    data_config_path: str | Path = "configs/data_sim.yaml",
    duct_config_path: str | Path = "configs/duct.yaml",
) -> dict[str, object]:
    """현재 canonical recorded QA가 반드시 참조해야 하는 binding을 만든다.

    이 함수는 raw recording을 열지 않는다. manifest/P/S/config 바이트와 strict
    ``TrainingTimingContract``만 읽는다. 따라서 기존 report audit 전에도 안전하게
    호출할 수 있다.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise RecordedQAReissueError(f"repository root가 디렉터리가 아닙니다: {root}")

    expected_manifest = (root / CANONICAL_RECORDED_MANIFEST).resolve()
    manifest, manifest_relative = _relative_file(
        manifest_path, root, label="recorded QA manifest"
    )
    if manifest != expected_manifest:
        raise RecordedQAReissueError(
            "canonical recorded QA는 정확히 "
            f"{CANONICAL_RECORDED_MANIFEST}만 허용합니다: {manifest_relative}"
        )

    data_config, data_relative = _relative_file(
        data_config_path, root, label="recorded QA data config"
    )
    duct_config, duct_relative = _relative_file(
        duct_config_path, root, label="recorded QA duct config"
    )
    data_cfg = load_yaml(data_config)
    duct_cfg = load_yaml(duct_config)
    contract, plant = _strict_measured_timing_contract(
        root=root, data_cfg=data_cfg, duct_cfg=duct_cfg
    )
    contract_payload = contract.model_dump()

    return {
        "schema": RECORDED_QA_REISSUE_SCHEMA,
        "manifest": {
            "path": manifest_relative,
            "sha256": sha256_file(manifest),
        },
        "inputs": {
            "data_config": {
                "path": data_relative,
                "sha256": sha256_file(data_config),
            },
            "duct_config": {
                "path": duct_relative,
                "sha256": sha256_file(duct_config),
            },
            # QA의 strict timing source가 surrogate pretrain 기본값으로 오해되지
            # 않게 explicit하게 남긴다. 숫자 lead를 여기에 따로 쓰지 않는다.
            "primary_path_mode": "measured",
        },
        "timing": {
            "training_timing_contract": contract_payload,
            "training_timing_contract_sha256": contract.digest(),
            **plant,
        },
    }


def _require_exact_mapping(value: object, *, expected_keys: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise RecordedQAReissueError(f"{label}가 object가 아닙니다")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        raise RecordedQAReissueError(
            f"{label} key가 정확하지 않습니다: expected={sorted(expected_keys)}, "
            f"actual={sorted(actual_keys)}"
        )
    return value


def validate_current_recorded_qa_report(
    report: object,
    *,
    expected_provenance: dict[str, object],
    repo_root: str | Path = REPO_ROOT,
) -> None:
    """QA report가 지금의 canonical manifest/P/S/lead evidence인지 검사한다.

    ``None`` 반환만 성공이다. historical QA는 ``provenance``가 없거나 manifest/lead
    중 하나가 달라 즉시 예외가 난다. 검증은 report가 PASS인 경우만 authority로
    인정한다.
    """

    if not isinstance(report, dict):
        raise RecordedQAReissueError("recorded QA report 최상위가 object가 아닙니다")
    if report.get("ok") is not True:
        raise RecordedQAReissueError("recorded QA report가 PASS가 아니므로 authority가 아닙니다")

    provenance = _require_exact_mapping(
        report.get("provenance"),
        expected_keys={"schema", "manifest", "inputs", "timing"},
        label="recorded QA provenance",
    )
    if provenance != expected_provenance:
        raise RecordedQAReissueError(
            "recorded QA provenance가 현재 canonical manifest/P/S/config와 다릅니다"
        )

    expected_manifest = expected_provenance["manifest"]
    if not isinstance(expected_manifest, dict):  # pragma: no cover - build 함수 invariant
        raise RuntimeError("expected provenance manifest 형식 오류")
    expected_path = (Path(repo_root).resolve() / str(expected_manifest["path"])).resolve()
    report_manifest = report.get("manifest")
    if not isinstance(report_manifest, str) or not report_manifest:
        raise RecordedQAReissueError("recorded QA report manifest 경로가 없습니다")
    visible_path = Path(report_manifest).expanduser()
    if not visible_path.is_absolute():
        visible_path = Path(repo_root).resolve() / visible_path
    if visible_path.resolve() != expected_path:
        raise RecordedQAReissueError(
            "recorded QA report의 visible manifest가 provenance canonical manifest와 다릅니다"
        )

    settings = report.get("settings")
    if not isinstance(settings, dict):
        raise RecordedQAReissueError("recorded QA settings가 없습니다")
    timing = expected_provenance["timing"]
    if not isinstance(timing, dict):  # pragma: no cover - build 함수 invariant
        raise RuntimeError("expected provenance timing 형식 오류")
    contract = timing["training_timing_contract"]
    if not isinstance(contract, dict):  # pragma: no cover - build 함수 invariant
        raise RuntimeError("expected provenance timing contract 형식 오류")
    expected_lead = int(contract["digital_reference_lead_samples"])
    expected_sample_rate = int(contract["sample_rate"])
    if settings.get("reference_mode") != "digital":
        raise RecordedQAReissueError("recorded QA settings.reference_mode=digital이 아닙니다")
    if settings.get("sample_rate") != expected_sample_rate:
        raise RecordedQAReissueError(
            "recorded QA sample_rate가 strict timing contract와 다릅니다"
        )
    if settings.get("digital_reference_lead_samples") != expected_lead:
        raise RecordedQAReissueError(
            "recorded QA lead가 strict P/S의 PlantDelays.lead() 유도값과 다릅니다: "
            f"report={settings.get('digital_reference_lead_samples')!r}, expected={expected_lead}"
        )
    segment_samples = settings.get("segment_samples")
    minimum_frames = settings.get("minimum_frames")
    if (
        isinstance(segment_samples, bool)
        or not isinstance(segment_samples, int)
        or isinstance(minimum_frames, bool)
        or not isinstance(minimum_frames, int)
        or minimum_frames != segment_samples + expected_lead + 1
    ):
        raise RecordedQAReissueError(
            "recorded QA minimum_frames가 segment + current lead + 1 규약과 다릅니다"
        )


def reissue_current_recorded_qa(
    *,
    repo_root: str | Path = REPO_ROOT,
    manifest_path: str | Path = CANONICAL_RECORDED_MANIFEST,
    data_config_path: str | Path = "configs/data_sim.yaml",
    duct_config_path: str | Path = "configs/duct.yaml",
    block_frames: int = 262_144,
    clip_threshold: float = 0.99,
    max_clip_ratio: float = 0.005,
    min_mic_rms_dbfs: float = -80.0,
    min_source_rms_dbfs: float = -80.0,
) -> dict:
    """현행 manifest/P/S binding을 포함해 QA를 다시 실행한다.

    caller가 반환 report를 저장하기 전 다시 :func:`validate_current_recorded_qa_report`
    로 확인한다. 그래서 reissue 구현이 future change로 old-like report를 만들면 파일을
    쓰기 전에 실패한다.
    """

    root = Path(repo_root).resolve()
    provenance = build_current_recorded_qa_provenance(
        repo_root=root,
        manifest_path=manifest_path,
        data_config_path=data_config_path,
        duct_config_path=duct_config_path,
    )
    manifest, _ = _relative_file(manifest_path, root, label="recorded QA manifest")
    data_config, _ = _relative_file(data_config_path, root, label="recorded QA data config")
    data_cfg = copy.deepcopy(load_yaml(data_config))
    timing = provenance["timing"]
    if not isinstance(timing, dict):  # pragma: no cover - build 함수 invariant
        raise RuntimeError("reissue provenance timing 형식 오류")
    contract = timing["training_timing_contract"]
    if not isinstance(contract, dict):  # pragma: no cover - build 함수 invariant
        raise RuntimeError("reissue timing contract 형식 오류")

    # settings_from_data_config는 QA 길이를 이 값에서 읽는다. 이 값은 사용자 입력이
    # 아니라 바로 위 strict P/S `TrainingTimingContract`에서 투영된다.
    data_cfg["reference_mode"] = "digital"
    data_cfg["digital_reference_lead_samples"] = int(
        contract["digital_reference_lead_samples"]
    )
    settings = settings_from_data_config(
        data_cfg,
        block_frames=block_frames,
        clip_threshold=clip_threshold,
        max_clip_ratio=max_clip_ratio,
        min_mic_rms_dbfs=min_mic_rms_dbfs,
        min_source_rms_dbfs=min_source_rms_dbfs,
        required_splits=VALID_SPLITS,
        allow_incomplete_family_coverage=False,
    )
    report = validate_recorded_sessions(
        read_manifest(manifest), settings, manifest_path=str(manifest)
    )
    report["settings"]["reference_mode"] = "digital"
    report["provenance"] = provenance
    if report.get("ok") is True:
        validate_current_recorded_qa_report(
            report, expected_provenance=provenance, repo_root=root
        )
    return report


def render_current_recorded_qa_markdown(report: dict) -> str:
    """기존 QA 표 뒤에 사람이 바로 볼 수 있는 authority binding을 붙인다."""

    rendered = render_recorded_qa_markdown(report).rstrip() + "\n"
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        return rendered
    manifest = provenance.get("manifest") or {}
    timing = provenance.get("timing") or {}
    contract = timing.get("training_timing_contract") if isinstance(timing, dict) else {}
    if not isinstance(manifest, dict) or not isinstance(contract, dict):
        return rendered
    return (
        rendered
        + "\n## Current authority binding\n\n"
        + f"- schema: `{provenance.get('schema', '')}`\n"
        + f"- canonical manifest: `{manifest.get('path', '')}`\n"
        + f"- manifest SHA-256: `{manifest.get('sha256', '')}`\n"
        + "- strict P/S-derived lead: "
        + f"{contract.get('digital_reference_lead_samples', '')} samples\n"
        + "- timing contract SHA-256: "
        + f"`{timing.get('training_timing_contract_sha256', '')}`\n"
    )
