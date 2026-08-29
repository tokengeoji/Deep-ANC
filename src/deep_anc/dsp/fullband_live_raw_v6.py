"""v6 immutable live raw의 세대 명시 public wrapper.

파일/dirfd writer의 구현은 검증된 공용 v5-origin primitive를 재사용하지만, signal
binding이 v6가 아니면 공용 validator가 fail-closed한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .fullband_live_raw_v5 import (
    LIVE_RAW_SCHEMA_V6 as LIVE_RAW_SCHEMA,
    POST_CAPTURE_BINDING_SCHEMA_V6 as POST_CAPTURE_BINDING_SCHEMA,
    PREFLIGHT_IDENTITY_SCHEMA,
    PREFLIGHT_REPORT_SCHEMA,
    RAW_ARRAY_FIELDS,
    SESSION_SCHEMA_V6 as SESSION_SCHEMA,
    TELEMETRY_ARRAY_FIELDS,
    load_live_raw_v5,
    publish_live_raw_v5,
)
from .fullband_live_authority_v6 import PLAN_ENVELOPE_SCHEMA


def _require_v6_bindings(value: Any, *, label: str) -> Mapping[str, Any]:
    """Public v6 namespace에서 v5 generation으로의 silent dispatch를 막는다."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label}에 v6 bindings mapping이 필요합니다")
    signal_plan = value.get("signal_plan")
    if (
        not isinstance(signal_plan, Mapping)
        or signal_plan.get("schema") != PLAN_ENVELOPE_SCHEMA
    ):
        raise ValueError(f"{label}가 exact v6 signal-plan binding이 아닙니다")
    return value


def publish_live_raw_v6(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _require_v6_bindings(kwargs.get("bindings"), label="v6 raw publish")
    return publish_live_raw_v5(*args, **kwargs)


def load_live_raw_v6(
    path: str | Path,
    *,
    repository_root: str | Path,
    expected_bindings: Mapping[str, Any],
    expected_raw_file_sha256: str,
    require_analysis_admission: bool = False,
) -> dict[str, Any]:
    _require_v6_bindings(expected_bindings, label="v6 raw load")
    return load_live_raw_v5(
        path,
        repository_root=repository_root,
        expected_bindings=expected_bindings,
        expected_raw_file_sha256=expected_raw_file_sha256,
        require_analysis_admission=require_analysis_admission,
    )


__all__ = [
    "LIVE_RAW_SCHEMA",
    "POST_CAPTURE_BINDING_SCHEMA",
    "PREFLIGHT_IDENTITY_SCHEMA",
    "PREFLIGHT_REPORT_SCHEMA",
    "RAW_ARRAY_FIELDS",
    "SESSION_SCHEMA",
    "TELEMETRY_ARRAY_FIELDS",
    "load_live_raw_v6",
    "publish_live_raw_v6",
]
