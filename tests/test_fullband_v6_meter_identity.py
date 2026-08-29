from __future__ import annotations

from copy import deepcopy

import pytest

from deep_anc.dsp import fullband_v6_meter as meter


def _execution_identity() -> dict[str, object]:
    return {
        "repository_commit": "1" * 40,
        "repository_branch": "work/v6-clock-checkpoints",
        "repository_dirty": False,
        "script_path": "scripts/data/set_amp_level.py",
        "script_file_sha256": "2" * 64,
    }


def test_v6_meter_repository_execution_binding_accepts_exact_current_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    expected = _execution_identity()
    monkeypatch.setattr(
        meter, "repository_execution_identity", lambda *_args: deepcopy(expected)
    )

    result = meter._validate_repository_execution_binding_v6(
        {"repository_execution": deepcopy(expected)}, repository_root=tmp_path
    )

    assert result == expected


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("repository_commit", "3" * 40),
        ("repository_branch", "tampered-branch"),
        ("script_path", "scripts/data/not_set_amp_level.py"),
        ("script_file_sha256", "4" * 64),
    ],
)
def test_v6_meter_repository_execution_binding_rejects_saved_identity_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path, field: str, replacement: object
) -> None:
    current = _execution_identity()
    saved = deepcopy(current)
    saved[field] = replacement
    monkeypatch.setattr(
        meter, "repository_execution_identity", lambda *_args: deepcopy(current)
    )

    with pytest.raises(ValueError, match="current clean checkout"):
        meter._validate_repository_execution_binding_v6(
            {"repository_execution": saved}, repository_root=tmp_path
        )


def test_v6_meter_repository_execution_binding_rejects_missing_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        meter,
        "repository_execution_identity",
        lambda *_args: _execution_identity(),
    )

    with pytest.raises(ValueError, match="current clean checkout"):
        meter._validate_repository_execution_binding_v6({}, repository_root=tmp_path)


def test_v6_meter_repository_execution_binding_rejects_current_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    saved = _execution_identity()
    current = deepcopy(saved)
    current["repository_commit"] = "5" * 40
    monkeypatch.setattr(
        meter, "repository_execution_identity", lambda *_args: deepcopy(current)
    )

    with pytest.raises(ValueError, match="current clean checkout"):
        meter._validate_repository_execution_binding_v6(
            {"repository_execution": saved}, repository_root=tmp_path
        )
