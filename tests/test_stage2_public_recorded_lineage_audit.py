from __future__ import annotations

import pytest

from deep_anc.data.stage2_public_recorded_lineage_audit import (
    Stage2PublicRecordedLineageError,
    audit_lineage_rows,
)


def _row(
    basename: str,
    *keys: str,
    split: str | None = None,
    content: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "basename": basename,
        "content_sha256": content,
        "lineage_keys": list(keys),
    }
    if split is not None:
        row["split"] = split
    return row


def test_transitive_component_overlap_and_split_crossing_are_fail_closed() -> None:
    result = audit_lineage_rows(
        recorded_rows=[_row("held.wav", "artist:a", "album:x")],
        public_rows_by_manifest={
            "music.jsonl": [
                _row("other.wav", "artist:b", "album:x", split="train"),
                _row("bridge.wav", "artist:b", "album:y", split="val"),
                _row("third.wav", "artist:c", "album:y", split="test"),
            ]
        },
    )
    inventory = result["inventories"][0]
    assert result["status"] == "BLOCKED"
    assert inventory["direct_recorded_identity_rows"] == 1
    assert inventory["transitive_recorded_component_rows"] == 3
    assert inventory["components_crossing_public_splits"] == 1
    assert inventory["public_rows_in_cross_split_components"] == 3


def test_exact_basename_is_identity_even_without_public_content_sha() -> None:
    result = audit_lineage_rows(
        recorded_rows=[
            _row("same.flac", "reader:1", content="a" * 64)
        ],
        public_rows_by_manifest={
            "speech.jsonl": [
                _row("same.flac", "reader:9", split="train")
            ]
        },
    )
    inventory = result["inventories"][0]
    assert inventory["exact_recorded_basename_rows"] == 1
    assert inventory["transitive_recorded_component_rows"] == 1


@pytest.mark.parametrize("split", ["", "training", None])
def test_noncanonical_split_is_rejected(split: str | None) -> None:
    with pytest.raises(Stage2PublicRecordedLineageError, match="split"):
        audit_lineage_rows(
            recorded_rows=[_row("held.wav", "source:1")],
            public_rows_by_manifest={
                "public.jsonl": [_row("other.wav", "source:2", split=split)]
            },
        )


def test_duplicate_public_basename_is_rejected() -> None:
    with pytest.raises(Stage2PublicRecordedLineageError, match="중복"):
        audit_lineage_rows(
            recorded_rows=[_row("held.wav", "source:1")],
            public_rows_by_manifest={
                "public.jsonl": [
                    _row("same.wav", "source:2", split="train"),
                    _row("same.wav", "source:3", split="val"),
                ]
            },
        )
