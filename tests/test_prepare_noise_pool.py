"""``scripts/data/prepare_noise_pool.py`` 의 빌드 게이트 짝.

⚠ 2026-08-06 통합 검증에서 추가됐다. 이 게이트(선언한 태그를 못 만들면 종료코드 1)는
게이트 레지스트리 정비와 **같은 변경 안에서 선언 없이 만들어졌다** — ``grep -rn
prepare_noise_pool tests/`` 0건, ``gates_for_owner(...)`` 빈 튜플이었다. 즉 "모든
게이트는 짝 없이 존재할 수 없다"가 선언된 게이트에 대해서만 참이었고, 새 게이트를
선언 없이 만드는 발생기는 그 변경 안에서 다시 돌았다.

여기서는 양방향을 다 본다: 선언 태그가 없으면 실패하고(negative), 전부 있으면
성공한다(positive). 후자가 없으면 "항상 실패하는 게이트" 와 구별되지 않는다.
"""

from __future__ import annotations

import importlib.util
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

import deep_anc.data.manifest_contract as manifest_contract

from deep_anc.data.holdout_contract import EXPECTED_HISTORICAL_BUILDERS
from deep_anc.data.decoder_audit import (
    DEFAULT_AUDIO_EXTENSIONS,
    DEFAULT_SEGMENT_FRAMES,
    DEFAULT_SEGMENT_GRID_DENOMINATOR,
    DEFAULT_SEQUENTIAL_CHUNK_FRAMES,
    MAX_DECODED_PCM_ABS,
    MIN_DECODED_RMS,
    decoder_fingerprint,
)
from deep_anc.data.public_lineage import (
    DNS_MARKER_TAG_ROOTS,
    DNS_NOISE_MARKERS,
    DNS_SPEECH_MARKER,
    ESC50_METADATA,
    FMA_TRACKS,
    LIBRISPEECH_CHAPTERS,
    PUBLIC_LINEAGE_SCHEMA,
    PublicLineageBuild,
    canonical_json_sha256,
    validate_dns_marker_partition,
)
from deep_anc.data.manifest import scan_wavs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/prepare_noise_pool.py"
FS = 48_000


def _load_script():
    spec = importlib.util.spec_from_file_location("_prepare_noise_pool_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_clips(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for index in range(count):
        data = rng.standard_normal(FS).astype(np.float32) * 0.05
        sf.write(directory / f"clip_{index:03d}.wav", data, FS)


def _build_tree(tmp_path: Path, present_tags: dict[str, float], ratios: dict[str, float]):
    """가짜 REPO_ROOT 를 만든다. 태그별 원본 유무를 시험이 직접 정한다."""

    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/data_sim.yaml").write_text(
        yaml.safe_dump({"source_mix_ratio": ratios}), encoding="utf-8"
    )
    for tag in present_tags:
        # data/raw/<계열>/<tag>/ — 스크립트가 깊이를 가정하지 않는지도 함께 본다.
        _write_clips(tmp_path / "data/raw" / f"{tag}_family" / tag, 6)
    csv_hashes = {}
    csv_paths = []
    for pool_name in ("source_pool", "source_pool_v2"):
        csv_path = tmp_path / "data" / pool_name / "sources.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(
            "source_family,session_index,group_id,path,seconds,clips\n"
            f'environment,0,g0,{pool_name}/environment_000.wav,1.0,"[]"\n',
            encoding="utf-8",
        )
        csv_paths.append(f"data/{pool_name}/sources.csv")
        csv_hashes[pool_name] = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    report_path = tmp_path / "results/provenance/source_pool_provenance_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "authority": "historical_builder_reproduction_plus_pcm_validation",
                "recorded_tree_protection": {"status": "PASS", "file_count": 1},
                "historical_builders": EXPECTED_HISTORICAL_BUILDERS,
                "post_repair_csv_sha256": csv_hashes,
                "downstream_gates": {
                    "active_holdout": {
                        "status": "PASS",
                        "active_session_count": 1,
                        "active_source_row_count": 1,
                        "total_clips": 4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    holdout = tmp_path / "data/manifests/recorded_holdout.json"
    holdout.parent.mkdir(parents=True, exist_ok=True)
    families = {
        "environment": ["held-env.wav"],
        "machine": ["held-machine.wav"],
        "music": ["held-music.wav"],
        "speech": ["held-speech.wav"],
    }
    clip_rows = [
        {
            "family": family,
            "clip": clips[0],
            "content_sha256": f"{index + 1:x}" * 64,
            "lineage_keys": [f"fixture_holdout:{family}"],
        }
        for index, (family, clips) in enumerate(sorted(families.items()))
    ]
    fixture_metadata = {
        "librispeech_chapters": {
            "path": LIBRISPEECH_CHAPTERS,
            "sha256": "a" * 64,
            "size": 1,
        },
        "fma_tracks": {"path": FMA_TRACKS, "sha256": "b" * 64, "size": 1},
        "esc50": {"path": ESC50_METADATA, "sha256": "c" * 64, "size": 1},
    }
    holdout.write_text(
        json.dumps(
            {
                "purpose": "test canonical active recorded provenance",
                "scope": "active_sessions_only",
                "active_session_count": 1,
                "active_source_row_count": 1,
                "source_rows": ["data/source_pool/environment/environment_000.wav"],
                "sources_csv": csv_paths,
                "sources_csv_sha256": csv_hashes,
                "provenance_report": "results/provenance/source_pool_provenance_report.json",
                "families": families,
                "clip_lineage": {
                    "schema_version": 1,
                    "metadata": fixture_metadata,
                    "clips": clip_rows,
                    "clips_sha256": canonical_json_sha256(clip_rows),
                },
                "total_clips": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def _write_decoder_audit(
    root: Path,
    *,
    decisions: dict[str, str] | None = None,
    path: str = "results/decoder_audit.json",
) -> str:
    """prepare fixture의 raw inventory를 canonical decoder-audit v1 형식으로 쓴다."""

    decisions = decisions or {}
    inventory = []
    for audio in sorted((root / "data/raw").rglob("*.wav")):
        relative = audio.relative_to(root).as_posix()
        inventory.append(
            {
                "relative_path": relative,
                "content_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "content_size": audio.stat().st_size,
                "decision": decisions.get(relative, "accept"),
            }
        )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "audit_policy": {
            "audio_extensions": sorted(DEFAULT_AUDIO_EXTENSIONS),
            "sequential_chunk_frames": list(DEFAULT_SEQUENTIAL_CHUNK_FRAMES),
            "segment_frames": DEFAULT_SEGMENT_FRAMES,
            "segment_grid_denominator": DEFAULT_SEGMENT_GRID_DENOMINATOR,
            "max_decoded_pcm_abs": MAX_DECODED_PCM_ABS,
            "min_decoded_rms": MIN_DECODED_RMS,
        },
        "decoder_fingerprint": decoder_fingerprint(),
        "inventory": inventory,
    }
    payload["decoder_fingerprint_sha256"] = canonical_json_sha256(
        payload["decoder_fingerprint"]
    )
    payload["inventory_sha256"] = canonical_json_sha256(inventory)
    payload["accepted_inventory_sha256"] = canonical_json_sha256(
        [
            {
                "relative_path": row["relative_path"],
                "content_sha256": row["content_sha256"],
                "content_size": row["content_size"],
            }
            for row in inventory
            if row["decision"] == "accept"
        ]
    )
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _run(
    module,
    root: Path,
    monkeypatch,
    *,
    decoder_audit: str | None = "auto",
    hash_workers: int = 1,
) -> int:
    monkeypatch.setattr(module, "REPO_ROOT", root)
    holdout = root / "data/manifests/recorded_holdout.json"
    expected = hashlib.sha256(holdout.read_bytes()).hexdigest() if holdout.is_file() else "0" * 64
    if decoder_audit == "auto":
        decoder_audit = _write_decoder_audit(root)

    def fixture_holdout_validator(path, *, repo_root, expected_sha256=None):
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise module.HoldoutContractError("fixture holdout SHA mismatch")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return {
            "sha256": actual,
            "families": payload["families"],
            "clip_lineage": payload["clip_lineage"],
        }

    def fixture_public_lineage(
        entries_by_tag,
        *,
        tag_roots,
        repo_root,
        holdout_lineage,
        **lineage_kwargs,
    ):
        rows_by_tag = {}
        components = {}
        for tag, entries in sorted(entries_by_tag.items()):
            rows_by_tag[tag] = []
            for entry in entries:
                item = dict(entry)
                digest = item["content_sha256"]
                lineage_key = f"fixture_content:{digest}"
                group = "public-lineage-" + canonical_json_sha256(
                    {
                        "lineage_keys": [lineage_key],
                        "content_sha256": [digest],
                    }
                )
                item["lineage_schema"] = PUBLIC_LINEAGE_SCHEMA
                item["lineage_keys"] = [lineage_key]
                item["group_id"] = group
                rows_by_tag[tag].append(item)
                components.setdefault(
                    group,
                    {
                        "members": [],
                        "tags": [],
                        "lineage_keys": [lineage_key],
                        "content_sha256": [digest],
                        "excluded_by_holdout": False,
                        "overlap": {
                            "basename": [],
                            "content_sha256": [],
                            "lineage_keys": [],
                        },
                    },
                )
                components[group]["members"].append(
                    f"{tag}:{len(components[group]['members'])}"
                )
                components[group]["tags"] = sorted(
                    set(components[group]["tags"] + [tag])
                )
        metadata_path = repo_root / "configs/data_sim.yaml"
        metadata_raw = metadata_path.read_bytes()
        evidence = {
            "schema_version": 1,
            "lineage_schema": PUBLIC_LINEAGE_SCHEMA,
            "metadata": {
                "fixture": {
                    "path": "configs/data_sim.yaml",
                    "sha256": hashlib.sha256(metadata_raw).hexdigest(),
                    "size": len(metadata_raw),
                }
            },
            "component_count": len(components),
            "component_membership_sha256": canonical_json_sha256(
                {key: components[key] for key in sorted(components)}
            ),
            "components": {key: components[key] for key in sorted(components)},
            "holdout_clips_sha256": holdout_lineage["clips_sha256"],
            "excluded_by_tag": {tag: 0 for tag in sorted(entries_by_tag)},
        }
        # 실제 public lineage parser를 쓰지 않는 이 fixture도 canonical v4의 DNS
        # marker partition/postcommit 경계를 통과해야 한다. marker member는 raw
        # source tag root 기준으로 만들며, rejected member는 caller가 audit inventory
        # 전체에서 투영한 값만 사용한다.
        rejected_members = lineage_kwargs.get("decoder_rejected_members_by_tag")
        inventory_sha = lineage_kwargs.get("decoder_audit_inventory_sha256")
        dns_tags = {
            tag
            for tag, relative in DNS_MARKER_TAG_ROOTS.items()
            if any(
                Path(item) == repo_root / relative
                for item in tag_roots.get(tag, [])
            )
            or isinstance(rejected_members, dict) and tag in rejected_members
        }
        if dns_tags:
            for tag in sorted(dns_tags):
                source_root = repo_root / DNS_MARKER_TAG_ROOTS[tag]
                assert source_root in [Path(item) for item in tag_roots.get(tag, [])]
                accepted = {
                    Path(item["path"]).relative_to(source_root).as_posix()
                    for item in entries_by_tag.get(tag, [])
                    if source_root in Path(item["path"]).parents
                }
                rejected = set((rejected_members or {}).get(tag, ()))
                members = sorted(accepted | rejected)
                if tag == "speech":
                    marker = repo_root / DNS_SPEECH_MARKER
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("\n".join(members) + "\n", encoding="utf-8")
                else:
                    # tiny fixture에는 single shard만 쓰되 public checker의 two-marker
                    # layout을 그대로 만든다.
                    for index, relative in enumerate(DNS_NOISE_MARKERS):
                        marker = repo_root / relative
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.write_text(
                            ("\n".join(members) + "\n") if index == 0 else "",
                            encoding="utf-8",
                        )
            partition = validate_dns_marker_partition(
                entries_by_tag,
                tag_roots=tag_roots,
                repo_root=repo_root,
                decoder_rejected_members_by_tag=rejected_members,
                decoder_audit_inventory_sha256=inventory_sha,
            )
            assert partition is not None
            evidence["decoder_rejected_marker_partition"] = partition
        return PublicLineageBuild(
            rows_by_tag,
            {tag: 0 for tag in sorted(entries_by_tag)},
            evidence,
        )

    monkeypatch.setattr(module, "validate_holdout_contract", fixture_holdout_validator)
    monkeypatch.setattr(module, "build_public_lineage", fixture_public_lineage)
    argv = [
        "prepare_noise_pool.py",
        "--out",
        "data/manifests",
        "--expected-holdout-sha256",
        expected,
        "--hash-workers",
        str(hash_workers),
    ]
    if decoder_audit is not None:
        argv.extend(["--decoder-audit", decoder_audit])
    monkeypatch.setattr(sys, "argv", argv)
    return module.main()


def test_missing_declared_tag_fails_the_build(tmp_path, monkeypatch, capsys):
    """비율 > 0 인데 원본이 없는 태그가 있으면 **종료코드 1**.

    이것이 없으면 ``synth_dataset`` 이 그 태그를 로그 없이 합성원으로 폴백하고,
    학습은 선언한 ``source_mix_ratio`` 와 다른 데이터로 돈다. 출하 상태가 실제로
    이렇다 — ``dns_fullband``/``demand``/``machine`` 원본이 유실됐다.
    """

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 0.4, "speech": 0.3},
        ratios={"esc50": 0.4, "speech": 0.3, "machine": 0.3, "synthetic": 0.1},
    )

    assert _run(module, root, monkeypatch) == 1

    err = capsys.readouterr().err
    assert "machine" in err and "[실패]" in err
    # 원인을 진단으로 남겨야 한다 — 조용한 폴백이 이 게이트가 막는 것이다.
    assert "폴백" in err
    assert not list((root / "data/manifests").glob("*.jsonl")), (
        "필수 태그가 빠진 실패 실행은 일부 manifest도 남기면 안 됩니다"
    )


def test_every_declared_tag_present_builds_all_manifests(tmp_path, monkeypatch, capsys):
    """**positive 짝** — 선언 태그를 정확히 다 채운 최소 구성에서 통과한다.

    경계까지 몰아본다: 비율 0 인 태그(``demand``)는 원본이 없어도 요구하지 않아야
    한다(``required_tags()`` 가 비율 > 0 만 센다). 그 태그까지 요구하면 게이트가
    "항상 실패" 로 굳고, 그러면 다음 사람이 게이트째로 끈다.
    """

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 0.4, "speech": 0.3, "music": 0.3},
        ratios={
            "esc50": 0.4,
            "speech": 0.3,
            "music": 0.3,
            "demand": 0.0,  # 선언은 남아 있으나 지금은 쓰지 않는다
            "synthetic": 0.1,
        },
    )

    assert _run(module, root, monkeypatch) == 0

    out = capsys.readouterr().out
    assert "완료: manifest 3개" in out
    for tag in ("esc50", "speech", "music"):
        assert (root / "data/manifests" / f"{tag}.jsonl").is_file()
    sidecar = json.loads(
        (root / "data/manifests/manifest_generation.json").read_text(encoding="utf-8")
    )
    assert sidecar["schema_version"] == 4
    assert sidecar["training_eligible"] is True
    assert sidecar["holdout_sha256"] == hashlib.sha256(
        (root / "data/manifests/recorded_holdout.json").read_bytes()
    ).hexdigest()
    assert set(sidecar["manifests"]) == {"esc50", "speech", "music"}
    for tag, metadata in sidecar["manifests"].items():
        assert metadata["sha256"] == hashlib.sha256(
            (root / "data/manifests" / f"{tag}.jsonl").read_bytes()
        ).hexdigest()
    audit_binding = sidecar["decoder_audit"]
    copied_audit = root / "data/manifests/decoder_audit.json"
    assert copied_audit.is_file()
    assert audit_binding["file"] == "decoder_audit.json"
    assert audit_binding["sha256"] == hashlib.sha256(copied_audit.read_bytes()).hexdigest()
    assert audit_binding["size"] == copied_audit.stat().st_size
    # 비율 0 인 태그는 만들지도, 요구하지도 않는다.
    assert not (root / "data/manifests/demand.jsonl").exists()


def test_parallel_raw_hash_workers_preserve_a_canonical_generation(
    tmp_path, monkeypatch, capsys
):
    """병렬 SHA도 transaction postcondition까지 통과해야만 사용할 수 있다."""

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 0.5, "speech": 0.5},
        ratios={"esc50": 0.5, "speech": 0.5, "synthetic": 0.1},
    )

    assert _run(module, root, monkeypatch, hash_workers=2) == 0
    capsys.readouterr()
    sidecar = json.loads(
        (root / "data/manifests/manifest_generation.json").read_text(encoding="utf-8")
    )
    assert sidecar["schema_version"] == 4
    assert sidecar["training_eligible"] is True
    assert set(sidecar["manifests"]) == {"esc50", "speech"}


def test_hash_workers_outside_the_bounded_range_fail_before_manifest_write(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )

    assert _run(module, root, monkeypatch, hash_workers=0) == 2
    assert "--hash-workers" in capsys.readouterr().err
    assert not (root / "data/manifests/esc50.jsonl").exists()


def test_missing_holdout_fails_before_writing_any_manifest(tmp_path, monkeypatch, capsys):
    """학습용 manifest는 recorded holdout 없이 생성할 수 없다."""

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    (root / "data/manifests/recorded_holdout.json").unlink()

    assert _run(module, root, monkeypatch) == 1
    assert "held-out 목록이 없습니다" in capsys.readouterr().err
    assert not (root / "data/manifests/esc50.jsonl").exists()


def test_training_generation_requires_completed_decoder_audit_before_writing(
    tmp_path, monkeypatch, capsys
):
    """v3까지는 raw decoder 결과 없이도 canonical generation을 쓸 수 있었다."""

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )

    assert _run(module, root, monkeypatch, decoder_audit=None) == 2
    assert "--decoder-audit" in capsys.readouterr().err
    assert not (root / "data/manifests/esc50.jsonl").exists()


def test_decoder_reject_rows_are_excluded_and_copied_audit_is_bound(
    tmp_path, monkeypatch
):
    """audit가 reject한 파일은 retry 후보로 남기지 않고 manifest에서 제거한다."""

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    rejected = "data/raw/esc50_family/esc50/clip_000.wav"
    audit = _write_decoder_audit(root, decisions={rejected: "reject"})

    assert _run(module, root, monkeypatch, decoder_audit=audit) == 0
    manifest = root / "data/manifests/esc50.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert all(Path(row["path"]).relative_to(root).as_posix() != rejected for row in rows)

    sidecar = json.loads(
        (root / "data/manifests/manifest_generation.json").read_text(encoding="utf-8")
    )
    copied = root / "data/manifests/decoder_audit.json"
    assert sidecar["decoder_audit"]["sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert sidecar["decoder_audit"]["accepted_inventory_sha256"] == canonical_json_sha256(
        [
            {
                "relative_path": row["relative_path"],
                "content_sha256": row["content_sha256"],
                "content_size": row["content_size"],
            }
            for row in json.loads(copied.read_text(encoding="utf-8"))["inventory"]
            if row["decision"] == "accept"
        ]
    )


def test_decoder_reject_marker_projection_uses_full_audit_not_scan_results(tmp_path):
    """sf.info가 못 여는 WAV도 audit reject면 DNS marker partition에 남아야 한다."""

    root = tmp_path
    source = root / "data/raw/noise/speech"
    _write_clips(source, 1)
    accepted_path = source / "book_12_chp_3_reader_4.wav"
    (source / "clip_000.wav").rename(accepted_path)
    rejected_member = "nested/book_13_chp_4_reader_5.wav"
    rejected_path = source / rejected_member
    rejected_path.parent.mkdir(parents=True)
    # 확장자는 WAV이지만 soundfile metadata를 읽을 수 없는 raw — scan_wavs가 생략한다.
    rejected_path.write_bytes(b"not a decodable wav")
    audit_path = _write_decoder_audit(
        root,
        decisions={rejected_path.relative_to(root).as_posix(): "reject"},
    )
    audit = manifest_contract.read_decoder_audit(
        root / audit_path,
        repo_root=root,
        label="broken DNS marker fixture audit",
    )
    scanned = scan_wavs(source, "speech")
    assert [Path(entry["path"]).name for entry in scanned] == [accepted_path.name]
    projection = manifest_contract.derive_decoder_rejected_members_by_tag(
        audit,
        tag_roots={"speech": [source]},
        repo_root=root,
        label="broken DNS marker fixture projection",
    )
    assert projection == {"speech": (rejected_member,)}
    marker = root / DNS_SPEECH_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{accepted_path.name}\n{rejected_member}\n", encoding="utf-8")
    partition = validate_dns_marker_partition(
        {"speech": scanned},
        tag_roots={"speech": [source]},
        repo_root=root,
        decoder_rejected_members_by_tag=projection,
        decoder_audit_inventory_sha256=audit["inventory_sha256"],
    )
    assert partition is not None
    assert partition["tags"]["speech"]["rejected_members"] == [rejected_member]


def test_prepare_binds_scan_skipped_dns_reject_to_generation_partition(
    tmp_path, monkeypatch
):
    """producer가 scan 결과가 아닌 audit inventory projection을 build evidence로 넘긴다."""

    from deep_anc.data.manifest_contract import validate_manifest_generation

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"speech": 1.0},
        ratios={"speech": 1.0},
    )
    generic_source = root / "data/raw/speech_family/speech"
    source = root / "data/raw/noise/speech"
    source.parent.mkdir(parents=True, exist_ok=True)
    generic_source.rename(source)
    rejected_member = "nested/undecodable.wav"
    rejected = source / rejected_member
    rejected.parent.mkdir(parents=True)
    rejected.write_bytes(b"not a decodable wav")
    audit = _write_decoder_audit(
        root,
        decisions={rejected.relative_to(root).as_posix(): "reject"},
    )

    assert _run(module, root, monkeypatch, decoder_audit=audit) == 0
    sidecar = json.loads(
        (root / "data/manifests/manifest_generation.json").read_text(encoding="utf-8")
    )
    partition = sidecar["public_lineage"]["decoder_rejected_marker_partition"]
    assert partition["tags"]["speech"]["rejected_members"] == [rejected_member]
    validate_manifest_generation(
        root / "data/manifests",
        required_tags={"speech"},
        repo_root=root,
    )


def test_prepare_keeps_librispeech_out_of_dns_marker_reject_partition(
    tmp_path, monkeypatch
):
    """Elice처럼 DNS와 LibriSpeech root가 같이 있어도 DNS marker만 exact partition한다."""

    from deep_anc.data.manifest_contract import validate_manifest_generation

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"speech": 1.0},
        ratios={"speech": 1.0},
    )
    generic_source = root / "data/raw/speech_family/speech"
    dns_source = root / "data/raw/noise/speech"
    dns_source.parent.mkdir(parents=True, exist_ok=True)
    generic_source.rename(dns_source)
    libri_source = root / "data/raw/speech/LibriSpeech"
    _write_clips(libri_source, 1)
    dns_rejected_member = "nested/dns-undecodable.wav"
    dns_rejected = dns_source / dns_rejected_member
    dns_rejected.parent.mkdir(parents=True)
    dns_rejected.write_bytes(b"not a decodable wav")
    libri_rejected = libri_source / "libri-undecodable.wav"
    libri_rejected.write_bytes(b"not a decodable wav")
    audit = _write_decoder_audit(
        root,
        decisions={
            dns_rejected.relative_to(root).as_posix(): "reject",
            libri_rejected.relative_to(root).as_posix(): "reject",
        },
    )

    assert _run(module, root, monkeypatch, decoder_audit=audit) == 0
    sidecar = json.loads(
        (root / "data/manifests/manifest_generation.json").read_text(encoding="utf-8")
    )
    partition = sidecar["public_lineage"]["decoder_rejected_marker_partition"]
    assert partition["tags"]["speech"]["tag_roots"] == ["data/raw/noise/speech"]
    assert partition["tags"]["speech"]["rejected_members"] == [dns_rejected_member]
    validate_manifest_generation(
        root / "data/manifests",
        required_tags={"speech"},
        repo_root=root,
    )


def test_decoder_audit_path_index_is_cached_once_per_raw_root_context(
    tmp_path, monkeypatch
):
    """큰 corpus도 entry마다 inventory 전체를 다시 index하지 않아야 한다."""

    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    audit_path = _write_decoder_audit(root)
    audit = manifest_contract.read_decoder_audit(
        root / audit_path,
        repo_root=root,
        label="cache fixture audit",
    )
    entries = [
        {
            "path": str(path),
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content_size": path.stat().st_size,
        }
        for path in sorted((root / "data/raw").rglob("*.wav"))
    ]
    original = manifest_contract._decoder_audit_index
    calls = 0

    def counted_index(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(manifest_contract, "_decoder_audit_index", counted_index)
    for index, entry in enumerate(entries):
        manifest_contract.validate_decoder_audit_manifest_entry(
            audit,
            entry,
            repo_root=root,
            raw_roots=[root / "data/raw"],
            label=f"cache fixture entry #{index}",
        )
    assert calls == 1

    # 동일 audit라도 raw-root 경계가 달라지면 stale absolute-path cache를 재사용하지 않는다.
    manifest_contract.validate_decoder_audit_manifest_entry(
        audit,
        entries[0],
        repo_root=root,
        raw_roots=[root / "data"],
        label="cache fixture changed root",
    )
    assert calls == 2


def test_failed_raw_inventory_verification_does_not_leave_a_path_index(
    tmp_path,
):
    """불완전 audit는 다음 caller가 파생 cache를 신뢰하게 해서는 안 된다."""

    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    audit_path = _write_decoder_audit(root)
    audit = manifest_contract.read_decoder_audit(
        root / audit_path,
        repo_root=root,
        label="failed cache fixture audit",
    )
    _write_clips(root / "data/raw/unlisted", 1)

    with pytest.raises(ValueError, match="raw inventory가 audit와 다릅니다"):
        manifest_contract.validate_decoder_audit_raw_inventory(
            audit,
            repo_root=root,
            raw_roots=[root / "data/raw"],
            label="failed cache fixture",
        )
    assert "_index_by_raw_path" not in audit
    assert "_index_context" not in audit


def test_tampered_decoder_audit_inventory_digest_blocks_generation(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    audit = _write_decoder_audit(root)
    path = root / audit
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inventory"][0]["content_sha256"] = "0" * 64
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    assert _run(module, root, monkeypatch, decoder_audit=audit) == 1
    assert "inventory SHA 불일치" in capsys.readouterr().err
    assert not (root / "data/manifests/esc50.jsonl").exists()


def test_decoder_audit_must_match_the_current_decoder_runtime(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    audit = _write_decoder_audit(root)
    path = root / audit
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decoder_fingerprint"]["soundfile"] = "not-the-current-runtime"
    payload["decoder_fingerprint_sha256"] = canonical_json_sha256(
        payload["decoder_fingerprint"]
    )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    assert _run(module, root, monkeypatch, decoder_audit=audit) == 1
    assert "현재 runtime" in capsys.readouterr().err


def test_decoder_audit_without_full_scan_recipe_blocks_generation(
    tmp_path, monkeypatch, capsys
):
    """inventory만 그럴듯한 얕은 header audit은 canonical evidence가 아니다."""

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    audit = _write_decoder_audit(root)
    path = root / audit
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["audit_policy"]["sequential_chunk_frames"] = [65_536]
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    assert _run(module, root, monkeypatch, decoder_audit=audit) == 1
    assert "full sequential scan" in capsys.readouterr().err
    assert not (root / "data/manifests/esc50.jsonl").exists()


def test_decoder_audit_raw_inventory_addition_blocks_generation(
    tmp_path, monkeypatch, capsys
):
    """태그 밖 raw 추가도 audit 전체성 계약을 무효화해야 한다."""

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    audit = _write_decoder_audit(root)
    _write_clips(root / "data/raw/unclassified", 1)

    assert _run(module, root, monkeypatch, decoder_audit=audit) == 1
    assert "raw inventory가 audit와 다릅니다" in capsys.readouterr().err
    assert not (root / "data/manifests/esc50.jsonl").exists()


def test_accept_row_with_rejected_content_duplicate_blocks_generation(
    tmp_path, monkeypatch, capsys
):
    """경로를 바꾼 duplicate가 reject raw를 다시 학습 분포에 넣으면 안 된다."""

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    rejected = root / "data/raw/esc50_family/esc50/clip_000.wav"
    accepted_duplicate = root / "data/raw/esc50_family/esc50/clip_001.wav"
    accepted_duplicate.write_bytes(rejected.read_bytes())
    audit = _write_decoder_audit(
        root,
        decisions={rejected.relative_to(root).as_posix(): "reject"},
    )

    assert _run(module, root, monkeypatch, decoder_audit=audit) == 1
    assert "reject 행과 중복" in capsys.readouterr().err
    assert not (root / "data/manifests/esc50.jsonl").exists()


def test_schema_v3_generation_is_diagnostic_only_and_cannot_feed_training(
    tmp_path, monkeypatch
):
    from deep_anc.data.manifest_contract import validate_manifest_generation

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    assert _run(module, root, monkeypatch) == 0
    sidecar_path = root / "data/manifests/manifest_generation.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["schema_version"] = 3
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostic-only"):
        validate_manifest_generation(
            root / "data/manifests", required_tags={"esc50"}, repo_root=root
        )


def test_allow_corpus_leak_is_an_explicit_diagnostic_only_escape_hatch(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    (root / "data/manifests/recorded_holdout.json").unlink()
    monkeypatch.setattr(module, "REPO_ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_noise_pool.py",
            "--out",
            "results/diagnostics/leaky-pool",
            "--allow-corpus-leak",
        ],
    )

    assert module.main() == module.DIAGNOSTIC_ONLY_EXIT
    assert "진단 전용" in capsys.readouterr().err
    diagnostic = root / "results/diagnostics/leaky-pool"
    assert (diagnostic / "esc50.jsonl").is_file()
    sidecar = json.loads(
        (diagnostic / "manifest_generation.json").read_text(encoding="utf-8")
    )
    assert sidecar["training_eligible"] is False
    assert sidecar["holdout"] is None
    assert not (root / "data/manifests/esc50.jsonl").exists()


def test_allow_corpus_leak_cannot_write_official_manifest_directory(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    monkeypatch.setattr(module, "REPO_ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_noise_pool.py", "--allow-corpus-leak"],
    )

    assert module.main() == 2
    assert "official data/manifests" in capsys.readouterr().err
    assert not list((root / "data/manifests").glob("*.jsonl"))


def _seed_old_generation(root: Path, tags: tuple[str, ...]) -> dict[str, bytes]:
    out = root / "data/manifests"
    out.mkdir(parents=True, exist_ok=True)
    expected: dict[str, bytes] = {}
    for tag in tags:
        payload = f'{{"old": "{tag}"}}\n'.encode()
        (out / f"{tag}.jsonl").write_bytes(payload)
        expected[f"{tag}.jsonl"] = payload
    sidecar = b'{"build_id":"old"}\n'
    (out / "manifest_generation.json").write_bytes(sidecar)
    expected["manifest_generation.json"] = sidecar
    return expected


def test_staging_write_failure_preserves_entire_previous_generation(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 0.5, "speech": 0.5},
        ratios={"esc50": 0.5, "speech": 0.5},
    )
    expected = _seed_old_generation(root, ("esc50", "speech"))
    real_write = module.write_manifest
    calls = 0

    def fail_second_write(entries, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second staging write failure")
        return real_write(entries, path)

    monkeypatch.setattr(module, "write_manifest", fail_second_write)

    assert _run(module, root, monkeypatch) == 1
    assert "기존 세대를 복구" in capsys.readouterr().err
    for name, payload in expected.items():
        assert (root / "data/manifests" / name).read_bytes() == payload


def test_commit_rename_failure_rolls_back_every_installed_file(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 0.5, "speech": 0.5},
        ratios={"esc50": 0.5, "speech": 0.5},
    )
    expected = _seed_old_generation(root, ("esc50", "speech"))
    real_replace = os.replace
    installs = 0

    def fail_second_install(source, target):
        nonlocal installs
        source_path = Path(source)
        if source_path.parent.name == "new":
            installs += 1
            if installs == 2:
                raise OSError("injected second commit rename failure")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second_install)

    assert _run(module, root, monkeypatch) == 1
    assert "기존 세대를 복구" in capsys.readouterr().err
    for name, payload in expected.items():
        assert (root / "data/manifests" / name).read_bytes() == payload


def test_postcondition_failure_rolls_back_complete_previous_generation(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 0.5, "speech": 0.5},
        ratios={"esc50": 0.5, "speech": 0.5},
    )
    expected = _seed_old_generation(root, ("esc50", "speech"))

    def reject_postcondition(*_args, **_kwargs):
        raise RuntimeError("injected committed-generation postcondition failure")

    monkeypatch.setattr(module, "_verify_committed_generation", reject_postcondition)
    assert _run(module, root, monkeypatch) == 1
    assert "기존 세대를 복구" in capsys.readouterr().err
    for name, payload in expected.items():
        assert (root / "data/manifests" / name).read_bytes() == payload


def test_raw_tree_symlink_is_rejected_before_manifest_write(tmp_path, monkeypatch, capsys):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0},
    )
    outside = tmp_path / "outside-raw"
    outside.mkdir()
    (root / "data/raw/linked").symlink_to(outside, target_is_directory=True)

    assert _run(module, root, monkeypatch) == 1
    assert "symlink" in capsys.readouterr().err
    assert not list((root / "data/manifests").glob("*.jsonl"))


def test_concurrent_prepare_process_lock_fails_before_staging(tmp_path, monkeypatch):
    module = _load_script()
    root = tmp_path
    monkeypatch.setattr(module, "REPO_ROOT", root)
    parent = root / "data"
    parent.mkdir()
    config = root / "config.yaml"
    config.write_text("source_mix_ratio: {}\n", encoding="utf-8")
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(module.ManifestTransactionError, match="다른 manifest prepare"):
            module.write_generation_transactionally(
                {},
                out_dir=parent / "manifests",
                data_config=config,
                holdout_path=None,
                seed=1,
                training_eligible=False,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not list(parent.glob(".noise-manifest-stage-*"))


def test_diagnostic_symlink_cannot_alias_official_manifest_directory(
    tmp_path, monkeypatch, capsys
):
    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    official = root / "data/manifests"
    sentinel = official / "sentinel.jsonl"
    sentinel.write_text("preserve\n", encoding="utf-8")
    diagnostics = root / "results/diagnostics"
    diagnostics.symlink_to("../data/manifests", target_is_directory=True)
    monkeypatch.setattr(module, "REPO_ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_noise_pool.py",
            "--out",
            "results/diagnostics/leaky",
            "--allow-corpus-leak",
        ],
    )

    assert module.main() == 2
    assert "symlink" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_generation_binds_raw_audio_content_and_returns_one_read_snapshot(
    tmp_path, monkeypatch
):
    from deep_anc.data.manifest_contract import validate_manifest_generation

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0, "synthetic": 0.1},
    )
    assert _run(module, root, monkeypatch) == 0
    manifest_dir = root / "data/manifests"
    snapshot = validate_manifest_generation(
        manifest_dir,
        required_tags={"esc50"},
        repo_root=root,
    )
    entries = snapshot["_validated_entries"]["esc50"]
    assert entries and all("content_sha256" in row for row in entries)
    assert snapshot["schema_version"] == 4
    assert snapshot["_canonical_decoder_audited"] is True

    # validator가 반환한 entry는 검증했던 동일 byte snapshot이다. 이후 JSONL 교체가
    # snapshot을 바꾸지 않으며, raw bytes 교체는 다음 소비 검증에서 즉시 실패한다.
    original_path = entries[0]["path"]
    (manifest_dir / "esc50.jsonl").write_text('{"forged": true}\n', encoding="utf-8")
    assert snapshot["_validated_entries"]["esc50"][0]["path"] == original_path
    audio = Path(original_path)
    audio.write_bytes(audio.read_bytes() + b"tamper")
    # manifest SHA 오류가 raw hash보다 먼저 나지 않게 검증된 manifest bytes를 복구한다.
    (manifest_dir / "esc50.jsonl").write_bytes(
        snapshot["_validated_manifest_bytes"]["esc50"]
    )
    with pytest.raises(ValueError, match="raw (inventory SHA/size|content SHA).*(다릅니다|불일치)"):
        validate_manifest_generation(
            manifest_dir,
            required_tags={"esc50"},
            repo_root=root,
        )


def test_manifest_validator_rederives_dns_reject_partition_after_sidecar_tamper(
    tmp_path, monkeypatch
):
    """build_id를 다시 계산해도 forged marker reject evidence는 통과하면 안 된다."""

    from deep_anc.data.manifest_contract import validate_manifest_generation

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"speech": 1.0},
        ratios={"speech": 1.0},
    )
    generic_source = root / "data/raw/speech_family/speech"
    dns_source = root / "data/raw/noise/speech"
    dns_source.parent.mkdir(parents=True, exist_ok=True)
    generic_source.rename(dns_source)
    assert _run(module, root, monkeypatch) == 0
    manifest_dir = root / "data/manifests"
    validate_manifest_generation(
        manifest_dir,
        required_tags={"speech"},
        repo_root=root,
    )
    sidecar_path = manifest_dir / "manifest_generation.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    partition = sidecar["public_lineage"]["decoder_rejected_marker_partition"]
    partition["tags"]["speech"]["rejected_members"] = ["forged/missing.wav"]
    partition["tags"]["speech"]["rejected_member_count"] = 1
    partition["tags"]["speech"]["rejected_members_sha256"] = canonical_json_sha256(
        ["forged/missing.wav"]
    )
    basis = {
        key: value
        for key, value in sidecar.items()
        if key not in {"build_id", "created_at"}
    }
    sidecar["build_id"] = hashlib.sha256(
        manifest_contract._canonical_json_bytes(basis)
    ).hexdigest()
    sidecar_path.write_bytes(manifest_contract._canonical_json_bytes(sidecar))

    with pytest.raises(ValueError, match="marker partition.*재계산 결과"):
        validate_manifest_generation(
            manifest_dir,
            required_tags={"speech"},
            repo_root=root,
        )


def test_noise_pool_rejects_raw_retarget_after_generation_validation(
    tmp_path, monkeypatch
):
    from deep_anc.data.manifest_contract import validate_manifest_generation
    from deep_anc.data.noise_pool import NoisePool

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0},
    )
    assert _run(module, root, monkeypatch) == 0
    snapshot = validate_manifest_generation(
        root / "data/manifests", required_tags={"esc50"}, repo_root=root
    )
    entry = dict(snapshot["_validated_entries"]["esc50"][0])
    entry["split"] = "train"
    pool = NoisePool([], "train", FS, seed=1, validated_entries=[entry])

    audio_path = Path(entry["path"])
    audio, rate = sf.read(audio_path, dtype="float32")
    sf.write(audio_path, audio * 0.5, rate)
    with pytest.raises(RuntimeError, match="변경/retarget"):
        pool.sample_segment(1024)


def test_noise_pool_rejects_raw_root_symlink_after_validation(tmp_path, monkeypatch):
    from deep_anc.data.manifest_contract import validate_manifest_generation
    from deep_anc.data.noise_pool import NoisePool

    module = _load_script()
    root = _build_tree(
        tmp_path,
        present_tags={"esc50": 1.0},
        ratios={"esc50": 1.0},
    )
    assert _run(module, root, monkeypatch) == 0
    snapshot = validate_manifest_generation(
        root / "data/manifests", required_tags={"esc50"}, repo_root=root
    )
    entry = dict(snapshot["_validated_entries"]["esc50"][0])
    entry["split"] = "train"
    pool = NoisePool([], "train", FS, seed=1, validated_entries=[entry])

    raw_root = root / "data/raw"
    moved = root / "data/raw-original"
    raw_root.rename(moved)
    raw_root.symlink_to(moved, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        pool.sample_segment(1024)


def test_rollback_continues_after_one_restore_failure_and_preserves_both_errors(
    tmp_path, monkeypatch
):
    module = _load_script()
    root = tmp_path
    monkeypatch.setattr(module, "REPO_ROOT", root)
    config = root / "configs/data_sim.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("source_mix_ratio: {}\n", encoding="utf-8")
    out = root / "data/manifests"
    out.mkdir(parents=True)
    old = _seed_old_generation(root, ("esc50", "speech"))
    prepared = {
        tag: (
            [
                {
                    "path": f"/{tag}.wav",
                    "duration_s": 1.0,
                    "sample_rate": FS,
                    "channels": 1,
                    "tag": tag,
                    "split": "train",
                    "content_sha256": "a" * 64,
                }
            ],
            0,
            [],
        )
        for tag in ("esc50", "speech")
    }
    real_replace = os.replace
    installs = 0

    def fail_commit_then_one_rollback(source, target):
        nonlocal installs
        source_path = Path(source)
        if source_path.parent.name == "new":
            installs += 1
            if installs == 3:
                raise OSError("original commit failure")
        if source_path.parent.name == "old" and source_path.name == "speech.jsonl":
            raise OSError("injected speech rollback failure")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_commit_then_one_rollback)
    with pytest.raises(module.ManifestTransactionError) as captured:
        module.write_generation_transactionally(
            prepared,
            out_dir=out,
            data_config=config,
            holdout_path=None,
            seed=1,
            training_eligible=False,
        )
    error = captured.value
    assert "original commit failure" in str(error)
    assert "injected speech rollback failure" in str(error)
    assert len(error.rollback_errors) == 1
    # speech 복구 실패 뒤에도 esc50 복구를 계속했다.
    assert (out / "esc50.jsonl").read_bytes() == old["esc50.jsonl"]
    assert error.recovery_dir.is_dir()


def test_the_gate_is_declared_in_the_registry():
    """스크립트 exit 게이트도 레지스트리에 있어야 한다 — 이 파일이 생긴 이유다."""

    from deep_anc.ops.gate_registry import gates_for_owner

    declared = gates_for_owner("scripts/data/prepare_noise_pool.py")
    assert [gate.gate_id for gate in declared] == [
        "noise_pool_declared_tags_exist",
        "noise_pool_recorded_holdout_required",
    ]
    assert all(
        gate.negative_fixture.startswith("tests/test_prepare_noise_pool.py::")
        and gate.positive_fixture.startswith("tests/test_prepare_noise_pool.py::")
        for gate in declared
    )


def test_the_shipped_config_declares_a_nonempty_public_pool_contract():
    """출하 config의 계약만 검사하고 ignored ``data/raw`` host 상태는 가정하지 않는다.

    결손/완전 동작은 위의 독립 tmp fixture가 각각 고정한다. Elice bootstrap이 공개
    코퍼스를 모두 받은 뒤에도 저장소 전체 pytest가 성공해야 하므로, 실제 머신에
    적어도 하나가 없기를 요구하는 테스트는 빌드 완료와 모순이다.
    """

    from deep_anc.config import REPO_ROOT

    module = _load_script()
    pools = module.declared_pools(REPO_ROOT / "configs/data_sim.yaml")
    required = set(module.PoolPlan(pools=pools, roots=("data/raw",)).required_tags())

    assert required
    assert "synthetic" not in required
    assert required <= {pool.tag for pool in pools}
