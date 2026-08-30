from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from deep_anc.data.broadband_population_contract_v3 import (
    CausalPrimaryOperatorV3,
    LocalFileReferenceV3,
    MAX_QUALIFIED_ITEMS_V3,
    MIN_DENSITY_RATIO,
    POPULATION_V3_AUTHORITY,
    POPULATION_V3_SCAFFOLD_BLOCKERS,
    PopulationAuditV3,
    PopulationCandidateV3,
    PopulationBatchPlanV3,
    PopulationCoverageContractV3,
    PopulationItemClaimV3,
    PopulationManifestV3,
    PopulationV3Blocked,
    UntouchedLevel5PolicyV3,
    apply_causal_primary_v3,
    audit_population_manifest_v3,
    current_population_v3_gate,
    density_ratios_v3,
    plan_structural_batch_v3,
)
from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3


_CONTROL_V3_SHA256 = "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
_POPULATION_V3_SHA256 = "8f0d1e3897a2ace87059cecd584ea5da3ed0ecdb01a45a5855b2475cfe6e05c1"
_FAMILIES = ("speech", "music", "environment", "machine")
_SPLITS = ("train", "val", "test")
_SEGMENT_SAMPLES = 8192


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _write_reference(root: Path, relative: str, payload: bytes) -> LocalFileReferenceV3:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return LocalFileReferenceV3(
        path=relative,
        size_bytes=len(payload),
        sha256=_sha256_bytes(payload),
    )


def _band_limited_signal(*, lower_hz: float, upper_hz: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(_SEGMENT_SAMPLES, d=1.0 / 48_000)
    spectrum = np.zeros(frequencies.size, dtype=np.complex128)
    selected = (frequencies >= lower_hz) & (frequencies <= upper_hz)
    spectrum[selected] = rng.standard_normal(selected.sum()) + 1j * rng.standard_normal(
        selected.sum()
    )
    signal = np.fft.irfft(spectrum, n=_SEGMENT_SAMPLES)
    signal = signal / np.max(np.abs(signal)) * 0.02
    return np.asarray(signal, dtype="<f4")


def _claim_for_signal(*, item_id: str, err: np.ndarray) -> PopulationItemClaimV3:
    control = BroadbandFullOctaveContractV3.canonical()
    physical = density_ratios_v3(
        err,
        sample_rate=48_000,
        bands_hz=control.physical_identification_subbands_hz,
    )
    objective = density_ratios_v3(
        err,
        sample_rate=48_000,
        bands_hz=control.equal_weight_octave_objective_bands_hz,
    )
    return PopulationItemClaimV3(
        item_id=item_id,
        start_frame=0,
        n_frames=_SEGMENT_SAMPLES,
        physical_density_ratios=physical,
        physical_valid_bands=tuple(value >= MIN_DENSITY_RATIO for value in physical),
        objective_octave_density_ratios=objective,
        objective_octave_valid_bands=tuple(
            value >= MIN_DENSITY_RATIO for value in objective
        ),
    )


def _build_population_fixture(root: Path) -> PopulationManifestV3:
    control = BroadbandFullOctaveContractV3.canonical()
    contract = PopulationCoverageContractV3.canonical()
    fir = np.asarray([1.0], dtype="<f4")
    fir_ref = _write_reference(root, "plant/causal_p.f32", fir.tobytes())
    operator_receipt = _sha256_text("synthetic-test-only-causal-p-receipt")
    operator = CausalPrimaryOperatorV3(
        control_band_contract_sha256=control.digest(),
        fir_file=fir_ref,
        delay_samples=0,
        verified_lower_hz=80.0,
        verified_upper_hz=11_400.0,
        operator_receipt_sha256=operator_receipt,
    )

    candidates: list[PopulationCandidateV3] = []
    sequence = 0
    for split in _SPLITS:
        for family in _FAMILIES:
            # 각 family에 low-only 네 component와 high-only 네 component를 둔다.
            # 어느 clip도 모든 band를 통과하지 않지만 population은 모든 band를 덮는다.
            for spectral_role in ("low", "high"):
                for component_index in range(4):
                    sequence += 1
                    if spectral_role == "low":
                        decoded = _band_limited_signal(
                            lower_hz=90.0,
                            upper_hz=1590.0,
                            seed=sequence,
                        )
                        native_rate = 6000
                    else:
                        decoded = _band_limited_signal(
                            lower_hz=1610.0,
                            upper_hz=11_300.0,
                            seed=sequence,
                        )
                        native_rate = 24_000
                    err = apply_causal_primary_v3(decoded, fir=fir, delay_samples=0)
                    stem = f"{split}_{family}_{spectral_role}_{component_index}"
                    native_payload = (
                        f"immutable-native-test-fixture:{stem}:{sequence}".encode("utf-8")
                    )
                    native_ref = _write_reference(
                        root, f"native/{stem}.bin", native_payload
                    )
                    decoded_ref = _write_reference(
                        root, f"decoded/{stem}.f32", decoded.tobytes(order="C")
                    )
                    err_ref = _write_reference(
                        root, f"err/{stem}.f32", err.tobytes(order="C")
                    )
                    candidates.append(
                        PopulationCandidateV3(
                            candidate_id=stem,
                            split=split,
                            source_family=family,
                            lineage_component_id=f"component_{stem}",
                            immutable_native_source=native_ref,
                            native_sample_rate_hz=native_rate,
                            native_nyquist_hz=native_rate / 2.0,
                            native_probe_receipt_sha256=_sha256_text(
                                f"native-probe:{stem}"
                            ),
                            decoded_pcm_file=decoded_ref,
                            decoded_frames=decoded.size,
                            decoded_transform_receipt_sha256=_sha256_text(
                                f"decode:{stem}"
                            ),
                            p_applied_err_file=err_ref,
                            p_operator_receipt_sha256=operator_receipt,
                            valid_prefix_samples=0,
                            items=(_claim_for_signal(item_id=f"item_{stem}", err=err),),
                        )
                    )

    return PopulationManifestV3(
        contract=contract,
        contract_sha256=contract.digest(),
        causal_primary=operator,
        segment_samples=_SEGMENT_SAMPLES,
        candidates=tuple(candidates),
        untouched_level5_policy=UntouchedLevel5PolicyV3(
            reservation_receipt_sha256=_sha256_text(
                "synthetic-test-only-level5-reservation"
            )
        ),
    )


@pytest.fixture(scope="module")
def population_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("population_v3")
    manifest = _build_population_fixture(root)
    audit = audit_population_manifest_v3(manifest, repository_root=root)
    return root, manifest, audit


def _validated_manifest_with_update(
    manifest: PopulationManifestV3,
    *,
    candidate_index: int,
    candidate_updates: dict[str, object],
) -> PopulationManifestV3:
    payload = manifest.model_dump(mode="json")
    payload["candidates"][candidate_index].update(candidate_updates)
    return PopulationManifestV3.model_validate(payload)


def test_v3_contract_is_exact_inline_and_digest_locked_without_authority() -> None:
    control = BroadbandFullOctaveContractV3.canonical()
    contract = PopulationCoverageContractV3.canonical()

    assert control.digest() == _CONTROL_V3_SHA256
    assert contract.control_band_contract.model_dump(mode="json") == control.model_dump(
        mode="json"
    )
    assert contract.control_band_contract_sha256 == _CONTROL_V3_SHA256
    assert contract.digest() == _POPULATION_V3_SHA256
    assert contract.minimum_density_ratio == 0.25
    assert contract.all_bands_per_clip_required is False
    assert contract.adaptive_eq_or_band_shaping_allowed is False
    assert POPULATION_V3_AUTHORITY is None

    gate = current_population_v3_gate()
    assert gate.status == "BLOCKED"
    assert gate.authority is None
    assert gate.contract_sha256 == _POPULATION_V3_SHA256
    assert any("actual bytes" in blocker for blocker in gate.blockers)
    assert any("causal P" in blocker for blocker in gate.blockers)


def test_partial_band_counterexample_passes_population_without_lowering_threshold(
    population_fixture,
) -> None:
    _root, manifest, audit = population_fixture

    assert len(manifest.candidates) == 3 * 4 * 8
    # v2식 all-seven-per-clip라면 이 자연 분포 반례는 96/96 모두 탈락한다.
    assert all(
        not all(candidate.items[0].physical_valid_bands)
        and not all(candidate.items[0].objective_octave_valid_bands)
        for candidate in manifest.candidates
    )
    # v3는 같은 0.25 threshold를 유지하고 population 독립 component로 보장한다.
    assert manifest.contract.minimum_density_ratio == 0.25
    assert audit.density_threshold_lowered is False
    assert audit.structural_status == "PASS"
    assert len(audit.coverage) == 3 * 4 * (8 + 7)
    assert all(row.passed for row in audit.coverage)
    assert min(row.independent_lineage_components for row in audit.coverage) >= 4

    # 실제 bytes 재검산 PASS도 canonical training authority를 만들지 않는다.
    assert audit.canonical_status == "BLOCKED"
    assert audit.authority is None
    assert audit.role == "local_recomputation_scaffold_not_external_raw_authority"
    assert audit.external_manifest_authority_bound is False
    assert audit.connected_component_authority_bound is False
    assert audit.interval_alias_authority_bound is False
    assert audit.actual_raw_manifest_authority_bound is False
    assert set(POPULATION_V3_SCAFFOLD_BLOCKERS).issubset(audit.blockers)
    assert any("AUTHORITY is None" in blocker for blocker in audit.blockers)


def test_population_audit_schema_has_qualified_item_upper_bound() -> None:
    metadata = PopulationAuditV3.model_fields["qualified_items"].metadata
    assert any(
        getattr(constraint, "max_length", None) == MAX_QUALIFIED_ITEMS_V3
        for constraint in metadata
    )


def test_family_balanced_batch_covers_every_physical_and_objective_band(
    population_fixture,
) -> None:
    _root, _manifest, audit = population_fixture
    plan = plan_structural_batch_v3(
        audit,
        split="train",
        batch_size=32,
        batch_index=7,
        seed=20260828,
    )

    assert plan.family_counts == tuple((family, 8) for family in _FAMILIES)
    assert len(plan.selected_item_ids) == len(set(plan.selected_item_ids)) == 32
    assert min(plan.physical_valid_item_counts) >= 4
    assert min(plan.objective_octave_valid_item_counts) >= 4
    assert min(plan.physical_distinct_lineage_counts) >= 4
    assert min(plan.objective_octave_distinct_lineage_counts) >= 4
    assert plan.structural_status == "PASS"
    assert plan.canonical_training_status == "BLOCKED"
    assert plan.authority is None

    forged = plan.model_dump(mode="json")
    forged["physical_valid_item_counts"][0] = 3
    with pytest.raises(ValidationError, match="physical valid item count"):
        PopulationBatchPlanV3.model_validate(forged)

    with pytest.raises(ValueError, match="batch_size=4"):
        plan_structural_batch_v3(
            audit,
            split="train",
            batch_size=4,
            batch_index=0,
            seed=1,
        )


def test_audit_rejects_forged_lineage_coverage_count(population_fixture) -> None:
    _root, _manifest, audit = population_fixture
    payload = audit.model_dump(mode="python")
    payload["coverage"][0]["independent_lineage_components"] += 1
    with pytest.raises(ValidationError, match="lineage count"):
        PopulationAuditV3.model_validate(payload)


def test_population_requires_four_independent_components_per_split_family_band(
    population_fixture,
) -> None:
    root, manifest, _audit = population_fixture
    # train/speech의 네 번째 high candidate를 첫 번째 high component에 합친다.
    updated = _validated_manifest_with_update(
        manifest,
        candidate_index=7,
        candidate_updates={
            "lineage_component_id": manifest.candidates[4].lineage_component_id
        },
    )
    audit = audit_population_manifest_v3(updated, repository_root=root)

    assert audit.structural_status == "BLOCKED"
    failed = [row for row in audit.coverage if not row.passed]
    assert failed
    assert all(row.split == "train" and row.source_family == "speech" for row in failed)
    assert any(row.independent_lineage_components == 3 for row in failed)
    with pytest.raises(PopulationV3Blocked, match="structural coverage"):
        plan_structural_batch_v3(
            audit,
            split="train",
            batch_size=32,
            batch_index=0,
            seed=1,
        )


def test_native_nyquist_is_checked_for_each_claimed_high_band(
    population_fixture,
) -> None:
    root, manifest, _audit = population_fixture
    # candidate #4는 1.6--11.3 kHz high-only item이다.
    updated = _validated_manifest_with_update(
        manifest,
        candidate_index=4,
        candidate_updates={
            "native_sample_rate_hz": 16_000,
            "native_nyquist_hz": 8_000.0,
        },
    )
    with pytest.raises(ValueError, match="native Nyquist"):
        audit_population_manifest_v3(updated, repository_root=root)


def test_actual_p_applied_err_bytes_are_recomputed_and_must_match(
    population_fixture,
) -> None:
    root, manifest, _audit = population_fixture
    original = manifest.candidates[0].p_applied_err_file
    tampered = bytearray((root / original.path).read_bytes())
    tampered[17] ^= 0x01
    tampered_ref = _write_reference(root, "err/tampered_candidate_0.f32", bytes(tampered))
    updated = _validated_manifest_with_update(
        manifest,
        candidate_index=0,
        candidate_updates={"p_applied_err_file": tampered_ref.model_dump(mode="json")},
    )

    with pytest.raises(ValueError, match="P-applied ERR"):
        audit_population_manifest_v3(updated, repository_root=root)


def test_claimed_density_and_mask_are_recomputed_from_actual_err_bytes(
    population_fixture,
) -> None:
    root, manifest, _audit = population_fixture
    candidate = manifest.candidates[0]
    bad_item = candidate.items[0].model_dump(mode="json")
    bad_item["physical_density_ratios"][0] += 0.1
    updated = _validated_manifest_with_update(
        manifest,
        candidate_index=0,
        candidate_updates={"items": [bad_item]},
    )

    with pytest.raises(ValueError, match="density/mask"):
        audit_population_manifest_v3(updated, repository_root=root)


def test_same_source_bytes_cannot_cross_lineage_components_or_splits(
    population_fixture,
) -> None:
    root, manifest, _audit = population_fixture
    train_source = manifest.candidates[0].immutable_native_source
    test_speech_index = 2 * 4 * 8
    updated = _validated_manifest_with_update(
        manifest,
        candidate_index=test_speech_index,
        candidate_updates={
            "immutable_native_source": train_source.model_dump(mode="json")
        },
    )
    with pytest.raises(ValueError, match="native source SHA"):
        audit_population_manifest_v3(updated, repository_root=root)


def test_population_item_ids_are_globally_unique(population_fixture) -> None:
    _root, manifest, _audit = population_fixture
    duplicate_id = manifest.candidates[0].items[0].item_id
    item = manifest.candidates[1].items[0].model_dump(mode="json")
    item["item_id"] = duplicate_id
    with pytest.raises(ValidationError, match="item_id가 중복"):
        _validated_manifest_with_update(
            manifest,
            candidate_index=1,
            candidate_updates={"items": [item]},
        )


def test_legacy_v2_and_level5_cannot_be_auto_promoted_into_population(
    population_fixture,
) -> None:
    _root, manifest, _audit = population_fixture
    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["legacy_v2_manifest_sha256"] = "0" * 64
    manifest_payload["legacy_v2_automatic_promotion"] = True
    with pytest.raises(ValidationError):
        PopulationManifestV3.model_validate(manifest_payload)

    candidate_payload = manifest.candidates[0].model_dump(mode="json")
    candidate_payload["legacy_v2_promoted"] = True
    candidate_payload["unmodified_level5_challenge"] = True
    with pytest.raises(ValidationError):
        PopulationCandidateV3.model_validate(candidate_payload)

    v2_shaped_payload = {
        "schema_version": "broadband_source_manifest_v2",
        "contract_sha256": "0" * 64,
        "candidates": [],
    }
    with pytest.raises(ValidationError):
        PopulationManifestV3.model_validate(v2_shaped_payload)


def test_inline_control_contract_or_sha_cannot_be_substituted() -> None:
    contract = PopulationCoverageContractV3.canonical()
    payload = contract.model_dump(mode="json")
    payload["control_band_contract_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="payload.*SHA"):
        PopulationCoverageContractV3.model_validate(payload)

    payload = contract.model_dump(mode="json")
    payload["control_band_contract"]["excitation_lower_hz"] = 81.0
    with pytest.raises(ValidationError):
        PopulationCoverageContractV3.model_validate(payload)
