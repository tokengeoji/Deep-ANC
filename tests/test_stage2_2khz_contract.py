from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_anc.dsp.control_band_contract import BroadbandFullOctaveContractV3
from deep_anc.dsp.stage2_2khz_contract import (
    STAGE2_2KHZ_DNH_BANDS_HZ,
    STAGE2_2KHZ_OBJECTIVE_BANDS_HZ,
    STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ,
    Stage2TwoKilohertzContract,
)


_STAGE2_FROZEN_DIGEST = (
    "70fc33d20a43bedaa5a51f8e19aed12fff687d8fb3901501f4a49bf2746d97cf"
)
_FULL_OCTAVE_V3_FROZEN_DIGEST = (
    "53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2"
)


def test_stage2_contract_is_separate_exact_and_frozen() -> None:
    contract = Stage2TwoKilohertzContract.canonical()

    assert contract.contract_id == "broadband_2khz_octave_88_2828_v1"
    assert contract.physical_identification_subbands_hz == (
        STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ
    )
    assert contract.octave_objective_centers_hz == (
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
    )
    assert contract.octave_objective_bands_hz == STAGE2_2KHZ_OBJECTIVE_BANDS_HZ
    assert contract.do_no_harm_octave_centers_hz == (4000.0, 8000.0)
    assert contract.do_no_harm_octave_bands_hz == STAGE2_2KHZ_DNH_BANDS_HZ
    assert contract.required_excitation_lower_hz == 80.0
    assert contract.required_excitation_upper_hz == 2828.4271247462
    assert contract.source_families == (
        "speech",
        "music",
        "environment",
        "machine",
    )
    assert contract.two_khz_octave_minimum_attenuation_db == 3.0
    assert contract.do_no_harm_max_worst10_amplification_db == 1.0
    assert contract.single_point_only is True
    assert contract.spatial_quiet_zone_claim_allowed is False
    assert contract.legacy_automatic_promotion_allowed is False
    assert contract.full_octave_v3_automatic_promotion_allowed is False
    assert contract.digest() == _STAGE2_FROZEN_DIGEST

    with pytest.raises(ValidationError, match="frozen"):
        contract.required_excitation_upper_hz = 3000.0


def test_stage2_addition_does_not_change_full_octave_v3_digest() -> None:
    assert (
        BroadbandFullOctaveContractV3.canonical().digest()
        == _FULL_OCTAVE_V3_FROZEN_DIGEST
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "physical_identification_subbands_hz",
            STAGE2_2KHZ_PHYSICAL_IDENTIFICATION_SUBBANDS_HZ[:-1],
            "physical identification",
        ),
        (
            "octave_objective_bands_hz",
            STAGE2_2KHZ_OBJECTIVE_BANDS_HZ[:-1],
            "objective octave band",
        ),
        (
            "do_no_harm_octave_bands_hz",
            STAGE2_2KHZ_DNH_BANDS_HZ[:1],
            "do-no-harm",
        ),
        ("required_excitation_lower_hz", 80.0000000001, "80 Hz 이하"),
        ("required_excitation_upper_hz", 2828.4271247461, "2,828"),
        ("source_families", ("speech", "music"), "네 family"),
        ("minimum_groups_per_family_octave", 3, "하한 4"),
        ("two_khz_octave_minimum_attenuation_db", 2.99, "최소 감쇠 3"),
        ("do_no_harm_max_worst10_amplification_db", 1.01, "증폭 한도 1"),
    ],
)
def test_stage2_contract_rejects_weakened_or_relabelled_payload(
    field: str, value: object, match: str
) -> None:
    payload = Stage2TwoKilohertzContract.canonical().model_dump(mode="python")
    payload[field] = value
    with pytest.raises(ValidationError, match=match):
        Stage2TwoKilohertzContract.model_validate(payload)


def test_full_octave_v3_payload_cannot_be_parsed_as_stage2() -> None:
    with pytest.raises(ValidationError):
        Stage2TwoKilohertzContract.model_validate(
            BroadbandFullOctaveContractV3.canonical().model_dump(mode="python")
        )
