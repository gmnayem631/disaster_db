"""Regression tests for claim_validator.non_people_noun_after."""
import pytest

from extraction.branch_b.candidate_generator import generate_candidates
from extraction.branch_b.claim_validator import (
    non_people_noun_after, validate_claims, REJECTED,
)
from extraction.branch_b.schemas import (
    BranchBResult, ImpactClaim, ImpactType, AggregationType, CertaintyType,
)


def _find(text, raw):
    """Run the real candidate generator and return (candidate, sentences_by_id)."""
    sentences, candidates = generate_candidates(text)
    matches = [c for c in candidates if c.kind == "QUANTITY" and c.raw == raw]
    assert matches, f"no candidate with raw={raw!r}; got {[c.raw for c in candidates]}"
    return matches[0], {s.sentence_id: s for s in sentences}, sentences, candidates


@pytest.mark.parametrize("text, raw, expected", [
    ("The death toll has risen to 31 in Feni district.", "31", None),
    ("Floods hit 73 upazilas of 11 flood-hit districts.", "73", "upazilas"),
    ("Floods killed 39,531 domestic animals.", "39,531", "animals"),
    ("There were four more deaths in Cumilla.", "four", None),
])
def test_non_people_noun_after(text, raw, expected):
    cand, by_id, _, _ = _find(text, raw)
    assert non_people_noun_after(cand, by_id) == expected


def test_feni_fatality_claim_not_rejected():
    text = "The death toll has risen to 31 in Feni district."
    cand, _, sentences, candidates = _find(text, "31")
    result = BranchBResult(claims=[ImpactClaim(
        candidate_id=cand.candidate_id,
        impact_type=ImpactType.FATALITIES,
        evidence_sentence_ids=[cand.sentence_id],
        location_ids=[],
        aggregation=AggregationType.CUMULATIVE,
        certainty=CertaintyType.REPORTED,
        unit="people",
    )])
    [record] = validate_claims(result, candidates, sentences, [])
    assert record["validation_status"] != REJECTED
    assert not any(f.startswith("NOT_A_PEOPLE_COUNT") for f in record["validation_flags"])


@pytest.mark.parametrize("unit, impact, expected", [
    ("medical teams", "RELIEF", True),
    ("metric tons of rice", "RELIEF", True),
    ("shelter centres", "RELIEF", True),
    ("food packets", "RELIEF", True),
    ("districts", "RELIEF", False),
    ("families", "DISPLACED", True),
    ("villages", "AFFECTED_PEOPLE", False),
    ("null", "FATALITIES", True),          # no real unit: nothing to check
    ("houses", "HOUSING_DAMAGE", True),
    ("people", "HOUSING_DAMAGE", False),
])
def test_unit_is_compatible_word_by_word(unit, impact, expected):
    from extraction.branch_b.claim_validator import unit_is_compatible
    assert unit_is_compatible(unit, impact) is expected
