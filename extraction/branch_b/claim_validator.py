import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from extraction.branch_b.schemas import ImpactClaim, BranchBResult
from extraction.branch_b.candidate_generator import Sentence, Candidate
from extraction.branch_b.quantity_normalizer import normalize_quantity
from extraction.branch_b.impact_cues import NEGATION_CUES, HISTORICAL_CUES, FORECAST_CUES


# ── Validation outcomes ────────────────────────────────────────────────────
ACCEPTED        = "ACCEPTED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
REJECTED        = "REJECTED"


# ── Unit compatibility ─────────────────────────────────────────────────────
UNIT_COMPATIBLE = {
    "FATALITIES":             ["people", "persons", "individuals", "victims"],
    "INJURED":                ["people", "persons", "individuals"],
    "MISSING":                ["people", "persons", "individuals"],
    "AFFECTED_PEOPLE":        ["people", "persons", "families", "households"],
    "DISPLACED":              ["people", "persons", "families", "households"],
    "EVACUATED":              ["people", "persons", "families", "households"],
    "MAROONED":               ["people", "persons", "families", "households"],
    "HOUSING_DAMAGE":         ["houses", "homes", "structures", "buildings", "units"],
    "INFRASTRUCTURE_DAMAGE":  ["roads", "bridges", "km", "kilometers", "structures"],
    "AGRICULTURAL_DAMAGE":    ["hectares", "acres", "bigha", "crops", "tonnes"],
    "LIVESTOCK_LOSS":         ["animals", "cattle", "livestock", "heads"],
    "ECONOMIC_LOSS":          ["taka", "tk", "crore", "lakh", "usd"],
    "RELIEF":                 ["taka", "tk", "crore", "lakh", "tonnes", "kg",
                               "packets", "families", "people"],
    "OTHER":                  [],
}


def validate_claims(
    branch_b_result: BranchBResult,
    candidates: list[Candidate],
    sentences: list[Sentence],
    branch_a_locations: list[dict]
) -> list[dict]:
    """
    Validate each LLM-generated claim against supplied data.
    Returns list of validated claim records with status and flags.
    """
    # Build lookup maps
    candidates_by_id  = {c.candidate_id: c for c in candidates}
    sentences_by_id   = {s.sentence_id: s for s in sentences}
    location_ids      = {loc["location_id"] for loc in branch_a_locations}

    validated = []

    for claim in branch_b_result.claims:
        flags   = []
        status  = ACCEPTED

        # ── 1. Candidate ID verification ──────────────────────────────────
        if claim.candidate_id not in candidates_by_id:
            validated.append(_rejected_claim(
                claim, "INVALID_CANDIDATE_ID",
                f"Candidate {claim.candidate_id} was not supplied"
            ))
            continue

        candidate = candidates_by_id[claim.candidate_id]

        # ── 2. Evidence sentence verification ─────────────────────────────
        for sid in claim.evidence_sentence_ids:
            if sid not in sentences_by_id:
                flags.append(f"INVALID_SENTENCE_ID:{sid}")
                status = REJECTED

        if status == REJECTED:
            validated.append(_build_result(
                claim, candidate, None, status, flags
            ))
            continue

        # ── 3. Candidate appears in evidence sentence ──────────────────────
        primary_sid = (claim.evidence_sentence_ids[0]
                       if claim.evidence_sentence_ids
                       else candidate.sentence_id)

        if primary_sid in sentences_by_id:
            evidence_text = sentences_by_id[primary_sid].text
            if (candidate.kind == "QUANTITY" and
                    candidate.raw.lower() not in evidence_text.lower()):
                flags.append("CANDIDATE_NOT_IN_EVIDENCE")
                status = REVIEW_REQUIRED
        else:
            flags.append("EVIDENCE_SENTENCE_MISSING")
            status = REVIEW_REQUIRED

        # ── 4. Location ID verification ────────────────────────────────────
        for lid in claim.location_ids:
            if lid not in location_ids:
                flags.append(f"INVALID_LOCATION_ID:{lid}")
                status = REVIEW_REQUIRED

        # ── 5. Quantity normalization ──────────────────────────────────────
        normalized = None
        if candidate.kind == "QUANTITY":
            normalized = normalize_quantity(candidate.raw, claim.unit)

        # ── 6. Unit compatibility ──────────────────────────────────────────
        impact_type_str = claim.impact_type.value
        if claim.unit and impact_type_str in UNIT_COMPATIBLE:
            compatible_units = UNIT_COMPATIBLE[impact_type_str]
            if (compatible_units and
                    claim.unit.lower() not in compatible_units):
                flags.append(
                    f"UNIT_MISMATCH:{claim.unit}"
                    f"_not_compatible_with_{impact_type_str}"
                )
                status = REVIEW_REQUIRED

        # ── 7. Negation check ──────────────────────────────────────────────
        evidence_text = ""
        if primary_sid in sentences_by_id:
            evidence_text = sentences_by_id[primary_sid].text.lower()

        if any(cue in evidence_text for cue in NEGATION_CUES):
            flags.append("NEGATION_DETECTED")
            status = REVIEW_REQUIRED

        # ── 8. Historical reference check ─────────────────────────────────
        if any(cue in evidence_text for cue in HISTORICAL_CUES):
            flags.append("HISTORICAL_REFERENCE")
            status = REVIEW_REQUIRED

        # ── 9. Forecast check ─────────────────────────────────────────────
        if any(cue in evidence_text for cue in FORECAST_CUES):
            flags.append("FORECAST_DETECTED")
            status = REVIEW_REQUIRED

        # ── 10. Candidate flags from generator ────────────────────────────
        for f in candidate.flags:
            flags.append(f"GENERATOR_FLAG:{f}")
            if status == ACCEPTED:
                status = REVIEW_REQUIRED

        validated.append(_build_result(
            claim, candidate, normalized, status, flags
        ))

    return validated


def _build_result(
    claim: ImpactClaim,
    candidate: Candidate,
    normalized,
    status: str,
    flags: list
) -> dict:
    """Build a validated claim record."""
    return {
        "candidate_id":        claim.candidate_id,
        "impact_type":         claim.impact_type.value,
        "raw_value":           candidate.raw,
        "normalized_quantity": {
            "raw":        normalized.raw_value        if normalized else None,
            "value":      normalized.normalized_value if normalized else None,
            "modifier":   normalized.modifier         if normalized else None,
            "unit":       normalized.unit             if normalized else None,
            "qualitative": normalized.is_qualitative  if normalized else False,
        },
        "aggregation":         claim.aggregation.value,
        "certainty":           claim.certainty.value,
        "location_ids":        claim.location_ids,
        "evidence_sentence_ids": claim.evidence_sentence_ids,
        "validation_status":   status,
        "validation_flags":    flags
    }


def _rejected_claim(
    claim: ImpactClaim,
    flag: str,
    reason: str
) -> dict:
    """Build a rejected claim record."""
    return {
        "candidate_id":        claim.candidate_id,
        "impact_type":         claim.impact_type.value,
        "raw_value":           None,
        "normalized_quantity": None,
        "aggregation":         claim.aggregation.value,
        "certainty":           claim.certainty.value,
        "location_ids":        claim.location_ids,
        "evidence_sentence_ids": claim.evidence_sentence_ids,
        "validation_status":   REJECTED,
        "validation_flags":    [flag, reason]
    }


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from extraction.branch_b.schemas import (
        ImpactType, AggregationType, CertaintyType
    )

    # Mock Branch B LLM output
    mock_result = BranchBResult(claims=[
        ImpactClaim(
            candidate_id         = "Q1",
            impact_type          = ImpactType.FATALITIES,
            evidence_sentence_ids = ["S2"],
            location_ids         = [],
            aggregation          = AggregationType.CUMULATIVE,
            certainty            = CertaintyType.REPORTED,
            unit                 = "people"
        ),
        ImpactClaim(
            candidate_id         = "Q2",
            impact_type          = ImpactType.FATALITIES,
            evidence_sentence_ids = ["S2"],
            location_ids         = ["L1"],
            aggregation          = AggregationType.INCREMENTAL,
            certainty            = CertaintyType.REPORTED,
            unit                 = "people"
        ),
        ImpactClaim(
            candidate_id         = "Q3",
            impact_type          = ImpactType.AFFECTED_PEOPLE,
            evidence_sentence_ids = ["S3"],
            location_ids         = [],
            aggregation          = AggregationType.POINT_ESTIMATE,
            certainty            = CertaintyType.REPORTED,
            unit                 = "people"
        ),
        ImpactClaim(
            candidate_id         = "Q99",  # invalid — should be rejected
            impact_type          = ImpactType.FATALITIES,
            evidence_sentence_ids = ["S2"],
            location_ids         = [],
            aggregation          = AggregationType.UNKNOWN,
            certainty            = CertaintyType.REPORTED,
            unit                 = "people"
        ),
        ImpactClaim(
            candidate_id         = "Q7",  # S7 has negation
            impact_type          = ImpactType.FATALITIES,
            evidence_sentence_ids = ["S7"],
            location_ids         = [],
            aggregation          = AggregationType.UNKNOWN,
            certainty            = CertaintyType.REPORTED,
            unit                 = "people"
        ),
    ])

    # Mock candidates and sentences
    from extraction.branch_b.candidate_generator import generate_candidates

    test_text = """
    Floods have devastated several districts of Bangladesh.
    The nationwide death toll has risen to 31 after three more people died in Feni district.
    At least 58 lakh people have been affected across 11 districts.
    Around 10,000 families remained marooned in Companiganj upazila.
    Several houses were washed away when the embankment collapsed.
    The government allocated Tk 4.52 crore for relief operations.
    No deaths were reported in Sylhet division yesterday.
    """

    sentences, candidates = generate_candidates(test_text)

    branch_a_locations = [
        {"location_id": "L1", "name": "Feni",        "type": "BD_DISTRICT"},
        {"location_id": "L2", "name": "Companiganj", "type": "BD_UPAZILA"},
    ]

    results = validate_claims(
        mock_result,
        candidates,
        sentences,
        branch_a_locations
    )

    print(f"Validated {len(results)} claims:\n")
    for r in results:
        print(f"  {r['candidate_id']} | {r['impact_type']:<20} | "
              f"{r['validation_status']:<15} | "
              f"value={r['normalized_quantity']['value'] if r['normalized_quantity'] else 'N/A'} | "
              f"flags={r['validation_flags']}")