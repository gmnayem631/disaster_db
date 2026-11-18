from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ImpactType(str, Enum):
    FATALITIES         = "FATALITIES"
    INJURED            = "INJURED"
    MISSING            = "MISSING"
    AFFECTED_PEOPLE    = "AFFECTED_PEOPLE"
    DISPLACED          = "DISPLACED"    # includes marooned, stranded, evacuated (decided 2026-11-17)
    HOUSING_DAMAGE     = "HOUSING_DAMAGE"
    INFRASTRUCTURE_DAMAGE = "INFRASTRUCTURE_DAMAGE"
    AGRICULTURAL_DAMAGE = "AGRICULTURAL_DAMAGE"
    LIVESTOCK_LOSS     = "LIVESTOCK_LOSS"
    ECONOMIC_LOSS      = "ECONOMIC_LOSS"
    RELIEF             = "RELIEF"
    OTHER              = "OTHER"


class AggregationType(str, Enum):
    CUMULATIVE     = "CUMULATIVE"
    INCREMENTAL    = "INCREMENTAL"
    POINT_ESTIMATE = "POINT_ESTIMATE"
    RANGE          = "RANGE"
    LOWER_BOUND    = "LOWER_BOUND"
    UPPER_BOUND    = "UPPER_BOUND"
    UNKNOWN        = "UNKNOWN"


class CertaintyType(str, Enum):
    REPORTED   = "REPORTED"
    ESTIMATED  = "ESTIMATED"
    CONFIRMED  = "CONFIRMED"
    EXPECTED   = "EXPECTED"
    PREDICTED  = "PREDICTED"
    POSSIBLE   = "POSSIBLE"


class ImpactClaim(BaseModel):
    """One answer of the LLM. Only candidate_id and impact_type are required,
    so an OTHER answer can be one short line (writing tokens is the slow part
    on CPU). The defaults are deliberately "weak": an impact claim that leaves
    out its aggregation gets UNKNOWN and never feeds a headline number, and
    one that leaves out its evidence is flagged for review."""
    candidate_id:          Optional[str]
    impact_type:           ImpactType
    evidence_sentence_ids: list[str] = []
    location_ids:          list[str] = []
    aggregation:           AggregationType = AggregationType.UNKNOWN
    certainty:             CertaintyType = CertaintyType.REPORTED
    unit:                  Optional[str] = None


class BranchBResult(BaseModel):
    claims: list[ImpactClaim]


class NormalizedQuantity(BaseModel):
    raw_value:        str
    normalized_value: Optional[float]
    modifier:         Optional[str]
    unit:             Optional[str]
    is_qualitative:   bool = False
    qualitative_text: Optional[str] = None
    lower_bound:      Optional[float] = None
    upper_bound:      Optional[float] = None