from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ImpactType(str, Enum):
    FATALITIES         = "FATALITIES"
    INJURED            = "INJURED"
    MISSING            = "MISSING"
    AFFECTED_PEOPLE    = "AFFECTED_PEOPLE"
    DISPLACED          = "DISPLACED"
    EVACUATED          = "EVACUATED"
    MAROONED           = "MAROONED"
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
    candidate_id:          Optional[str]
    impact_type:           ImpactType
    evidence_sentence_ids: list[str]
    location_ids:          list[str]
    aggregation:           AggregationType
    certainty:             CertaintyType
    unit:                  Optional[str]


class BranchBResult(BaseModel):
    claims: list[ImpactClaim]


class NormalizedQuantity(BaseModel):
    raw_value:        str
    normalized_value: Optional[float]
    modifier:         Optional[str]
    unit:             Optional[str]
    is_qualitative:   bool = False
    qualitative_text: Optional[str] = None