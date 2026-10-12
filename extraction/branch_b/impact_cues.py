# Impact cue lexicons for candidate generation
# These are high-recall detectors only — not classifiers

IMPACT_CUES = {
    "human_impacts": [
        "died", "killed", "death toll", "drowned", "dead",
        "injured", "wounded", "hurt", "missing", "unaccounted",
        "affected", "displaced", "marooned", "stranded",
        "evacuated", "homeless", "sheltered", "rescued"
    ],
    "housing_damage": [
        "houses collapsed", "homes destroyed", "houses damaged",
        "washed away", "dwelling", "shanties", "structures damaged",
        "buildings collapsed", "houses inundated"
    ],
    "infrastructure": [
        "road damaged", "bridge collapsed", "embankment breached",
        "roads submerged", "culvert damaged", "highway blocked",
        "rail disrupted", "power outage", "communication disrupted"
    ],
    "agriculture": [
        "crops damaged", "farmland inundated", "crop losses",
        "livestock killed", "fish farm", "paddy field",
        "agricultural land", "harvest lost", "cropland submerged"
    ],
    "relief": [
        "relief", "aid distributed", "allocated", "disbursed",
        "food assistance", "cash support", "shelter opened",
        "emergency supplies", "relief camp", "rescue operation",
        "tarpaulin", "dry food", "rice distributed"
    ]
}

# Negation patterns — candidates in these contexts need flagging
NEGATION_CUES = [
    "no deaths", "no casualties", "not reported",
    "denied", "no fatalities", "none reported",
    "no injuries"
]

# Historical reference patterns
HISTORICAL_CUES = [
    "in 2022", "in 2023", "in 2024", "last year",
    "previously", "earlier flood", "past disaster",
    "compared to", "historically"
]

# Forecast patterns
FORECAST_CUES = [
    "expected", "predicted", "forecast", "may rise",
    "could reach", "anticipated", "warning issued"
]