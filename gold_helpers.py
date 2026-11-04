import re

PEOPLE_TYPES = {"FATALITIES", "INJURED", "MISSING", "AFFECTED_PEOPLE", "DISPLACED", "EVACUATED", "MAROONED"}

UNIT_WORDS = {
    "people": "people", "persons": "people", "person": "people", "residents": "people",
    "villagers": "people", "children": "people", "women": "people", "men": "people",
    "individuals": "people", "victims": "people", "fishermen": "people", "farmers": "people",
    "workers": "people", "bodies": "people", "deaths": "people", "dead": "people",
    "families": "families", "family": "families", "households": "families",
    "houses": "houses", "homes": "houses", "tonnes": "tonnes", "tons": "tonnes",
    "kg": "kg", "packets": "packets", "pieces": "pieces", "taka": "taka",
    "hectares": "hectares", "acres": "acres", "animals": "animals", "cattle": "animals",
    "villages": "villages", "unions": "unions", "upazilas": "upazilas",
    "districts": "districts", "municipalities": "municipalities",
    "centres": "centres", "centers": "centres", "teams": "teams",
}
NON_PEOPLE_UNITS = {"villages", "unions", "upazilas", "districts", "municipalities",
                    "centres", "teams", "animals", "houses", "tonnes", "kg", "packets",
                    "pieces", "taka", "hectares", "acres"}

INCREMENT_WORDS = {"more", "another", "additional", "fresh", "further", "new"}
CUMULATIVE_CUES = ["toll", "total", "so far", "risen to", "rose to", "climbed to",
                   "reached", "stands at", "tally", "altogether", "overall", "in all"]


def guess_unit(before, after):
    """before/after = the sentence text on each side of the number."""
    last = re.findall(r"[a-zA-Z]+", before.lower())[-1:]
    if last and last[0] in {"tk", "taka", "bdt"}:
        return "taka"
    for w in re.findall(r"[a-zA-Z]+", after.lower())[:4]:
        if w in UNIT_WORDS:
            return UNIT_WORDS[w]
    return ""


def guess_aggregation(before, after):
    near = set(re.findall(r"[a-zA-Z]+", before.lower())[-2:]) | set(re.findall(r"[a-zA-Z]+", after.lower())[:2])
    if near & INCREMENT_WORDS:
        return "INCREMENTAL"
    window = " ".join(re.findall(r"[a-zA-Z]+", before.lower())[-5:])
    if any(cue in window for cue in CUMULATIVE_CUES):
        return "CUMULATIVE"
    return "POINT_ESTIMATE"


def review_reason(impact_type, raw, unit):
    if not raw:
        return "no number found (qualitative claim or generator miss)"
    if not unit:
        return "unit not found"
    if impact_type in PEOPLE_TYPES and unit in NON_PEOPLE_UNITS:
        return f"number counts {unit}, not people - probably OTHER"
    return ""