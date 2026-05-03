import re
import spacy
import os

# ── Load model ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xlmroberta_final")

print("Loading XLM-RoBERTa model...")
nlp = spacy.load(MODEL_PATH)
print("Model loaded.")

# ── Number extractor ───────────────────────────────────────────────────────
def extract_number(text):
    text = text.replace(',', '')
    lakh_match  = re.search(r'(\d+\.?\d*)\s*lakh', text, re.IGNORECASE)
    crore_match = re.search(r'(\d+\.?\d*)\s*crore', text, re.IGNORECASE)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)
    if crore_match:
        return int(float(crore_match.group(1)) * 10000000)
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return None

# ── Rule-based fallback patterns ───────────────────────────────────────────
FATALITY_PATTERNS = [
    r'(\d[\d,]*)\s+people\s+(died|killed|dead)',
    r'death toll\s+\w+\s+to\s+(\d[\d,]*)',
    r'at least\s+(\d[\d,]*)\s+(dead|killed|deaths)',
    r'(\d[\d,]*)\s+(deaths?|fatalities)',
    r'killed\s+(\d[\d,]*)',
]

DISPLACED_PATTERNS = [
    r'(\d[\d,]*\s*(?:lakh|thousand|million)?)\s+(?:people\s+)?(?:displaced|stranded|evacuated|homeless)',
    r'(\d[\d,]*)\s+families\s+(?:displaced|stranded|homeless)',
]

AFFECTED_PATTERNS = [
    r'(\d[\d,]*\s*(?:lakh|thousand|million)?)\s+(?:people\s+)?affected',
    r'affecting\s+(\d[\d,]*\s*(?:lakh|thousand|million)?)\s+people',
]

def apply_rule_based(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group and re.search(r'\d', group):
                    return extract_number(group)
    return None

# ── Main entity linker ─────────────────────────────────────────────────────
def extract_entities(text, source_url, publish_date=None):
    doc = nlp(text)

    record = {
        "source_url":        source_url,
        "publish_date":      publish_date,
        "disaster_type":     [],
        "locations": {
            "districts":     [],
            "upazilas":      [],
            "unions":        []
        },
        "fatalities":        None,
        "displaced":         None,
        "affected_people":   None,
        "missing":           None,
        "relief_info":       [],
        "agencies_involved": [],
        "raw_entities":      []
    }

    fatality_mentions  = []
    displaced_mentions = []
    affected_mentions  = []

    for ent in doc.ents:
        label = ent.label_
        text_ = ent.text.strip()

        record["raw_entities"].append({
            "text":   text_,
            "label":  label,
            "source": "model"
        })

        if label == "DISASTER_TYPE":
            if text_.lower() not in [d["value"].lower() for d in record["disaster_type"]]:
                record["disaster_type"].append({
                    "value":  text_,
                    "source": "model"
                })

        elif label == "BD_DISTRICT":
            if text_ not in [d["value"] for d in record["locations"]["districts"]]:
                record["locations"]["districts"].append({
                    "value":  text_,
                    "source": "model"
                })

        elif label == "BD_UPAZILA":
            if text_ not in [d["value"] for d in record["locations"]["upazilas"]]:
                record["locations"]["upazilas"].append({
                    "value":  text_,
                    "source": "model"
                })

        elif label == "BD_UNION":
            if text_ not in [d["value"] for d in record["locations"]["unions"]]:
                record["locations"]["unions"].append({
                    "value":  text_,
                    "source": "model"
                })

        elif label == "FATALITIES":
            fatality_mentions.append(text_)

        elif label == "DISPLACED":
            displaced_mentions.append(text_)

        elif label == "AFFECTED_PEOPLE":
            affected_mentions.append(text_)

        elif label == "MISSING":
            num = extract_number(text_)
            if num is not None and (record["missing"] is None or num > record["missing"].get("value", 0)):
                record["missing"] = {
                    "value":  num,
                    "source": "model"
                }
            elif num is None and record["missing"] is None:
                record["missing"] = {
                    "value":  text_,
                    "source": "model"
                }

        elif label == "RELIEF_INFO":
            if text_ not in [r["value"] for r in record["relief_info"]]:
                record["relief_info"].append({
                    "value":  text_,
                    "source": "model"
                })

        elif label == "AGENCIES_INVOLVED":
            if text_ not in [a["value"] for a in record["agencies_involved"]]:
                record["agencies_involved"].append({
                    "value":  text_,
                    "source": "model"
                })

    # ── Process numerical entities ─────────────────────────────────────────
    if fatality_mentions:
        best = max(fatality_mentions, key=lambda x: extract_number(x) or 0)
        num  = extract_number(best)
        if num:
            record["fatalities"] = {
                "value":  num,
                "source": "model"
            }

    if record["fatalities"] is None:
        num = apply_rule_based(text, FATALITY_PATTERNS)
        if num:
            record["fatalities"] = {
                "value":  num,
                "source": "rule_based"
            }

    if displaced_mentions:
        best = max(displaced_mentions, key=lambda x: extract_number(x) or 0)
        num  = extract_number(best)
        if num:
            record["displaced"] = {
                "value":  num,
                "source": "model"
            }

    if record["displaced"] is None:
        num = apply_rule_based(text, DISPLACED_PATTERNS)
        if num:
            record["displaced"] = {
                "value":  num,
                "source": "rule_based"
            }

    if affected_mentions:
        best = max(affected_mentions, key=lambda x: extract_number(x) or 0)
        num  = extract_number(best)
        if num:
            record["affected_people"] = {
                "value":  num,
                "source": "model"
            }

    if record["affected_people"] is None:
        num = apply_rule_based(text, AFFECTED_PATTERNS)
        if num:
            record["affected_people"] = {
                "value":  num,
                "source": "rule_based"
            }

    return record

# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_text = """
    The flood situation in Sylhet and Sunamganj has worsened.
    The death toll has risen to 31 with four more deaths in Cumilla and Feni district,
    according to the Disaster Management and Relief Ministry.
    At least 58 lakh people have been affected across 11 districts.
    Some 10,000 families have been displaced in Companiganj upazila.
    Two people remain missing in Moulvibazar district.
    A total of Tk 4.52 crore has been allocated for relief.
    """

    result = extract_entities(test_text, "https://test-url.com", "2024-07-15")

    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))