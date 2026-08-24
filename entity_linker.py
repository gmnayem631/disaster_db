import re
import spacy
import os
import json

# Load model
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "xlmroberta_final")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xlmroberta_v3") # updated xlmroberta model

# print("Loading XLM-RoBERTa model...")
# nlp = spacy.load(MODEL_PATH)
# print("Model loaded.")

print("Loading XLM-RoBERTa model...")
nlp = spacy.load(MODEL_PATH)

# Load gazetteer and add to pipeline
GAZETTEER_PATH = os.path.join(BASE_DIR, "data", "gazetteer", "bangladesh_gazetteer.json")

with open(GAZETTEER_PATH, "r", encoding="utf-8") as f:
    gazetteer = json.load(f)

if "entity_ruler" in nlp.pipe_names:
    ruler = nlp.get_pipe("entity_ruler")
else:
    ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = []
for district in gazetteer["districts"]:
    patterns.append({"label": "BD_DISTRICT", "pattern": district})
    patterns.append({"label": "BD_DISTRICT", "pattern": district.lower()})

for upazila in gazetteer["upazilas"]:
    patterns.append({"label": "BD_UPAZILA", "pattern": upazila})
    patterns.append({"label": "BD_UPAZILA", "pattern": upazila.lower()})

for union in gazetteer["unions"]:
    patterns.append({"label": "BD_UNION", "pattern": union})
    patterns.append({"label": "BD_UNION", "pattern": union.lower()})

ruler.add_patterns(patterns)
print(f"Gazetteer loaded: {len(patterns)} patterns added.")

# Number extractor 
def extract_number(text):
    """Extract number from entity text including words and South Asian formats."""
    text_clean = text.replace(',', '').strip()

    # Word to number mapping
    word_numbers = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
        'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
        'ninety': 90, 'hundred': 100, 'thousand': 1000,
    }

    # South Asian formats
    lakh_match  = re.search(r'(\d+\.?\d*)\s*lakh', text_clean, re.IGNORECASE)
    crore_match = re.search(r'(\d+\.?\d*)\s*crore', text_clean, re.IGNORECASE)
    million_match = re.search(r'(\d+\.?\d*)\s*million', text_clean, re.IGNORECASE)
    thousand_match = re.search(r'(\d+\.?\d*)\s*thousand', text_clean, re.IGNORECASE)

    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)
    if crore_match:
        return int(float(crore_match.group(1)) * 10000000)
    if million_match:
        return int(float(million_match.group(1)) * 1000000)
    if thousand_match:
        return int(float(thousand_match.group(1)) * 1000)

    # Regular digits
    digit_match = re.search(r'\d+', text_clean)
    if digit_match:
        return int(digit_match.group())

    # Word numbers
    text_lower = text_clean.lower()
    for word, value in word_numbers.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            return value

    return None

# Rule-based fallback patterns 
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

# Main entity linker 
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
            if num is not None:
                existing = record["missing"]
                existing_val = existing.get("value", 0) if existing else 0
                if isinstance(existing_val, str):
                    existing_val = 0
                if existing is None or num > existing_val:
                    record["missing"] = {
                        "value":  num,
                        "source": "model"
                    }
            elif record["missing"] is None:
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

    # Process numerical entities 
    if fatality_mentions:
        cumulative_keywords = ['total', 'toll', 'risen to', 'climbed to',
                            'reached', 'now', 'so far']
        cumulative = [m for m in fatality_mentions
                    if any(kw in m.lower() for kw in cumulative_keywords)]
        best = cumulative[-1] if cumulative else fatality_mentions[-1]
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
        cumulative_keywords = ['total', 'displaced', 'stranded',
                               'marooned', 'evacuated', 'so far']
        cumulative = [m for m in displaced_mentions
                      if any(kw in m.lower() for kw in cumulative_keywords)]
        best = cumulative[-1] if cumulative else displaced_mentions[-1]
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
        cumulative_keywords = ['total', 'affected', 'impacted',
                               'so far', 'now', 'across']
        cumulative = [m for m in affected_mentions
                      if any(kw in m.lower() for kw in cumulative_keywords)]
        best = cumulative[-1] if cumulative else affected_mentions[-1]
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

# Test 
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