import re
import spacy
import os
from datetime import datetime

# Load model
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xlmroberta_final")

print("Loading XLM-RoBERTa model...")
nlp = spacy.load(MODEL_PATH)
print("Model loaded.")

# Number extractor
def extract_number(text):
    """Extract first number from entity text including lakh/crore formats."""
    # Remove commas from numbers
    text = text.replace(',', '')
    
    # Find numbers with lakh/crore multipliers
    lakh_match = re.search(r'(\d+\.?\d*)\s*lakh', text, re.IGNORECASE)
    crore_match = re.search(r'(\d+\.?\d*)\s*crore', text, re.IGNORECASE)
    
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)
    if crore_match:
        return int(float(crore_match.group(1)) * 10000000)
    
    # Regular number
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
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
    """Apply regex patterns to extract numbers when model misses them."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Get the number group
            for group in match.groups():
                if group and re.search(r'\d', group):
                    return extract_number(group)
    return None

# Main entity linker
def extract_entities(text, source_url, publish_date=None):
    """Run NER model and rule-based extraction, return structured record."""
    
    doc = nlp(text)
    
    # Initialize record
    record = {
        "source_url":          source_url,
        "publish_date":        publish_date,
        "disaster_type":       [],
        "locations": {
            "districts":       [],
            "upazilas":        [],
            "unions":          []
        },
        "fatalities":          None,
        "displaced":           None,
        "affected_people":     None,
        "missing":             None,
        "relief_info":         [],
        "agencies_involved":   [],
        "raw_entities":        []
    }
    
    # Track what model found
    fatality_mentions  = []
    displaced_mentions = []
    affected_mentions  = []
    
    for ent in doc.ents:
        label = ent.label_
        text_  = ent.text.strip()
        
        # Store all raw entities for reference
        record["raw_entities"].append({"text": text_, "label": label})
        
        if label == "DISASTER_TYPE":
            if text_.lower() not in [d.lower() for d in record["disaster_type"]]:
                record["disaster_type"].append(text_)
        
        elif label == "BD_DISTRICT":
            if text_ not in record["locations"]["districts"]:
                record["locations"]["districts"].append(text_)
        
        elif label == "BD_UPAZILA":
            if text_ not in record["locations"]["upazilas"]:
                record["locations"]["upazilas"].append(text_)
        
        elif label == "BD_UNION":
            if text_ not in record["locations"]["unions"]:
                record["locations"]["unions"].append(text_)
        
        elif label == "FATALITIES":
            fatality_mentions.append(text_)
        
        elif label == "DISPLACED":
            displaced_mentions.append(text_)
        
        elif label == "AFFECTED_PEOPLE":
            affected_mentions.append(text_)
        
        elif label == "MISSING":
            num = extract_number(text_)
            if num is not None and (record["missing"] is None or num > record["missing"]):
                record["missing"] = num
            elif num is None and record["missing"] is None:
                record["missing"] = text_
        
        elif label == "RELIEF_INFO":
            if text_ not in record["relief_info"]:
                record["relief_info"].append(text_)
        
        elif label == "AGENCIES_INVOLVED":
            if text_ not in record["agencies_involved"]:
                record["agencies_involved"].append(text_)
    
    # Extract highest fatality number from model mentions
    if fatality_mentions:
        numbers = [extract_number(m) for m in fatality_mentions]
        numbers = [n for n in numbers if n is not None]
        if numbers:
            record["fatalities"] = max(numbers)
    
    # Rule-based fallback for fatalities
    if record["fatalities"] is None:
        record["fatalities"] = apply_rule_based(text, FATALITY_PATTERNS)
    
    # Extract displaced
    if displaced_mentions:
        numbers = [extract_number(m) for m in displaced_mentions]
        numbers = [n for n in numbers if n is not None]
        if numbers:
            record["displaced"] = max(numbers)
    
    if record["displaced"] is None:
        record["displaced"] = apply_rule_based(text, DISPLACED_PATTERNS)
    
    # Extract affected people
    if affected_mentions:
        numbers = [extract_number(m) for m in affected_mentions]
        numbers = [n is not None and n for n in numbers]
        numbers = [n for n in numbers if n]
        if numbers:
            record["affected_people"] = max(numbers)
    
    if record["affected_people"] is None:
        record["affected_people"] = apply_rule_based(text, AFFECTED_PATTERNS)
    
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