import re
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from extraction.branch_b.schemas import NormalizedQuantity


# ── Word number mapping ────────────────────────────────────────────────────
WORD_NUMBERS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
    'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
}

# ── Modifier detection ─────────────────────────────────────────────────────
MODIFIERS = [
    'nearly', 'almost', 'about', 'around', 'approximately',
    'over', 'more than', 'at least', 'up to', 'fewer than',
    'less than', 'some', 'nearly', 'just over', 'just under',
]

# ── Qualitative expressions ────────────────────────────────────────────────
QUALITATIVE = [
    'several', 'dozens', 'hundreds', 'thousands', 'many',
    'numerous', 'scores', 'a number of', 'a few',
]


def extract_modifier(text: str) -> Optional[str]:
    """Extract approximation modifier from raw quantity text."""
    text_lower = text.lower()
    for mod in sorted(MODIFIERS, key=len, reverse=True):
        if mod in text_lower:
            return mod.upper().replace(' ', '_')
    return None


def parse_word_number(text: str) -> Optional[float]:
    """
    Parse compound word numbers like 'twenty five', 'one hundred'.
    Returns None if no word number found.
    """
    text_lower = text.lower().strip()
    text_lower = re.sub(r'[-]', ' ', text_lower)
    words      = text_lower.split()

    total  = 0
    current = 0
    found  = False

    for word in words:
        if word in WORD_NUMBERS:
            found = True
            val   = WORD_NUMBERS[word]
            if val == 100:
                current = current * 100 if current else 100
            elif val == 1000:
                total  += (current if current else 1) * 1000
                current = 0
            else:
                current += val

    if found:
        return round(float(total + current))
    return None


def normalize_quantity(raw_text: str, unit: Optional[str] = None) -> NormalizedQuantity:
    """
    Convert a raw quantity string to a normalized numeric value.
    Retains the original raw text and modifier.
    Never invents values for qualitative expressions.
    """
    text  = raw_text.strip()
    clean = text.replace(',', '').lower()

    modifier       = extract_modifier(clean)
    normalized_val = None
    is_qualitative = False
    qualitative_text = None

    # Check qualitative first
    for qual in QUALITATIVE:
        if qual in clean:
            is_qualitative   = True
            qualitative_text = qual
            return NormalizedQuantity(
                raw_value        = raw_text,
                normalized_value = None,
                modifier         = modifier,
                unit             = unit,
                is_qualitative   = True,
                qualitative_text = qualitative_text
            )

    # South Asian formats
    crore_match   = re.search(r'(\d+\.?\d*)\s*crore', clean)
    lakh_match    = re.search(r'(\d+\.?\d*)\s*lakh', clean)
    million_match = re.search(r'(\d+\.?\d*)\s*million', clean)
    billion_match = re.search(r'(\d+\.?\d*)\s*billion', clean)
    thousand_match = re.search(r'(\d+\.?\d*)\s*thousand', clean)

    if crore_match:
        normalized_val = round(float(crore_match.group(1)) * 10_000_000)
    elif lakh_match:
        normalized_val = round(float(lakh_match.group(1)) * 100_000)
    elif billion_match:
        normalized_val = round(float(billion_match.group(1)) * 1_000_000_000)
    elif million_match:
        normalized_val = round(float(million_match.group(1)) * 1_000_000)
    elif thousand_match:
        normalized_val = round(float(thousand_match.group(1)) * 1_000)
    else:
        # Regular digit
        digit_match = re.search(r'\d+\.?\d*', clean)
        if digit_match:
            normalized_val = round(float(digit_match.group()))
        else:
            # Word number
            normalized_val = parse_word_number(clean)

    return NormalizedQuantity(
        raw_value        = raw_text,
        normalized_value = normalized_val,
        modifier         = modifier,
        unit             = unit,
        is_qualitative   = False,
        qualitative_text = None
    )


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("58 lakh", "people"),
        ("4.52 crore", "taka"),
        ("1.2 million", "people"),
        ("50 thousand", "families"),
        ("twelve", "people"),
        ("twenty five", "people"),
        ("one hundred", "people"),
        ("nearly 31", "people"),
        ("at least 10,000", "families"),
        ("several hundred", "people"),
        ("dozens", "houses"),
        ("31", "people"),
    ]

    print(f"{'Raw':<25} {'Normalized':>15} {'Modifier':<20} {'Qualitative'}")
    print("-" * 75)
    for raw, unit in test_cases:
        result = normalize_quantity(raw, unit)
        norm   = str(int(result.normalized_value)) if result.normalized_value else "None"
        mod    = result.modifier or "-"
        qual   = str(result.is_qualitative)
        print(f"{raw:<25} {norm:>15} {mod:<20} {qual}")