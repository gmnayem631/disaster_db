import re
from typing import Optional
from extraction.branch_b.schemas import NormalizedQuantity

WORD_NUMBERS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
}
MODIFIERS = ['nearly', 'almost', 'about', 'around', 'approximately', 'over', 'more than',
             'at least', 'up to', 'fewer than', 'less than', 'some', 'just over', 'just under']
QUALITATIVE = ['several', 'dozens', 'hundreds', 'thousands', 'many', 'numerous',
               'scores', 'a number of', 'a few']
SCALES = {'lakh': 100_000, 'crore': 10_000_000, 'million': 1_000_000,
          'billion': 1_000_000_000, 'thousand': 1_000}
RANGE_RE = re.compile(r'(\d+\.?\d*)\s*(?:to|-|–)\s*(\d+\.?\d*)\s*(lakh|crore|million|billion|thousand)?')


def extract_modifier(text):
    t = text.lower()
    for mod in sorted(MODIFIERS, key=len, reverse=True):
        if mod in t:
            return mod.upper().replace(' ', '_')
    return None


def parse_word_number(text):
    words = re.sub(r'[-]', ' ', text.lower().strip()).split()
    total, current, found = 0, 0, False
    for w in words:
        if w in WORD_NUMBERS:
            found = True
            v = WORD_NUMBERS[w]
            if v == 100:
                current = current * 100 if current else 100
            elif v == 1000:
                total += (current if current else 1) * 1000
                current = 0
            else:
                current += v
    return float(total + current) if found else None


def normalize_quantity(raw_text, unit=None):
    text = raw_text.strip()
    # remove commas AND a space after a comma: "5, 40,510" -> "540510"
    clean = re.sub(r',\s*', '', text).lower()
    modifier = extract_modifier(clean)

    for qual in QUALITATIVE:
        if qual in clean:
            return NormalizedQuantity(raw_value=raw_text, normalized_value=None, modifier=modifier,
                                      unit=unit, is_qualitative=True, qualitative_text=qual)

    # Ranges: keep both ends, never pick one silently
    rm = RANGE_RE.search(clean)
    if rm:
        scale = SCALES.get(rm.group(3), 1) if rm.group(3) else 1
        lo, hi = round(float(rm.group(1)) * scale), round(float(rm.group(2)) * scale)
        return NormalizedQuantity(raw_value=raw_text, normalized_value=None, modifier=modifier,
                                  unit=unit, lower_bound=lo, upper_bound=hi)

    val = None
    for name in ('crore', 'lakh', 'billion', 'million', 'thousand'):
        m = re.search(r'(\d+\.?\d*)\s*' + name, clean)
        if m:
            val = round(float(m.group(1)) * SCALES[name])
            break
    if val is None:
        m = re.search(r'\d+\.?\d*', clean)
        val = float(m.group()) if m else parse_word_number(clean)

    return NormalizedQuantity(raw_value=raw_text, normalized_value=val, modifier=modifier, unit=unit)