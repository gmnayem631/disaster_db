import re
import sys
import os
import spacy
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from extraction.branch_b.impact_cues import IMPACT_CUES, NEGATION_CUES, HISTORICAL_CUES, FORECAST_CUES

# Lightweight sentencizer
_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")


# Data structures 
@dataclass
class Sentence:
    sentence_id: str
    text: str
    start_char: int
    end_char: int


@dataclass
class Candidate:
    candidate_id: str
    kind: str  # QUANTITY or CUE
    raw: str
    sentence_id: str
    start_char: int
    end_char: int
    context_sentence_ids: list = field(default_factory=list)
    flags: list = field(default_factory=list)


# Quantity patterns 
QUANTITY_PATTERNS = [
    # Bengali/South Asian comma format: 12,27,554 or 1,23,456
    r'\b\d{1,2}(?:,\d{2})*(?:,\d{3})\b',
    # South Asian formats
    r'\d+\.?\d*\s*(?:lakh|crore|million|billion|thousand)',
    # Ranges
    r'\d[\d,]*\s*(?:to|-)\s*\d[\d,]*',
    # Approximations with digits
    r'(?:nearly|almost|about|around|approximately|over|at least|up to|more than|some)\s+\d[\d,]*',
    # Plain digits with commas
    r'\b\d{1,3}(?:,\d{3})+\b',
    # Plain digits
    r'\b\d+\b',
    # Word numbers
    r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
    r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|'
    r'thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b',
    # Compound word numbers
    r'\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
    r'[\s-](?:one|two|three|four|five|six|seven|eight|nine)\b',
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in QUANTITY_PATTERNS]


def get_sentences(text: str) -> list[Sentence]:
    """Segment text into sentences with IDs and character offsets."""
    doc = _nlp(text)
    sentences = []
    for i, sent in enumerate(doc.sents):
        sentences.append(Sentence(
            sentence_id = f"S{i+1}",
            text        = sent.text,
            start_char  = sent.start_char,
            end_char    = sent.end_char
        ))
    return sentences


def detect_flags(text: str) -> list[str]:
    """Detect negation, historical, forecast flags in sentence."""
    flags = []
    text_lower = text.lower()
    if any(cue in text_lower for cue in NEGATION_CUES):
        flags.append("NEGATION")
    if any(cue in text_lower for cue in HISTORICAL_CUES):
        flags.append("HISTORICAL")
    if any(cue in text_lower for cue in FORECAST_CUES):
        flags.append("FORECAST")
    return flags


def has_impact_cue(text: str) -> bool:
    """Check if sentence contains any impact cue."""
    text_lower = text.lower()
    for cues in IMPACT_CUES.values():
        if any(cue in text_lower for cue in cues):
            return True
    return False

# Remove smaller spans that are substrings of larger detected spans
def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    Remove overlapping candidates.
    Priority: South Asian formats > approximations > plain digits > word numbers.
    """
    quantities = [c for c in candidates if c.kind == "QUANTITY"]
    cues       = [c for c in candidates if c.kind == "CUE"]

    def priority(c: Candidate) -> int:
        raw = c.raw.lower()
        if any(unit in raw for unit in ['lakh', 'crore', 'million', 'billion', 'thousand']):
            return 4  # highest — South Asian/large formats
        if re.search(r'\d{1,3}(?:,\d{3})+', raw):
            return 3  # comma-formatted numbers
        if re.search(r'\d', raw):
            return 2  # plain digits
        return 1      # word numbers

    # Sort by priority descending, then by length descending as tiebreaker
    quantities_sorted = sorted(
        quantities,
        key=lambda c: (priority(c), c.end_char - c.start_char),
        reverse=True
    )

    accepted       = []
    accepted_spans = []

    for c in quantities_sorted:
        overlaps = False
        for s, e in accepted_spans:
            if c.start_char < e and c.end_char > s:
                overlaps = True
                break
        if not overlaps:
            accepted.append(c)
            accepted_spans.append((c.start_char, c.end_char))

    # Restore sentence order
    accepted_ordered = sorted(accepted, key=lambda c: c.start_char)

    return accepted_ordered + cues

def generate_candidates(text: str) -> tuple[list[Sentence], list[Candidate]]:
    """
    Generate all quantity and cue candidates from article text.
    Returns sentences and candidates.
    """
    sentences   = get_sentences(text)
    sentence_map = {s.sentence_id: s for s in sentences}
    candidates  = []
    q_counter   = 1
    c_counter   = 1
    seen_spans  = set()

    for i, sentence in enumerate(sentences):
        flags = detect_flags(sentence.text)

        # Context window
        context_ids = []
        if i > 0:
            context_ids.append(sentences[i-1].sentence_id)
        context_ids.append(sentence.sentence_id)
        if i < len(sentences) - 1:
            context_ids.append(sentences[i+1].sentence_id)

        # Quantity candidates
        found_quantities = []
        for pattern in COMPILED_PATTERNS:
            for match in pattern.finditer(sentence.text):
                abs_start = sentence.start_char + match.start()
                abs_end   = sentence.start_char + match.end()
                span_key  = (abs_start, abs_end)

                if span_key in seen_spans:
                    continue

                # Skip if fully contained in a larger seen span
                dominated = any(
                    s[0] <= abs_start and abs_end <= s[1]
                    for s in seen_spans
                    if s != span_key
                )
                if dominated:
                    continue

                seen_spans.add(span_key)
                found_quantities.append(Candidate(
                    candidate_id         = f"Q{q_counter}",
                    kind                 = "QUANTITY",
                    raw                  = match.group().strip(),
                    sentence_id          = sentence.sentence_id,
                    start_char           = abs_start,
                    end_char             = abs_end,
                    context_sentence_ids = context_ids,
                    flags                = flags
                ))
                q_counter += 1

        candidates.extend(found_quantities)

        # Cue-only candidates — impact sentences without quantities
        if has_impact_cue(sentence.text) and not found_quantities:
            candidates.append(Candidate(
                candidate_id         = f"C{c_counter}",
                kind                 = "CUE",
                raw                  = sentence.text.strip(),
                sentence_id          = sentence.sentence_id,
                start_char           = sentence.start_char,
                end_char             = sentence.end_char,
                context_sentence_ids = context_ids,
                flags                = flags
            ))
            c_counter += 1

        candidates = deduplicate_candidates(candidates)
    return sentences, candidates


def format_for_llm(
    sentences: list[Sentence],
    candidates: list[Candidate],
    branch_a_locations: list[dict]
) -> str:
    """
    Format candidates and context into a structured prompt for Branch B LLM.
    Sends candidates in batches with their context sentences only.
    """
    # Build sentence lookup
    sentence_lookup = {s.sentence_id: s.text for s in sentences}

    # Collect all relevant sentence IDs
    relevant_ids = set()
    for c in candidates:
        for sid in c.context_sentence_ids:
            relevant_ids.add(sid)

    # Also include flagged sentences even with no candidates
    for s in sentences:
        if detect_flags(s.text):
            relevant_ids.add(s.sentence_id)

    # Format locations
    loc_lines = "\n".join(
        f"{loc['location_id']} = {loc['name']} [{loc['type']}]"
        for loc in branch_a_locations
    ) if branch_a_locations else "None identified"

    # Format candidates
    cand_lines = "\n".join(
        f"{c.candidate_id} = \"{c.raw}\" in {c.sentence_id}"
        + (f" [FLAGS: {', '.join(c.flags)}]" if c.flags else "")
        for c in candidates
    )

    # Format relevant sentences only
    sent_lines = "\n".join(
        f"{sid}: {sentence_lookup[sid]}"
        for sid in sorted(relevant_ids,
                         key=lambda x: int(x[1:]))
        if sid in sentence_lookup
    )

    return f"""Known locations:
{loc_lines}

Candidate quantities and cues:
{cand_lines}

Relevant sentences:
{sent_lines}

Classify each candidate as a disaster impact claim.
Use only the IDs provided above.
Do not invent quantities, locations or evidence.
If a candidate is not a disaster impact, label it as OTHER.
If aggregation cannot be determined, use UNKNOWN."""


# Test 
if __name__ == "__main__":
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

    print(f"Sentences found: {len(sentences)}")
    print(f"Candidates found: {len(candidates)}\n")

    for c in candidates:
        flag_str = f" [{', '.join(c.flags)}]" if c.flags else ""
        print(f"  {c.candidate_id} ({c.kind}): \"{c.raw}\" — {c.sentence_id}{flag_str}")

    print("\n--- LLM Prompt Preview ---\n")
    branch_a_mock = [
        {"location_id": "L1", "name": "Feni", "type": "BD_DISTRICT"},
        {"location_id": "L2", "name": "Companiganj", "type": "BD_UPAZILA"},
    ]
    prompt = format_for_llm(sentences, candidates, branch_a_mock)
    print(prompt)