import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import ollama
from extraction.branch_b.schemas import BranchBResult

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

SYSTEM_PROMPT = """You are a disaster information extraction engine for Bangladesh flood news.
Extract only information explicitly stated in the supplied text.
Never invent quantities, locations, dates or evidence.

Impact type guide:
- FATALITIES: people who died or were killed
- AFFECTED_PEOPLE: people affected, impacted, or in distress
- DISPLACED: people displaced, marooned, stranded, evacuated
- MISSING: people missing or unaccounted for
- RELIEF: aid, food, money distributed or allocated

Aggregation guide:
- CUMULATIVE: total running count (death toll rose TO 31)
- INCREMENTAL: new addition (THREE MORE died)
- POINT_ESTIMATE: single reported figure

Unit guide: use 'people' for persons, 'families' for households, 
'taka' for money, 'tonnes' for food/goods.

Return only valid JSON matching the required schema."""

def run_branch_b(prompt: str, model: str = "qwen2.5:3b") -> BranchBResult:
    """Send structured prompt to local LLM and return validated claims."""
    
    response = ollama.chat(
        model   = model,
        messages = [
            {
                "role":    "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role":    "user",
                "content": prompt
            }
        ],
        format  = BranchBResult.model_json_schema(),
        options = {"temperature": 0}
    )

    return BranchBResult.model_validate_json(
        response.message.content
    )


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_prompt = """
Known locations:
L1 = Feni [district]
L2 = Sylhet [district]

Candidate quantities:
Q1 = "31" in S2
Q2 = "58 lakh" in S3
Q3 = "10,000" in S4

Sentences:
S1: Floods have been devastating several districts of Bangladesh.
S2: The nationwide death toll has risen to 31.
S3: At least 58 lakh people have been affected across 11 districts.
S4: Around 10,000 families remained marooned in Feni district.

Classify each candidate quantity as a disaster impact claim.
Use only the IDs provided above.
Do not invent quantities, locations or evidence.
"""

    print("Sending to local LLM...")
    result = run_branch_b(test_prompt)
    
    print(f"\nExtracted {len(result.claims)} claims:\n")
    for claim in result.claims:
        print(f"  Candidate : {claim.candidate_id}")
        print(f"  Type      : {claim.impact_type}")
        print(f"  Evidence  : {claim.evidence_sentence_ids}")
        print(f"  Locations : {claim.location_ids}")
        print(f"  Aggregation: {claim.aggregation}")
        print(f"  Certainty : {claim.certainty}")
        print(f"  Unit      : {claim.unit}")
        print()