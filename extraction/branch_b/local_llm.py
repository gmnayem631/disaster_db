import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import ollama
from extraction.branch_b.schemas import BranchBResult

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

SYSTEM_PROMPT = """You are a disaster information extraction engine for Bangladeshi news
about floods, cyclones, fires, landslides, road and boat accidents, building
collapses, earthquakes, lightning, disease outbreaks and other disasters.
For each candidate number or cue, decide what it describes.
Use only what the supplied sentences say. Never invent quantities, locations or evidence.

Impact types:
- FATALITIES: people killed or dead ("death toll rose to 31", "3 bodies recovered")
- INJURED: people physically injured or hurt (wounds, burns, fractures). NOT people who are ill.
- MISSING: people missing or unaccounted for
- AFFECTED_PEOPLE: people or families affected, hit, in distress. For disease outbreaks:
  cases, infected people and patients admitted to hospital are AFFECTED_PEOPLE, never INJURED
- DISPLACED: people or families displaced, marooned, stranded, waterlogged in their homes,
  evacuated, moved to shelters, homeless
- HOUSING_DAMAGE: houses, homes, shops or other buildings damaged, destroyed, gutted, washed away
- INFRASTRUCTURE_DAMAGE: roads, bridges, embankments, culverts, power lines, schools damaged
- AGRICULTURAL_DAMAGE: crops, cropland, fish ponds, hectares or tonnes of produce damaged or lost
- LIVESTOCK_LOSS: cattle, goats, poultry or other animals killed or lost
- ECONOMIC_LOSS: money value of losses or damage ("goods worth Tk 50 lakh were burnt")
- RELIEF: aid given or allocated: money, rice, food packets, shelters opened, relief teams
- OTHER: anything that is not a disaster impact. Always OTHER: dates, years, times, durations,
  ages, distances, wind speeds, rainfall, water levels, signal numbers, phone numbers,
  numbers of fire service units or firefighters, and counts of districts or villages.

Aggregation:
- CUMULATIVE: a running total ("the death toll rose TO 31", "so far 136 have died")
- INCREMENTAL: a new addition ("THREE MORE died", "1,537 new patients")
- POINT_ESTIMATE: a single reported figure that is neither of the above
- RANGE: "10 to 15", "between 5 and 7"
- LOWER_BOUND: "at least", "more than", "over"
- UPPER_BOUND: "up to", "at most", "less than"
- UNKNOWN: cannot be decided

Certainty: CONFIRMED (officially confirmed), REPORTED (stated as fact), ESTIMATED
("about", "nearly", "estimated"), EXPECTED or PREDICTED (forecasts), POSSIBLE ("may", "feared").

Unit: what is counted, in one plain word: people, families, houses, shops, roads, bridges,
hectares, tonnes, animals, taka, packets. Use null if nothing is counted.

Answer every candidate. For an OTHER candidate give ONLY its id and type, nothing else:
{"candidate_id": "Q3", "impact_type": "OTHER"}
For every other impact type give all fields: candidate_id, impact_type,
evidence_sentence_ids, location_ids, aggregation, certainty, unit.

Return only valid JSON matching the required schema."""

def run_branch_b(prompt: str, model: str = "qwen3:8b") -> BranchBResult:
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
        options = {
            "temperature": 0,
            "num_predict": 4096,
            "num_gpu": 0
        },
        think   = False
    )

    content = response.message.content
    if not content or not content.strip():
        raise ValueError("Empty response from model")

    return BranchBResult.model_validate_json(
        response.message.content
    )


def run_branch_b_batched(
    sentences,
    candidates,
    branch_a_locations,
    batch_size: int = 6,
    model: str = "qwen3:8b"
) -> list[dict]:
    """Process candidates in batches and collect all claims."""
    from extraction.branch_b.candidate_generator import format_for_llm
    from extraction.branch_b.schemas import BranchBResult

    all_claims = []

    # Split candidates into batches
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        print(f"  Processing batch {i//batch_size + 1} "
              f"({len(batch)} candidates)...")

        prompt = format_for_llm(sentences, batch, branch_a_locations)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = run_branch_b(prompt, model)
                all_claims.extend(result.claims)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Batch {i//batch_size + 1} attempt {attempt + 1} failed, retrying...")
                else:
                    raise RuntimeError(
                        f"Branch B batch {i//batch_size + 1} failed "
                        f"after {max_retries} attempts"
                    ) from e
                   

    # Wrap all claims in BranchBResult
    return BranchBResult(claims=all_claims)

# Quick test 
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