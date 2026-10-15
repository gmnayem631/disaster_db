import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraction.branch_b.candidate_generator import generate_candidates, format_for_llm
from extraction.branch_b.local_llm import run_branch_b
from extraction.branch_b.claim_validator import validate_claims
from entity_linker import extract_entities

# ── Load a real cleaned article ────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "articles", "cleaned")

files = [f for f in os.listdir(CLEANED_DIR) if f.endswith('.json')]

# Find an article likely to have impact numbers
selected = None
for f in files:
    fp = os.path.join(CLEANED_DIR, f)
    with open(fp, "r", encoding="utf-8") as fh:
        a = json.load(fh)
    if any(kw in a.get("title", "").lower()
           for kw in ["death toll", "killed", "affected", "displaced"]):
        selected = fp
        break

filepath = selected if selected else os.path.join(CLEANED_DIR, files[0])


with open(filepath, "r", encoding="utf-8") as f:
    article = json.load(f)

print(f"Article: {article['title']}\n")
print("=" * 60)

# ── Branch A — stable entities ─────────────────────────────────────────────
print("\n[Branch A] Running XLM-RoBERTa + Gazetteer...")
branch_a = extract_entities(
    text        = article["text"],
    source_url  = article["source_url"],
    publish_date = article.get("publish_date")
)

branch_a_locations = []
loc_counter = 1
for d in branch_a["locations"]["districts"]:
    branch_a_locations.append({
        "location_id": f"L{loc_counter}",
        "name":        d["value"],
        "type":        "BD_DISTRICT"
    })
    loc_counter += 1
for u in branch_a["locations"]["upazilas"]:
    branch_a_locations.append({
        "location_id": f"L{loc_counter}",
        "name":        u["value"],
        "type":        "BD_UPAZILA"
    })
    loc_counter += 1

print(f"Locations found: {[l['name'] for l in branch_a_locations]}")
print(f"Disaster type  : {[d['value'] for d in branch_a['disaster_type']]}")

# ── Branch B — candidate generation ───────────────────────────────────────
print("\n[Branch B] Generating candidates...")
sentences, candidates = generate_candidates(article["text"])
print(f"Sentences : {len(sentences)}")
print(f"Candidates: {len(candidates)}")
for c in candidates:
    print(f"  {c.candidate_id} ({c.kind}): \"{c.raw}\" — {c.sentence_id}")

# ── Branch B — LLM interpretation ─────────────────────────────────────────
print("\n[Branch B] Sending to Qwen3 8B...")
prompt = format_for_llm(sentences, candidates, branch_a_locations)
result = run_branch_b(prompt)
print(f"Claims returned: {len(result.claims)}")

# ── Validation ─────────────────────────────────────────────────────────────
print("\n[Validator] Running deterministic checks...")
validated = validate_claims(result, candidates, sentences, branch_a_locations)

print(f"\nFinal validated claims:")
print("-" * 70)
for v in validated:
    norm = v["normalized_quantity"]
    val  = str(norm["value"]) if norm and norm["value"] is not None else "N/A"
    print(f"  {v['candidate_id']:<5} | {v['impact_type']:<20} | "
          f"value={val:<12} | "
          f"{v['validation_status']:<15} | "
          f"flags={v['validation_flags']}")
