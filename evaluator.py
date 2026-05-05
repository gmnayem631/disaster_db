import json
import os
import time
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from datetime import datetime

# Paths
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "xlmroberta_final")
DEV_DATA_PATH  = os.path.join(BASE_DIR, "data", "annotations", "dev.spacy")
OUTPUT_FILE    = os.path.join(BASE_DIR, "data", "evaluation_results.json")

# Load model
print("Loading model...")
nlp = spacy.load(MODEL_PATH)
print("Model loaded.")

# Load dev data
def load_dev_data():
    doc_bin = DocBin().from_disk(DEV_DATA_PATH)
    docs    = list(doc_bin.get_docs(nlp.vocab))
    return docs

# Entity level evaluation
def evaluate_entities(dev_docs):
    """Calculate precision, recall, F1 per entity type."""
    
    entity_stats = {}
    total_time   = 0
    
    for doc in dev_docs:
        # Measure processing time
        start     = time.time()
        pred_doc  = nlp(doc.text)
        end       = time.time()
        total_time += (end - start)
        
        # Gold entities
        gold_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}
        
        # Predicted entities
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents}
        
        # Get all labels
        all_labels = set([e[2] for e in gold_ents] + [e[2] for e in pred_ents])
        
        for label in all_labels:
            if label not in entity_stats:
                entity_stats[label] = {"tp": 0, "fp": 0, "fn": 0}
            
            gold_label = {e for e in gold_ents if e[2] == label}
            pred_label = {e for e in pred_ents if e[2] == label}
            
            tp = len(gold_label & pred_label)
            fp = len(pred_label - gold_label)
            fn = len(gold_label - pred_label)
            
            entity_stats[label]["tp"] += tp
            entity_stats[label]["fp"] += fp
            entity_stats[label]["fn"] += fn
    
    # Calculate metrics
    results     = {}
    total_tp    = 0
    total_fp    = 0
    total_fn    = 0

    for label, stats in entity_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[label] = {
            "precision": round(precision * 100, 2),
            "recall":    round(recall * 100, 2),
            "f1":        round(f1 * 100, 2),
            "support":   tp + fn
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1        = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    avg_time = total_time / len(dev_docs) if dev_docs else 0

    return results, overall_precision, overall_recall, overall_f1, avg_time

# Rule-based contribution
def evaluate_rule_based_contribution(dev_docs):
    """Measure how many entities came from model vs rule-based."""
    from entity_linker import extract_entities

    model_count      = 0
    rule_based_count = 0

    for doc in dev_docs[:20]:  # Sample 20 articles
        result = extract_entities(doc.text, "eval_test")

        # Check numerical fields
        for field in ["fatalities", "displaced", "affected_people"]:
            val = result.get(field)
            if val:
                if val.get("source") == "model":
                    model_count += 1
                elif val.get("source") == "rule_based":
                    rule_based_count += 1

    return model_count, rule_based_count

# Print results
def print_results(results, overall_p, overall_r, overall_f1, avg_time, model_count, rule_count):
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS — XLM-RoBERTa NER Model")
    print(f"Evaluated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print(f"\n{'Entity Type':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)

    for label, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{label:<25} {metrics['precision']:>9.2f}% {metrics['recall']:>9.2f}% {metrics['f1']:>9.2f}% {metrics['support']:>9}")

    print("-" * 65)
    print(f"{'OVERALL':<25} {overall_p*100:>9.2f}% {overall_r*100:>9.2f}% {overall_f1*100:>9.2f}%")

    print(f"\nAverage processing time per article : {avg_time:.3f} seconds")
    print(f"Model extracted entities            : {model_count}")
    print(f"Rule-based extracted entities       : {rule_count}")
    print(f"Rule-based contribution             : {rule_count/(model_count+rule_count)*100:.1f}%" if (model_count+rule_count) > 0 else "N/A")
    print("=" * 65)

# Save results
def save_results(results, overall_p, overall_r, overall_f1, avg_time, model_count, rule_count):
    output = {
        "model":                  "XLM-RoBERTa",
        "evaluation_date":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dev_set_size":           56,
        "overall_precision":      round(overall_p * 100, 2),
        "overall_recall":         round(overall_r * 100, 2),
        "overall_f1":             round(overall_f1 * 100, 2),
        "avg_processing_time_sec": round(avg_time, 3),
        "model_extracted":        model_count,
        "rule_based_extracted":   rule_count,
        "per_entity_results":     results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_FILE}")

# Main
if __name__ == "__main__":
    print("Loading dev data...")
    dev_docs = load_dev_data()
    print(f"Loaded {len(dev_docs)} dev articles.")

    print("Running entity evaluation...")
    results, overall_p, overall_r, overall_f1, avg_time = evaluate_entities(dev_docs)

    print("Measuring rule-based contribution...")
    model_count, rule_count = evaluate_rule_based_contribution(dev_docs)

    print_results(results, overall_p, overall_r, overall_f1, avg_time, model_count, rule_count)
    save_results(results, overall_p, overall_r, overall_f1, avg_time, model_count, rule_count)