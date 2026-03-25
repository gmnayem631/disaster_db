import csv
import json
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
GAZETTEER_DIR = os.path.join(BASE_DIR, "data", "gazetteer")
OUTPUT_FILE   = os.path.join(GAZETTEER_DIR, "bangladesh_gazetteer.json")

FILES = {
    "district": os.path.join(GAZETTEER_DIR, "districts.csv"),
    "upazila":  os.path.join(GAZETTEER_DIR, "upazilas.csv"),
    "union":    os.path.join(GAZETTEER_DIR, "unions.csv"),
}


# ── Builder ────────────────────────────────────────────────────────────────
def build_gazetteer():
    gazetteer = {
        "districts": [],
        "upazilas":  [],
        "unions":    []
    }

    for level, filepath in FILES.items():
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                name = row[2].strip()
                if name:
                    gazetteer[level + "s"].append(name)

    # Remove duplicates while preserving order
    for key in gazetteer:
        gazetteer[key] = list(dict.fromkeys(gazetteer[key]))

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(gazetteer, f, ensure_ascii=False, indent=2)

    print(f"Districts : {len(gazetteer['districts'])}")
    print(f"Upazilas  : {len(gazetteer['upazilas'])}")
    print(f"Unions    : {len(gazetteer['unions'])}")
    print(f"Saved to  : {OUTPUT_FILE}")


if __name__ == "__main__":
    build_gazetteer()