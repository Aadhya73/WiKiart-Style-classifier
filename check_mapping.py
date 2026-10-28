
import json
from pathlib import Path

folder = Path("prepared_data")
mapping_file = folder / "class_mapping.json"

if not mapping_file.exists():
    print("Mapping file is missing. Run prepare_data.py first.")
    raise SystemExit(1)

with mapping_file.open(encoding="utf-8") as f:
    mapping = json.load(f)

print("Entries in saved mapping:", len(mapping))
print("\nStyle ID -> Style name")

for style_id, name in mapping.items():
    print(f"{style_id:>3} -> {name}")
