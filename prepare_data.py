
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


# ============================================================
# WIKIART-10K: COMPLETE DATA PREPARATION
# ============================================================

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("prepared_data")
RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(DATASET_DIR.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"No Parquet files found in {DATASET_DIR.resolve()}"
    )

print("=" * 65)
print("WIKIART-10K DATA PREPARATION")
print("=" * 65)


# ============================================================
# 1. READ LABELS AND COUNT RECORDS
# ============================================================

all_original_labels = []
total_records = 0

for parquet_path in parquet_files:
    parquet_file = pq.ParquetFile(parquet_path)
    table = parquet_file.read(columns=["style"])

    labels = table.column("style").to_pylist()

    if any(label is None for label in labels):
        raise ValueError(
            f"Missing style labels found in {parquet_path.name}"
        )

    all_original_labels.extend(labels)
    total_records += len(labels)

    print(
        f"Read {parquet_path.name}: "
        f"{len(labels):,} records"
    )

original_labels = np.asarray(
    all_original_labels, dtype=np.int64
)

print(f"\nTotal records: {total_records:,}")


# ============================================================
# 2. READ ORIGINAL STYLE NAMES FROM DATASET METADATA
# ============================================================

schema = pq.read_schema(parquet_files[0])
schema_metadata = schema.metadata or {}

huggingface_metadata = schema_metadata.get(b"huggingface")

if huggingface_metadata is None:
    raise ValueError(
        "Hugging Face metadata was not found in the Parquet files."
    )

metadata = json.loads(
    huggingface_metadata.decode("utf-8")
)

features = metadata.get("info", {}).get("features", {})
style_feature = features.get("style", {})

if style_feature.get("_type") != "ClassLabel":
    raise ValueError(
        "The style column is not stored as a ClassLabel feature. "
        "Inspect the dataset metadata before proceeding."
    )

original_style_names = style_feature.get("names")

if not original_style_names:
    raise ValueError("No original style names were found.")

print("Original style vocabulary:", len(original_style_names))


# ============================================================
# 3. CREATE CONTIGUOUS MODEL LABELS
# ============================================================

observed_original_ids = sorted(
    set(int(label) for label in original_labels)
)

for original_id in observed_original_ids:
    if not 0 <= original_id < len(original_style_names):
        raise ValueError(
            f"Invalid original style ID: {original_id}"
        )

original_to_model = {
    original_id: model_id
    for model_id, original_id in enumerate(
        observed_original_ids
    )
}

model_to_original = {
    model_id: original_id
    for original_id, model_id in original_to_model.items()
}

model_labels = np.asarray(
    [original_to_model[int(label)] for label in original_labels],
    dtype=np.int64
)

number_of_classes = len(observed_original_ids)

print("Observed style classes:", number_of_classes)


# ============================================================
# 4. COUNT IMAGES PER STYLE
# ============================================================

original_counts = Counter(
    int(label) for label in original_labels
)

print("\nIMAGES PER STYLE")
print("-" * 65)

for original_id in observed_original_ids:
    style_name = original_style_names[original_id]
    count = original_counts[original_id]
    model_id = original_to_model[original_id]

    print(
        f"Model ID {model_id:>2} | "
        f"Original ID {original_id:>2} | "
        f"{style_name:<35} | {count:>5}"
    )

print("-" * 65)


# ============================================================
# 5. SAVE CLASS MAPPING
# ============================================================

class_mapping = {
    str(model_id): {
        "original_id": original_id,
        "style_name": original_style_names[original_id]
    }
    for model_id, original_id in model_to_original.items()
}

mapping_path = OUTPUT_DIR / "class_mapping.json"

with mapping_path.open("w", encoding="utf-8") as file:
    json.dump(
        class_mapping,
        file,
        indent=2,
        ensure_ascii=False
    )

print(f"\nSaved class mapping: {mapping_path}")


# ============================================================
# 6. CREATE TRAIN / VALIDATION / TEST SPLITS
# ============================================================

indices = np.arange(total_records)

# First, separate 70% training from 30% remaining.
train_indices, remaining_indices = train_test_split(
    indices,
    test_size=VALIDATION_RATIO + TEST_RATIO,
    random_state=RANDOM_STATE,
    stratify=model_labels
)

remaining_labels = model_labels[remaining_indices]

# Split the remaining 30% equally into validation and test.
validation_indices, test_indices = train_test_split(
    remaining_indices,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=remaining_labels
)

# Check that all classes appear in all three splits.
for split_name, split_indices in [
    ("training", train_indices),
    ("validation", validation_indices),
    ("testing", test_indices)
]:
    split_classes = set(
        model_labels[split_indices].tolist()
    )

    missing_classes = set(range(number_of_classes)) - split_classes

    if missing_classes:
        raise ValueError(
            f"Some classes are missing from the {split_name} split: "
            f"{sorted(missing_classes)}"
        )


# ============================================================
# 7. SAVE SPLIT INDICES AND CONFIGURATION
# ============================================================

split_data = {
    "dataset": "wikiart-10k",
    "target_column": "style",
    "random_state": RANDOM_STATE,
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VALIDATION_RATIO,
    "test_ratio": TEST_RATIO,
    "total_records": total_records,
    "number_of_classes": number_of_classes,
    "train": train_indices.tolist(),
    "validation": validation_indices.tolist(),
    "test": test_indices.tolist()
}

split_path = OUTPUT_DIR / "split_indices.json"

with split_path.open("w", encoding="utf-8") as file:
    json.dump(split_data, file)

print(f"Saved split indices: {split_path}")


# ============================================================
# 8. DISPLAY SPLIT SUMMARY
# ============================================================

print("\nDATASET SPLIT SUMMARY")
print("-" * 40)
print(f"Total:      {total_records:>6,}")
print(f"Training:   {len(train_indices):>6,}")
print(f"Validation: {len(validation_indices):>6,}")
print(f"Testing:    {len(test_indices):>6,}")
print(f"Classes:    {number_of_classes:>6}")

print("\nClass imbalance check:")
print(
    "Largest class:",
    max(original_counts.values()),
    "images"
)
print(
    "Smallest class:",
    min(original_counts.values()),
    "images"
)

print("\nIMPORTANT:")
print("- Original style IDs are preserved in class_mapping.json.")
print("- Model labels are remapped to consecutive IDs starting at 0.")
print("- Split indices refer to row positions across sorted Parquet files.")
print("- Original Parquet files have not been modified.")
print("- No image copies were created.")

print("\nDATA PREPARATION COMPLETED SUCCESSFULLY.")
