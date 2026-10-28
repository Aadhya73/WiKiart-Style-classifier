
from pathlib import Path
import json

import pandas as pd
from PIL import Image


# ==================================================
# VERIFY PROCESSED WIKIART DATASET
# ==================================================

ROOT = Path("processed_data")
MAPPING_FILE = Path("prepared_data/class_mapping.json")
METADATA_FILE = ROOT / "metadata.csv"

if not MAPPING_FILE.exists():
    raise FileNotFoundError(f"Missing: {MAPPING_FILE}")

if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Missing: {METADATA_FILE}")

with MAPPING_FILE.open(encoding="utf-8") as file:
    class_mapping = json.load(file)

metadata = pd.read_csv(METADATA_FILE)

required_columns = {
    "dataset_index",
    "image_path",
    "split",
    "original_style_id",
    "model_label",
    "style_name",
    "width",
    "height",
}

missing_columns = required_columns - set(metadata.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("=" * 55)
print("PROCESSED DATASET VERIFICATION")
print("=" * 55)

print("Metadata records:", len(metadata))
print("Observed classes:", len(class_mapping))

# Verify split sizes.
print("\nSplit sizes:")
print(metadata["split"].value_counts().to_string())

expected_splits = {"train", "validation", "test"}

if set(metadata["split"].unique()) != expected_splits:
    raise ValueError("Unexpected split names found.")

# Verify labels against the saved class mapping.
allowed_labels = {int(key) for key in class_mapping}

actual_labels = set(
    metadata["model_label"].astype(int).unique()
)

if actual_labels != allowed_labels:
    raise ValueError(
        "Labels in metadata do not match class_mapping.json."
    )

for model_label, details in class_mapping.items():
    label_rows = metadata[
        metadata["model_label"] == int(model_label)
    ]

    if not label_rows.empty:
        if not (
            label_rows["original_style_id"]
            == details["original_id"]
        ).all():
            raise ValueError(
                f"Original ID mismatch for model label {model_label}"
            )

        if not (
            label_rows["style_name"]
            == details["style_name"]
        ).all():
            raise ValueError(
                f"Style name mismatch for model label {model_label}"
            )

print("\nClass mapping: valid")

# Check duplicate row indices and paths.
if metadata["dataset_index"].duplicated().any():
    raise ValueError("Duplicate dataset indices found.")

if metadata["image_path"].duplicated().any():
    raise ValueError("Duplicate image paths found.")

# Check a sample of images from each split.
print("\nChecking sample images:")

for split_name in ["train", "validation", "test"]:
    split_rows = metadata[metadata["split"] == split_name]

    if split_rows.empty:
        raise ValueError(f"No images found for {split_name}")

    for _, row in split_rows.head(3).iterrows():
        image_path = ROOT / row["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            if image.size != (224, 224):
                raise ValueError(
                    f"Unexpected size {image.size}: {image_path}"
                )

            if image.mode != "RGB":
                raise ValueError(
                    f"Unexpected mode {image.mode}: {image_path}"
                )

        print(
            f"{split_name}: {image_path.name} "
            f"| label={row['model_label']} "
            f"| style={row['style_name']}"
        )

# Verify that row indices do not overlap across splits.
indices_by_split = {
    split_name: set(
        metadata.loc[
            metadata["split"] == split_name,
            "dataset_index"
        ].astype(int)
    )
    for split_name in expected_splits
}

if indices_by_split["train"] & indices_by_split["validation"]:
    raise ValueError("Train/validation overlap found.")

if indices_by_split["train"] & indices_by_split["test"]:
    raise ValueError("Train/test overlap found.")

if indices_by_split["validation"] & indices_by_split["test"]:
    raise ValueError("Validation/test overlap found.")

print("\nNo overlapping dataset indices.")
print("Sample images have the expected dimensions and RGB mode.")
print("\nVERIFICATION SUCCESSFUL.")
