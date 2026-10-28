
from pathlib import Path
from io import BytesIO
from collections import Counter
import csv
import json

import pyarrow.parquet as pq
from PIL import Image


# ============================================================
# WIKIART-10K: RESIZE AND ORGANIZE DATASET
# ============================================================

DATASET_DIR = Path("dataset")
PREPARED_DIR = Path("prepared_data")
OUTPUT_DIR = Path("processed_data")

IMAGE_SIZE = (224, 224)
JPEG_QUALITY = 90

PARQUET_BATCH_SIZE = 16

SPLITS = ("train", "validation", "test")


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

parquet_files = sorted(DATASET_DIR.glob("*.parquet"))

required_files = [
    PREPARED_DIR / "class_mapping.json",
    PREPARED_DIR / "split_indices.json",
    PREPARED_DIR / "valid_indices.json",
]

if not parquet_files:
    raise FileNotFoundError(
        "No Parquet files found in dataset/."
    )

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

print("=" * 65)
print("WIKIART-10K IMAGE PREPARATION")
print("=" * 65)


# ============================================================
# 2. LOAD CLASS MAPPING AND SPLITS
# ============================================================

with (PREPARED_DIR / "class_mapping.json").open(
    encoding="utf-8"
) as file:
    class_mapping = json.load(file)

with (PREPARED_DIR / "split_indices.json").open(
    encoding="utf-8"
) as file:
    split_data = json.load(file)

with (PREPARED_DIR / "valid_indices.json").open(
    encoding="utf-8"
) as file:
    valid_indices = set(json.load(file))


# Original style ID -> contiguous model label
original_to_model = {}

for model_label, details in class_mapping.items():
    original_id = int(details["original_id"])
    original_to_model[original_id] = int(model_label)


# Dataset row index -> split name
split_for_index = {}

for split_name in SPLITS:
    for index in split_data[split_name]:
        index = int(index)

        if index in split_for_index:
            raise ValueError(
                f"Index {index} appears in multiple splits."
            )

        split_for_index[index] = split_name


# ============================================================
# 3. CREATE OUTPUT FOLDERS
# ============================================================

for split_name in SPLITS:
    for details in class_mapping.values():
        style_name = details["style_name"]

        (OUTPUT_DIR / split_name / style_name).mkdir(
            parents=True,
            exist_ok=True
        )

print("Output directory:", OUTPUT_DIR.resolve())
print("Image dimensions:", IMAGE_SIZE)
print("Observed classes:", len(class_mapping))
print("Valid source images:", len(valid_indices))


# ============================================================
# 4. PROCESS EACH IMAGE
# ============================================================

global_index = 0
processed_count = 0
skipped_count = 0

split_counts = Counter()
class_counts = Counter()
metadata_path = OUTPUT_DIR / "metadata.csv"

with metadata_path.open(
    "w", newline="", encoding="utf-8"
) as metadata_file:

    writer = csv.DictWriter(
        metadata_file,
        fieldnames=[
            "dataset_index",
            "image_path",
            "split",
            "original_style_id",
            "model_label",
            "style_name",
            "width",
            "height"
        ]
    )

    writer.writeheader()

    for file_number, parquet_path in enumerate(
        parquet_files, start=1
    ):

        print(
            f"\nReading file {file_number}/{len(parquet_files)}: "
            f"{parquet_path.name}"
        )

        parquet_file = pq.ParquetFile(parquet_path)

        for batch in parquet_file.iter_batches(
            batch_size=PARQUET_BATCH_SIZE,
            columns=["image", "style"]
        ):
            image_values = batch.column("image").to_pylist()
            style_values = batch.column("style").to_pylist()

            for image_value, original_style_id in zip(
                image_values, style_values
            ):
                current_index = global_index
                global_index += 1

                # Exclude images that failed the earlier validation.
                if current_index not in valid_indices:
                    skipped_count += 1
                    continue

                if original_style_id is None:
                    raise ValueError(
                        f"Missing style at row {current_index}"
                    )

                original_style_id = int(original_style_id)

                if original_style_id not in original_to_model:
                    raise ValueError(
                        f"Unmapped style ID {original_style_id} "
                        f"at row {current_index}"
                    )

                if current_index not in split_for_index:
                    raise ValueError(
                        f"No split assigned to row {current_index}"
                    )

                split_name = split_for_index[current_index]
                model_label = original_to_model[original_style_id]

                style_name = class_mapping[
                    str(model_label)
                ]["style_name"]

                image_bytes = (
                    image_value.get("bytes")
                    if image_value
                    else None
                )

                if not image_bytes:
                    raise ValueError(
                        f"Missing image bytes at row {current_index}"
                    )

                # Decode, convert to RGB and resize.
                with Image.open(BytesIO(image_bytes)) as source:
                    image = source.convert("RGB")
                    image = image.resize(
                        IMAGE_SIZE,
                        Image.Resampling.LANCZOS
                    )

                relative_path = (
                    Path(split_name)
                    / style_name
                    / f"image_{current_index:05d}.jpg"
                )

                output_path = OUTPUT_DIR / relative_path

                image.save(
                    output_path,
                    format="JPEG",
                    quality=JPEG_QUALITY,
                    optimize=True
                )

                writer.writerow({
                    "dataset_index": current_index,
                    "image_path": relative_path.as_posix(),
                    "split": split_name,
                    "original_style_id": original_style_id,
                    "model_label": model_label,
                    "style_name": style_name,
                    "width": IMAGE_SIZE[0],
                    "height": IMAGE_SIZE[1]
                })

                processed_count += 1
                split_counts[split_name] += 1
                class_counts[(split_name, style_name)] += 1

                if processed_count % 100 == 0:
                    print(
                        f"Processed {processed_count:,} images",
                        end="\r"
                    )


# ============================================================
# 5. SAVE PROCESSING CONFIGURATION
# ============================================================

configuration = {
    "dataset": "WikiArt-10K",
    "source_records": global_index,
    "processed_images": processed_count,
    "skipped_invalid_images": skipped_count,
    "image_width": IMAGE_SIZE[0],
    "image_height": IMAGE_SIZE[1],
    "color_mode": "RGB",
    "format": "JPEG",
    "jpeg_quality": JPEG_QUALITY,
    "target_column": "style",
    "number_of_observed_classes": len(class_mapping),
    "normalization": (
        "Not applied to saved JPEGs; pixel scaling "
        "will be applied in the model input pipeline."
    )
}

with (OUTPUT_DIR / "preprocessing_config.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(configuration, file, indent=2)


# ============================================================
# 6. PRINT SUMMARY
# ============================================================

print("\n\n" + "=" * 65)
print("IMAGE PREPARATION SUMMARY")
print("=" * 65)

print(f"Source records:       {global_index:,}")
print(f"Processed images:     {processed_count:,}")
print(f"Skipped invalid:      {skipped_count:,}")

for split_name in SPLITS:
    print(
        f"{split_name.capitalize():<12}: "
        f"{split_counts[split_name]:,}"
    )

print("\nImages per class and split:")

for style_name in sorted(
    details["style_name"] for details in class_mapping.values()
):
    print(f"\n{style_name}")

    for split_name in SPLITS:
        print(
            f"  {split_name:<12}: "
            f"{class_counts[(split_name, style_name)]:,}"
        )

print("\nSaved metadata:", metadata_path)
print("Saved configuration:", OUTPUT_DIR / "preprocessing_config.json")

print("\nOriginal Parquet files remain unchanged.")
print("IMAGE PREPARATION COMPLETED.")
