
from pathlib import Path
from io import BytesIO
import csv

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError


# ============================================================
# WIKIART-10K IMAGE VALIDATION
# ============================================================

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("prepared_data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "image_validation.csv"
VALID_PATH = OUTPUT_DIR / "valid_indices.json"
INVALID_PATH = OUTPUT_DIR / "invalid_indices.json"

parquet_files = sorted(DATASET_DIR.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"No Parquet files found in {DATASET_DIR.resolve()}"
    )

print("=" * 65)
print("WIKIART-10K IMAGE VALIDATION")
print("=" * 65)

total_images = 0
valid_images = 0
invalid_images = 0
global_index = 0

valid_indices = []
invalid_indices = []

mode_counts = {}
dimension_counts = {}

with REPORT_PATH.open(
    "w", newline="", encoding="utf-8"
) as report_file:

    writer = csv.DictWriter(
        report_file,
        fieldnames=[
            "dataset_index",
            "parquet_file",
            "style_id",
            "valid",
            "width",
            "height",
            "mode",
            "error"
        ]
    )

    writer.writeheader()

    for file_number, parquet_path in enumerate(parquet_files, start=1):

        print(
            f"\nProcessing file {file_number}/{len(parquet_files)}: "
            f"{parquet_path.name}"
        )

        parquet_file = pq.ParquetFile(parquet_path)

        # Read small batches so all image bytes are not held in RAM.
        for batch in parquet_file.iter_batches(
            batch_size=16,
            columns=["image", "style"]
        ):
            image_values = batch.column("image").to_pylist()
            style_values = batch.column("style").to_pylist()

            for image_value, style_id in zip(
                image_values, style_values
            ):
                current_index = global_index
                global_index += 1
                total_images += 1

                width = ""
                height = ""
                mode = ""
                error = ""

                try:
                    if style_id is None:
                        raise ValueError("Missing style label")

                    if not image_value:
                        raise ValueError("Missing image field")

                    image_bytes = image_value.get("bytes")

                    if not image_bytes:
                        raise ValueError("Missing image bytes")

                    # Verify that the stored bytes represent an image.
                    with Image.open(BytesIO(image_bytes)) as image:
                        width, height = image.size
                        mode = image.mode
                        image.verify()

                    if width <= 0 or height <= 0:
                        raise ValueError("Invalid image dimensions")

                    valid_images += 1
                    valid_indices.append(current_index)

                    mode_counts[mode] = (
                        mode_counts.get(mode, 0) + 1
                    )

                    dimensions = f"{width}x{height}"
                    dimension_counts[dimensions] = (
                        dimension_counts.get(dimensions, 0) + 1
                    )

                    is_valid = True

                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                    TypeError,
                    AttributeError
                ) as exc:
                    invalid_images += 1
                    invalid_indices.append(current_index)
                    error = str(exc)
                    is_valid = False

                writer.writerow({
                    "dataset_index": current_index,
                    "parquet_file": parquet_path.name,
                    "style_id": style_id,
                    "valid": is_valid,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "error": error
                })

                if total_images % 100 == 0:
                    print(
                        f"Checked {total_images:,} images | "
                        f"Valid: {valid_images:,} | "
                        f"Invalid: {invalid_images:,}",
                        end="\r"
                    )

        print(
            f"Finished {parquet_path.name}. "
            f"Images checked so far: {total_images:,}"
        )


# ============================================================
# SAVE VALID AND INVALID ROW INDICES
# ============================================================

import json

with VALID_PATH.open("w", encoding="utf-8") as file:
    json.dump(valid_indices, file)

with INVALID_PATH.open("w", encoding="utf-8") as file:
    json.dump(invalid_indices, file)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n\n" + "=" * 65)
print("IMAGE VALIDATION SUMMARY")
print("=" * 65)

print(f"Total records checked: {total_images:,}")
print(f"Valid images:          {valid_images:,}")
print(f"Invalid images:        {invalid_images:,}")

if total_images:
    print(f"Valid percentage:      {valid_images / total_images * 100:.2f}%")

print("\nImage color modes:")
for mode, count in sorted(mode_counts.items()):
    print(f"{mode}: {count:,}")

print("\nMost common image dimensions:")
for dimensions, count in sorted(
    dimension_counts.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]:
    print(f"{dimensions}: {count:,}")

print("\nSaved reports:")
print(REPORT_PATH)
print(VALID_PATH)
print(INVALID_PATH)

print("\nImage validation completed.")
