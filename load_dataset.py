
import fsspec
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO

url = (
    "https://huggingface.co/datasets/"
    "tellow-it-1/wikiart-10k/resolve/main/"
    "data/train-00000-of-00010.parquet"
)

print("Connecting to WikiArt...", flush=True)

try:
    with fsspec.open(url, "rb", block_size=65536) as f:
        parquet = pq.ParquetFile(f)

        print("Available Arrow columns:",
              parquet.schema_arrow.names, flush=True)

        print("Reading first image...", flush=True)

        # Read the nested image column and its metadata
        table = parquet.read_row_group(
            0,
            columns=["image", "artist", "genre", "style"]
        ).slice(0, 1)

        row = table.to_pylist()[0]

        print("Metadata:", {
            k: v for k, v in row.items() if k != "image"
        }, flush=True)

        image_data = row["image"]

        if isinstance(image_data, dict):
            image_data = image_data.get("bytes")

        if not image_data:
            raise ValueError("Image bytes are empty.")

        image = Image.open(BytesIO(image_data)).convert("RGB")
        image.save("sample_image.png")

        print("SUCCESS! Saved sample_image.png", flush=True)
        image.show()

except Exception as e:
    print("ERROR:", repr(e), flush=True)
