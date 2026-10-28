
from pathlib import Path
import pyarrow.parquet as pq

DATASET_DIR = Path("dataset")
parquet_files = sorted(DATASET_DIR.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError("No Parquet files found in dataset/")

print("Checking Parquet metadata...\n")

total_rows = 0

for file in parquet_files:
    parquet_file = pq.ParquetFile(file)
    rows = parquet_file.metadata.num_rows
    total_rows += rows

    print(f"{file.name}: {rows:,} rows")

print(f"\nTotal records: {total_rows:,}")

print("\nDataset schema:")
print(pq.read_schema(parquet_files[0]))

print("\nColumn names:")
print(pq.read_schema(parquet_files[0]).names)
