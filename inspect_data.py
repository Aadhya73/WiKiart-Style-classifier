
from pathlib import Path

DATASET_DIR = Path("dataset")

print("Dataset folder:", DATASET_DIR.resolve())

if not DATASET_DIR.exists():
    raise FileNotFoundError("The dataset folder was not found.")

files = [p for p in DATASET_DIR.iterdir() if p.is_file()]

print("\nTotal files:", len(files))

for file in files[:15]:
    print(f"{file.name} | {file.stat().st_size / (1024**2):.2f} MB")

if files:
    print("\nFile extensions:")
    extensions = {}
    for file in files:
        ext = file.suffix.lower() or "(no extension)"
        extensions[ext] = extensions.get(ext, 0) + 1

    for ext, count in extensions.items():
        print(f"{ext}: {count}")
