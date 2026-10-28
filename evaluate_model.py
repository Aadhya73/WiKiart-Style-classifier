from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow.keras.applications.resnet50 import preprocess_input

# Evaluates the existing checkpoint; does not train or modify model weights.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed_data"
METADATA_FILE = DATA_DIR / "metadata.csv"
MODEL_PATH = PROJECT_DIR / "models" / "best_wikiart_model.keras"
OUTPUT_DIR = PROJECT_DIR / "models"
IMAGE_SIZE = 224
BATCH_SIZE = 8


def resolve_image_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else DATA_DIR / path


def load_test_metadata():
    if not METADATA_FILE.is_file():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    df = pd.read_csv(METADATA_FILE)
    required = {"image_path", "split", "model_label", "style_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")

    df["model_label"] = pd.to_numeric(df["model_label"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["style_name"] = df["style_name"].astype(str).str.strip()

    labels = sorted(df["model_label"].unique().tolist())
    if labels != list(range(len(labels))):
        raise ValueError("model_label values must be contiguous from 0.")

    class_names = []
    for label in labels:
        names = df.loc[df["model_label"] == label, "style_name"].unique()
        if len(names) != 1:
            raise ValueError(f"Label {label} has inconsistent style names.")
        class_names.append(str(names[0]))

    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    if test_df.empty:
        raise ValueError("No rows found with split='test' in metadata.csv.")

    test_df["full_path"] = test_df["image_path"].apply(resolve_image_path)
    missing_paths = test_df.loc[~test_df["full_path"].map(Path.is_file), "full_path"]
    if not missing_paths.empty:
        raise FileNotFoundError(
            "Some test images are missing:\n" +
            "\n".join(str(p) for p in missing_paths.head(10))
        )
    return test_df, class_names


def make_test_dataset(frame):
    paths = frame["full_path"].astype(str).to_numpy()
    labels = frame["model_label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            method="bilinear",
            antialias=False,
        )
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.int32)

    return (
        ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    test_df, class_names = load_test_metadata()
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
    )

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} classes but metadata has "
            f"{len(class_names)} classes."
        )

    test_ds = make_test_dataset(test_df)
    y_true_batches, y_pred_batches = [], []

    for images, labels in test_ds:
        probs = np.asarray(model.predict(images, verbose=0))
        if probs.ndim != 2 or probs.shape[1] != len(class_names):
            raise ValueError("Unexpected model output shape.")
        if not np.isfinite(probs).all():
            raise ValueError("Model produced NaN or infinite predictions.")
        y_true_batches.extend(labels.numpy().tolist())
        y_pred_batches.extend(np.argmax(probs, axis=1).tolist())

    y_true = np.asarray(y_true_batches, dtype=np.int32)
    y_pred = np.asarray(y_pred_batches, dtype=np.int32)

    if len(y_true) != len(test_df) or len(y_pred) != len(test_df):
        raise RuntimeError("Evaluated sample count does not match test metadata.")

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(
        y_true, y_pred, average="macro", labels=list(range(len(class_names))),
        zero_division=0
    ))
    weighted_f1 = float(f1_score(
        y_true, y_pred, average="weighted", labels=list(range(len(class_names))),
        zero_division=0
    ))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv", index=True
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )
    pd.DataFrame(
        matrix, index=class_names, columns=class_names
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=True)

    summary = {
        "checkpoint": str(MODEL_PATH.relative_to(PROJECT_DIR)),
        "number_of_classes": len(class_names),
        "test_images": int(len(y_true)),
        "test_accuracy": accuracy,
        "test_accuracy_percent": accuracy * 100.0,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "evaluation_split": "test",
        "note": "Metrics computed from the checkpoint and test split above."
    }
    with (OUTPUT_DIR / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nFINAL TEST EVALUATION")
    print("---------------------")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Test images: {len(y_true)}")
    print(f"Classes: {len(class_names)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nSaved:")
    print(OUTPUT_DIR / "evaluation_summary.json")
    print(OUTPUT_DIR / "classification_report.csv")
    print(OUTPUT_DIR / "confusion_matrix.csv")


if __name__ == "__main__":
    main()
