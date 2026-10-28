from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# ============================================================
# WIKIART-10K: BALANCED-SAMPLING EXPERIMENT
# Keeps the original architecture and saves all outputs separately.
# Training samples classes with equal probability. Validation/test
# splits are unchanged. Class weights are intentionally not used too.
# ============================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
HEAD_EPOCHS = 25
FINE_TUNE_EPOCHS = 20
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_LAST_LAYERS = 30

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed_data"
METADATA_FILE = DATA_DIR / "metadata.csv"
OUTPUT_DIR = PROJECT_DIR / "models" / "balanced_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 68)
print("WIKIART-10K: BALANCED-SAMPLING EXPERIMENT")
print("=" * 68)
print("TensorFlow version:", tf.__version__)
print("Project directory:", PROJECT_DIR)
print("Experiment output directory:", OUTPUT_DIR)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# 1. LOAD AND VALIDATE METADATA
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

metadata = pd.read_csv(METADATA_FILE)
required_columns = {"image_path", "split", "model_label", "style_name"}
missing_columns = required_columns - set(metadata.columns)
if missing_columns:
    raise ValueError(f"Missing metadata columns: {sorted(missing_columns)}")

metadata["model_label"] = metadata["model_label"].astype(int)
metadata["split"] = metadata["split"].astype(str)
metadata["style_name"] = metadata["style_name"].astype(str)

def resolve_image_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else DATA_DIR / path

metadata["full_path"] = metadata["image_path"].apply(resolve_image_path)
exists_mask = metadata["full_path"].map(Path.is_file)
if not exists_mask.all():
    missing = metadata.loc[~exists_mask, "full_path"].head(10)
    raise FileNotFoundError(
        "Some image files could not be found. Examples:\n"
        + "\n".join(str(path) for path in missing)
    )

expected_splits = {"train", "validation", "test"}
actual_splits = set(metadata["split"].unique())
if actual_splits != expected_splits:
    raise ValueError(f"Expected split names {expected_splits}, found {actual_splits}.")

class_labels = sorted(metadata["model_label"].unique().tolist())
if class_labels != list(range(len(class_labels))):
    raise ValueError("Model labels must be contiguous from 0.")

num_classes = len(class_labels)
class_names = []
for label in class_labels:
    names = metadata.loc[metadata["model_label"] == label, "style_name"].unique()
    if len(names) != 1:
        raise ValueError(f"Label {label} has inconsistent style names: {names}")
    class_names.append(str(names[0]))

train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)
val_df = metadata[metadata["split"] == "validation"].reset_index(drop=True)
test_df = metadata[metadata["split"] == "test"].reset_index(drop=True)
if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError("One of the dataset splits is empty.")

print("\nDataset records:", len(metadata))
print("Number of style classes:", num_classes)
print("\nSplit counts:")
print(metadata["split"].value_counts().to_string())
print("\nTraining images per class:")
print(train_df["style_name"].value_counts().sort_values().to_string())

with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as file:
    json.dump({str(i): name for i, name in enumerate(class_names)}, file, indent=4)

# 2. INPUT PIPELINE
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.int32)

def make_regular_dataset(frame):
    """Normal non-balanced pipeline for validation and test."""
    paths = frame["full_path"].astype(str).to_numpy()
    labels = frame["model_label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    return ds.prefetch(AUTOTUNE)

def make_balanced_training_dataset(frame):
    """
    Sample equally from every class. Rare classes are oversampled and
    common classes are sampled less often. Repeated examples are not
    new data, so monitor validation performance for overfitting.
    """
    class_datasets = []
    class_sampling_weights = []

    for label in class_labels:
        class_frame = frame[frame["model_label"] == label]
        if class_frame.empty:
            raise ValueError(
                f"No training images for class {label} ({class_names[label]})."
            )

        paths = class_frame["full_path"].astype(str).to_numpy()
        labels = class_frame["model_label"].to_numpy(dtype=np.int32)
        class_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        class_ds = class_ds.shuffle(
            buffer_size=max(1, len(class_frame)),
            seed=SEED + int(label),
            reshuffle_each_iteration=True,
        )
        class_ds = class_ds.repeat()
        class_datasets.append(class_ds)
        class_sampling_weights.append(1.0 / num_classes)

    ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=class_sampling_weights,
        seed=SEED,
        stop_on_empty_dataset=False,
    )
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    return ds.prefetch(AUTOTUNE)

train_ds = make_balanced_training_dataset(train_df)
val_ds = make_regular_dataset(val_df)
test_ds = make_regular_dataset(test_df)

steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

print("\nInput pipelines created.")
print("Training images in source split:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))
print("Batch size:", BATCH_SIZE)
print("Balanced training steps per epoch:", steps_per_epoch)
print("Each of the classes has equal sampling probability.")
print("Class weights are disabled because balanced sampling is the correction.")

# 3. ATTENTION MODULE (PRESERVED)
def attention_module(features, reduction_ratio=16):
    channels = int(features.shape[-1])
    spatial_attention = layers.Conv2D(
        1, kernel_size=1, activation="sigmoid", padding="same",
        name="spatial_attention_map"
    )(features)
    spatial_features = layers.Multiply(name="spatial_attention_features")(
        [features, spatial_attention]
    )

    channel_weights = layers.GlobalAveragePooling2D(
        name="channel_global_average_pooling"
    )(features)
    channel_weights = layers.Dense(
        max(channels // reduction_ratio, 32), activation="relu",
        name="channel_attention_dense_1"
    )(channel_weights)
    channel_weights = layers.Dense(
        channels, activation="sigmoid", name="channel_attention_dense_2"
    )(channel_weights)
    channel_weights = layers.Reshape(
        (1, 1, channels), name="channel_attention_reshape"
    )(channel_weights)
    channel_features = layers.Multiply(name="channel_attention_features")(
        [features, channel_weights]
    )
    return layers.Add(name="combined_attention_features")(
        [spatial_features, channel_features]
    )

# 4. BUILD MODEL (PRESERVED)
def build_model(number_of_classes):
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="artwork_image")

    x = layers.RandomFlip("horizontal", seed=SEED, name="random_flip")(inputs)
    x = layers.RandomRotation(0.04, seed=SEED, name="random_rotation")(x)
    x = layers.RandomZoom(0.08, seed=SEED, name="random_zoom")(x)
    x = layers.RandomTranslation(
        0.04, 0.04, seed=SEED, name="random_translation"
    )(x)
    x = layers.Lambda(preprocess_input, name="resnet_preprocessing")(x)

    backbone = ResNet50(
        include_top=False, weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    backbone.trainable = False
    features = backbone(x, training=False)

    attended = attention_module(features)
    residual = layers.Conv2D(
        512, kernel_size=1, padding="same", use_bias=False, name="aar_conv_1"
    )(attended)
    residual = layers.BatchNormalization(name="aar_batch_norm_1")(residual)
    residual = layers.LeakyReLU(negative_slope=0.1, name="aar_leaky_relu")(residual)
    residual = layers.Conv2D(
        int(features.shape[-1]), kernel_size=1, padding="same",
        use_bias=False, name="aar_conv_2"
    )(residual)
    residual = layers.BatchNormalization(name="aar_batch_norm_2")(residual)

    features = layers.Add(name="aar_residual_add")([features, residual])
    features = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = layers.Dense(
        256, use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="classification_dense"
    )(features)
    features = layers.BatchNormalization(name="classification_batch_norm")(features)
    features = layers.LeakyReLU(
        negative_slope=0.1, name="classification_leaky_relu"
    )(features)
    features = layers.Dropout(0.4, name="classification_dropout")(features)
    outputs = layers.Dense(
        number_of_classes, activation="softmax", name="style_probabilities"
    )(features)

    model = Model(inputs=inputs, outputs=outputs, name="WikiArt_AAR_ResNet50_Balanced")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, backbone

model, backbone = build_model(num_classes)
print("\nModel architecture:")
model.summary()

# 5. CALLBACKS: separate folder means original checkpoint is untouched
best_model_path = OUTPUT_DIR / "best_wikiart_model_balanced.keras"
last_model_path = OUTPUT_DIR / "last_wikiart_model_balanced.keras"
history_path = OUTPUT_DIR / "training_history_balanced.csv"
fine_history_path = OUTPUT_DIR / "fine_tuning_history_balanced.csv"

def make_callbacks(csv_path, patience):
    return [
        ModelCheckpoint(
            filepath=str(best_model_path), monitor="val_loss",
            save_best_only=True, mode="min", verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, mode="min", verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-7, mode="min", verbose=1
        ),
        CSVLogger(filename=str(csv_path), append=False),
    ]

# 6. TRAIN CLASSIFICATION HEAD
print("\n" + "=" * 68)
print("PHASE 1: TRAINING CLASSIFICATION HEAD WITH BALANCED SAMPLING")
print("=" * 68)
model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=HEAD_EPOCHS,
    callbacks=make_callbacks(history_path, patience=6),
    verbose=1,
)

# 7. FINE-TUNE LAST BACKBONE LAYERS
print("\n" + "=" * 68)
print("PHASE 2: FINE-TUNING LAST RESNET50 LAYERS")
print("=" * 68)
backbone.trainable = True
for layer in backbone.layers:
    layer.trainable = False
for layer in backbone.layers[-FINE_TUNE_LAST_LAYERS:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=make_callbacks(fine_history_path, patience=5),
    verbose=1,
)

# 8. LOAD BEST CHECKPOINT AND EVALUATE THE UNCHANGED TEST SPLIT
print("\nLoading best balanced-sampling checkpoint for final evaluation...")
if not best_model_path.exists():
    raise FileNotFoundError(f"Best checkpoint was not created: {best_model_path}")

best_model = tf.keras.models.load_model(
    best_model_path,
    custom_objects={"preprocess_input": preprocess_input},
    compile=False,
)
best_model.save(last_model_path)

print("\n" + "=" * 68)
print("TEST SET EVALUATION — BALANCED-SAMPLING EXPERIMENT")
print("=" * 68)

y_true_batches = []
y_pred_batches = []
for images, labels in test_ds:
    probabilities = best_model.predict(images, verbose=0)
    y_true_batches.extend(labels.numpy().tolist())
    y_pred_batches.extend(np.argmax(probabilities, axis=1).tolist())

y_true = np.asarray(y_true_batches)
y_pred = np.asarray(y_pred_batches)

overall_accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"\nTest accuracy: {overall_accuracy:.4f}")
print(f"Test accuracy (%): {overall_accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

report = classification_report(
    y_true, y_pred, labels=list(range(num_classes)),
    target_names=class_names, zero_division=0, output_dict=True
)
report_path = OUTPUT_DIR / "classification_report_balanced.csv"
pd.DataFrame(report).transpose().to_csv(report_path, index=True)

matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
matrix_path = OUTPUT_DIR / "confusion_matrix_balanced.csv"
pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(matrix_path)

summary = {
    "experiment": "balanced_sampling",
    "number_of_classes": num_classes,
    "training_images_in_source_split": len(train_df),
    "validation_images": len(val_df),
    "test_images": len(test_df),
    "test_accuracy": float(overall_accuracy),
    "test_accuracy_percent": float(overall_accuracy * 100),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "batch_size": BATCH_SIZE,
    "head_epochs_requested": HEAD_EPOCHS,
    "fine_tune_epochs_requested": FINE_TUNE_EPOCHS,
    "backbone": "ResNet50 ImageNet pretrained",
    "architecture": "Existing attention and residual head",
    "sampling": "Equal class sampling with replacement through repeat",
    "class_weights": "Not used; balanced sampling is the correction",
    "seed": SEED,
    "checkpoint": str(best_model_path),
}
summary_path = OUTPUT_DIR / "evaluation_summary_balanced.json"
with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nSaved experiment files:")
print("Best checkpoint:", best_model_path)
print("Final model copy:", last_model_path)
print("Head history:", history_path)
print("Fine-tuning history:", fine_history_path)
print("Classification report:", report_path)
print("Confusion matrix:", matrix_path)
print("Evaluation summary:", summary_path)
print("\nBALANCED-SAMPLING EXPERIMENT COMPLETED.")
