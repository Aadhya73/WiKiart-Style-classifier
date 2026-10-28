from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

# ============================================================
# WIKIART-10K: ATTENTION-RESIDUAL STYLE CLASSIFICATION
# Uses the existing processed_data/metadata.csv dataset split.
# ============================================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8

# More epochs are allowed, but EarlyStopping will stop when
# validation performance no longer improves.
HEAD_EPOCHS = 25
FINE_TUNE_EPOCHS = 20

HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
MAX_CLASS_WEIGHT = 5.0
FINE_TUNE_LAST_LAYERS = 30

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed_data"
METADATA_FILE = DATA_DIR / "metadata.csv"
OUTPUT_DIR = PROJECT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 65)
print("WIKIART-10K MODEL TRAINING")
print("=" * 65)
print("TensorFlow version:", tf.__version__)
print("Project directory:", PROJECT_DIR)
print("Output directory:", OUTPUT_DIR)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# ============================================================
# 1. LOAD AND VALIDATE METADATA
# ============================================================

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}\n"
        "Expected processed_data/metadata.csv."
    )

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
    raise ValueError(
        f"Expected split names {expected_splits}, found {actual_splits}."
    )

class_labels = sorted(metadata["model_label"].unique().tolist())
if class_labels != list(range(len(class_labels))):
    raise ValueError(
        "Model labels must be contiguous from 0. Check "
        "prepared_data/class_mapping.json and processed_data/metadata.csv."
    )

num_classes = len(class_labels)
class_names = []
for label in class_labels:
    names = metadata.loc[metadata["model_label"] == label, "style_name"].unique()
    if len(names) != 1:
        raise ValueError(f"Label {label} has inconsistent style names: {names}")
    class_names.append(str(names[0]))

print("\nDataset records:", len(metadata))
print("Number of style classes:", num_classes)
print("\nSplit counts:")
print(metadata["split"].value_counts().to_string())
print("\nTraining images per class:")
print(
    metadata[metadata["split"] == "train"]["style_name"]
    .value_counts()
    .sort_index()
    .to_string()
)

with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as file:
    json.dump({str(i): name for i, name in enumerate(class_names)}, file, indent=4)

train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)
val_df = metadata[metadata["split"] == "validation"].reset_index(drop=True)
test_df = metadata[metadata["split"] == "test"].reset_index(drop=True)

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError("One of the dataset splits is empty.")

# ============================================================
# 2. INPUT PIPELINE
# Images stay in [0, 255] here. Augmentation and ResNet
# preprocessing are handled inside the model.
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(frame, training=False):
    paths = frame["full_path"].astype(str).to_numpy()
    labels = frame["model_label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.int32)

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(frame), 3000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    return ds.prefetch(AUTOTUNE)

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

print("\nInput pipeline created.")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))
print("Batch size:", BATCH_SIZE)

# Class weights are calculated from TRAINING labels only.
train_labels = train_df["model_label"].to_numpy()
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels,
)
class_weights = {
    int(class_id): float(min(weight, MAX_CLASS_WEIGHT))
    for class_id, weight in enumerate(weights)
}
with open(OUTPUT_DIR / "class_weights.json", "w", encoding="utf-8") as file:
    json.dump(class_weights, file, indent=2)
print("\nClass weights saved to:", OUTPUT_DIR / "class_weights.json")

# ============================================================
# 3. ATTENTION MODULE
# ============================================================

def attention_module(features, reduction_ratio=16):
    channels = int(features.shape[-1])

    spatial_attention = layers.Conv2D(
        1, kernel_size=1, activation="sigmoid", padding="same",
        name="spatial_attention_map"
    )(features)
    spatial_features = layers.Multiply(
        name="spatial_attention_features"
    )([features, spatial_attention])

    channel_weights = layers.GlobalAveragePooling2D(
        name="channel_global_average_pooling"
    )(features)
    channel_weights = layers.Dense(
        max(channels // reduction_ratio, 32),
        activation="relu",
        name="channel_attention_dense_1",
    )(channel_weights)
    channel_weights = layers.Dense(
        channels, activation="sigmoid", name="channel_attention_dense_2"
    )(channel_weights)
    channel_weights = layers.Reshape(
        (1, 1, channels), name="channel_attention_reshape"
    )(channel_weights)
    channel_features = layers.Multiply(
        name="channel_attention_features"
    )([features, channel_weights])

    return layers.Add(name="combined_attention_features")(
        [spatial_features, channel_features]
    )

# ============================================================
# 4. BUILD MODEL
# ============================================================

def build_model(number_of_classes):
    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="artwork_image"
    )

    # Moderate augmentation. Keras random augmentation layers are
    # active during training and inactive during validation/inference.
    x = layers.RandomFlip("horizontal", seed=SEED, name="random_flip")(inputs)
    x = layers.RandomRotation(0.04, seed=SEED, name="random_rotation")(x)
    x = layers.RandomZoom(0.08, seed=SEED, name="random_zoom")(x)
    x = layers.RandomTranslation(
        0.04, 0.04, seed=SEED, name="random_translation"
    )(x)

    x = layers.Lambda(preprocess_input, name="resnet_preprocessing")(x)

    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    backbone.trainable = False
    features = backbone(x, training=False)

    attended = attention_module(features)

    residual = layers.Conv2D(
        512, kernel_size=1, padding="same", use_bias=False, name="aar_conv_1"
    )(attended)
    residual = layers.BatchNormalization(name="aar_batch_norm_1")(residual)
    residual = layers.LeakyReLU(
        negative_slope=0.1, name="aar_leaky_relu"
    )(residual)
    residual = layers.Conv2D(
        int(features.shape[-1]),
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="aar_conv_2",
    )(residual)
    residual = layers.BatchNormalization(name="aar_batch_norm_2")(residual)

    features = layers.Add(name="aar_residual_add")([features, residual])
    features = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(features)
    features = layers.Dense(
        256,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="classification_dense",
    )(features)
    features = layers.BatchNormalization(name="classification_batch_norm")(
        features
    )
    features = layers.LeakyReLU(
        negative_slope=0.1, name="classification_leaky_relu"
    )(features)
    features = layers.Dropout(0.4, name="classification_dropout")(features)
    outputs = layers.Dense(
        number_of_classes, activation="softmax", name="style_probabilities"
    )(features)

    model = Model(inputs=inputs, outputs=outputs, name="WikiArt_AAR_ResNet50")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, backbone

model, backbone = build_model(num_classes)
print("\nModel architecture:")
model.summary()

# ============================================================
# 5. CALLBACKS
# ============================================================

best_model_path = OUTPUT_DIR / "best_wikiart_model.keras"
last_model_path = OUTPUT_DIR / "last_wikiart_model.keras"
history_path = OUTPUT_DIR / "training_history.csv"
fine_history_path = OUTPUT_DIR / "fine_tuning_history.csv"

def make_callbacks(csv_path, patience=6):
    return [
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            mode="min",
            verbose=1,
        ),
        CSVLogger(filename=str(csv_path), append=False),
    ]

# ============================================================
# 6. TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n" + "=" * 65)
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 65)

head_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(history_path, patience=6),
    verbose=1,
)

# ============================================================
# 7. FINE-TUNE LAST BACKBONE LAYERS
# ============================================================

print("\n" + "=" * 65)
print("PHASE 2: FINE-TUNING LAST RESNET50 LAYERS")
print("=" * 65)

backbone.trainable = True
for layer in backbone.layers:
    layer.trainable = False

# Unfreeze only the last N layers; keep BatchNorm frozen for stability.
for layer in backbone.layers[-FINE_TUNE_LAST_LAYERS:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

fine_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(fine_history_path, patience=5),
    verbose=1,
)

# Load the best validation-loss checkpoint from either phase.
print("\nLoading best saved checkpoint for final evaluation...")
from tensorflow.keras.applications.resnet50 import preprocess_input

best_model = tf.keras.models.load_model(
    best_model_path,
    custom_objects={"preprocess_input": preprocess_input}
)
best_model.save(last_model_path)

# ============================================================
# 8. TEST EVALUATION
# Test set is used only after training/fine-tuning is complete.
# ============================================================

print("\n" + "=" * 65)
print("TEST SET EVALUATION")
print("=" * 65)

test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
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

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {overall_accuracy:.4f}")
print(f"Test accuracy (%): {overall_accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0,
    output_dict=True,
)
report_df = pd.DataFrame(report).transpose()
report_path = OUTPUT_DIR / "classification_report.csv"
report_df.to_csv(report_path, index=True)

matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
matrix_path = OUTPUT_DIR / "confusion_matrix.csv"
pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(matrix_path)

summary = {
    "number_of_classes": num_classes,
    "training_images": len(train_df),
    "validation_images": len(val_df),
    "test_images": len(test_df),
    "test_loss": float(test_loss),
    "test_accuracy": float(overall_accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "batch_size": BATCH_SIZE,
    "head_epochs_requested": HEAD_EPOCHS,
    "fine_tune_epochs_requested": FINE_TUNE_EPOCHS,
    "backbone": "ResNet50 ImageNet pretrained",
    "architecture": "AAR-style attention and residual head with fine-tuning",
    "seed": SEED,
}
with open(OUTPUT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\nSaved evaluation files:")
print(best_model_path)
print(last_model_path)
print(history_path)
print(fine_history_path)
print(report_path)
print(matrix_path)
print(OUTPUT_DIR / "evaluation_summary.json")
print("\nMODEL TRAINING AND EVALUATION COMPLETED.")
