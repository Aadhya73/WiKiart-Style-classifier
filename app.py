from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input

# WikiArt Style Classifier — loads an existing checkpoint; it does not train.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_wikiart_model.keras"
DATA_DIR = PROJECT_DIR / "processed_data"
METADATA_PATH = DATA_DIR / "metadata.csv"
RESULTS_DIR = PROJECT_DIR / "models"
IMAGE_SIZE = 224

st.set_page_config(
    page_title="WikiArt Style Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Check that models/best_wikiart_model.keras exists."
        )
    # preprocess_input is registered because the model may contain a Lambda layer.
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
    )


@st.cache_data
def load_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    required = {"image_path", "split", "model_label", "style_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"metadata.csv is missing required columns: {sorted(missing)}"
        )

    df["model_label"] = pd.to_numeric(df["model_label"], errors="raise").astype(int)
    df["style_name"] = df["style_name"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    mapping = (
        df[["model_label", "style_name"]]
        .drop_duplicates()
        .sort_values("model_label")
    )
    if mapping["model_label"].duplicated().any():
        raise ValueError("One model_label maps to more than one style_name.")
    if mapping["model_label"].tolist() != list(range(len(mapping))):
        raise ValueError("model_label values must be contiguous, starting at 0.")
    if mapping["style_name"].duplicated().any():
        raise ValueError("Duplicate style names found in the label mapping.")

    return df, mapping["style_name"].tolist()


def prepare_image_for_prediction(image):
    """Match training: RGB, 224x224 bilinear resize, float32 [0, 255]."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_array = np.asarray(image, dtype=np.uint8)
    image_tensor = tf.convert_to_tensor(image_array)
    image_tensor = tf.image.resize(
        image_tensor,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
        antialias=False,
    )
    image_tensor = tf.cast(image_tensor, tf.float32)
    # Do NOT call preprocess_input here: it is already inside the saved model.
    return tf.expand_dims(image_tensor, axis=0)


try:
    model = load_model()
    metadata, class_names = load_metadata()
except Exception as exc:
    st.error(f"Could not load the model or metadata: {exc}")
    st.stop()

if model.output_shape[-1] != len(class_names):
    st.error(
        f"Model outputs {model.output_shape[-1]} classes, but metadata contains "
        f"{len(class_names)} style names. Verify the checkpoint and metadata match."
    )
    st.stop()

page = st.sidebar.radio(
    "Navigate", ["Predict Style", "Model Performance", "About Project"]
)
st.sidebar.divider()
st.sidebar.caption(f"WikiArt-10K • {len(class_names)} style classes")
st.sidebar.caption("ResNet50 + attention + residual classification head")

# -------------------------------------------------------------------
# PAGE 1 — PREDICTION
# -------------------------------------------------------------------
if page == "Predict Style":
    st.title("🎨 WikiArt Style Classifier")
    st.header("Predict an Art Style")
    st.write(
        "Upload a painting to receive the model's most likely style and its "
        "top three predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload a painting",
        type=["jpg", "jpeg", "png", "webp"],
        key="painting_upload",
    )

    if uploaded_file is None:
        st.info("Upload a JPG, JPEG, PNG, or WEBP painting to get started.")
    else:
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception:
            st.error("This file could not be opened as an image. Try another file.")
            st.stop()

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.subheader("Uploaded Painting")
            st.image(image, use_container_width=True)

        try:
            with st.spinner("Analyzing painting..."):
                image_batch = prepare_image_for_prediction(image)
                probabilities = np.asarray(
                    model.predict(image_batch, verbose=0)[0], dtype=np.float64
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        if (
            probabilities.ndim != 1
            or len(probabilities) != len(class_names)
            or not np.isfinite(probabilities).all()
        ):
            st.error("The model returned invalid output. Check the checkpoint.")
            st.stop()

        # Softmax outputs should be probabilities. Do not silently normalize logits.
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            st.error(
                "The model output is not a probability distribution. "
                "Confirm that the saved model ends with a softmax layer."
            )
            st.stop()

        predicted_index = int(np.argmax(probabilities))
        predicted_style = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])
        top_indices = np.argsort(probabilities)[::-1][: min(3, len(class_names))]

        with right:
            st.subheader("Prediction")
            st.metric("Predicted Art Style", predicted_style)
            st.metric("Model Confidence", f"{confidence * 100:.2f}%")
            st.caption(
                "This is the model's predicted probability, not a guarantee "
                "that the style is correct."
            )

            if confidence < 0.50:
                st.warning(
                    "The top prediction has relatively low model confidence. "
                    "Consider the alternatives below."
                )

            st.subheader("Top Predictions")
            for index in top_indices:
                style = class_names[int(index)]
                score = float(probabilities[index])
                st.write(f"**{style}** — {score * 100:.2f}%")
                st.progress(float(np.clip(score, 0.0, 1.0)))

        st.divider()
        st.subheader("Interpreting This Result")
        st.write(
            f"The highest-probability prediction is **{predicted_style}**. "
            "Art styles can share visual features, and predictions may be less "
            "reliable for classes with very few training examples."
        )
        st.info(
            "Educational use only. The model predicts among its trained styles; "
            "it does not verify the artist, date, provenance, or authenticity."
        )

# -------------------------------------------------------------------
# PAGE 2 — MODEL PERFORMANCE
# -------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.write(
        "These metrics are loaded from saved evaluation files for the selected "
        "checkpoint. They are not computed from the currently uploaded image."
    )

    summary_path = RESULTS_DIR / "evaluation_summary.json"
    report_path = RESULTS_DIR / "classification_report.csv"
    matrix_path = RESULTS_DIR / "confusion_matrix.csv"

    if not summary_path.is_file():
        st.warning(
            "Evaluation summary not found. Run evaluate_model.py to generate "
            "metrics for the checkpoint currently selected in this app."
        )
        st.stop()

    try:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
    except Exception as exc:
        st.error(f"Could not read evaluation summary: {exc}")
        st.stop()

    # Supports both the older key ('test_accuracy') and percentage key.
    accuracy_percent = summary.get("test_accuracy_percent")
    if accuracy_percent is None and summary.get("test_accuracy") is not None:
        accuracy_percent = float(summary["test_accuracy"]) * 100.0

    macro_f1 = summary.get("macro_f1")
    weighted_f1 = summary.get("weighted_f1")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Test Accuracy",
        f"{float(accuracy_percent):.2f}%"
        if accuracy_percent is not None else "Not available",
    )
    col2.metric(
        "Macro F1-score",
        f"{float(macro_f1):.4f}" if macro_f1 is not None else "Not available",
    )
    col3.metric(
        "Weighted F1-score",
        f"{float(weighted_f1):.4f}" if weighted_f1 is not None else "Not available",
    )

    st.write(f"**Test images:** {summary.get('test_images', 'Not recorded')}")
    st.write(
        f"**Style classes:** "
        f"{summary.get('number_of_classes', len(class_names))}"
    )
    st.caption(
        f"Checkpoint configured in app: {MODEL_PATH.relative_to(PROJECT_DIR)}"
    )

    st.markdown(
        """
        - **Accuracy:** proportion of test images classified correctly.
        - **Macro F1:** equally averages the F1-score of each class.
        - **Weighted F1:** averages class F1-scores weighted by their test support.
        """
    )

    if report_path.is_file():
        st.subheader("Per-class Evaluation")
        try:
            report_df = pd.read_csv(report_path, index_col=0)
            st.dataframe(report_df, use_container_width=True)
            st.download_button(
                "Download classification report",
                data=report_path.read_bytes(),
                file_name="classification_report.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.warning(f"Could not display classification report: {exc}")
    else:
        st.info("Classification report not found. Run evaluate_model.py.")

    if matrix_path.is_file():
        st.subheader("Confusion Matrix")
        try:
            cm_df = pd.read_csv(matrix_path, index_col=0)
            st.dataframe(cm_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(12, 9))
            plot = ax.imshow(cm_df.to_numpy(), aspect="auto")
            ax.set_xticks(np.arange(len(cm_df.columns)))
            ax.set_xticklabels(cm_df.columns, rotation=90, fontsize=7)
            ax.set_yticks(np.arange(len(cm_df.index)))
            ax.set_yticklabels(cm_df.index, fontsize=7)
            ax.set_xlabel("Predicted style")
            ax.set_ylabel("True style")
            ax.set_title("Confusion Matrix")
            fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.download_button(
                "Download confusion matrix",
                data=matrix_path.read_bytes(),
                file_name="confusion_matrix.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.warning(f"Could not display confusion matrix: {exc}")
    else:
        st.info("Confusion matrix not found. Run evaluate_model.py.")

# -------------------------------------------------------------------
# PAGE 3 — ABOUT PROJECT
# -------------------------------------------------------------------
else:
    st.title("About the Project")
    st.write(
        "WikiArt Style Classifier is a computer vision application that "
        "predicts an artistic style from an uploaded painting."
    )

    st.subheader("Dataset")
    st.write(
        f"The project uses WikiArt-10K with {len(class_names)} style classes. "
        "Images are converted to RGB and resized to 224 × 224 pixels. "
        "The metadata file records labels and train/validation/test splits."
    )

    st.subheader("Model Architecture")
    st.markdown(
        """
        1. **Input:** RGB image resized to 224 × 224.
        2. **ResNet50 backbone:** ImageNet-pretrained visual feature extractor.
        3. **Attention module:** combines spatial and channel attention.
        4. **Residual feature block:** additional convolutional layers with a skip connection.
        5. **Classification head:** global average pooling, dense layers, batch normalization,
           LeakyReLU, dropout, and a softmax output.
        """
    )

    st.subheader("Training Strategy")
    st.markdown(
        """
        - Transfer learning using ImageNet-pretrained ResNet50.
        - Train the classification head with the backbone frozen.
        - Fine-tune selected backbone layers with a lower learning rate.
        - Use validation data for checkpoint selection.
        - Use the held-out test split for final evaluation.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - The model predicts only styles represented in its training labels.
        - Rare styles may be classified unreliably due to limited training examples.
        - Visually similar styles can be confused.
        - Softmax confidence is not a calibrated guarantee of correctness.
        - The model does not establish authorship or artwork authenticity.
        """
    )

    st.subheader("Technology Stack")
    st.write(
        "Python • TensorFlow/Keras • ResNet50 • Streamlit • "
        "Pandas • NumPy • Matplotlib • scikit-learn"
    )
    st.success("This app loads an existing model and does not retrain it.")
