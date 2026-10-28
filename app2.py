from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# WikiArt Style Classifier — uses the existing trained model; no training occurs.
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

st.title("🎨 WikiArt Style Classifier")
st.write(
    "Upload a painting to predict its artistic style using a trained "
    "ResNet50-based deep learning model."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
    )


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    required = {"model_label", "style_name", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")

    df["model_label"] = df["model_label"].astype(int)
    df["style_name"] = df["style_name"].astype(str)
    df["split"] = df["split"].astype(str)
    mapping = (
        df[["model_label", "style_name"]]
        .drop_duplicates()
        .sort_values("model_label")
    )
    if mapping["model_label"].duplicated().any():
        raise ValueError("A model label maps to multiple style names.")
    if mapping["model_label"].tolist() != list(range(len(mapping))):
        raise ValueError("Model labels in metadata.csv are not contiguous from 0.")
    return df, mapping["style_name"].tolist()


try:
    model = load_model()
    metadata, class_names = load_metadata()
except Exception as exc:
    st.error(f"Could not load the model or metadata: {exc}")
    st.stop()

if int(model.output_shape[-1]) != len(class_names):
    st.error(
        f"Model outputs {model.output_shape[-1]} classes, but metadata lists "
        f"{len(class_names)} classes. Check that the model and metadata match."
    )
    st.stop()

page = st.sidebar.radio(
    "Navigate", ["Predict Style", "Model Performance", "About Project"]
)
st.sidebar.divider()
st.sidebar.caption("WikiArt-10K • 16 style classes")
st.sidebar.caption("ResNet50 + attention + residual classification head")


# PAGE 1 — PREDICTION
if page == "Predict Style":
    st.header("Predict an Art Style")
    st.write(
        "Upload a JPG, JPEG, PNG, or WEBP image. The model returns its most "
        "likely style and the top three predicted classes."
    )
    uploaded_file = st.file_uploader(
        "Upload a painting",
        type=["jpg", "jpeg", "png", "webp"],
        key="painting_upload",
    )
    if uploaded_file is None:
        st.info("Upload a painting to get started.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("This file could not be opened as an image. Try another file.")
        st.stop()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Uploaded Painting")
        st.image(image, use_container_width=True)

    # ResNet preprocessing is already included inside the saved model.
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_batch = np.expand_dims(
        np.asarray(resized, dtype=np.float32), axis=0
    )

    try:
        with st.spinner("Analyzing painting..."):
            probabilities = np.asarray(
                model.predict(image_batch, verbose=0)[0], dtype=np.float64
            )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    if len(probabilities) != len(class_names):
        st.error("Prediction output does not match the class list.")
        st.stop()

    predicted_index = int(np.argmax(probabilities))
    predicted_style = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    top_indices = np.argsort(probabilities)[::-1][:3]

    with right:
        st.subheader("Prediction")
        st.metric("Predicted Art Style", predicted_style)
        st.metric("Model Confidence", f"{confidence * 100:.2f}%")
        st.caption(
            "Confidence is the model's predicted probability, not a guarantee "
            "that the classification is correct."
        )
        st.subheader("Top 3 Predictions")
        for index in top_indices:
            style = class_names[int(index)]
            score = float(probabilities[index])
            st.write(f"**{style}** — {score * 100:.2f}%")
            st.progress(float(np.clip(score, 0.0, 1.0)))

    st.divider()
    st.subheader("Interpreting This Result")
    st.write(
        f"The model's highest-probability prediction is **{predicted_style}**. "
        "It uses visual patterns learned from labelled WikiArt images. Art "
        "styles can overlap, and predictions may be less reliable for classes "
        "with very few training examples."
    )
    st.info(
        "Educational use only. This model predicts among its 16 learned styles; "
        "it does not verify an artwork's creator, date, or authenticity."
    )


# PAGE 2 — MODEL PERFORMANCE
elif page == "Model Performance":
    st.header("Model Performance")
    st.write(
        "These metrics describe the saved model's evaluation on the held-out "
        "test set. They are not calculated from the image currently uploaded."
    )
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    report_path = RESULTS_DIR / "classification_report.csv"
    matrix_path = RESULTS_DIR / "confusion_matrix.csv"

    if not summary_path.exists():
        st.warning(
            f"Evaluation summary not found at {summary_path}. "
            "Run evaluate_model.py once to generate the metrics."
        )
        st.stop()

    try:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
    except Exception as exc:
        st.error(f"Could not read evaluation summary: {exc}")
        st.stop()

    accuracy = summary.get("test_accuracy_percent")
    macro_f1 = summary.get("macro_f1")
    weighted_f1 = summary.get("weighted_f1")
    test_images = summary.get("test_images")
    num_classes = summary.get("number_of_classes")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Test Accuracy",
        f"{float(accuracy):.2f}%" if accuracy is not None else "Not available",
    )
    col2.metric(
        "Macro F1-score",
        f"{float(macro_f1):.4f}" if macro_f1 is not None else "Not available",
    )
    col3.metric(
        "Weighted F1-score",
        f"{float(weighted_f1):.4f}" if weighted_f1 is not None else "Not available",
    )

    info1, info2 = st.columns(2)
    info1.write(f"**Test images:** {test_images if test_images is not None else 'Not recorded'}")
    info2.write(f"**Style classes:** {num_classes if num_classes is not None else len(class_names)}")

    st.divider()
    st.subheader("What the metrics mean")
    st.markdown(
        """
        - **Accuracy:** proportion of test images classified correctly.
        - **Macro F1-score:** F1 is calculated for each class and then averaged
          equally, so rare styles count as much as common styles.
        - **Weighted F1-score:** class F1-scores are averaged with more weight
          given to classes that have more test images.
        """
    )

    if report_path.exists():
        try:
            report_df = pd.read_csv(report_path, index_col=0)
            st.subheader("Per-class Evaluation")
            st.write(
                "Precision, recall, and F1-score for each style. The summary "
                "rows at the bottom are aggregate metrics."
            )
            st.dataframe(report_df, use_container_width=True)
            st.download_button(
                "Download classification report",
                data=report_path.read_bytes(),
                file_name="classification_report.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.warning(f"Could not display the classification report: {exc}")
    else:
        st.info("Classification report not found. Run evaluate_model.py to create it.")

    if matrix_path.exists():
        st.subheader("Confusion Matrix")
        st.write(
            "Rows represent the true styles and columns represent the predicted "
            "styles. Larger off-diagonal values indicate types of confusion."
        )
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
            st.warning(f"Could not display the confusion matrix: {exc}")
    else:
        st.info("Confusion matrix not found. Run evaluate_model.py to create it.")

    st.caption(
        "Metrics are read from saved evaluation files. If files are missing or "
        "outdated, regenerate them with evaluate_model.py before presenting."
    )


# PAGE 3 — ABOUT PROJECT
elif page == "About Project":
    st.header("About the Project")
    st.write(
        "WikiArt Style Classifier is a computer vision application that "
        "predicts an artistic style from an uploaded painting."
    )

    st.subheader("Dataset")
    st.write(
        f"The project uses the WikiArt-10K dataset with {len(class_names)} "
        "style classes. The preparation pipeline validates images, resizes "
        "them to 224 × 224 pixels, converts them to RGB, and records labels "
        "and train/validation/test splits in metadata.csv."
    )

    st.subheader("Model Architecture")
    st.markdown(
        """
        1. **Input:** RGB painting resized to 224 × 224.
        2. **ResNet50 backbone:** ImageNet-pretrained feature extractor.
        3. **Attention module:** combines spatial and channel attention.
        4. **Residual feature block:** additional convolutional layers and a
           residual connection.
        5. **Classification head:** dense layer, batch normalization, dropout,
           and a softmax output over the art-style classes.
        """
    )

    st.subheader("Training Strategy")
    st.markdown(
        """
        - Transfer learning with an ImageNet-pretrained ResNet50.
        - Train the classification head while the backbone is frozen.
        - Fine-tune selected backbone layers using a lower learning rate.
        - Use a held-out validation split for early stopping and checkpoint selection.
        - Evaluate final performance on a separate test split.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - The model only predicts styles represented in its training labels.
        - Styles with very few examples may be classified unreliably.
        - Visually similar styles can be confused.
        - Predicted probabilities are not guarantees of correctness.
        - The project classifies visual style; it does not establish authorship
          or artwork authenticity.
        """
    )

    st.subheader("Technology Stack")
    st.write(
        "Python • TensorFlow/Keras • ResNet50 • Streamlit • "
        "Pandas • NumPy • Matplotlib • scikit-learn"
    )
    st.success(
        "This app loads the existing trained model. Opening this page does not "
        "retrain the model."
    )
