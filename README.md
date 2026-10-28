# 🎨 WikiArt Style Classifier

A deep learning project that classifies paintings into different artistic styles using computer vision and transfer learning. The project uses the WikiArt-10K dataset and a modified ResNet50 architecture with spatial attention, channel attention, and a custom classification head.

The aim is to explore how deep learning can identify visual patterns associated with different art movements, while evaluating the challenges caused by similarities between styles and an imbalanced dataset.

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Class Imbalance](#-handling-class-imbalance)
- [Experimental Results](#-experimental-results)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Model Evaluation](#-model-evaluation)
- [Running the Web Application](#-running-the-web-application)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Reference](#-reference)

## 🧠 Project Overview

Artistic styles are often distinguished by characteristics such as brushwork, colour palettes, composition, texture, and representation. However, several art movements share similar visual characteristics, making automatic classification a challenging computer vision task.

This project develops an image classification pipeline that takes a painting as input and predicts its artistic style using a convolutional neural network based on ResNet50.

The workflow includes:

1. Dataset inspection and image validation.
2. Image preprocessing and dataset splitting.
3. Transfer learning using an ImageNet-pretrained ResNet50 backbone.
4. Spatial and channel attention mechanisms.
5. Training with class weighting and a separate balanced-sampling experiment.
6. Evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.
7. An interactive Streamlit application for artwork classification.

## ✨ Features

- **Art style classification:** Predicts the artistic style of an uploaded painting.
- **Transfer learning:** Uses a pretrained ResNet50 backbone to extract visual features.
- **Attention mechanisms:** Combines spatial and channel attention to refine feature representations.
- **Two training experiments:** Compares class-weighted training with balanced sampling.
- **Model evaluation:** Generates classification reports, confusion matrices, and evaluation summaries.
- **Interactive application:** Provides an interface for uploading paintings and viewing predictions.
- **Reproducible workflow:** Includes separate scripts for dataset validation, preprocessing, training, and evaluation.

## 🖼️ Dataset

The project uses the [WikiArt-10K dataset on Hugging Face](https://huggingface.co/datasets/tellow-it-1/wikiart-10k).

The dataset contains approximately 10,000 artwork records across 16 observed style classes in the version used for this project.

### Dataset characteristics

| Property | Description |
|---|---|
| Dataset | WikiArt-10K |
| Source | Hugging Face Datasets |
| Original records | Approximately 10,000 |
| Observed style classes | 16 |
| Image format after preprocessing | RGB |
| Input image size | 224 × 224 pixels |
| Train/validation/test split | 70% / 15% / 15% |
| Split strategy | Stratified splitting |
| Random state | 42 |

The final number of usable images and the number of examples in each split depend on the output of the preprocessing pipeline.

### Data preprocessing

The preprocessing pipeline is designed to:

- Validate image records and identify invalid images.
- Convert usable images to RGB format.
- Resize images to 224 × 224 pixels.
- Create training, validation, and test splits.
- Store image metadata, including image paths, split assignments, numerical labels, and style names.

The dataset exhibits considerable class imbalance. Some styles have thousands of examples, while others have very few. This can affect the model's ability to generalize across all classes.

**Note:** The raw dataset and processed images are not included in this repository. Obtain the dataset from its original source before running the complete pipeline.

## 🏗️ Model Architecture

The classification model is based on an ImageNet-pretrained ResNet50 backbone, modified with attention mechanisms and a custom classification head.

### Architecture components

**1. ResNet50 backbone**

A pretrained ResNet50 extracts hierarchical visual features from the input painting. Transfer learning reduces the need to learn all visual representations from scratch.

**2. Spatial attention**

A convolutional layer followed by a sigmoid activation generates a spatial attention map. This map reweights feature locations to emphasize potentially relevant regions of the painting.

**3. Channel attention**

Global average pooling and dense layers generate channel-wise attention weights. These weights adjust the relative importance of different feature channels.

**4. Residual refinement block**

The attended features are refined using convolutional layers, batch normalization, and Leaky ReLU activation.

**5. Classification head**

The classification head uses global average pooling, a dense layer with 256 units, batch normalization, Leaky ReLU activation, and dropout with a rate of 0.4.

**6. Output layer**

A softmax output layer produces class probabilities for the 16 observed art styles.

### Training strategy

Training uses a two-stage transfer learning approach:

- **Stage 1 — Feature extraction:** Freeze the ResNet50 backbone and train the classification head.
- **Stage 2 — Fine-tuning:** Unfreeze selected backbone layers and continue training with a lower learning rate.

Callbacks such as early stopping, learning-rate reduction, model checkpointing, and CSV logging are used to support training and retain useful experiment outputs.

## ⚖️ Handling Class Imbalance

The dataset contains substantially different numbers of examples across art styles. Without appropriate handling, the model may favor frequently occurring classes.

Two approaches were explored.

### Experiment 1: Class-weighted training

The training loss is adjusted using class weights so that examples from underrepresented classes can have a greater influence on model optimization. The weights are capped to limit excessively large values for rare classes.

### Experiment 2: Balanced sampling

A separate training script samples classes more evenly during training. This increases the frequency with which underrepresented classes are encountered by repeating existing examples where necessary.

The balanced-sampling experiment is maintained separately from the original training experiment so that its model checkpoint and outputs do not overwrite those of the original run.

## 📊 Experimental Results

Two training experiments were evaluated on the held-out test split.

| Metric | Class-weighted experiment | Balanced-sampling experiment |
|---|---:|---:|
| Test accuracy | 67.53% | 68.73% |
| Macro F1-score | 0.5786 | 0.5513 |
| Weighted F1-score | 0.6893 | 0.6971 |

### Results interpretation

- The balanced-sampling experiment achieved a test accuracy approximately 1.20 percentage points higher than the class-weighted experiment.
- The class-weighted experiment achieved a higher macro F1-score, indicating better aggregate performance when each class receives equal weight in the metric.
- The balanced-sampling experiment achieved a slightly higher weighted F1-score.
- These results illustrate that improved overall accuracy does not necessarily translate into better performance across every class.

Macro F1-score is particularly relevant for this dataset because the number of images varies considerably between styles. Per-class precision, recall, F1-scores, and the confusion matrix should also be examined to understand which art movements are difficult to distinguish.

**Reproducibility note:** The metrics above are the results recorded for these experiments. They are not a guarantee that a new training run will produce identical results. Outcomes may vary with software versions, hardware, random operations, and available data.

The evaluation script `evaluate_model.py` evaluates the original checkpoint at `models/best_wikiart_model.keras`. The balanced-sampling checkpoint is stored separately under `models/balanced_experiment/` and requires the appropriate evaluation configuration if you want to evaluate that experiment independently.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning model development and training |
| ResNet50 | Transfer learning and feature extraction |
| NumPy | Numerical operations |
| Pandas | Dataset metadata and tabular processing |
| Scikit-learn | Dataset splitting and evaluation metrics |
| Pillow | Image processing |
| PyArrow | Reading Parquet datasets |
| Matplotlib | Visualization |
| Streamlit | Interactive web application |

## 📁 Repository Structure

The main project files are organized as follows:

```text
WiKiart-Style-classifier/
│
├── app.py
├── validate_images.py
├── prepare_data.py
├── create_processed_dataset.py
├── verify_processed_data.py
├── train_model.py
├── train_model_balanced.py
├── evaluate_model.py
├── check_mapping.py
├── inspect_data.py
├── inspect_schema.py
├── load_dataset.py
│
├── dataset/                     # Raw dataset files; not included
├── processed_data/              # Processed images and metadata; generated
├── models/                      # Trained model checkpoints; generated
│   └── balanced_experiment/     # Separate balanced-sampling outputs
│
├── requirements.txt
├── README.md
└── docs/                        # Project documentation and reports, if present
```

This is a logical overview rather than a guarantee that every folder is tracked in Git. Generated folders may only appear after the relevant scripts have been executed.

## 🚀 Installation

### Prerequisites

- Python 3.11 recommended.
- Git.
- Sufficient disk space for the dataset, processed images, and model checkpoints.
- A compatible TensorFlow environment. A supported GPU is optional; training can be slower on a CPU.

### 1. Clone the repository

```bash
git clone https://github.com/Aadhya73/WiKiart-Style-classifier.git
cd WiKiart-Style-classifier
```

If the repository URL or capitalization differs in your GitHub account, use the URL displayed on the repository's **Code** button.

### 2. Create a virtual environment

On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -c "import tensorflow, numpy, pandas, sklearn, PIL, pyarrow, streamlit, matplotlib; print('All imports successful')"
```

If an import fails, check the Python version and the compatibility of the installed packages.

## 📂 Dataset Preparation

The raw dataset is not committed to the repository. Download it from the [WikiArt-10K dataset page](https://huggingface.co/datasets/tellow-it-1/wikiart-10k) and place the Parquet files in the location expected by the dataset-loading scripts.

For a typical local setup, the raw files may be organized as follows:

```text
dataset/
├── file-00000-of-00010.parquet
├── file-00001-of-00010.parquet
└── ...
```

The actual filenames and expected paths must match the paths configured in the scripts.

Run the data preparation scripts in the following order, checking each script's output before continuing:

```bash
python validate_images.py
python prepare_data.py
python create_processed_dataset.py
python verify_processed_data.py
```

The scripts perform the validation, splitting, image creation, and verification stages implemented in the project.

After preparation, confirm that the metadata and processed image paths are correct before training. The metadata file is expected to be available at:

```text
processed_data/metadata.csv
```

If your local dataset or output folders are configured differently in the scripts, use those configured paths instead.

## 🏋️ Training

### Experiment 1: Class-weighted training

Run:

```bash
python train_model.py
```

This script trains the attention-enhanced ResNet50 model using the class-weighted training approach. The saved checkpoint and training logs are written to the locations configured in the script.

The original best model is expected at:

```text
models/best_wikiart_model.keras
```

### Experiment 2: Balanced sampling

Run:

```bash
python train_model_balanced.py
```

This launches a separate experiment using balanced sampling. Its outputs are saved under:

```text
models/balanced_experiment/
```

The best balanced-sampling checkpoint is expected at:

```text
models/balanced_experiment/best_wikiart_model_balanced.keras
```

Training time depends on the hardware, batch size, dataset size, and number of epochs completed before early stopping. GPU acceleration is recommended for faster training when a compatible environment is available.

## 📈 Model Evaluation

To evaluate the original class-weighted checkpoint, run:

```bash
python evaluate_model.py
```

The script uses the test split defined by the prepared metadata and evaluates the original model checkpoint.

Evaluation outputs include files such as:

```text
models/
├── classification_report.csv
├── confusion_matrix.csv
└── evaluation_summary.json
```

The precise output paths and filenames are determined by the current evaluation script. The balanced-sampling experiment has separate evaluation output files when its corresponding evaluation workflow is run.

The evaluation process reports classification performance using metrics such as:

- Test accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Weighted F1-score.
- Confusion matrix.

Use the held-out test set for final evaluation rather than repeatedly tuning the model against test results.

## 🖥️ Running the Web Application

The project includes a Streamlit application for interactive predictions and model performance visualization.

### 1. Check the required files

Before launching the application, ensure that the model checkpoint expected by `app.py` exists:

```text
models/best_wikiart_model.keras
```

The application may also require the processed metadata and evaluation outputs for its performance page. Generate these files using the preparation and evaluation workflows described above.

### 2. Launch Streamlit

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

### 3. Use the application

Depending on the features implemented in `app.py`, you can:

- Upload a painting for classification.
- View the predicted art style and associated confidence.
- Explore model performance information and evaluation results.

The application requires the trained model files locally. They are not automatically included when cloning the repository.

## ⚠️ Limitations

- **Class imbalance:** Some art styles have very few training examples, which limits the model's ability to learn reliable representations for those classes.
- **Visual similarity:** Different artistic movements can share colors, textures, brushwork, and compositions.
- **Overfitting:** The model may learn patterns specific to the training dataset that do not generalize to unseen paintings.
- **Dataset dependence:** Performance depends on the dataset's labels, image quality, class distribution, and preprocessing.
- **Confidence interpretation:** A high softmax probability should not automatically be treated as a calibrated measure of certainty.
- **Reproducibility:** Exact results may vary across software environments, hardware, and training runs.
- **Model availability:** Predictions require a locally available trained checkpoint.
- **Research comparison:** Results from this project should not be compared directly with other papers unless their datasets, class definitions, splits, and evaluation procedures are sufficiently aligned.

This project is intended for academic experimentation and educational use. Its predictions should not be treated as definitive art-historical judgments.

## 🔮 Future Improvements

Potential directions for future development include:

- Improving representation of underrepresented styles with additional data.
- Comparing class weighting, balanced sampling, and other imbalance-handling techniques.
- Evaluating alternative pretrained architectures.
- Exploring stronger data augmentation and regularization.
- Improving performance on rare classes using per-class error analysis.
- Adding explainability visualizations to investigate which image regions influence predictions.
- Calibrating prediction probabilities and displaying uncertainty more clearly.
- Improving experiment configuration and recording exact package versions and random seeds.
- Adding automated tests and a streamlined end-to-end setup process.

## 📚 Reference

This project is inspired by the research paper:

Zhang, H. (2025). *Improved Deep Convolutional Neural Network for Digital Art Image Classification and Identification*. International Journal of Computational Intelligence Systems, 18, Article 249.

https://doi.org/10.1007/s44196-025-00996-0

The implementation in this repository is an adaptation for the project's dataset and experimental setup; it should not be interpreted as an exact reproduction of the paper's architecture or reported results.

## 👩‍💻 Project Information

**Project:** WikiArt Style Classifier  
**Domain:** Computer Vision and Deep Learning  
**Task:** Multiclass image classification  
**Dataset:** WikiArt-10K  
**Framework:** TensorFlow / Keras  
**Application:** Streamlit

---

*Developed as an academic computer vision project exploring deep learning for art-style classification.*