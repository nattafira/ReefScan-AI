# ReefScan AI 🌊🪸

**ReefScan AI** is a deep learning project for multi-label classification of
coral reef conditions from underwater images.

🌐 **Live Demo:** https://nattafira.pythonanywhere.com

The system uses **MobileNetV2** with transfer learning and fine-tuning. Because the dataset is multi-label, a single image patch can contain multiple coral conditions or stressors at the same time.

> **Current best test result:** Macro F1 **63.9%**

---

## 🎯 Project Overview

ReefScan AI is designed to support the analysis of coral reef conditions from underwater survey imagery.

Instead of forcing every image into a single class, the model can identify multiple conditions within the same image patch according to the dataset's multi-label annotations.

### Model Outputs

The final model uses 7 labels:

1. **Healthy coral**
2. **Compromised coral**
3. **Dead coral**
4. **Rubble**
5. **Competition**
6. **Disease**
7. **Physical issues**

The **Predation** label was excluded from the final model because it had substantially fewer samples than the other labels and was removed during dataset preparation.

---

## 🧠 Model

The main model pipeline is:

```text
Input image
    ↓
Resize to 224 × 224
    ↓
MobileNetV2 pretrained backbone
    ↓
Global Average Pooling
    ↓
Dense(256) + Dropout
    ↓
7 sigmoid outputs
    ↓
Per-label thresholds
    ↓
Multi-label predictions
```

### Training Strategy

- A pretrained **MobileNetV2** backbone is used for transfer learning.
- A frozen-backbone baseline was trained first.
- A **weighted binary cross-entropy** experiment was evaluated for class imbalance, but it was not selected as the final approach.
- The final model uses **fine-tuning** on the later part of the MobileNetV2 backbone with a small learning rate.
- Decision thresholds were optimized separately for each label using the validation set.

### Final Decision Thresholds

| Label | Threshold |
|---|---:|
| Healthy coral | 0.45 |
| Compromised coral | 0.35 |
| Dead coral | 0.25 |
| Rubble | 0.40 |
| Competition | 0.30 |
| Disease | 0.30 |
| Physical issues | 0.35 |

---

## 📊 Evaluation

The final evaluation was performed on **3,162 test patches**.

| Label | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Healthy coral | 0.87 | 0.95 | 0.91 |
| Compromised coral | 0.58 | 0.78 | 0.67 |
| Dead coral | 0.61 | 0.82 | 0.70 |
| Rubble | 0.59 | 0.55 | 0.57 |
| Competition | 0.49 | 0.52 | 0.51 |
| Disease | 0.57 | 0.66 | 0.62 |
| Physical issues | 0.48 | 0.53 | 0.50 |

### Overall Metrics

| Metric | Score |
|---|---:|
| **Macro F1** | **0.6391** |
| **Micro F1** | **0.7358** |

Macro F1 is used as an important summary metric because it gives equal weight to every label and therefore provides a clearer view of performance on less frequent labels than accuracy alone.

---

## 🗂️ Dataset Preparation

The dataset contains coral image patches collected from underwater field surveys in **Koh Tao, Thailand**. Before training, the images were audited for validity, dimensions, brightness, contrast, and blur quality.

After quality filtering and label preparation:

- **20,914 patches** were retained for model development.
- The data was split into:
  - **14,651 training patches**
  - **3,101 validation patches**
  - **3,162 test patches**
- Original image patches are **512 × 512**.
- Model input images are resized to **224 × 224 × 3** for MobileNetV2.

The dataset is multi-label, meaning that a single patch may contain multiple condition or stressor annotations.

---

## 🔎 Error Analysis

Error analysis was performed after model training to understand the remaining weaknesses of the classifier.

Key observations:

- **Healthy coral** is the strongest-performing label, but it may co-occur with other condition or stressor labels because the dataset uses multi-label annotations..
- **Disease, Competition, and Physical issues** are more difficult to distinguish because they often co-occur with other labels and their visual characteristics can overlap with surrounding coral conditions.
- Per-label threshold tuning improved the balance between precision and recall compared with using a fixed threshold of 0.5 for every label.
- The final model should be interpreted as a **multi-label decision-support system**, not as a definitive ecological diagnosis.

---

## 🌐 Web Application

The project includes a simple **Flask-based web application** that connects the trained model to a browser-based interface.

For deployment, model inference is performed using **LiteRT / TensorFlow Lite** to reduce the runtime dependency footprint while preserving the prediction behavior of the original Keras model.

```text
User uploads image
      ↓
Flask API /predict
      ↓
LiteRT inference
      ↓
Per-label thresholds
      ↓
JSON response
      ↓
ReefScan AI web interface
```

### Main Application Files

```text
app/
├── app.py          # Flask API and web server
├── predict.py      # LiteRT model loading and inference logic
├── templates/
│   └── index.html  # ReefScan AI web interface
└── static/
    ├── audio/      # Underwater ambience assets
    └── video/      # Hero video assets
```

---

## 📁 Project Structure

```text
ReefScan-AI/
│
├── app/
│   ├── app.py
│   ├── predict.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── audio/
│       └── video/
│
├── models/
│   ├── reefscan_model.tflite
│   └── reefscan_thresholds.json
│
├── notebooks/
│   ├── 01_Audit_Dataset.ipynb
│   ├── 02_Prepare_Dataset.ipynb
│   └── 03_Finalize_Inference.ipynb
│
├── .gitignore
├── .python-version
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

### 1. Create and activate the virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Flask application

Run from the project root:

```bash
python -m app.app
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 🧪 Running Model Inference

The inference logic is implemented in `app/predict.py`.

The deployed application uses the LiteRT model:

```text
models/reefscan_model.tflite
```

The model returns an independent probability for each of the 7 labels and then applies the saved per-label thresholds from:

```text
models/reefscan_thresholds.json
```

The LiteRT model was validated against the original Keras model to ensure that the prediction outputs remain consistent after conversion, with negligible numerical differences.

---

## 📚 Notebooks

| Notebook | Purpose |
|---|---|
| `01_Audit_Dataset.ipynb` | Dataset audit, image quality inspection, label analysis, and filtering |
| `02_Prepare_Dataset.ipynb` | Train/validation/test preparation, model training, threshold tuning, and evaluation |
| `03_Finalize_Inference.ipynb` | Final model loading, threshold export, and inference validation |

---

## 🛠️ Tech Stack

### Model Development

- **Python**
- **TensorFlow / Keras**
- **MobileNetV2**
- **OpenCV / PIL**
- **NumPy / Pandas**
- **scikit-learn**
- **Jupyter Notebook**

### Deployment

- **LiteRT / TensorFlow Lite**
- **Flask**
- **HTML**
- **Tailwind CSS**
- **JavaScript**

---

### Deployment Environment

For deployment, the application uses a lightweight Python environment with
**LiteRT** for model inference instead of the full TensorFlow runtime.

The production environment reuses the host-provided Flask, NumPy, and Pillow
packages where available, while LiteRT is installed separately.

## ⚠️ Limitations

- The model was trained on coral image patches from a specific survey dataset and may not generalize equally well to arbitrary underwater or aquarium photographs.
- Multi-label predictions can contain several conditions simultaneously.
- The system is intended as a **portfolio/research prototype and decision-support tool**, not as a replacement for expert ecological assessment.
- Some labels have lower F1 scores than Healthy coral because of class imbalance, label co-occurrence, and overlapping visual characteristics.

---

## 📌 Dataset & References

This project is based on the **CoralConditionDataset** from the XL-SHAO repository and the associated research work on multi-label coral reef condition classification.

- **Dataset:** [CoralConditionDataset](https://github.com/XL-SHAO/CoralConditionDataset)
- **Research paper:** [DOI: 10.1002/aqc.4241](https://doi.org/10.1002/aqc.4241)

---

## 👩‍💻 Project Status

**Current status:** Deployed and publicly accessible.

The ReefScan AI model was developed using **MobileNetV2 transfer learning with fine-tuning**, then converted from Keras to **LiteRT** for lightweight deployment.

The converted LiteRT model was validated against the original Keras model with negligible prediction differences.

The model is integrated into a **Flask-based web application** and deployed on PythonAnywhere.

### 🌐 Live Demo

**ReefScan AI:** https://nattafira.pythonanywhere.com
