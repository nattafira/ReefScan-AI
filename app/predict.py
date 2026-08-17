from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "mobilenetv2_multilabel_finetuned.keras"
THRESHOLD_PATH = MODEL_DIR / "reefscan_thresholds.json"


# ==========================================
# MODEL CONFIG
# ==========================================

IMG_SIZE = 224

LABEL_NAMES = [
    "Healthy coral",
    "Compromised coral",
    "Dead coral",
    "Rubble",
    "Competition",
    "Disease",
    "Physical issues",
]


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    THRESHOLDS = json.load(file)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(
        image_array
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    results = []

    for label_name, probability in zip(
        LABEL_NAMES,
        probabilities
    ):
        threshold = THRESHOLDS[label_name]

        results.append({
            "label": label_name,
            "probability": float(probability),
            "threshold": float(threshold),
            "predicted": bool(
                probability >= threshold
            )
        })

    return results