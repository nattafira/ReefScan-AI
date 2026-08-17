from pathlib import Path
import json

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "reefscan_model.tflite"
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
# LOAD TFLITE MODEL
# ==========================================

interpreter = Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()


# ==========================================
# LOAD THRESHOLDS
# ==========================================

with open(
    THRESHOLD_PATH,
    "r",
    encoding="utf-8"
) as file:
    THRESHOLDS = json.load(file)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_image(image_path):

    # --------------------------------------
    # Load image
    # --------------------------------------

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------
    # MobileNetV2 preprocessing
    # --------------------------------------

    image_array = (image_array / 127.5) - 1.0

    # --------------------------------------
    # TFLite inference
    # --------------------------------------

    interpreter.set_tensor(
        INPUT_DETAILS[0]["index"],
        image_array
    )

    interpreter.invoke()

    probabilities = interpreter.get_tensor(
        OUTPUT_DETAILS[0]["index"]
    )[0]

    # --------------------------------------
    # Apply thresholds
    # --------------------------------------

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