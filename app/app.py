from pathlib import Path
import tempfile

from flask import Flask, jsonify, request, render_template

from app.predict import predict_image


app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "No image file provided"
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    image_path = None

    try:
        suffix = Path(image_file.filename).suffix or ".jpg"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:

            image_path = Path(temp_file.name)

        image_file.save(image_path)

        results = predict_image(image_path)

        detected_labels = [
            result["label"]
            for result in results
            if result["predicted"]
        ]

        return jsonify({
            "success": True,
            "detected_labels": detected_labels,
            "predictions": results
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500

    finally:
        if image_path:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )