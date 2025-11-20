# app.py
import os
from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model at startup
MODEL_PATH = "savedmodel.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("savedmodel.pth not found. Run train.py first.")
model = joblib.load(MODEL_PATH)

def preprocess_image(file) -> np.ndarray:
    """
    Convert uploaded image to 64x64 grayscale and flatten to (1, 4096)
    to match Olivetti faces input format.
    """
    img = Image.open(file).convert("L")   # grayscale
    img = img.resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0  # normalize like faces data
    arr = arr.flatten().reshape(1, -1)
    return arr

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file part"
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "No selected file"
            else:
                try:
                    x = preprocess_image(file)
                    pred = model.predict(x)[0]
                    prediction = int(pred)
                except Exception as e:
                    error = f"Error processing image: {e}"

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
