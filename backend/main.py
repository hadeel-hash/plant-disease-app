
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# 1. CREATE FASTAPI APP
# ============================================================

app = FastAPI()


# ============================================================
# 2. ENABLE CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "potato_mobilenetv2_from_notebook.keras"
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 5. LOAD CLASS NAMES
# ============================================================

CLASS_NAMES_PATH = (
    BASE_DIR
    / "potato_class_names.json"
)

with open(
    CLASS_NAMES_PATH,
    "r"
) as f:

    class_names = json.load(f)
# ============================================================
# 6. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    contents = await file.read()


    # --------------------------------------------------------
    # Convert uploaded bytes into an image
    # --------------------------------------------------------

    image = Image.open(
        io.BytesIO(contents)
    )


    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")


    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image = image.resize(
        (IMG_WIDTH, IMG_HEIGHT)
    )


    # --------------------------------------------------------
    # Convert image to NumPy array
    # --------------------------------------------------------

    image_array = np.array(image)


    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    print("\n==============================")
    print("IMAGE INFORMATION")
    print("==============================")

    print("Image shape:", image_array.shape)
    print("Image dtype:", image_array.dtype)
    print("Image min:", image_array.min())
    print("Image max:", image_array.max())


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )


    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = predictions[0]


    print("\n==============================")
    print("PREDICTION")
    print("==============================")

    for i, probability in enumerate(probabilities):
        print(
            f"{class_names[i]}: "
            f"{probability:.6f} "
            f"({probability * 100:.2f}%)"
        )


    # --------------------------------------------------------
    # Find predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )


    # --------------------------------------------------------
    # Get class name
    # --------------------------------------------------------

    predicted_class = class_names[
        predicted_index
    ]


    # --------------------------------------------------------
    # Get confidence
    # --------------------------------------------------------

    confidence = float(
        probabilities[predicted_index]
    )


    print("\nPredicted class:", predicted_class)
    print("Confidence:", confidence)
    print("==============================\n")


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "disease": predicted_class,
        "confidence": confidence
    }

