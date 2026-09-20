import json
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------
# 1. Load the model
# -----------------------------

model = tf.keras.models.load_model(
    "potato_mobilenetv2.keras"
)

print("Model loaded successfully.")


# -----------------------------
# 2. Load class names
# -----------------------------

with open("potato_class_names.json", "r") as f:
    class_names = json.load(f)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# -----------------------------
# 3. Image paths
# -----------------------------

images = {
    "Early blight": r"D:\EX2\PtatoDataSet\test\Potato___Early_blight\0e0a1b51-f61c-4934-bc57-a820af1faacb___RS_Early.B 7147.JPG",

    "Late blight": r"D:\EX2\PtatoDataSet\test\Potato___Late_blight\0c2628d4-8d64-48a9-a157-19a9c902e304___RS_LB 4590.JPG",

    "Healthy": r"D:\EX2\PtatoDataSet\test\Potato___healthy\ff700844-68ad-4e99-8427-58a39c07f817___RS_HL 1860.JPG"
}


# -----------------------------
# 4. Predict each image
# -----------------------------

for true_class, image_path in images.items():

    print("\n" + "=" * 60)
    print("Actual class:", true_class)
    print("Image:", image_path)

    # Open image
    image = Image.open(image_path)

    print("Original size:", image.size)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize exactly as the API does
    image = image.resize((224, 224))

    # Convert to NumPy
    image_array = np.array(image)

    # MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities = predictions[0]

    # Print all probabilities
    print("\nProbabilities:")

    for i, probability in enumerate(probabilities):

        print(
            f"{class_names[i]}: "
            f"{probability:.6f} "
            f"({probability * 100:.2f}%)"
        )

    # Predicted class
    predicted_index = int(
        np.argmax(probabilities)
    )

    print("\nPredicted class:")
    print(class_names[predicted_index])

    print(
        f"Confidence: "
        f"{probabilities[predicted_index] * 100:.2f}%"
    )