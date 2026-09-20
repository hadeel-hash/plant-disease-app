import tensorflow as tf
import numpy as np

# --------------------------------------------------
# Settings
# --------------------------------------------------

TEST_DIR = r"D:\EX2\PtatoDataSet\test"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = tf.keras.models.load_model(
    "potato_mobilenetv2.keras"
)

print("Model loaded.")


# --------------------------------------------------
# Load test dataset EXACTLY like training code
# --------------------------------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nClasses:")
print(test_dataset.class_names)


# --------------------------------------------------
# Get predictions
# --------------------------------------------------

predictions = model.predict(
    test_dataset,
    verbose=1
)


# --------------------------------------------------
# Get true labels
# --------------------------------------------------

true_classes = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
])


predicted_classes = np.argmax(
    predictions,
    axis=1
)


# --------------------------------------------------
# Find the exact Early blight image
# --------------------------------------------------

target_name = (
    "0e0a1b51-f61c-4934-bc57-a820af1faacb"
    "___RS_Early.B 7147.JPG"
)

# Get files in the same sorted order used by the dataset
import os

class_names = test_dataset.class_names

all_files = []

for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ])

    for filename in files:
        all_files.append(
            (
                os.path.join(class_dir, filename),
                class_index
            )
        )


# --------------------------------------------------
# Find target
# --------------------------------------------------

for index, (filepath, true_label) in enumerate(all_files):

    if os.path.basename(filepath) == target_name:

        print("\n======================================")
        print("TARGET IMAGE FOUND")
        print("======================================")

        print("Index:", index)
        print("File:", filepath)

        print("\nTrue class:")
        print(class_names[true_label])

        print("\nPrediction probabilities:")

        for i, probability in enumerate(
            predictions[index]
        ):
            print(
                f"{class_names[i]}: "
                f"{probability:.6f} "
                f"({probability * 100:.2f}%)"
            )

        print("\nPredicted class:")
        print(
            class_names[
                np.argmax(predictions[index])
            ]
        )

        break
else:
    print("Target image was not found.")