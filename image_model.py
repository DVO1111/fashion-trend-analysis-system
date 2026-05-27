"""CNN image classifier for fashion categories.

Builds a small CNN (the architecture from the project spec) and exposes:
- build_model(num_classes): construct + compile the model
- predict_style(model, image_path, labels): classify a single image
- save_model / load_model: persist to models/cnn_model.h5

Run directly to build and save an untrained model skeleton.
"""

import json
import os
import numpy as np
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

MODEL_PATH = "models/cnn_model.h5"
LABELS_PATH = "models/cnn_labels.json"
IMG_SIZE = (224, 224)
DEFAULT_LABELS = ["ankara_fusion", "streetwear", "casual", "alte", "corporate_chic"]


def load_labels(path: str = LABELS_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return DEFAULT_LABELS


def build_model(num_classes: int = 5) -> Sequential:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_style(model, image_path: str, labels=None):
    import cv2

    labels = labels or DEFAULT_LABELS
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    # cv2.imread returns BGR; convert to RGB to match training (Keras image_dataset_from_directory)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    # Pass raw [0, 255] floats — the model's first layer (Lambda + preprocess_input)
    # handles MobileNetV2's [0, 255] -> [-1, 1] mapping internally.
    x = np.expand_dims(image.astype("float32"), axis=0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {"label": labels[idx], "confidence": float(probs[idx])}


def save(model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def load(path: str = MODEL_PATH):
    # The trained transfer-learning model has a Lambda(preprocess_input) layer.
    # Keras 3 needs both safe_mode=False and an explicit custom_objects entry
    # for the Lambda's wrapped function, since the saved config records
    # `preprocess_input` without its module path.
    from keras.applications.mobilenet_v2 import preprocess_input
    return keras_load_model(
        path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
    )


if __name__ == "__main__":
    model = build_model(num_classes=len(DEFAULT_LABELS))
    model.summary()
    save(model)
    print(f"CNN model saved to {MODEL_PATH}")
