"""Train the fashion CNN on a labelled image folder.

Expected layout:

    dataset/images/
        ankara_fusion/
            img001.jpg
            img002.jpg
            ...
        streetwear/
            ...
        casual/
        alte/
        corporate_chic/

Run:
    python train_image_model.py                   # transfer learning (MobileNetV2)
    python train_image_model.py --simple-cnn      # train the spec's plain CNN

Outputs:
    models/cnn_model.h5         trained weights, loadable via image_model.load()
    models/cnn_labels.json      class index -> label mapping
    models/cnn_report.txt       accuracy + per-class precision/recall +
                                confusion matrix (paste into thesis)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
DATA_DIR = "dataset/images"
MODEL_PATH = "models/cnn_model.h5"
LABELS_PATH = "models/cnn_labels.json"
REPORT_PATH = "models/cnn_report.txt"


def load_datasets(batch_size: int, val_split: float, seed: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )
    # NOTE: both subsets must use shuffle=True with the SAME seed. With
    # shuffle=False the split is taken from the unshuffled file list, which
    # is grouped by class folder — that produces a single-class val set.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )
    return train_ds, val_ds


def build_transfer_model(num_classes: int) -> models.Model:
    base = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.Lambda(preprocess_input)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_simple_cnn(num_classes: int) -> models.Model:
    """The architecture from the original project spec."""
    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=IMG_SIZE + (3,)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def augment(ds):
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    return ds.map(lambda x, y: (aug(x, training=True), y),
                  num_parallel_calls=tf.data.AUTOTUNE)


def evaluate(model, val_ds, labels) -> str:
    y_true_idx, y_pred_idx = [], []
    for x_batch, y_batch in val_ds:
        probs = model.predict(x_batch, verbose=0)
        y_true_idx.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred_idx.extend(np.argmax(probs, axis=1))
    y_true_idx = np.array(y_true_idx)
    y_pred_idx = np.array(y_pred_idx)

    acc = float((y_true_idx == y_pred_idx).mean())
    report = classification_report(
        y_true_idx, y_pred_idx,
        target_names=labels, digits=3, zero_division=0,
    )
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(labels))))

    lines = [
        f"Overall validation accuracy: {acc:.3%}",
        "",
        "Per-class precision / recall / F1:",
        report,
        "",
        "Confusion matrix (rows = true, cols = predicted):",
        "labels: " + ", ".join(labels),
        np.array2string(cm),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simple-cnn", action="store_true",
                        help="Use the plain 2-layer CNN from the spec instead of MobileNetV2")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    args = parser.parse_args()

    if not Path(DATA_DIR).is_dir():
        raise SystemExit(
            f"{DATA_DIR} does not exist. Create one subfolder per fashion "
            "category and drop labelled images inside."
        )

    train_ds, val_ds = load_datasets(args.batch_size, args.val_split, args.seed)
    labels = train_ds.class_names
    num_classes = len(labels)

    train_count = sum(1 for _ in train_ds.unbatch())
    val_count = sum(1 for _ in val_ds.unbatch())
    print(f"Classes ({num_classes}): {labels}")
    print(f"Train images: {train_count}    Val images: {val_count}")

    if not args.no_augment:
        train_ds = augment(train_ds)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds_cached = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    builder = build_simple_cnn if args.simple_cnn else build_transfer_model
    model = builder(num_classes)
    model.summary()

    model.fit(train_ds, validation_data=val_ds_cached, epochs=args.epochs)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    report = evaluate(model, val_ds, labels)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nSaved: {MODEL_PATH}\nSaved: {LABELS_PATH}\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
