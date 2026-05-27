"""Image preprocessing: resize all images in dataset/images/ to 224x224."""

import os
import cv2

IMAGE_DIR = "dataset/images"
TARGET_SIZE = (224, 224)
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_dir(image_dir: str = IMAGE_DIR) -> int:
    if not os.path.isdir(image_dir):
        print(f"Image folder not found: {image_dir}")
        return 0

    processed = 0
    for name in os.listdir(image_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext not in VALID_EXT:
            continue

        path = os.path.join(image_dir, name)
        image = cv2.imread(path)
        if image is None:
            print(f"Skipped unreadable file: {name}")
            continue

        resized = cv2.resize(image, TARGET_SIZE)
        cv2.imwrite(path, resized)
        processed += 1

    print(f"Image preprocessing completed: {processed} files")
    return processed


if __name__ == "__main__":
    preprocess_dir()
