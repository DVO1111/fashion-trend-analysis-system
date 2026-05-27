"""Download a public fashion image dataset and arrange it into the
per-class folder layout expected by train_image_model.py.

Dataset: ashraq/fashion-product-images-small (44k items, originally from
Myntra / Kaggle). We use the `usage` attribute to derive class labels:

    Casual  -> casual
    Sports  -> streetwear
    Ethnic  -> ankara_fusion
    Formal  -> corporate_chic

The 5th category from the project spec (alté) has no clean equivalent in
public fashion datasets and is left for Phase 2 (Lagos-specific data
collection).

Usage:
    python download_dataset.py                 # 1500 images per class
    python download_dataset.py --per-class 500 # smaller, faster
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DATASET = "ashraq/fashion-product-images-small"
OUT_DIR = Path("dataset/images")

USAGE_TO_CLASS = {
    "Casual": "casual",
    "Sports": "streetwear",
    "Ethnic": "ankara_fusion",
    "Formal": "corporate_chic",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1500,
                        help="Max images per class (default 1500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET, split="train")
    print(f"Loaded {len(ds)} items")

    rng = random.Random(args.seed)
    indices_by_class = defaultdict(list)
    for i, usage in enumerate(ds["usage"]):
        target = USAGE_TO_CLASS.get(usage)
        if target is not None:
            indices_by_class[target].append(i)

    for cls, idxs in indices_by_class.items():
        rng.shuffle(idxs)
        indices_by_class[cls] = idxs[: args.per_class]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for cls, idxs in indices_by_class.items():
        cls_dir = OUT_DIR / cls
        cls_dir.mkdir(exist_ok=True)
        for j, i in enumerate(idxs):
            row = ds[i]
            img = row["image"].convert("RGB")
            img.save(cls_dir / f"{row['id']}.jpg", "JPEG", quality=85)
            if (j + 1) % 200 == 0:
                print(f"  {cls}: {j + 1}/{len(idxs)}")
        print(f"Saved {len(idxs)} images to {cls_dir}")
        total_saved += len(idxs)

    print(f"\nDone: {total_saved} images across {len(indices_by_class)} classes")
    print("Next: python train_image_model.py")


if __name__ == "__main__":
    main()
