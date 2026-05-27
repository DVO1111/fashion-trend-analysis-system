"""Social media fashion data collector.

Real scraping of Instagram/TikTok/X requires authenticated APIs and is bound
by each platform's ToS. This module ships a mock collector that produces a
realistic CSV so the downstream pipeline can be exercised end-to-end. Swap
the `collect_mock` body for real Selenium/API calls when credentials are
available.
"""

import os
import random
import pandas as pd

HASHTAGS = [
    "#CampusStyle",
    "#AlteFashion",
    "#AnkaraStyle",
    "#LagosFashion",
    "#NaijaStyle",
    "#StudentDrip",
    "#CorporateChic",
    "#MinimalStyle",
]

CAPTION_TEMPLATES = [
    "Campus fashion inspiration {tag}",
    "Loving the {tag} energy today",
    "Owambe ready {tag}",
    "Lecture-day fit {tag}",
    "Trying the {tag} aesthetic this week",
]


def collect_mock(n_per_tag: int = 25, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for tag in HASHTAGS:
        for _ in range(n_per_tag):
            rows.append({
                "hashtag": tag,
                "caption": rng.choice(CAPTION_TEMPLATES).format(tag=tag),
                "likes": rng.randint(40, 900),
                "comments": rng.randint(2, 150),
                "shares": rng.randint(0, 80),
            })
    return pd.DataFrame(rows)


def main():
    out_dir = "dataset"
    os.makedirs(out_dir, exist_ok=True)
    df = collect_mock()
    out_path = os.path.join(out_dir, "fashion_posts.csv")
    df.to_csv(out_path, index=False)
    print(f"Fashion data collected: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
