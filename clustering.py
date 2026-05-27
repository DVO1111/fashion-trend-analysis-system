"""K-Means clustering of fashion engagement records into style communities."""

import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/clustering_model.pkl"
ENGAGEMENT_CSV = "dataset/engagement.csv"
FEATURES = ["likes", "comments", "shares"]


def fit(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    X = df[FEATURES].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(Xs)
    return {"scaler": scaler, "model": km}


def predict(bundle, df: pd.DataFrame):
    Xs = bundle["scaler"].transform(df[FEATURES].values)
    return bundle["model"].predict(Xs)


def save(bundle, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load(path: str = MODEL_PATH):
    return joblib.load(path)


def main():
    df = pd.read_csv(ENGAGEMENT_CSV)
    bundle = fit(df, n_clusters=min(4, len(df)))
    labels = predict(bundle, df)
    df_out = df.copy()
    df_out["cluster"] = labels
    print(df_out)
    save(bundle)
    print(f"\nClustering model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
