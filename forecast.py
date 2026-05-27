"""LSTM trend forecasting on time-series engagement data."""

import os
import numpy as np
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import LSTM, Dense

MODEL_PATH = "models/lstm_model.h5"
WINDOW = 10


def build_model(window: int = WINDOW) -> Sequential:
    model = Sequential([
        LSTM(50, input_shape=(window, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def make_windows(series, window: int = WINDOW):
    series = np.asarray(series, dtype="float32")
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def forecast_next(model, recent_window) -> float:
    arr = np.asarray(recent_window, dtype="float32").reshape(1, -1, 1)
    return float(model.predict(arr, verbose=0)[0, 0])


def trend_direction(history) -> str:
    h = np.asarray(history, dtype="float32")
    if len(h) < 2:
        return "INSUFFICIENT_DATA"
    slope = np.polyfit(np.arange(len(h)), h, 1)[0]
    if slope > 0:
        return "INCREASING POPULARITY"
    if slope < 0:
        return "DECLINING POPULARITY"
    return "STABLE"


def save(model, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def load(path: str = MODEL_PATH):
    return keras_load_model(path)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    months = 24
    series = 100 + np.arange(months) * 8 + rng.normal(0, 10, months)

    model = build_model()
    X, y = make_windows(series)
    model.fit(X, y, epochs=15, verbose=0)

    next_val = forecast_next(model, series[-WINDOW:])
    direction = trend_direction(series[-6:])
    print(f"Next-month engagement forecast: {next_val:.1f}")
    print(f"Trend direction (last 6 months): {direction}")

    save(model)
    print(f"LSTM model saved to {MODEL_PATH}")
