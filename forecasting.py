"""
PV output forecasting — optional extension.

Trains a small RandomForestRegressor to predict next-hour PV output from
the last few hours plus time-of-day, using a completed simulation run's
own PV series as training data. This is a forecasting demo, not a
production forecaster — it's here to show the ML-extension path without
overreaching on claims.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def build_training_data(pv_series: pd.Series, lookback: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from a PV series: each row of X is [pv(t-1), ..., pv(t-lookback),
    hour_of_day(t)], and y is pv(t). Rows before `lookback` hours of history
    are dropped since they don't have enough lag features yet.
    """
    values = pv_series.values
    hours_of_day = np.arange(len(values)) % 24

    X, y = [], []
    for t in range(lookback, len(values)):
        lags = values[t - lookback:t][::-1]  # most recent lag first
        X.append(np.concatenate([lags, [hours_of_day[t]]]))
        y.append(values[t])

    return np.array(X), np.array(y)


def train_forecast_model(pv_series: pd.Series, lookback: int = 3,
                          test_size: float = 0.3, seed: int = 42):
    """
    Train a RandomForestRegressor on the given PV series.

    Returns (model, X_test, y_test, y_pred, r2) so the caller can both
    reuse the model and immediately show how well it did.
    """
    X, y = build_training_data(pv_series, lookback=lookback)

    if len(X) < 10:
        raise ValueError(
            "Not enough data points to train a forecast model — "
            "run the simulation for more days first."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False,  # keep time order intact
    )

    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, r2


def forecast_next_hour(model, recent_pv_values: list[float], hour_of_day: int) -> float:
    """
    Predict the next hour's PV output given the most recent lag values
    (most recent first, length must match the model's lookback) and the
    hour-of-day for the hour being predicted.
    """
    features = np.array([list(recent_pv_values) + [hour_of_day]])
    return float(model.predict(features)[0])


if __name__ == "__main__":
    # Quick smoke test: train on a standalone synthetic PV profile
    from simulation import generate_pv_profile

    pv = generate_pv_profile(hours=24 * 14, pv_capacity_kw=10)
    model, X_test, y_test, y_pred, r2 = train_forecast_model(pv)
    print(f"Trained on {len(pv)} hours of PV data. Test R^2 = {r2:.3f}")
