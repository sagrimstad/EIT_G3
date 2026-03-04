import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    *,
    target_col: str = "consumption",
    temp_col: str | None = "temperature",
    delay_days: int = 5,
    drop_cols: list[str] | None = None,
    cons_lags: tuple[int, ...] = (120, 168),
    cons_rolling_windows: tuple[int, ...] = (168, 336),
    temp_lags: tuple[int, ...] = (24, 168),
    temp_rolling_windows: tuple[int, ...] = (24, 168),
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    out = df.copy()
    delay_h = delay_days * 24

    # Calendar primitives (used to build cyclical features)
    hour = out.index.hour
    dow = out.index.dayofweek
    month = out.index.month
    woy = out.index.isocalendar().week.astype(int)

    out["is_weekend"] = (dow >= 5).astype(int)

    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    out["woy_sin"] = np.sin(2 * np.pi * (woy - 1) / 52.0)
    out["woy_cos"] = np.cos(2 * np.pi * (woy - 1) / 52.0)

    # Temperature features (always available)
    if temp_col is not None and temp_col in out.columns:
        out[temp_col] = pd.to_numeric(out[temp_col], errors="coerce")

        for k in temp_lags:
            out[f"temp_lag_{k}"] = out[temp_col].shift(k)

        shifted_temp = out[temp_col].shift(1)
        for w in temp_rolling_windows:
            out[f"temp_roll_mean_{w}"] = shifted_temp.rolling(w, min_periods=w).mean()
            out[f"temp_roll_std_{w}"] = shifted_temp.rolling(w, min_periods=w).std()

        out["heating_degree_18"] = np.maximum(0.0, 18.0 - out[temp_col])

    # Delayed consumption features (must respect 5-day delay)
    if target_col not in out.columns:
        raise KeyError(f"Missing target column: {target_col}")

    for k in cons_lags:
        if k < delay_h:
            raise ValueError(f"consumption lag {k}h violates delay constraint ({delay_h}h)")
        out[f"lag_{k}"] = out[target_col].shift(k)

    avail = out[target_col].shift(delay_h)
    for w in cons_rolling_windows:
        out[f"avail_roll_mean_{w}"] = avail.rolling(w, min_periods=w).mean()
        out[f"avail_roll_std_{w}"] = avail.rolling(w, min_periods=w).std()

    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    # Drop rows created by lag/rolling NaNs
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    return out