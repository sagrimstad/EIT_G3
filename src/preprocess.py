import pandas as pd


def load_raw_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def preprocess_data(
    df: pd.DataFrame,
    *,
    city: str,
    time_col: str = "time",
    location_col: str = "location",
    starting_date: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=["datetime"])

    mask = out[location_col].astype(str).str.lower() == city.lower()
    out = out.loc[mask].copy()

    out = out.sort_values("datetime").set_index("datetime")

    if starting_date is not None:
        out = out.loc[out.index >= pd.Timestamp(starting_date)]

    return out


def cutoff_last_complete_day(df: pd.DataFrame, *, last_complete_day: str) -> pd.DataFrame:
    """
    Keep rows up to and including last_complete_day 23:59:59...
    """
    end_ts = pd.Timestamp(last_complete_day) + pd.Timedelta(days=1)
    return df.loc[df.index < end_ts].copy()