from dataclasses import dataclass
import pandas as pd


@dataclass
class ValDaySplit:
    forecast_date: pd.Timestamp
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val_day: pd.DataFrame
    y_val_day: pd.Series


@dataclass
class FinalTrainPredictSplit:
    test_day: pd.Timestamp
    X_fit: pd.DataFrame       # train + validation (all before test day)
    y_fit: pd.Series
    X_test: pd.DataFrame      # chosen test day
    y_test: pd.Series | None  # optional if available


def _day_bounds(day: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(day).normalize()
    end = start + pd.Timedelta(days=1)
    return start, end


def _slice_day(df: pd.DataFrame, day: str | pd.Timestamp) -> pd.DataFrame:
    start, end = _day_bounds(day)
    return df.loc[(df.index >= start) & (df.index < end)]


def build_train_val_test_windows(
    df: pd.DataFrame,
    *,
    test_day: str,
    history_days: int = 180,
    val_days: int = 14,
    target_col: str = "consumption",
    feature_cols: list[str],
):
    """
    Build:
      1) validation walk-forward splits across the 14 days before test_day
      2) final fit + predict split (fit on all data before test_day, predict test_day)
    """
    data = df.sort_index()
    test_day_ts = pd.Timestamp(test_day).normalize()

    val_end = test_day_ts                          # exclusive
    val_start = val_end - pd.Timedelta(days=val_days)

    train_end = val_start                          # exclusive
    train_start = train_end - pd.Timedelta(days=history_days)

    # Base windows
    train_base = data.loc[(data.index >= train_start) & (data.index < train_end)]
    val_window = data.loc[(data.index >= val_start) & (data.index < val_end)]
    test_day_df = _slice_day(data, test_day_ts)

    if train_base.empty:
        raise ValueError("Training window is empty.")
    if val_window.empty:
        raise ValueError("Validation window is empty.")
    if test_day_df.empty:
        raise ValueError(f"Test day {test_day} has no rows.")

    # Build walk-forward validation splits (one split per validation day)
    val_splits: list[ValDaySplit] = []
    for i in range(val_days):
        day_i = val_start + pd.Timedelta(days=i)

        train_i = data.loc[(data.index >= train_start) & (data.index < day_i)]
        val_day_i = _slice_day(data, day_i)

        if train_i.empty or val_day_i.empty:
            raise ValueError(f"Invalid val split for day {day_i.date()} (empty train/val)")

        val_splits.append(
            ValDaySplit(
                forecast_date=day_i,
                X_train=train_i[feature_cols],
                y_train=train_i[target_col],
                X_val_day=val_day_i[feature_cols],
                y_val_day=val_day_i[target_col],
            )
        )

    # Final fit split: all available data before test day (train + val)
    fit_df = data.loc[(data.index >= train_start) & (data.index < test_day_ts)]

    final_split = FinalTrainPredictSplit(
        test_day=test_day_ts,
        X_fit=fit_df[feature_cols],
        y_fit=fit_df[target_col],
        X_test=test_day_df[feature_cols],
        y_test=test_day_df[target_col] if target_col in test_day_df.columns else None,
    )

    meta = {
        "train_start": train_start,
        "train_end_exclusive": train_end,
        "val_start": val_start,
        "val_end_exclusive": val_end,
        "test_day": test_day_ts,
        "n_train_base": len(train_base),
        "n_val_window": len(val_window),
        "n_fit_final": len(fit_df),
        "n_test": len(test_day_df),
    }

    return val_splits, final_split, meta