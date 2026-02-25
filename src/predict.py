from pathlib import Path
import json
import joblib
import pandas as pd

from .preprocess import load_raw_data, preprocess_data, cutoff_last_complete_day
from .features import add_time_features
from .metrics import mae, rmse, smape
from .utils import log


def predict_test_day(
    *,
    model_dir: str,
    csv_path: str,
    location: str,
    test_day: str,
    last_complete_day: str,
    starting_date: str | None = None,
    target_col: str = "consumption",
    temp_col: str = "temperature",
    time_col: str = "time",
    location_col: str = "location",
):
    model_dir = Path(model_dir)

    model = joblib.load(model_dir / "model.joblib")
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    delay_days = metadata["delay_days"]

    # rebuild feature table exactly as in training
    raw = load_raw_data(csv_path)
    df_loc = preprocess_data(
        raw,
        city=location,
        time_col=time_col,
        location_col=location_col,
        starting_date=starting_date,
    )
    df_loc = cutoff_last_complete_day(df_loc, last_complete_day=last_complete_day)

    data = add_time_features(
        df_loc,
        target_col=target_col,
        temp_col=temp_col,
        delay_days=delay_days,
        drop_cols=["time", "location", "datetime"],
    )

    day_start = pd.Timestamp(test_day).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    day_df = data.loc[(data.index >= day_start) & (data.index < day_end)].copy()

    if day_df.empty:
        raise ValueError(f"No rows found for test_day={test_day}")

    missing = [c for c in feature_cols if c not in day_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns at prediction time: {missing}")

    X_test = day_df[feature_cols]
    y_pred = model.predict(X_test)

    pred_df = pd.DataFrame({
        "timestamp": day_df.index,
        "y_pred": y_pred,
    })

    # If true target exists, attach and evaluate
    if target_col in day_df.columns:
        pred_df["y_true"] = day_df[target_col].values
        metrics = {
            "MAE": mae(pred_df["y_true"], pred_df["y_pred"]),
            "RMSE": rmse(pred_df["y_true"], pred_df["y_pred"]),
            "SMAPE_%": smape(pred_df["y_true"], pred_df["y_pred"]),
        }
        log(f"Prediction metrics for {test_day}: {metrics}")
    else:
        metrics = None
        log(f"Predicted {len(pred_df)} rows for {test_day} (no ground truth available).")

    return pred_df, metrics