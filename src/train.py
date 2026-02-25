from pathlib import Path
import json
import joblib
import pandas as pd

from .utils import log
from .preprocessing import load_raw_data, preprocess_data, cutoff_last_complete_day
from .features import add_time_features
from .splits import build_train_val_test_windows
from .tuning import grid_then_refine_search
from .modeling import fit_final_and_predict  # we use only final fit part here


def train_for_test_day(
    *,
    csv_path: str,
    location: str,
    test_day: str,
    last_complete_day: str,
    starting_date: str | None = None,
    target_col: str = "consumption",
    temp_col: str = "temperature",
    time_col: str = "time",
    location_col: str = "location",
    history_days: int = 180,
    val_days: int = 14,
    delay_days: int = 5,
    metric_name: str = "mae",
    top_k: int = 3,
    early_stopping_rounds: int = 100,
    outdir: str = "artifacts/run",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load + preprocess
    raw = load_raw_data(csv_path)
    df_loc = preprocess_data(
        raw,
        city=location,
        time_col=time_col,
        location_col=location_col,
        starting_date=starting_date,
    )
    df_loc = cutoff_last_complete_day(df_loc, last_complete_day=last_complete_day)

    # 2) Features
    data = add_time_features(
        df_loc,
        target_col=target_col,
        temp_col=temp_col,
        delay_days=delay_days,
        drop_cols=["time", "location", "datetime"],
    )

    feature_cols = [c for c in data.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No feature columns found after feature engineering.")

    # 3) Build splits for this chosen test day
    val_splits, final_split, meta = build_train_val_test_windows(
        data,
        test_day=test_day,
        history_days=history_days,
        val_days=val_days,
        target_col=target_col,
        feature_cols=feature_cols,
    )

    log(f"Prepared validation splits: {len(val_splits)} days")
    log(f"Validation: {meta['val_start'].date()} -> {(meta['val_end_exclusive'] - pd.Timedelta(days=1)).date()}")
    log(f"Test day: {meta['test_day'].date()}")

    # 4) Tune on validation window
    best_params, tune_results = grid_then_refine_search(
        val_splits,
        metric_name=metric_name,
        top_k=top_k,
        early_stopping_rounds=early_stopping_rounds,
    )

    # 5) Retrain final model on ALL data before test day (train + val)
    # We reuse fit_final_and_predict and ignore predictions here
    _, final_model = fit_final_and_predict(final_split, params=best_params)

    # 6) Save artifacts
    joblib.dump(final_model, outdir / "model.joblib")
    tune_results.to_csv(outdir / "tune_results.csv", index=False)

    metadata = {
        "location": location,
        "test_day": str(pd.Timestamp(test_day).date()),
        "last_complete_day": str(pd.Timestamp(last_complete_day).date()),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "history_days": history_days,
        "val_days": val_days,
        "delay_days": delay_days,
        "best_params": best_params,
        "split_meta": {k: str(v) if hasattr(v, "isoformat") else v for k, v in meta.items()},
    }

    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save the exact final fit data schema columns (defensive)
    pd.Series(feature_cols, name="feature").to_csv(outdir / "feature_cols.csv", index=False)

    log(f"Saved model + metadata to {outdir}")

    return {
        "model": final_model,
        "best_params": best_params,
        "tune_results": tune_results,
        "metadata": metadata,
        "data": data,  # optional convenience
    }