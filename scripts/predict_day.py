#!/usr/bin/env python3
import argparse
from pathlib import Path
import json

from src.predict import predict_test_day

"""
example usage:
python -m scripts.predict_day \
  --model-dir artifacts/bergen_2023-04-01 \
  --csv data/consumption_temp.csv \
  --location Bergen \
  --test-day 2023-04-01 \
  --last-complete-day 2023-04-01 \
  --out outputs/bergen_2023-04-01_predictions.csv
"""


def main():
    p = argparse.ArgumentParser(description="Predict a single day using a trained model")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--location", required=True)
    p.add_argument("--test-day", required=True)
    p.add_argument("--last-complete-day", required=True)
    p.add_argument("--out", required=True, help="Output CSV for predictions")

    p.add_argument("--starting-date", default=None)
    p.add_argument("--target-col", default="consumption")
    p.add_argument("--temp-col", default="temperature")
    p.add_argument("--time-col", default="time")
    p.add_argument("--location-col", default="location")
    args = p.parse_args()

    pred_df, metrics = predict_test_day(
        model_dir=args.model_dir,
        csv_path=args.csv,
        location=args.location,
        test_day=args.test_day,
        last_complete_day=args.last_complete_day,
        starting_date=args.starting_date,
        target_col=args.target_col,
        temp_col=args.temp_col,
        time_col=args.time_col,
        location_col=args.location_col,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)

    if metrics is not None:
        with open(out_path.with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()