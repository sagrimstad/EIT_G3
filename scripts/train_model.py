#!/usr/bin/env python3
import argparse
from src.train import train_for_test_day

"""
example usage:
python scripts/train_model.py \
  --csv data/consumption_temp.csv \
  --location Bergen \
  --test-day 2023-04-01 \
  --last-complete-day 2023-04-01 \
  --history-days 180 \
  --val-days 14 \
  --outdir artifacts/bergen_2023-04-01
"""


def main():
    p = argparse.ArgumentParser(description="Tune + train final model for ONE chosen day-ahead test day")
    p.add_argument("--csv", required=True)
    p.add_argument("--location", required=True)
    p.add_argument("--test-day", required=True, help="Day to predict, e.g. 2023-04-01")
    p.add_argument("--last-complete-day", required=True, help="Latest complete day in dataset")
    p.add_argument("--outdir", required=True)

    p.add_argument("--starting-date", default=None)
    p.add_argument("--target-col", default="consumption")
    p.add_argument("--temp-col", default="temperature")
    p.add_argument("--time-col", default="time")
    p.add_argument("--location-col", default="location")

    p.add_argument("--history-days", type=int, default=180)
    p.add_argument("--val-days", type=int, default=14)
    p.add_argument("--delay-days", type=int, default=5)

    p.add_argument("--metric", default="mae", choices=["mae", "rmse", "smape"])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--early-stopping-rounds", type=int, default=5000)

    args = p.parse_args()

    train_for_test_day(
        csv_path=args.csv,
        location=args.location,
        test_day=args.test_day,
        last_complete_day=args.last_complete_day,
        starting_date=args.starting_date,
        target_col=args.target_col,
        temp_col=args.temp_col,
        time_col=args.time_col,
        location_col=args.location_col,
        history_days=args.history_days,
        val_days=args.val_days,
        delay_days=args.delay_days,
        metric_name=args.metric,
        top_k=args.top_k,
        early_stopping_rounds=args.early_stopping_rounds,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()