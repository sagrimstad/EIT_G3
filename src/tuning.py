import itertools
import pandas as pd

from .metrics import METRICS
from .modeling import fit_for_validation
from .utils import log


COARSE_SPACE = {
    "learning_rate": [0.03, 0.05],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 50, 100],
    "reg_lambda": [0.0, 1e-2, 1e-1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_alpha": [0.0],
}


def _grid_product(space: dict) -> list[dict]:
    keys = list(space.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[space[k] for k in keys])]


def score_params_on_val_splits(
    val_splits: list,
    *,
    params: dict,
    metric_name: str = "mae",
    early_stopping_rounds: int = 100,
    log_every: int = 0,
) -> float:
    metric_fn = METRICS[metric_name]
    scores = []

    for i, split in enumerate(val_splits, start=1):
        y_pred, _ = fit_for_validation(split, params=params, early_stopping_rounds=early_stopping_rounds)
        s = metric_fn(split.y_val_day.values, y_pred)
        scores.append(s)

        if log_every and (i == 1 or i == len(val_splits) or i % log_every == 0):
            log(f"  VAL {i:>2}/{len(val_splits)} day={split.forecast_date.date()} {metric_name.upper()}={s:.4f}")

    return float(sum(scores) / len(scores))


def _refine_space(base_params: dict) -> dict:
    lr = float(base_params["learning_rate"])
    leaves = int(base_params["num_leaves"])
    mcs = int(base_params["min_child_samples"])
    rl = float(base_params["reg_lambda"])

    def two(v, alt, lo, hi):
        alt = max(lo, min(hi, alt))
        return sorted(set([v, alt]))

    return {
        "learning_rate": two(lr, lr * 0.8, 0.005, 0.2),
        "num_leaves": two(leaves, leaves + 16, 15, 255),
        "min_child_samples": two(mcs, mcs + 20, 5, 300),
        "reg_lambda": two(rl, (rl * 10 if rl > 0 else 0.1), 0.0, 10.0),
        "subsample": [base_params["subsample"]],
        "colsample_bytree": [base_params["colsample_bytree"]],
        "reg_alpha": [base_params["reg_alpha"]],
    }


def grid_then_refine_search(
    val_splits: list,
    *,
    metric_name: str = "mae",
    top_k: int = 3,
    early_stopping_rounds: int = 100,
    log_every: int = 10,
) -> tuple[dict, pd.DataFrame]:
    # Stage A: coarse
    coarse_candidates = _grid_product(COARSE_SPACE)
    log(f"Stage A (coarse): {len(coarse_candidates)} configs")

    coarse_rows = []
    for i, params in enumerate(coarse_candidates, start=1):
        if i == 1 or i == len(coarse_candidates) or (log_every and i % log_every == 0):
            log(f"  coarse {i}/{len(coarse_candidates)} params={params}")

        score = score_params_on_val_splits(
            val_splits,
            params=params,
            metric_name=metric_name,
            early_stopping_rounds=early_stopping_rounds,
            log_every=0,
        )
        coarse_rows.append({"stage": "coarse", "score": score, **params})

    coarse_df = pd.DataFrame(coarse_rows).sort_values("score").reset_index(drop=True)
    log(f"Best coarse score: {coarse_df.loc[0, 'score']:.4f}")

    # Stage B: refine top_k
    top_candidates = coarse_df.head(top_k).to_dict(orient="records")
    refine_rows = []
    log(f"Stage B (refine): top_k={top_k}")

    for row in top_candidates:
        base = {k: row[k] for k in COARSE_SPACE.keys()}
        refine_candidates = _grid_product(_refine_space(base))
        log(f"  refine around {base} -> {len(refine_candidates)} configs")

        for params in refine_candidates:
            score = score_params_on_val_splits(
                val_splits,
                params=params,
                metric_name=metric_name,
                early_stopping_rounds=early_stopping_rounds,
                log_every=0,
            )
            refine_rows.append({"stage": "refine", "score": score, **params})

    refine_df = pd.DataFrame(refine_rows).sort_values("score").reset_index(drop=True)
    results_df = pd.concat([coarse_df, refine_df], ignore_index=True).sort_values("score").reset_index(drop=True)

    best_params = results_df.iloc[0].drop(labels=["stage", "score"]).to_dict()
    log(f"Best overall score: {results_df.loc[0, 'score']:.4f}")
    log(f"Best params: {best_params}")

    return best_params, results_df