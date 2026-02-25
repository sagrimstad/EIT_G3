import lightgbm as lgb
import pandas as pd


def make_lgbm_model(params: dict | None = None) -> lgb.LGBMRegressor:
    base_params = dict(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    if params:
        base_params.update(params)
    return lgb.LGBMRegressor(**base_params)


def fit_for_validation(split, params: dict, early_stopping_rounds: int = 100):
    """
    Fit on split.X_train/y_train and predict the validation day.
    """
    model = make_lgbm_model(params)
    model.fit(
        split.X_train,
        split.y_train,
        eval_set=[(split.X_val_day, split.y_val_day)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )
    y_pred = model.predict(split.X_val_day)
    return y_pred, model


def fit_final_and_predict(final_split, params: dict):
    """
    Fit on all available pre-test data and predict the chosen test day.
    """
    model = make_lgbm_model(params)
    model.fit(final_split.X_fit, final_split.y_fit)
    y_pred = model.predict(final_split.X_test)
    return y_pred, model