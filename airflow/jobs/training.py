# jobs/training.py
import numpy as np
from collections import defaultdict
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CALL_COST, CATEGORICAL, DEPOSIT_VALUE, FEATURES,
    FOREST_PARAMS, LOGREG_PARAMS, NUMERIC, SEED, THRESHOLD_GRID
)

# пайплайн
def make_pipeline(model):
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("pre", preprocessor), ("clf", model)])

def build_model(model_name: str):
    if model_name == "logreg":
        params = {k.removeprefix("clf__"): v for k, v in LOGREG_PARAMS.items()}
        return make_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED, **params),
        )
    if model_name == "random_forest":
        params = {k.removeprefix("clf__"): v for k, v in FOREST_PARAMS.items()}
        return make_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **params),
        )


# OOF и фолды
def fold_ids(cv, x, y):
    folds = np.empty(len(y), dtype=int)
    for fold, (_, test_idx) in enumerate(cv.split(x, y)):
        folds[test_idx] = fold
    return folds


def oof_proba(estimator, x, y, cv):
    matrix = cross_val_predict(clone(estimator), x, y, cv=cv, method="predict_proba", n_jobs=-1)
    classes = [int(c) for c in estimator.classes_]
    return matrix[:, classes.index(1)]


# бизнес-метрика
def campaign_profit(y, proba, threshold: float):
    call = proba >= threshold
    tp = int((call & (y == 1)).sum())
    return DEPOSIT_VALUE * tp - CALL_COST * int(call.sum())


def choose_threshold(y, proba, folds):
    best_t = THRESHOLD_GRID[0]
    best_p = -np.inf
    for t in THRESHOLD_GRID:
        profits = [
            campaign_profit(y[folds == f], proba[folds == f], t)
            for f in np.unique(folds)
        ]
        mean_profit = float(np.mean(profits))
        if mean_profit >= best_p :
            best_t, best_p = float(t), mean_profit
    return best_t, best_p


# метрики
def metrics_at(y, proba, folds, threshold, with_std: bool = False):
    """Средние метрики по фолдам OOF.
    При with_std=True — ещё и std (ключи с суффиксом '_std').
    """
    values = defaultdict(list)

    for fold in np.unique(folds):
        in_fold = folds == fold
        y_fold = y[in_fold]
        proba_fold = proba[in_fold]
        pred_fold = (proba_fold >= threshold).astype(int)

        values["profit"].append(campaign_profit(y_fold, proba_fold, threshold))
        values["roc_auc"].append(roc_auc_score(y_fold, proba_fold))
        values["pr_auc"].append(average_precision_score(y_fold, proba_fold))
        values["precision"].append(precision_score(y_fold, pred_fold, zero_division=0))
        values["recall"].append(recall_score(y_fold, pred_fold, zero_division=0))
        values["accuracy"].append(accuracy_score(y_fold, pred_fold))

    result = {name: float(np.mean(v)) for name, v in values.items()}
    if with_std:
        result.update({f"{name}_std": float(np.std(v)) for name, v in values.items()})
    return result

def fit_one(name, x, y, cv, folds):
    """Обучает одну модель, считает OOF и порог."""
    model = build_model(name)
    model.fit(x, y)
    oof = oof_proba(model, x, y, cv)
    threshold, _ = choose_threshold(y, oof, folds)
    return model, oof, threshold