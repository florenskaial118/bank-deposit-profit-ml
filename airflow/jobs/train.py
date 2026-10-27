# jobs/train.py

import joblib
import pandas as pd
from sqlalchemy import create_engine

from .config import FEATURES, SEED, DWH, MODELS
from .training import fold_ids, metrics_at, fit_one
from sklearn.model_selection import StratifiedKFold

def train():
    engine = create_engine(DWH)
    data = pd.read_sql("SELECT * FROM mart_campaign_features ORDER BY source_row", engine)
    x, y = data[FEATURES], data["subscribed"].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    
    folds = fold_ids(cv, x, y)
    for name in ("logreg", "random_forest"):
        model, oof, threshold = fit_one(name, x, y, cv, folds)
        results[name] = {"model": model, "oof": oof, "threshold": threshold}

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_name": "logreg", "pipeline": results["logreg"]["model"],
                "threshold": results["logreg"]["threshold"]},
                MODELS / "campaign_model.joblib")
    joblib.dump({"model_name": "random_forest", "pipeline": results["random_forest"]["model"],
                "threshold": results["random_forest"]["threshold"]},
                MODELS / "tree_model.joblib")

    report = pd.DataFrame([
    {"model_name": name,
     "threshold": r["threshold"],
     **metrics_at(y, r["oof"], folds, r["threshold"])}
    for name, r in results.items()
    ])
    report.to_sql("mart_model_metrics", engine, if_exists="replace", index=False)
    print(report.to_string(index=False))