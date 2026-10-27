# jobs/config.py
from pathlib import Path
import numpy as np

DWH = "postgresql+psycopg2://admin:admin@postgres-dwh:5432/dwh"
MODELS = Path("/opt/airflow/models")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

# бизнес-константы
CALL_COST = 2
DEPOSIT_VALUE = 10

# модель
SEED = 42
CATEGORICAL = [
    "job", "marital", "education", "credit_default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome",
]
NUMERIC = [
    "age", "campaign", "never_contacted", "previous",
    "cons_price_idx", "cons_conf_idx", "nr_employed",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "subscribed"

# зафиксировано по результатам ноутбука 02
LOGREG_PARAMS = {"clf__C": 0.1}
FOREST_PARAMS = {"clf__max_depth": 16, "clf__min_samples_leaf": 20,
                 "clf__max_features": "sqrt"}

# единая сетка порогов для choose_threshold и threshold_curve
THRESHOLD_GRID = np.round(np.arange(0.01, 0.81, 0.01), 2)
