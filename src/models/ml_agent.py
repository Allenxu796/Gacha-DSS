from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


@dataclass
class TrainResult:
    model_name: str
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    model: object


def _split(df: pd.DataFrame, label_col: str, test_size: float, seed: int):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def _metrics(y_true, y_prob) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def train_random_forest(df: pd.DataFrame, label_col: str, seed: int = 42) -> TrainResult:
    X_train, X_test, y_train, y_test = _split(df, label_col, test_size=0.2, seed=seed)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=seed,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test.values, y_prob)

    importances = dict(zip(X_train.columns.tolist(), model.feature_importances_.tolist()))

    return TrainResult("random_forest", metrics, importances, model)


def train_gbdt(df: pd.DataFrame, label_col: str, seed: int = 42) -> TrainResult:
    X_train, X_test, y_train, y_test = _split(df, label_col, test_size=0.2, seed=seed)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=3,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test.values, y_prob)

    importances = dict(zip(X_train.columns.tolist(), model.feature_importances_.tolist()))

    return TrainResult("gbdt", metrics, importances, model)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def select_features(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, str]:
    if label_col not in df.columns:
        raise ValueError(f"label column not found: {label_col}")
    return df.copy(), label_col
