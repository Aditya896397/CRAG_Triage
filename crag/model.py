"""Stage 2: risk models, calibration, alert threshold and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score)

from . import config as C


# ------------------------------------------------------------------ resampling
def resample(X: np.ndarray, y: np.ndarray, method: str, max_neg: int = 150_000):
    """Return (X, y, sample_weight). Applied to training rows only."""
    if method == "weights":
        pos = y.mean()
        w = np.where(y == 1, (1 - pos) / pos, 1.0).astype(np.float32)
        return X, y, w
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        rng = np.random.default_rng(C.SEED)
        neg_idx, pos_idx = np.where(y == 0)[0], np.where(y == 1)[0]
        if len(neg_idx) > max_neg:  # SMOTE's kNN is too slow on ~1M rows
            neg_idx = rng.choice(neg_idx, max_neg, replace=False)
        idx = np.concatenate([neg_idx, pos_idx])
        Xs, ys = SMOTE(sampling_strategy=0.3, random_state=C.SEED).fit_resample(X[idx], y[idx])
        return Xs, ys, None
    if method == "none":
        return X, y, None
    raise ValueError(method)


# ------------------------------------------------------------------ models
def train_lightgbm(X, y, w, Xv, yv, params: dict | None = None):
    import lightgbm as lgb
    p = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
             random_state=C.SEED, n_jobs=-1, verbose=-1)
    p.update(params or {})
    m = lgb.LGBMClassifier(**p)
    m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def train_xgboost(X, y, w, Xv, yv, params: dict | None = None):
    import xgboost as xgb
    p = dict(n_estimators=2000, learning_rate=0.05, max_depth=7, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist",
             eval_metric="aucpr", early_stopping_rounds=100, random_state=C.SEED, n_jobs=-1)
    p.update(params or {})
    m = xgb.XGBClassifier(**p)
    m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
    return m


class TabNetWrapper:
    """TabNet with standardisation; exposes predict_proba like sklearn."""

    def __init__(self, max_rows: int = 250_000, epochs: int = 30):
        self.max_rows, self.epochs = max_rows, epochs

    def fit(self, X, y, w, Xv, yv):
        from pytorch_tabnet.tab_model import TabNetClassifier
        rng = np.random.default_rng(C.SEED)
        if len(X) > self.max_rows:  # CPU budget
            idx = rng.choice(len(X), self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.m = TabNetClassifier(n_d=16, n_a=16, n_steps=4, seed=C.SEED, verbose=0)
        self.m.fit((X - self.mu) / self.sd, y, eval_set=[((Xv - self.mu) / self.sd, yv)],
                   eval_metric=["auc"], max_epochs=self.epochs, patience=6,
                   batch_size=4096, virtual_batch_size=512, weights=1)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(((X - self.mu) / self.sd).astype(np.float32))


def train_model(name: str, X, y, w, Xv, yv):
    if name == "lightgbm":
        return train_lightgbm(X, y, w, Xv, yv)
    if name == "xgboost":
        return train_xgboost(X, y, w, Xv, yv)
    if name == "tabnet":
        return TabNetWrapper().fit(X, y, w, Xv, yv)
    raise ValueError(name)


# ------------------------------------------------------------------ calibration / threshold
@dataclass
class RiskModel:
    name: str
    estimator: object
    features: list[str]
    calibrator: IsotonicRegression | None = None
    threshold: float = 0.5
    meta: dict = field(default_factory=dict)

    def raw_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.raw_proba(X)
        return self.calibrator.predict(p) if self.calibrator is not None else p

    def risk_band(self, p: float) -> str:
        if p >= self.threshold:
            return "high"
        return "moderate" if p >= 0.5 * self.threshold else "low"


def calibrate(model: RiskModel, Xv: np.ndarray, yv: np.ndarray) -> RiskModel:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model.raw_proba(Xv), yv)
    model.calibrator = iso
    return model


def pick_threshold(y: np.ndarray, p: np.ndarray, target_sens: float = C.TARGET_SENSITIVITY) -> float:
    """Highest threshold whose sensitivity is still >= target."""
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target_sens * len(pos))) - 1
    return float(pos[max(k, 0)])


# ------------------------------------------------------------------ metrics
def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Sensitivity": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
        "F1": f1_score(y, pred, zero_division=0),
        "Brier": brier_score_loss(y, p),
    }


def bootstrap_ci(df: pd.DataFrame, p: np.ndarray, n: int = 200, seed: int = C.SEED) -> dict:
    """Patient-level bootstrap 95% CI for AUROC and AUPRC."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"pid": df["patient_id"].to_numpy(), "y": df[C.LABEL].to_numpy(), "p": p})
    groups = {k: v.index.to_numpy() for k, v in d.groupby("pid")}
    keys = np.array(list(groups))
    aucs, aps = [], []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        y, s = d["y"].to_numpy()[idx], d["p"].to_numpy()[idx]
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, s)); aps.append(average_precision_score(y, s))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"AUROC_CI": q(aucs), "AUPRC_CI": q(aps)}
