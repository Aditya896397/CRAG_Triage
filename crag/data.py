"""Stage 1: loading, leakage-free preprocessing and window features.

Order of operations (paper Sec. III-D):
  1. split by hospital (A = train/val, B = external test)
  2. per-patient forward fill limited to a carry window
  3. trailing-window statistics (past and present only)
  4. MICE imputation fitted on training rows only
  5. resampling happens later, inside model training, on training rows only
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from . import config as C

STATS = ("last", "mean", "min", "max", "delta", "count")


# ------------------------------------------------------------------ loading
def load_hospital(hospital: str, limit: int = 0) -> pd.DataFrame:
    """Read all .psv files of one hospital into one frame."""
    folder = C.DATA_RAW / ("training_setA" if hospital == "A" else "training_setB")
    files = sorted(folder.glob("p*.psv"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .psv files in {folder}. Run scripts/download_data.py first.")
    frames = []
    for f in files:
        df = pd.read_csv(f, sep="|")
        df.insert(0, "patient_id", f.stem)
        df.insert(1, "hour", np.arange(len(df), dtype=np.int32))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(1, "hospital", hospital)
    return out


# ------------------------------------------------------------------ features
def _carry_forward(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("patient_id", sort=False)
    filled = df.copy()
    for v in C.DYNAMIC:
        filled[v] = g[v].ffill(limit=C.CARRY_HOURS[v])
    return filled


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per patient-hour with window features (NaNs kept)."""
    raw = raw.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    filled = _carry_forward(raw)
    g_raw = raw.groupby("patient_id", sort=False)
    g_fill = filled.groupby("patient_id", sort=False)
    w = C.WINDOW

    cols: dict[str, np.ndarray] = {}
    for v in C.DYNAMIC:
        roll = g_raw[v].rolling(w, min_periods=1)
        cols[f"{v}__last"] = filled[v].to_numpy(np.float32)
        cols[f"{v}__mean"] = roll.mean().to_numpy(np.float32)
        cols[f"{v}__min"] = roll.min().to_numpy(np.float32)
        cols[f"{v}__max"] = roll.max().to_numpy(np.float32)
        cols[f"{v}__count"] = roll.count().to_numpy(np.float32)
        prev = g_fill[v].shift(w - 1)
        cols[f"{v}__delta"] = (filled[v] - prev).to_numpy(np.float32)

    feats = pd.DataFrame(cols)
    for d in C.DEMOGRAPHICS:
        feats[d] = raw[d].astype(np.float32).to_numpy()
    meta = raw[["patient_id", "hospital", "hour", C.LABEL]].reset_index(drop=True)
    # raw last-observed values (before imputation) are kept for the SHAP-to-RAG bridge
    obs = filled[C.DYNAMIC].astype(np.float32).add_suffix("__obs").reset_index(drop=True)
    return pd.concat([meta, feats, obs], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if ("__" in c and not c.endswith("__obs")) or c in C.DEMOGRAPHICS]


# ------------------------------------------------------------------ imputation
class Imputer:
    """MICE on the 'last' columns + demographics; window stats derived from it."""

    def __init__(self, max_rows: int = 20_000, max_iter: int = 5):
        self.max_rows, self.max_iter = max_rows, max_iter
        self.cols = [f"{v}__last" for v in C.DYNAMIC] + C.DEMOGRAPHICS
        self.mice: IterativeImputer | None = None

    def fit(self, train: pd.DataFrame) -> "Imputer":
        sample = train[self.cols]
        if len(sample) > self.max_rows:
            sample = sample.sample(self.max_rows, random_state=C.SEED)
        self.mice = IterativeImputer(max_iter=self.max_iter, random_state=C.SEED,
                                     initial_strategy="median", skip_complete=True,
                                     n_nearest_features=12, keep_empty_features=True)
        self.mice.fit(sample)
        return self

    def transform(self, df: pd.DataFrame, batch: int = 200_000) -> pd.DataFrame:
        assert self.mice is not None, "call fit first"
        out = df.copy()
        parts = [self.mice.transform(out[self.cols].iloc[i:i + batch])
                 for i in range(0, len(out), batch)]
        out[self.cols] = np.vstack(parts).astype(np.float32)
        for v in C.DYNAMIC:
            last = out[f"{v}__last"]
            for s in ("mean", "min", "max"):
                out[f"{v}__{s}"] = out[f"{v}__{s}"].fillna(last)
            out[f"{v}__delta"] = out[f"{v}__delta"].fillna(0.0)
        return out


# ------------------------------------------------------------------ splits
def patient_split(df: pd.DataFrame, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-grouped train/validation split inside Hospital A (stratified by sepsis)."""
    rng = np.random.default_rng(C.SEED)
    septic = df.groupby("patient_id")[C.LABEL].max()
    val_ids = []
    for label in (0, 1):
        ids = septic.index[septic == label].to_numpy()
        rng.shuffle(ids)
        val_ids.extend(ids[: int(len(ids) * val_frac)])
    mask = df["patient_id"].isin(set(val_ids))
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    per_patient = df.groupby("patient_id")[C.LABEL].max()
    return {
        "patients": int(per_patient.size),
        "hourly_records": int(len(df)),
        "septic_patients": int(per_patient.sum()),
        "septic_patients_pct": round(100 * per_patient.mean(), 2),
        "positive_rows_pct": round(100 * df[C.LABEL].mean(), 2),
    }


def row_from_manual(values: dict[str, float], imputer: "Imputer", features: list[str],
                    demographics: dict[str, float] | None = None) -> pd.Series:
    """Build one feature row from manually entered current values (dashboard 'manual' mode).
    Window statistics equal the entered value; anything not entered is imputed by MICE."""
    row = {f: np.nan for f in features}
    for v in C.DYNAMIC:
        x = values.get(v)
        if x is None:
            row[f"{v}__count"] = 0.0
            continue
        for s in ("last", "mean", "min", "max"):
            row[f"{v}__{s}"] = float(x)
        row[f"{v}__delta"], row[f"{v}__count"] = 0.0, 1.0
    for d, val in (demographics or {}).items():
        row[d] = float(val)
    df = imputer.transform(pd.DataFrame([row])[features])
    out = df.iloc[0].copy()
    for v in C.DYNAMIC:
        out[f"{v}__obs"] = values.get(v, np.nan)
    return out


def cache_path(name: str) -> Path:
    return C.DATA_PROCESSED / f"{name}.pkl"
