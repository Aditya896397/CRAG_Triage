"""Stage 3: TreeSHAP attributions aggregated per raw clinical variable."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Driver:
    variable: str
    shap: float           # aggregated SHAP value (log-odds)
    value: float | None   # last observed raw value (None if never measured)


def raw_variable(feature: str) -> str:
    return feature.split("__")[0]


class Explainer:
    def __init__(self, risk_model):
        import shap
        est = risk_model.estimator
        if risk_model.name not in ("lightgbm", "xgboost"):
            raise ValueError("TreeSHAP needs a tree model; select LightGBM or XGBoost.")
        self.features = risk_model.features
        self.tree = shap.TreeExplainer(est)
        self.groups = pd.Series(self.features).map(raw_variable).to_numpy()

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        sv = self.tree.shap_values(X)
        if isinstance(sv, list):          # older SHAP returns [neg, pos]
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # (n, features, classes)
            sv = sv[..., 1]
        return sv

    def aggregate(self, sv_row: np.ndarray) -> pd.Series:
        """Sum feature attributions belonging to the same raw variable (Phi_v)."""
        return pd.Series(sv_row).groupby(self.groups).sum().sort_values(ascending=False)

    @property
    def base_value(self) -> float:
        ev = self.tree.expected_value
        return float(ev[1] if np.ndim(ev) else ev)


def select_drivers(phi: pd.Series, observed: dict[str, float | None],
                   k: int = C.SHAP_TOP_K, delta: float = C.SHAP_DELTA) -> list[Driver]:
    """Eq. (4): top-k clinical variables with positive aggregated SHAP above delta."""
    out = []
    for var, val in phi.items():
        if val <= delta or var in C.NON_CLINICAL:
            continue
        out.append(Driver(var, float(val), observed.get(var)))
        if len(out) == k:
            break
    return out
