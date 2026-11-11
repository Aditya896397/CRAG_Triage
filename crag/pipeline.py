"""End-to-end pipeline for one patient-hour (paper Algorithm 1)."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .bridge import build_query
from .corrective import build_graph, corrective_retrieve
from .explain import Explainer, select_drivers
from .generator import generate
from .kb import Retriever


def load_risk_model():
    path = C.MODEL_DIR / "risk_model.joblib"
    if not path.exists():
        raise FileNotFoundError("No trained model. Run: python scripts/train.py")
    return joblib.load(path)


def observed_values(row: pd.Series) -> dict[str, float | None]:
    out = {}
    for v in C.DYNAMIC:
        x = row.get(f"{v}__obs")
        out[v] = None if x is None or pd.isna(x) else float(x)
    return out


class TriagePipeline:
    def __init__(self, grader: str = "auto"):
        self.model = load_risk_model()
        self.explainer = Explainer(self.model)
        self.retriever = Retriever.load()
        self.graph = build_graph(self.retriever)
        self.grader = grader

    def predict(self, row: pd.Series) -> dict:
        x = row[self.model.features].to_numpy(np.float32)[None, :]
        prob = float(self.model.predict_proba(x)[0])
        phi = self.explainer.aggregate(self.explainer.shap_values(x)[0])
        return {"prob": prob, "band": self.model.risk_band(prob), "phi": phi,
                "threshold": self.model.threshold}

    def run(self, row: pd.Series, mode: str = "crag", retrieval_mode: str = "hybrid",
            force: bool = False) -> dict:
        t0 = time.time()
        pred = self.predict(row)
        obs = observed_values(row)
        result = {**pred, "observed": obs, "alert": pred["prob"] >= pred["threshold"]}
        if not result["alert"] and not force:
            result.update(status="no alert: risk below threshold", seconds=time.time() - t0)
            return result
        drivers = select_drivers(pred["phi"], obs)
        bridge = build_query(drivers, pred["band"])
        state = corrective_retrieve(self.graph, bridge.query, bridge.mesh_query, mode,
                                    self.grader, retrieval_mode)
        gen = generate(pred["prob"], pred["band"], bridge.concepts, state["evidence"], bridge.query)
        result.update(status="ok", drivers=drivers, bridge=bridge, route=state.get("route"),
                      best=state.get("best"), retrieved=state.get("passages", []),
                      evidence=state["evidence"], log=state.get("log", []), generation=gen,
                      seconds=time.time() - t0)
        return result
