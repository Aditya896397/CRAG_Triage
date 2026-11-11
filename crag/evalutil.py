"""Shared helpers for the evaluation scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .bridge import all_abnormal_query, build_query
from .explain import select_drivers
from .pipeline import TriagePipeline, observed_values


def sample_alerts(pipe: TriagePipeline, n: int, seed: int = C.SEED) -> list[dict]:
    """Sample N test alerts (one per patient), stratified round-robin by driver-concept set."""
    test = pd.read_pickle(C.DATA_PROCESSED / "test_imputed.pkl")
    X = test[pipe.model.features].to_numpy(np.float32)
    p = pipe.model.predict_proba(X)
    alerts = test[p >= pipe.model.threshold].copy()
    alerts["prob"] = p[p >= pipe.model.threshold]
    alerts = alerts.sample(frac=1, random_state=seed).drop_duplicates("patient_id")
    alerts = alerts.head(max(n * 6, 300))

    cases = []
    for _, row in alerts.iterrows():
        pred = pipe.predict(row)
        obs = observed_values(row)
        drivers = select_drivers(pred["phi"], obs)
        bridge = build_query(drivers, pred["band"])
        key = tuple(sorted(c.concept for c in bridge.concepts)) or ("generic",)
        cases.append({"alert_id": f"{row['patient_id']}_h{int(row['hour'])}", "row": row,
                      "prob": pred["prob"], "band": pred["band"], "bridge": bridge,
                      "all_abnormal": all_abnormal_query(obs, pred["band"]), "key": key,
                      "septic_patient": int(row[C.LABEL])})
    # round-robin over concept sets for diversity
    groups: dict[tuple, list] = {}
    for c in cases:
        groups.setdefault(c["key"], []).append(c)
    out, buckets = [], list(groups.values())
    while len(out) < n and any(buckets):
        for bkt in buckets:
            if bkt and len(out) < n:
                out.append(bkt.pop(0))
    return out


def ndcg_at_k(rels: list[float], k: int = 5) -> float:
    rels = np.asarray(rels[:k], dtype=float)
    if rels.size == 0:
        return 0.0
    disc = 1 / np.log2(np.arange(2, rels.size + 2))
    dcg = float(((2 ** rels - 1) * disc).sum())
    ideal = np.sort(rels)[::-1]
    idcg = float(((2 ** ideal - 1) * disc).sum())
    return dcg / idcg if idcg > 0 else 0.0


def llm_query(concepts: list[str], band: str) -> str | None:
    """Ablation: query written by the LLM from the same concepts (needs GROQ_API_KEY)."""
    if not C.GROQ_API_KEY:
        return None
    from .generator import groq_chat
    msg = ("Write one search query (max 25 words) to retrieve clinical guideline passages for an adult "
           f"ICU patient at {band} risk of sepsis with: {', '.join(concepts) or 'no specific abnormality'}. "
           "Return only the query.")
    return groq_chat([{"role": "user", "content": msg}], max_tokens=60).strip('"')
