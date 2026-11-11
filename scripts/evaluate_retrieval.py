"""Retrieval ablation (paper Table VI: query source x retriever).

Step 1 - build the annotation pool:
    python scripts/evaluate_retrieval.py pool --n 100
  -> artifacts/annotation/retrieval_pool.csv  (fill the 'relevance' column: 0, 1 or 2)

Step 2 - score with your human labels:
    python scripts/evaluate_retrieval.py score
  -> artifacts/results/table_retrieval.csv

Development shortcut (NOT for the paper): --silver fills relevance automatically with
an independent MS MARCO cross-encoder judged against the patient's full abnormal-concept
profile, so you can see provisional numbers before annotating.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crag  # noqa: E402,F401  loads onnxruntime before pyarrow (see crag/__init__.py)

import numpy as np
import pandas as pd

from crag import config as C
from crag.bridge import GENERIC_QUERY
from crag.evalutil import llm_query, ndcg_at_k, sample_alerts
from crag.grader import Grader
from crag.pipeline import TriagePipeline

ANN = C.ARTIFACTS / "annotation"
ANN.mkdir(exist_ok=True)
POOL, RUNS = ANN / "retrieval_pool.csv", ANN / "retrieval_runs.json"


def configs(case: dict) -> dict[str, tuple[str, str]]:
    concepts = [c.concept for c in case["bridge"].concepts]
    out = {"Generic | Hybrid": (GENERIC_QUERY, "hybrid"),
           "All abnormal (no SHAP) | Hybrid": (case["all_abnormal"], "hybrid")}
    q = llm_query(concepts, case["band"])
    if q:
        out["LLM-written | Hybrid"] = (q, "hybrid")
    shap_q = case["bridge"].query
    out.update({"SHAP-guided | BM25": (shap_q, "bm25"),
                "SHAP-guided | Dense (BGE)": (shap_q, "dense"),
                "SHAP-guided | Hybrid": (shap_q, "hybrid")})
    return out


def build_pool(n: int, silver: bool) -> None:
    pipe = TriagePipeline()
    cases = sample_alerts(pipe, n)
    runs, rows = {}, {}
    for case in cases:
        aid = case["alert_id"]
        profile = case["all_abnormal"]
        for name, (q, mode) in configs(case).items():
            ps = pipe.retriever.search(q, 5, mode)
            runs.setdefault(name, {})[aid] = [p.id for p in ps]
            for p in ps:
                rows[(aid, p.id)] = {"alert_id": aid, "patient_profile": profile,
                                     "shap_query": case["bridge"].query, "passage_id": p.id,
                                     "title": p.title, "text": p.text, "relevance": ""}
    pool = pd.DataFrame(rows.values())
    if silver:
        judge = Grader("msmarco")
        for aid, grp in pool.groupby("alert_id"):
            s = judge.score(grp["patient_profile"].iloc[0], grp["text"].tolist())
            pool.loc[grp.index, "relevance"] = np.select([s >= 0.6, s >= 0.2], [2, 1], 0)
        pool["label_source"] = "silver-msmarco"
    pool.to_csv(POOL, index=False)
    RUNS.write_text(json.dumps(runs))
    print(f"{len(cases)} alerts, {len(pool)} unique passages -> {POOL}")


def score() -> None:
    pool = pd.read_csv(POOL)
    if pool["relevance"].isna().any():
        raise SystemExit("Some rows have no relevance label yet.")
    rel = {(r.alert_id, r.passage_id): float(r.relevance) for r in pool.itertuples()}
    runs = json.loads(RUNS.read_text())
    rows = []
    for name, per_alert in runs.items():
        nd, pr = [], []
        for aid, ids in per_alert.items():
            r = [rel.get((aid, i), 0.0) for i in ids] + [0.0] * (5 - len(ids))
            nd.append(ndcg_at_k(r, 5)); pr.append(np.mean([x >= 1 for x in r[:5]]))
        q, ret = name.split(" | ")
        rows.append({"Query source": q, "Retriever": ret, "nDCG@5": round(np.mean(nd), 4),
                     "P@5": round(np.mean(pr), 4), "alerts": len(per_alert)})
    out = pd.DataFrame(rows)
    tag = "_SILVER" if "label_source" in pool.columns else ""
    path = C.RESULTS_DIR / f"table_retrieval{tag}.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False)); print("->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pool", "score"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--silver", action="store_true")
    a = ap.parse_args()
    build_pool(a.n, a.silver) if a.step == "pool" else score()
