"""Calibrate the routing thresholds tau_high / tau_low on the annotated retrieval pool.

For every alert, the "true" route is derived from the labels of the SHAP-guided hybrid top-5:
  correct   - at least one passage labelled 2
  ambiguous - best label is 1
  incorrect - all labels 0
A grid search picks (tau_low, tau_high) maximising macro-F1 between predicted and true routes.

    python scripts/calibrate_thresholds.py [--dev-frac 0.5]
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
from sklearn.metrics import f1_score

from crag import config as C
from crag.grader import get_grader

ANN = C.ARTIFACTS / "annotation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-frac", type=float, default=0.5)
    a = ap.parse_args()
    pool = pd.read_csv(ANN / "retrieval_pool.csv")
    runs = json.loads((ANN / "retrieval_runs.json").read_text())["SHAP-guided | Hybrid"]
    rel = {(r.alert_id, r.passage_id): r.relevance for r in pool.itertuples()}
    text = dict(zip(pool.passage_id, pool.text))
    query = dict(zip(pool.alert_id, pool.shap_query))
    g = get_grader()

    rows = []
    for aid, ids in runs.items():
        labels = [rel.get((aid, i), 0) for i in ids]
        best_label = max(labels) if labels else 0
        truth = "correct" if best_label >= 2 else ("ambiguous" if best_label == 1 else "incorrect")
        S = float(g.score(query[aid], [text[i] for i in ids]).max()) if ids else 0.0
        rows.append((aid, S, truth))
    df = pd.DataFrame(rows, columns=["alert", "S", "truth"]).sample(frac=1, random_state=C.SEED)
    dev = df.iloc[: int(len(df) * a.dev_frac)]
    test = df.iloc[int(len(df) * a.dev_frac):]

    def route(S, lo, hi):
        return np.where(S >= hi, "correct", np.where(S <= lo, "incorrect", "ambiguous"))

    best = (C.TAU_LOW, C.TAU_HIGH, -1)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for lo in grid:
        for hi in grid[grid > lo]:
            f = f1_score(dev["truth"], route(dev["S"].to_numpy(), lo, hi), average="macro")
            if f > best[2]:
                best = (float(lo), float(hi), float(f))
    lo, hi, f_dev = best
    f_test = f1_score(test["truth"], route(test["S"].to_numpy(), lo, hi), average="macro") if len(test) else float("nan")
    out = {"tau_low": lo, "tau_high": hi, "macro_f1_dev": round(f_dev, 4), "macro_f1_heldout": round(f_test, 4),
           "label_source": pool.get("label_source", pd.Series(["human"])).iloc[0]}
    # silver-label thresholds are saved separately and NOT picked up by the pipeline
    name = "thresholds.json" if out["label_source"] == "human" else "thresholds_SILVER.json"
    (C.RESULTS_DIR / name).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
