"""Generation ablation (paper Table VII) and routing distribution.

    python scripts/evaluate_generation.py --n 100

Metrics per configuration:
  faithfulness  - RAGAS-style: an LLM judge splits the answer into statements and checks each
                  against the evidence (needs GROQ_API_KEY); otherwise reported as n/a
  answer_rel    - grader relevance between the SHAP query and the answer
  cit_prec      - share of cited sentences supported by the passage they cite
  latency_s     - retrieval + generation time
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crag  # noqa: E402,F401  loads onnxruntime before pyarrow (see crag/__init__.py)

import numpy as np
import pandas as pd

from crag import config as C
from crag.corrective import corrective_retrieve
from crag.evalutil import sample_alerts
from crag.generator import generate
from crag.grader import get_grader
from crag.pipeline import TriagePipeline


def judge_faithfulness(answer: str, evidence) -> float:
    if not C.GROQ_API_KEY or not evidence:
        return float("nan")
    from crag.generator import groq_chat
    ctx = "\n\n".join(f"[P{i}] {p.text}" for i, p in enumerate(evidence, 1))
    prompt = ("Split the ANSWER into atomic factual statements (ignore headings and the risk "
              "probability). For each statement decide if it is directly supported by the CONTEXT. "
              'Reply with JSON only: {"supported": <int>, "total": <int>}.\n\n'
              f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}")
    m = re.search(r"\{.*\}", groq_chat([{"role": "user", "content": prompt}], max_tokens=60), re.S)
    try:
        d = json.loads(m.group(0))
        return d["supported"] / d["total"] if d["total"] else float("nan")
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--graders", default="msmarco,medcpt,biobert")
    ap.add_argument("--no-baselines", action="store_true", help="skip standard / filter rows")
    ap.add_argument("--append", action="store_true", help="merge rows into existing tables")
    a = ap.parse_args()

    pipe = TriagePipeline()
    cases = sample_alerts(pipe, a.n)
    cfgs = [] if a.no_baselines else [("Standard RAG", "standard", "medcpt"),
                                      ("Grader filter only", "filter", "medcpt")]
    names = {"msmarco": "MS MARCO", "medcpt": "MedCPT", "biobert": "BioBERT"}
    for g in a.graders.split(","):
        if g == "biobert" and not (C.BIOBERT_GRADER_DIR / "config.json").exists():
            print("BioBERT grader not trained yet - skipping (run scripts/finetune_biobert_grader.py)")
            continue
        cfgs.append((f"CRAG, {names[g]} grader", "crag", g))

    scorer = get_grader("auto")
    rows, routes = [], []
    for label, mode, g in cfgs:
        recs = []
        for case in cases:
            t0 = time.time()
            b = case["bridge"]
            st = corrective_retrieve(pipe.graph, b.query, b.mesh_query, mode, g)
            gen = generate(case["prob"], case["band"], b.concepts, st["evidence"], b.query)
            sec = time.time() - t0
            recs.append({"faith": judge_faithfulness(gen["raw"], st["evidence"]),
                         "rel": float(scorer.score(b.query, [gen["answer"]])[0]),
                         "cit": gen["citation_precision"], "sec": sec})
            if mode == "crag":
                routes.append({"config": label, "route": st.get("route"), "S": st.get("best")})
        d = pd.DataFrame(recs)
        rows.append({"Configuration": label, "Faithfulness": round(d["faith"].mean(), 4),
                     "Answer relevancy": round(d["rel"].mean(), 4),
                     "Citation precision": round(d["cit"].mean(), 4),
                     "Latency (s)": round(d["sec"].mean(), 2), "cases": len(d)})
        print(rows[-1], flush=True)
    table, rpath = C.RESULTS_DIR / "table_generation.csv", C.RESULTS_DIR / "routing_raw.csv"
    new = pd.DataFrame(rows)
    r = pd.DataFrame(routes)
    if a.append and table.exists():
        old = pd.read_csv(table)
        new = pd.concat([old[~old["Configuration"].isin(new["Configuration"])], new])
        if rpath.exists() and len(r):
            oldr = pd.read_csv(rpath)
            r = pd.concat([oldr[~oldr["config"].isin(r["config"])], r])
    new.to_csv(table, index=False)
    if len(r):
        r.to_csv(rpath, index=False)
        dist = (r.groupby("config")["route"].value_counts(normalize=True) * 100).round(1).unstack(fill_value=0)
        dist.to_csv(C.RESULTS_DIR / "routing_distribution.csv")
        print(dist)
    print("->", C.RESULTS_DIR / "table_generation.csv")


if __name__ == "__main__":
    main()
