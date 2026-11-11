"""Run the full pipeline on one test patient from the command line.

    python scripts/run_case.py              # highest-risk hour of a septic demo patient
    python scripts/run_case.py --patient p100123 --hour 30 --mode standard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crag  # noqa: E402,F401  loads onnxruntime before pyarrow (see crag/__init__.py)

import numpy as np
import pandas as pd

from crag import config as C
from crag.pipeline import TriagePipeline


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient")
    ap.add_argument("--hour", type=int)
    ap.add_argument("--mode", default="crag", choices=["crag", "filter", "standard"])
    a = ap.parse_args()
    demo = pd.read_pickle(C.DATA_PROCESSED / "demo_patients.pkl")
    pipe = TriagePipeline()
    if a.patient is None:
        septic = demo.groupby("patient_id")[C.LABEL].max()
        a.patient = septic[septic == 1].index[0]
    pdf = demo[demo["patient_id"] == a.patient].reset_index(drop=True)
    if a.hour is None:
        a.hour = int(np.argmax(pipe.model.predict_proba(pdf[pipe.model.features].to_numpy(np.float32))))
    res = pipe.run(pdf.iloc[a.hour], mode=a.mode, force=True)
    print(f"\nPatient {a.patient}, hour {a.hour}: risk {res['prob']:.3f} ({res['band']}), "
          f"threshold {res['threshold']:.3f}")
    print("Drivers:", [(d.variable, d.value, round(d.shap, 3)) for d in res["drivers"]])
    print("Query:  ", res["bridge"].query)
    print("Route:  ", res["route"], "S =", res["best"])
    for line in res["log"]:
        print("  log:", line)
    for i, p in enumerate(res["evidence"], 1):
        print(f"  [P{i}] ({p.origin}, {p.score:.3f}) {p.title}")
    print("\n" + res["generation"]["answer"])
    print(f"\nCitation precision {res['generation']['citation_precision']}, {res['seconds']:.1f}s, "
          f"backend {res['generation']['backend']}")


if __name__ == "__main__":
    main()
