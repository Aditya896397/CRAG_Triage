"""Train and compare risk models; writes the numbers for paper Tables II, VI and Sec. V-D.

    python scripts/train.py                         # full run
    python scripts/train.py --limit 3000 --models lightgbm,xgboost   # quick dev run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crag  # noqa: E402,F401  loads onnxruntime before pyarrow (see crag/__init__.py)

import joblib
import numpy as np
import pandas as pd

from crag import config as C
from crag import data as D
from crag import model as M
from crag.bridge import build_query
from crag.explain import Explainer, select_drivers
from crag.pipeline import observed_values


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def features_for(hospital: str, limit: int) -> pd.DataFrame:
    path = D.cache_path(f"features_{hospital}_{limit or 'all'}")
    if path.exists():
        return pd.read_pickle(path)
    log(f"loading hospital {hospital}")
    raw = D.load_hospital(hospital, limit)
    log(f"building window features for {raw['patient_id'].nunique()} patients ({len(raw)} rows)")
    f = D.build_features(raw)
    f.to_pickle(path)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="patients per hospital (0 = all)")
    ap.add_argument("--models", default="lightgbm,xgboost,tabnet")
    ap.add_argument("--resampling", default="weights,smote")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--shap-sample", type=int, default=2000)
    args = ap.parse_args()

    A, B = features_for("A", args.limit), features_for("B", args.limit)
    pd.DataFrame({"Hospital A (train/val)": D.dataset_summary(A),
                  "Hospital B (test)": D.dataset_summary(B)}).to_csv(C.RESULTS_DIR / "dataset_summary.csv")
    log("dataset summary written")

    train, val = D.patient_split(A)
    log(f"train rows {len(train)}, val rows {len(val)}, test rows {len(B)}")
    imp = D.Imputer().fit(train)
    log("MICE fitted on training rows")
    train, val, test = imp.transform(train), imp.transform(val), imp.transform(B)
    del A, B
    import gc; gc.collect()
    feats = D.feature_columns(train)
    Xtr, ytr = train[feats].to_numpy(np.float32), train[C.LABEL].to_numpy()
    Xva, yva = val[feats].to_numpy(np.float32), val[C.LABEL].to_numpy()
    Xte, yte = test[feats].to_numpy(np.float32), test[C.LABEL].to_numpy()
    log(f"{len(feats)} features")

    rows, fitted = [], []
    for name in args.models.split(","):
        methods = ["none"] if name == "tabnet" else args.resampling.split(",")
        for method in methods:
            log(f"training {name} + {method}")
            X, y, w = M.resample(Xtr, ytr, method)
            est = M.train_model(name, X, y, w, Xva, yva)
            rm = M.calibrate(M.RiskModel(name, est, feats), Xva, yva)
            pva = rm.predict_proba(Xva)
            rm.threshold = M.pick_threshold(yva, pva)
            rm.meta = {"resampling": method, "val_AUPRC": float(M.metrics(yva, pva, rm.threshold)["AUPRC"])}
            pte = rm.predict_proba(Xte)
            m = M.metrics(yte, pte, rm.threshold)
            label = f"{name} + {'class weights' if method == 'weights' else method}"
            rows.append({"Model": label, **{k: round(v, 4) for k, v in m.items()},
                         "val_AUPRC": round(rm.meta["val_AUPRC"], 4), "threshold": round(rm.threshold, 4)})
            fitted.append((rm, pte, label))
            log(f"  test AUROC {m['AUROC']:.4f}  AUPRC {m['AUPRC']:.4f}  val AUPRC {rm.meta['val_AUPRC']:.4f}")

    res = pd.DataFrame(rows)
    # select best TREE model by validation AUPRC (TreeSHAP requires a tree model)
    trees = [f for f in fitted if f[0].name in ("lightgbm", "xgboost")]
    best_rm, best_p, best_label = max(trees, key=lambda f: f[0].meta["val_AUPRC"])
    ci = M.bootstrap_ci(test, best_p, n=args.bootstrap)
    res["selected"] = res["Model"] == best_label
    res.to_csv(C.RESULTS_DIR / "predictive_metrics.csv", index=False)
    (C.RESULTS_DIR / "selected_model.json").write_text(json.dumps(
        {"model": best_label, "threshold": best_rm.threshold, **ci}, indent=2))
    log(f"selected {best_label}; AUROC 95% CI {ci['AUROC_CI']}")
    joblib.dump(best_rm, C.MODEL_DIR / "risk_model.joblib")
    joblib.dump(imp, C.MODEL_DIR / "imputer.joblib")

    # ---- explanation statistics on test alerts (paper Sec. V-D)
    ex = Explainer(best_rm)
    alerts = test[best_p >= best_rm.threshold]
    sample = alerts.sample(min(args.shap_sample, len(alerts)), random_state=C.SEED)
    sv = ex.shap_values(sample[feats].to_numpy(np.float32))
    freq, triggered = {}, 0
    for i in range(len(sample)):
        row = sample.iloc[i]
        drivers = select_drivers(ex.aggregate(sv[i]), observed_values(row))
        for d in drivers:
            freq[d.variable] = freq.get(d.variable, 0) + 1
        if build_query(drivers, "high").concepts:
            triggered += 1
    stats = {"alerts_analysed": len(sample),
             "driver_frequency": dict(sorted(freq.items(), key=lambda x: -x[1])),
             "bridge_trigger_rate_pct": round(100 * triggered / max(len(sample), 1), 2)}
    (C.RESULTS_DIR / "explanation_stats.json").write_text(json.dumps(stats, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        shap.summary_plot(sv, sample[feats], max_display=15, show=False)
        plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "shap_summary.png", dpi=200); plt.close()
    except Exception as e:  # plotting is optional
        log(f"SHAP plot skipped: {e}")

    # ---- demo set for the dashboard: hourly rows of a few test patients
    pids = test.groupby("patient_id")[C.LABEL].max()
    rng = np.random.default_rng(C.SEED)
    pick = list(rng.choice(pids[pids == 1].index, min(40, int(pids.sum())), replace=False)) + \
        list(rng.choice(pids[pids == 0].index, min(40, int((pids == 0).sum())), replace=False))
    test[test["patient_id"].isin(pick)].to_pickle(C.DATA_PROCESSED / "demo_patients.pkl")
    test.to_pickle(C.DATA_PROCESSED / "test_imputed.pkl")
    log("done")


if __name__ == "__main__":
    main()
