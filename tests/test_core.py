import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crag  # noqa: E402,F401  must precede pandas (pyarrow/onnxruntime order)

import numpy as np
import pandas as pd

from crag.bridge import GENERIC_QUERY, binned, build_query
from crag.evalutil import ndcg_at_k
from crag.explain import Driver, select_drivers
from crag.kb import chunk_words, rrf
from crag.model import pick_threshold


def test_bridge_builds_query_from_abnormal_drivers():
    drivers = [Driver("MAP", 0.8, 60.0), Driver("Lactate", 0.5, 4.3), Driver("HR", 0.3, 125.0)]
    out = build_query(drivers, "high")
    assert [c.concept for c in out.concepts] == ["hypotension", "raised lactate", "tachycardia"]
    assert "hypotension, raised lactate and tachycardia" in out.query
    assert '"Acidosis, Lactic"[MeSH Terms]' in out.mesh_query


def test_bridge_ignores_normal_values_and_dedupes():
    drivers = [Driver("Temp", 0.9, 37.0), Driver("MAP", 0.5, 60.0), Driver("SBP", 0.4, 80.0)]
    out = build_query(drivers, "high")
    assert [c.concept for c in out.concepts] == ["hypotension"]


def test_bridge_falls_back_to_generic():
    assert build_query([Driver("HR", 0.5, 80.0)], "high").query == GENERIC_QUERY


def test_select_drivers_skips_admin_and_negative():
    phi = pd.Series({"ICULOS": 2.0, "HR": 0.5, "Temp": -0.3, "MAP": 0.4, "Resp": 0.001})
    d = select_drivers(phi, {"HR": 120.0, "MAP": 60.0})
    assert [x.variable for x in d] == ["HR", "MAP"]


def test_rrf_rewards_agreement():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


def test_threshold_reaches_target_sensitivity():
    y = np.array([0] * 90 + [1] * 10)
    p = np.linspace(0, 1, 100)
    t = pick_threshold(y, p, 0.8)
    assert ((p >= t) & (y == 1)).sum() / 10 >= 0.8


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([2, 1, 0]) == 1.0
    assert ndcg_at_k([0, 0, 0]) == 0.0


def test_chunking_and_binning():
    chunks = chunk_words("This is a sentence about sepsis. " * 200)
    assert len(chunks) > 1 and all(len(c.split()) <= 200 for c in chunks)
    assert binned("Lactate", 4.3) == "4-5 mmol/L"
