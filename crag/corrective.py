"""Stage 5: corrective retrieval with three-way routing (paper Eq. 9, Fig. 2).

Modes (for the ablation in Table VII):
  standard  - top-k hybrid retrieval passed through unchanged (vanilla RAG)
  filter    - grader filtering only, no routing / PubMed
  crag      - full corrective routing with sentence strips and PubMed fallback
"""
from __future__ import annotations

import json
from typing import TypedDict

import numpy as np

from . import config as C
from . import pubmed
from .grader import get_grader
from .kb import Passage, Retriever, split_sentences


def thresholds() -> tuple[float, float]:
    f = C.RESULTS_DIR / "thresholds.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d["tau_high"], d["tau_low"]
    return C.TAU_HIGH, C.TAU_LOW


class State(TypedDict, total=False):
    query: str
    mesh_query: str
    mode: str
    retrieval_mode: str
    grader: str
    passages: list[Passage]
    best: float
    route: str
    evidence: list[Passage]
    log: list[str]


# ------------------------------------------------------------------ helpers
def _grade(query: str, passages: list[Passage], grader_name: str) -> list[Passage]:
    scores = get_grader(grader_name).score(query, [p.text for p in passages])
    for p, s in zip(passages, scores):
        p.score = float(s)
    return sorted(passages, key=lambda p: p.score, reverse=True)


def _strip_filter(query: str, passages: list[Passage], grader_name: str,
                  tau_keep: float, tau_strip: float = C.TAU_STRIP) -> list[Passage]:
    """Decompose kept passages into sentences and keep only relevant strips."""
    kept = []
    g = get_grader(grader_name)
    for p in passages:
        if p.score < tau_keep:
            continue
        sents = split_sentences(p.text)
        if len(sents) <= 1:
            kept.append(p)
            continue
        s = g.score(query, sents)
        keep = set(np.argsort(s)[::-1][:2]) | {i for i, sc in enumerate(s) if sc >= tau_strip}
        good = [x for i, x in enumerate(sents) if i in keep]   # original order preserved
        kept.append(Passage(**{**p.to_dict(), "text": " ".join(good)}))
    return kept


def _pubmed(query: str, term: str, grader_name: str, tau_low: float, log: list[str]) -> list[Passage]:
    try:
        docs = pubmed.search_and_fetch(term)
        if not docs:
            log.append("PubMed strict query empty -> relaxed MeSH rewrite")
            docs = pubmed.search_and_fetch(pubmed.relax(term))
    except RuntimeError as e:
        log.append(f"PubMed unavailable: {e}")
        return []
    ps = [Passage(f"pmid{d['pmid']}", d["abstract"][:1800], d["title"],
                  f"{d['journal']} ({d['year']}), PMID {d['pmid']}", d["year"], d["url"], "pubmed")
          for d in docs]
    ps = _grade(query, ps, grader_name)
    log.append(f"PubMed returned {len(ps)} abstracts, {sum(p.score > tau_low for p in ps)} above tau_low")
    return [p for p in ps if p.score > tau_low][:5]


# ------------------------------------------------------------------ graph nodes
def make_nodes(retriever: Retriever):
    def retrieve(s: State) -> State:
        ps = retriever.search(s["query"], C.TOP_K, s.get("retrieval_mode", "hybrid"))
        return {"passages": ps, "log": s.get("log", []) + [f"retrieved {len(ps)} local passages"]}

    def grade(s: State) -> State:
        if s["mode"] == "standard":
            return {"best": float("nan"), "route": "standard"}
        ps = _grade(s["query"], s["passages"], s["grader"])
        best = ps[0].score if ps else 0.0
        hi, lo = thresholds()
        if s["mode"] == "filter":
            route = "filter"
        else:
            route = "correct" if best >= hi else ("incorrect" if best <= lo else "ambiguous")
        return {"passages": ps, "best": best, "route": route,
                "log": s["log"] + [f"best grader score S={best:.3f} -> {route}"]}

    def standard(s: State) -> State:
        return {"evidence": s["passages"][:5]}

    def filt(s: State) -> State:
        _, lo = thresholds()
        return {"evidence": [p for p in s["passages"] if p.score > lo][:5]}

    def correct(s: State) -> State:
        _, lo = thresholds()
        return {"evidence": _strip_filter(s["query"], s["passages"], s["grader"], lo)[:5]}

    def ambiguous(s: State) -> State:
        _, lo = thresholds()
        log = list(s["log"])
        local = _strip_filter(s["query"], s["passages"], s["grader"], lo)[:3]
        web = _pubmed(s["query"], s["mesh_query"], s["grader"], lo, log)
        return {"evidence": local + web[:3], "log": log}

    def incorrect(s: State) -> State:
        _, lo = thresholds()
        log = list(s["log"]) + ["local evidence discarded; MeSH rewrite sent to PubMed"]
        return {"evidence": _pubmed(s["query"], s["mesh_query"], s["grader"], lo, log), "log": log}

    return {"retrieve": retrieve, "grade": grade, "standard": standard, "filter": filt,
            "correct": correct, "ambiguous": ambiguous, "incorrect": incorrect}


def build_graph(retriever: Retriever):
    """LangGraph state machine; falls back to a plain dispatcher if LangGraph is absent."""
    nodes = make_nodes(retriever)
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        def run(state: State) -> State:
            for name in ("retrieve", "grade"):
                state.update(nodes[name](state))
            state.update(nodes[state["route"]](state))
            return state
        return run

    g = StateGraph(State)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    branches = ["standard", "filter", "correct", "ambiguous", "incorrect"]
    g.add_conditional_edges("grade", lambda s: s["route"], {b: b for b in branches})
    for b in branches:
        g.add_edge(b, END)
    app = g.compile()
    return lambda state: app.invoke(state)


def corrective_retrieve(graph, query: str, mesh_query: str, mode: str = "crag",
                        grader: str = "auto", retrieval_mode: str = "hybrid") -> State:
    state: State = {"query": query, "mesh_query": mesh_query, "mode": mode, "grader": grader,
                    "retrieval_mode": retrieval_mode, "log": []}
    out = graph(state)
    out.setdefault("evidence", [])
    return out
