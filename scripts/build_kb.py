"""Build the local evidence corpus and the BM25 + dense (ChromaDB) index.

    python scripts/build_kb.py                    # fetch PubMed abstracts (cached) + index
    python scripts/build_kb.py --per-topic 120    # bigger corpus
    python scripts/build_kb.py --no-fetch         # index only what is already on disk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crag  # noqa: E402,F401  loads onnxruntime before pyarrow (see crag/__init__.py)

from crag import config as C
from crag import pubmed
from crag.kb import Retriever, load_guidelines, load_pubmed_corpus

REVIEW = '(guideline[pt] OR practice guideline[pt] OR review[pt] OR systematic review[pt])'
RECENT = '("last 10 years"[dp])'

# Sepsis-related topics (one per lexicon concept) ...
TOPICS = [
    '"Sepsis"[MeSH Terms] AND management',
    '"Shock, Septic"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND "Hypotension"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND ("Lactic Acid"[MeSH Terms] OR "Acidosis, Lactic"[MeSH Terms])',
    '"Sepsis"[MeSH Terms] AND "Tachycardia"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND ("Tachypnea"[MeSH Terms] OR "Respiratory Insufficiency"[MeSH Terms])',
    '"Sepsis"[MeSH Terms] AND ("Fever"[MeSH Terms] OR "Hypothermia"[MeSH Terms])',
    '"Sepsis"[MeSH Terms] AND "Hypoxia"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND ("Leukocytosis"[MeSH Terms] OR "Leukopenia"[MeSH Terms])',
    '"Sepsis"[MeSH Terms] AND "Thrombocytopenia"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND "Acute Kidney Injury"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND ("Hyperbilirubinemia"[MeSH Terms] OR "Liver Failure"[MeSH Terms])',
    '"Sepsis"[MeSH Terms] AND "Fluid Therapy"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND "Vasoconstrictor Agents"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND "Early Diagnosis"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND "Blood Coagulation Disorders"[MeSH Terms]',
    '"Sepsis"[MeSH Terms] AND "Hyperglycemia"[MeSH Terms]',
]
# ... plus unrelated critical-care topics, so retrieval has realistic distractors.
DISTRACTORS = [
    '"Delirium"[MeSH Terms] AND "Intensive Care Units"[MeSH Terms]',
    '"Diabetic Ketoacidosis"[MeSH Terms]',
    '"Stroke"[MeSH Terms] AND "Thrombolytic Therapy"[MeSH Terms]',
    '"Pancreatitis"[MeSH Terms] AND acute',
    '"Brain Injuries, Traumatic"[MeSH Terms] AND "Critical Care"[MeSH Terms]',
    '"Heart Failure"[MeSH Terms] AND acute',
    '"Asthma"[MeSH Terms] AND "Status Asthmaticus"[MeSH Terms]',
    '"Pulmonary Embolism"[MeSH Terms]',
]


def fetch_corpus(per_topic: int) -> None:
    seen, rows = set(), []
    if C.PUBMED_CACHE.exists():
        for line in C.PUBMED_CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line); seen.add(d["pmid"]); rows.append(d)
    for i, t in enumerate(TOPICS + DISTRACTORS, 1):
        term = f"({t}) AND {REVIEW} AND {RECENT}"
        try:
            ids = [x for x in pubmed.search(term, per_topic) if x not in seen]
            docs = pubmed.fetch(ids) if ids else []
        except RuntimeError as e:
            print(f"  topic {i} failed: {e}")
            continue
        for d in docs:
            seen.add(d["pmid"]); rows.append(d)
        print(f"  topic {i}/{len(TOPICS) + len(DISTRACTORS)}: +{len(docs)} (total {len(rows)})", flush=True)
    C.PUBMED_CACHE.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=60)
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if not args.no_fetch:
        print("Fetching PubMed abstracts ...")
        fetch_corpus(args.per_topic)
    g, p = load_guidelines(), load_pubmed_corpus()
    docs = len({x.id.split("#")[0] for x in g + p})
    print(f"{docs} documents -> {len(g)} guideline passages + {len(p)} PubMed passages")
    print("Embedding and indexing (first run downloads the BGE model) ...")
    Retriever.build(g + p)
    (C.RESULTS_DIR / "corpus_stats.json").write_text(json.dumps(
        {"documents": docs, "passages": len(g) + len(p), "guideline_passages": len(g),
         "pubmed_passages": len(p)}, indent=2))
    print("Index ready in", C.INDEX_DIR)


if __name__ == "__main__":
    main()
