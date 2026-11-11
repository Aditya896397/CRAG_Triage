"""Fine-tune BioBERT as a cross-encoder relevance grader (weak supervision from PubMed).

Training pairs come from the cached PubMed corpus:
  positive       - (article title, its own abstract)
  hard negative  - (title, top BM25 abstract from a *different* article)
  easy negative  - (title, random abstract)
CPU training of 1 epoch on ~3k titles takes roughly 30-60 minutes.

    python scripts/finetune_biobert_grader.py --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crag  # noqa: E402,F401  loads onnxruntime before pyarrow (see crag/__init__.py)

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from crag import config as C
from crag.kb import tokenize

BASE = "dmis-lab/biobert-base-cased-v1.2"


def make_pairs(max_titles: int) -> list[tuple[str, str, float]]:
    docs = [json.loads(l) for l in C.PUBMED_CACHE.read_text(encoding="utf-8").splitlines()]
    docs = [d for d in docs if d["title"] and len(d["abstract"]) > 200]
    random.Random(C.SEED).shuffle(docs)
    abstracts = [d["abstract"][:1500] for d in docs]
    bm25 = BM25Okapi([tokenize(a) for a in abstracts])
    rng = random.Random(C.SEED)
    pairs = []
    for i, d in enumerate(docs[:max_titles]):
        pairs.append((d["title"], abstracts[i], 1.0))
        order = np.argsort(bm25.get_scores(tokenize(d["title"])))[::-1]
        hard = next(j for j in order if j != i)
        pairs.append((d["title"], abstracts[hard], 0.0))
        pairs.append((d["title"], abstracts[rng.choice([j for j in range(len(docs)) if j != i])], 0.0))
    rng.shuffle(pairs)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-titles", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    torch.manual_seed(C.SEED)
    pairs = make_pairs(a.max_titles)
    split = int(0.9 * len(pairs))
    train, val = pairs[:split], pairs[split:]
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def batches(data):
        for i in range(0, len(data), a.batch):
            b = data[i:i + a.batch]
            enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_second",
                      max_length=256, padding=True, return_tensors="pt")
            yield enc, torch.tensor([x[2] for x in b])

    for ep in range(a.epochs):
        model.train()
        for step, (enc, y) in enumerate(batches(train), 1):
            loss = loss_fn(model(**enc).logits[:, 0], y)
            loss.backward(); opt.step(); opt.zero_grad()
            if step % 10 == 0:
                print(f"epoch {ep + 1} step {step}/{len(train) // a.batch} loss {loss.item():.4f}", flush=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for enc, y in batches(val):
            pred = (torch.sigmoid(model(**enc).logits[:, 0]) >= 0.5).float()
            correct += int((pred == y).sum()); total += len(y)
    print(f"validation accuracy {correct / total:.4f}")
    C.BIOBERT_GRADER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(C.BIOBERT_GRADER_DIR); tok.save_pretrained(C.BIOBERT_GRADER_DIR)
    (C.RESULTS_DIR / "biobert_grader.json").write_text(json.dumps(
        {"pairs": len(pairs), "val_accuracy": correct / total, "epochs": a.epochs}, indent=2))
    print("saved ->", C.BIOBERT_GRADER_DIR)


if __name__ == "__main__":
    main()
