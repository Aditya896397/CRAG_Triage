"""Cross-encoder relevance grader: s = sigmoid(g(q, p))."""
from __future__ import annotations

import numpy as np

from . import config as C

_CACHE: dict[str, "Grader"] = {}

# Temperature scaling s = sigmoid((logit - shift) / T). MedCPT and MS MARCO logits are
# unbounded (MedCPT spans roughly -16..+12), so a plain sigmoid saturates at 0 or 1 and the
# routing thresholds lose meaning. Values set by inspecting logits of relevant vs. irrelevant
# passages; calibrate_thresholds.py then tunes tau_low / tau_high on labelled data.
CALIBRATION = {"medcpt": (-2.0, 4.0), "msmarco": (0.0, 1.5), "biobert": (0.0, 1.0)}


def resolve(name: str = C.GRADER) -> str:
    if name == "auto":
        return "biobert" if (C.BIOBERT_GRADER_DIR / "config.json").exists() else "medcpt"
    return name


def model_path(name: str) -> str:
    return {"biobert": str(C.BIOBERT_GRADER_DIR), "medcpt": C.MEDCPT_GRADER,
            "msmarco": C.MSMARCO_GRADER}[name]


class Grader:
    def __init__(self, name: str = "auto", batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name = resolve(name)
        path = model_path(self.name)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.torch, self.bs = torch, batch_size
        self.shift, self.temp = CALIBRATION[self.name]

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.bs):
                batch = texts[i:i + self.bs]
                enc = self.tok([query] * len(batch), batch, truncation="only_second",
                               max_length=512, padding=True, return_tensors="pt")
                logits = self.model(**enc).logits
                logit = logits[:, -1] if logits.shape[-1] > 1 else logits[:, 0]
                out.append(self.torch.sigmoid((logit - self.shift) / self.temp).cpu().numpy())
        return np.concatenate(out)


def get_grader(name: str = "auto") -> Grader:
    key = resolve(name)
    if key not in _CACHE:
        _CACHE[key] = Grader(key)
    return _CACHE[key]
