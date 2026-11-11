"""Knowledge base: chunking, BM25 + BGE dense index (ChromaDB), RRF fusion."""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass

import numpy as np

from . import config as C

PAEDIATRIC = re.compile(r"\b(infant|neonat\w*|newborn|child\w*|paediatric|pediatric|"
                        r"adolescen\w*|preterm|young infants)\b", re.I)

STOP = set("""a an the of and or in on for to with without by at from as is are be was were
this that these those it its into than then which who whom what when where how all any
each other such no not only own same so too very can will just should now""".split())


@dataclass
class Passage:
    id: str
    text: str
    title: str
    source: str        # citation string shown to the clinician
    year: str
    url: str
    origin: str        # "local" | "pubmed"
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def split_sentences(text: str) -> list[str]:
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text.strip())
    return [x.strip() for x in s if len(x.strip()) > 20]


def chunk_words(text: str, size: int = C.CHUNK_TOKENS, overlap: int = C.CHUNK_OVERLAP) -> list[str]:
    """Approximate token windows (~0.75 words per token) on sentence boundaries."""
    max_words = int(size * 0.75)
    ov_words = int(overlap * 0.75)
    sents, chunks, cur = split_sentences(text) or [text], [], []
    for s in sents:
        w = s.split()
        if cur and len(cur) + len(w) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-ov_words:] if ov_words else []
        cur.extend(w)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ------------------------------------------------------------------ corpus
def load_guidelines() -> list[Passage]:
    """Markdown files: '# Title', 'Source: ...', 'Year: ...', '## Section' blocks."""
    out = []
    for f in sorted(C.GUIDELINE_DIR.glob("*.md")):
        txt = f.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", txt, re.M).group(1).strip()
        source = (re.search(r"^Source: (.+)$", txt, re.M) or [None, "Local guideline note"])[1]
        year = (re.search(r"^Year: (.+)$", txt, re.M) or [None, ""])[1]
        url = (re.search(r"^URL: (.+)$", txt, re.M) or [None, ""])[1]
        for sec in re.split(r"^## ", txt, flags=re.M)[1:]:
            head, _, body = sec.partition("\n")
            body = re.sub(r"\s+", " ", body).strip()
            for j, ch in enumerate(chunk_words(body)):
                out.append(Passage(f"{f.stem}#{len(out)}", ch, f"{title} — {head.strip()}",
                                   source.strip(), year.strip(), url.strip(), "local"))
    return out


def load_pubmed_corpus() -> list[Passage]:
    out = []
    if not C.PUBMED_CACHE.exists():
        return out
    for line in C.PUBMED_CACHE.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        for j, ch in enumerate(chunk_words(d["abstract"])):
            out.append(Passage(f"pmid{d['pmid']}#{j}", ch, d["title"],
                               f"{d['journal']} ({d['year']}), PMID {d['pmid']}",
                               d["year"], d["url"], "local"))
    return out


# ------------------------------------------------------------------ embeddings
_EMB = {}


def embedder():
    if "m" not in _EMB:
        from sentence_transformers import SentenceTransformer
        _EMB["m"] = SentenceTransformer(C.EMBED_MODEL, device="cpu")
    return _EMB["m"]


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    if query:
        texts = [C.BGE_QUERY_PREFIX + t for t in texts]
    return embedder().encode(texts, batch_size=32, normalize_embeddings=True,
                             show_progress_bar=len(texts) > 200)


# ------------------------------------------------------------------ index
class Retriever:
    def __init__(self, passages: list[Passage], bm25, collection):
        self.passages = passages
        self.by_id = {p.id: p for p in passages}
        # target population is adult ICU patients: paediatric literature is never returned
        self.excluded = {p.id for p in passages if PAEDIATRIC.search(p.title)}
        self.bm25 = bm25
        self.col = collection

    # ---- build / load
    @classmethod
    def build(cls, passages: list[Passage]) -> "Retriever":
        import chromadb
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([tokenize(p.text + " " + p.title) for p in passages], k1=C.BM25_K1, b=C.BM25_B)
        client = chromadb.PersistentClient(path=str(C.INDEX_DIR / "chroma"))
        try:
            client.delete_collection("guidelines")
        except Exception:
            pass
        col = client.create_collection("guidelines", metadata={"hnsw:space": "cosine"})
        vecs = embed([p.title + ". " + p.text for p in passages])
        for i in range(0, len(passages), 2000):
            sl = passages[i:i + 2000]
            col.add(ids=[p.id for p in sl], embeddings=vecs[i:i + 2000].tolist(),
                    documents=[p.text for p in sl])
        with open(C.INDEX_DIR / "passages.pkl", "wb") as fh:
            pickle.dump((passages, bm25), fh)
        return cls(passages, bm25, col)

    @classmethod
    def load(cls) -> "Retriever":
        import chromadb
        path = C.INDEX_DIR / "passages.pkl"
        if not path.exists():
            raise FileNotFoundError("Index missing. Run: python scripts/build_kb.py")
        with open(path, "rb") as fh:
            passages, bm25 = pickle.load(fh)
        client = chromadb.PersistentClient(path=str(C.INDEX_DIR / "chroma"))
        return cls(passages, bm25, client.get_collection("guidelines"))

    # ---- search
    def search_bm25(self, query: str, k: int = C.TOP_K) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:k]
        return [self.passages[i].id for i in idx if scores[i] > 0]

    def search_dense(self, query: str, k: int = C.TOP_K) -> list[str]:
        q = embed([query], query=True)[0].tolist()
        res = self.col.query(query_embeddings=[q], n_results=k)
        return res["ids"][0]

    def search(self, query: str, k: int = C.TOP_K, mode: str = "hybrid") -> list[Passage]:
        n = k + 10
        if mode == "bm25":
            ids = self.search_bm25(query, n)
        elif mode == "dense":
            ids = self.search_dense(query, n)
        else:
            ids = rrf([self.search_bm25(query, 3 * k), self.search_dense(query, 3 * k)])
        ids = [i for i in ids if i not in self.excluded][:k]
        return [Passage(**{**self.by_id[i].to_dict(), "score": 0.0}) for i in ids]


def rrf(rankings: list[list[str]], k: int = C.RRF_K) -> list[str]:
    """Reciprocal rank fusion (Cormack et al., 2009)."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + r)
    return sorted(score, key=score.get, reverse=True)
