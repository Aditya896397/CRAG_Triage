"""Stage 6: grounded generation and sentence-level citation support check.

Privacy: the prompt contains only the risk band, a calibrated probability,
binned driver values and public evidence text. No identifiers, timestamps
or raw time series are ever sent to the LLM.
"""
from __future__ import annotations

import re
import time

from . import config as C
from .bridge import Concept, binned
from .grader import get_grader
from .kb import Passage, split_sentences

SYSTEM = (
    "You are a clinical decision-support assistant for ICU clinicians. "
    "You summarise evidence; you do not diagnose or prescribe. "
    "Use ONLY the numbered evidence passages provided. Every factual sentence in the "
    "'Relevant evidence' and 'Considerations for clinician review' sections must end with "
    "one or more citations like [P1] or [P2][P3]. If the evidence does not cover a point, "
    "say that the retrieved evidence is insufficient. Do not give drug doses. Be concise."
)

SECTIONS = ["Risk summary", "Contributing observations", "Relevant evidence",
            "Considerations for clinician review"]


def build_prompt(prob: float, band: str, concepts: list[Concept], evidence: list[Passage]) -> str:
    obs = "\n".join(f"- {c.concept} ({c.variable} {binned(c.variable, c.value)})" for c in concepts) \
        or "- no triggered abnormal driver"
    ev = "\n\n".join(f"[P{i}] {p.title}\n{p.text}" for i, p in enumerate(evidence, start=1))
    return (
        f"Model output: {band} risk of sepsis within 6 hours (calibrated probability {prob:.3f}).\n"
        f"Main contributing observations (from SHAP):\n{obs}\n\n"
        f"Evidence passages:\n{ev}\n\n"
        "Write the summary with these four headings, in this order: "
        + "; ".join(SECTIONS) + "."
    )


def groq_chat(messages: list[dict], max_tokens: int = 700, retries: int = 6) -> str:
    """One Groq chat call with rate-limit back-off. For gpt-oss reasoning models the
    reasoning tokens count against max_tokens, so extra room is added and the
    reasoning text is hidden from the answer."""
    from groq import Groq, RateLimitError
    client = Groq(api_key=C.GROQ_API_KEY)
    kw = {}
    if "gpt-oss" in C.GROQ_MODEL:
        kw = {"reasoning_effort": C.GROQ_REASONING, "reasoning_format": "hidden"}
        max_tokens += 1500
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=C.GROQ_MODEL, temperature=0,
                                               max_tokens=max_tokens, messages=messages, **kw)
            return (r.choices[0].message.content or "").strip()
        except RateLimitError as e:
            wait = 5 * (attempt + 1)
            try:
                wait = float(e.response.headers.get("retry-after", wait)) + 1
            except Exception:
                pass
            time.sleep(min(wait, 60))
    raise RuntimeError("Groq rate limit: retries exhausted")


def call_groq(prompt: str) -> str:
    return groq_chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])


def offline_summary(prob: float, band: str, concepts: list[Concept], evidence: list[Passage],
                    query: str) -> str:
    """Extractive fallback when no LLM key is configured: picks the best-supported
    evidence sentences and cites them. Keeps the app usable offline."""
    g = get_grader()
    lines = [f"**Risk summary**\nThe model estimates a {band} risk of sepsis within 6 hours "
             f"(calibrated probability {prob:.3f}). This is decision support, not a diagnosis."]
    obs = "; ".join(f"{c.concept} ({c.variable} {binned(c.variable, c.value)})" for c in concepts)
    lines.append("**Contributing observations**\n" + (obs or "No abnormal driver triggered."))
    cands = []
    for i, p in enumerate(evidence, start=1):
        for s in split_sentences(p.text):
            cands.append((s, i))
    picked = []
    if cands:
        scores = g.score(query, [c[0] for c in cands])
        for (s, i), sc in sorted(zip(cands, scores), key=lambda x: -x[1])[:5]:
            picked.append(f"- {s.rstrip('.')}. [P{i}]")
    lines.append("**Relevant evidence**\n" + ("\n".join(picked) or
                 "The retrieved evidence is insufficient for this presentation."))
    lines.append("**Considerations for clinician review**\nReview the cited passages against the "
                 "full clinical picture; the system does not recommend specific treatment.")
    return "\n\n".join(lines)


CITE = re.compile(r"\[P(\d+)\]")


def support_check(answer: str, evidence: list[Passage], tau: float = C.TAU_CITE) -> dict:
    """Score every cited sentence against the passages it cites."""
    g = get_grader()
    kept, flagged, total = [], [], 0
    out_lines = []
    for line in answer.splitlines():
        if not CITE.search(line):
            out_lines.append(line)
            continue
        prefix = re.match(r"^\s*([-*]\s+|\d+\.\s+)?", line).group(0)
        pieces = re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", line[len(prefix):])
        rebuilt = []
        for piece in pieces:
            cites = [int(x) for x in CITE.findall(piece)]
            if not cites:
                rebuilt.append(piece)
                continue
            total += 1
            sent = CITE.sub("", piece).strip(" -*")
            valid = [c for c in cites if 1 <= c <= len(evidence)]
            best = max((float(g.score(sent, [evidence[c - 1].text])[0]) for c in valid), default=0.0)
            if best >= tau:
                kept.append(sent); rebuilt.append(piece)
            else:
                flagged.append({"sentence": sent, "support": best})
                rebuilt.append(f"~~{piece.strip()}~~ *(removed: not supported by cited passage)*")
        out_lines.append(prefix + " ".join(rebuilt))
    precision = len(kept) / total if total else float("nan")
    return {"answer": "\n".join(out_lines), "citation_precision": precision,
            "cited_sentences": total, "flagged": flagged}


def generate(prob: float, band: str, concepts: list[Concept], evidence: list[Passage],
             query: str) -> dict:
    t0 = time.time()
    if not evidence:
        text = ("**Insufficient evidence**\nNo retrieved passage passed the relevance check, "
                "so no evidence summary is generated. Please consult local protocols.")
        return {"raw": text, "answer": text, "backend": "none", "citation_precision": float("nan"),
                "cited_sentences": 0, "flagged": [], "seconds": time.time() - t0}
    if C.GROQ_API_KEY:
        raw, backend = call_groq(build_prompt(prob, band, concepts, evidence)), f"groq:{C.GROQ_MODEL}"
    else:
        raw, backend = offline_summary(prob, band, concepts, evidence, query), "offline-extractive"
    chk = support_check(raw, evidence)
    return {"raw": raw, "backend": backend, "seconds": time.time() - t0, **chk}
