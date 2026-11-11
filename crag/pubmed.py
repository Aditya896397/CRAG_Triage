"""PubMed access through NCBI E-utilities (used for corpus building and live fallback)."""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import requests

from . import config as C

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_last_call = [0.0]


def _params(extra: dict) -> dict:
    p = {"db": "pubmed", "tool": "crag-triage"}
    if C.NCBI_API_KEY:
        p["api_key"] = C.NCBI_API_KEY
    if C.NCBI_EMAIL:
        p["email"] = C.NCBI_EMAIL
    p.update(extra)
    return p


def _get(endpoint: str, params: dict, retries: int = 3) -> requests.Response:
    # NCBI allows 3 requests/s without a key, 10/s with one.
    gap = 0.11 if C.NCBI_API_KEY else 0.35
    for attempt in range(retries):
        wait = gap - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = requests.get(EUTILS + endpoint, params=_params(params), timeout=30)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    raise RuntimeError(f"PubMed request failed: {endpoint}")


def search(term: str, retmax: int = C.PUBMED_MAX, sort: str = "relevance") -> list[str]:
    r = _get("esearch.fcgi", {"term": term, "retmax": retmax, "retmode": "json", "sort": sort})
    return r.json().get("esearchresult", {}).get("idlist", [])


def fetch(pmids: list[str]) -> list[dict]:
    """Return [{pmid, title, abstract, year, journal, url}] for PMIDs with an abstract."""
    out = []
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i + 100]
        r = _get("efetch.fcgi", {"id": ",".join(batch), "retmode": "xml"})
        root = ET.fromstring(r.content)
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID", default="")
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            parts = []
            for ab in art.iter("AbstractText"):
                label = ab.get("Label")
                text = "".join(ab.itertext()).strip()
                parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(p for p in parts if p)
            if not abstract:
                continue
            year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
            out.append({
                "pmid": pmid, "title": title.strip(), "abstract": abstract,
                "year": year, "journal": art.findtext(".//Journal/Title", default=""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
    return out


def search_and_fetch(term: str, retmax: int = C.PUBMED_MAX) -> list[dict]:
    ids = search(term, retmax)
    return fetch(ids) if ids else []


def relax(mesh_query: str) -> str:
    """Rewrite used when the strict MeSH query returns nothing: drop the type filter
    and turn the MeSH conjunction into a disjunction."""
    tail = ' AND ("Sepsis"[MeSH Terms] OR sepsis[tiab]) AND ("last 10 years"[dp]) ' \
           'NOT ("Infant"[MeSH Terms] OR "Child"[MeSH Terms])'
    if not mesh_query.startswith('(("'):          # no concept terms: broad sepsis search
        return '("Sepsis"[MeSH Terms] AND management)' + tail
    core = mesh_query.split(' AND ("Sepsis"')[0].replace(" AND ", " OR ")
    return core + tail
