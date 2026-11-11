"""Central configuration. Values mirror Table V of the paper."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # silence ChromaDB telemetry
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------- paths
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
KNOWLEDGE_DIR = ROOT / "knowledge"
GUIDELINE_DIR = KNOWLEDGE_DIR / "guidelines"
PUBMED_CACHE = KNOWLEDGE_DIR / "pubmed_corpus.jsonl"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"
INDEX_DIR = ARTIFACTS / "index"
for _p in (DATA_PROCESSED, MODEL_DIR, RESULTS_DIR, INDEX_DIR):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------- dataset
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
DYNAMIC = VITALS + LABS
LABEL = "SepsisLabel"

# Administrative variables are model inputs but never retrieval drivers.
NON_CLINICAL = {"Unit1", "Unit2", "HospAdmTime", "ICULOS", "Gender", "Age"}

# Forward-fill carry window in hours (vitals short, labs long).
CARRY_HOURS = {v: 4 for v in VITALS} | {v: 24 for v in LABS}

# ---------------------------------------------------------------- features / model
WINDOW = 6                      # trailing window (hours)
TARGET_SENSITIVITY = 0.80       # alert threshold chosen to reach this on validation
SHAP_TOP_K = 3
SHAP_DELTA = 0.01               # minimum aggregated SHAP (log-odds) to count as a driver

# ---------------------------------------------------------------- retrieval
CHUNK_TOKENS = 256
CHUNK_OVERLAP = 32
BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60
TOP_K = 10
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Grader: fine-tuned BioBERT cross-encoder if present, else MedCPT cross-encoder.
BIOBERT_GRADER_DIR = MODEL_DIR / "biobert_grader"
MEDCPT_GRADER = "ncbi/MedCPT-Cross-Encoder"
MSMARCO_GRADER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GRADER = os.getenv("GRADER", "auto")  # auto | biobert | medcpt | msmarco

# Routing thresholds (initial; overwritten by artifacts/results/thresholds.json if calibrated)
TAU_HIGH = 0.75
TAU_LOW = 0.35
TAU_STRIP = 0.35                # keep sentence strips at or above this (best 2 always kept)
TAU_CITE = 0.30                 # cited sentence must reach this support score

# ---------------------------------------------------------------- PubMed / LLM
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
PUBMED_MAX = 8
PUBMED_YEARS = 10

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq retired its Llama-3.1 chat models in 2026; gpt-oss-20b is an open-weight
# (Apache-2.0) replacement that can also be self-hosted for MIMIC-IV experiments.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_REASONING = os.getenv("GROQ_REASONING", "low")   # gpt-oss reasoning effort
