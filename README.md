# Predictive Clinical Triage with SHAP-Guided Corrective RAG

A research prototype for explainable ICU clinical decision support. It predicts sepsis
risk from hourly vitals and labs, explains each prediction with SHAP, turns those
explanations into a clinical search query, then grades and corrects the retrieved
evidence before writing a cited summary.

> Decision support only. Not a medical device, and not for clinical use.

```
ICU vitals/labs ─► LightGBM risk ─► TreeSHAP drivers ─► SHAP-to-RAG query
                                                             │
     cited summary ◄─ gpt-oss-20b (Groq) ◄─ graded evidence ◄─ BM25 + BGE (RRF)
                                              ▲                  │
                                              └─ PubMed fallback ◄┘ cross-encoder routing
                                                                   (correct / ambiguous / incorrect)
```

## Project layout
| Path | Paper section | What it does |
|---|---|---|
| `crag/data.py` | III, IV-B | loading, split by hospital, carry-forward, 6-h window features, MICE |
| `crag/model.py` | IV-C | LightGBM / XGBoost / TabNet, SMOTE vs class weights, isotonic calibration, alert threshold, metrics |
| `crag/explain.py` | IV-D | TreeSHAP, aggregation per raw variable, driver selection |
| `crag/bridge.py` | IV-E, Table III | concept lexicon, fixed-template query, MeSH PubMed query |
| `crag/kb.py` | IV-F | chunking, BM25, BGE embeddings in ChromaDB, reciprocal rank fusion |
| `crag/grader.py` | IV-F | cross-encoder grader (BioBERT fine-tuned / MedCPT / MS MARCO) |
| `crag/corrective.py` | IV-F, Fig. 2 | LangGraph routing: correct / ambiguous / incorrect, sentence strips, PubMed fallback |
| `crag/pubmed.py` | IV-F | NCBI E-utilities client |
| `crag/generator.py` | IV-G | Groq gpt-oss-20b prompt (rate-limit retries), offline fallback, citation support check |
| `crag/pipeline.py` | Algorithm 1 | end-to-end run for one patient-hour |
| `app.py` | — | Streamlit dashboard |
| `knowledge/guidelines/` | III-C | 10 original guideline summary notes (source guideline cited in each) |
| `scripts/` | V | download, train, build index, evaluate, calibrate, fine-tune |
| `paper/` | — | the LaTeX paper (Overleaf project) |

## Setup (Windows, Python 3.11)
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env        # optional: add GROQ_API_KEY
```
Versions are pinned because Windows Smart App Control blocks some newer binary
wheels. Keep the project in a short path, such as `Documents\crag-triage`. Very long
paths break numba and ChromaDB.

## Run everything
```bash
python scripts/download_data.py          # PhysioNet 2019, 40,336 patients (resumable)
python scripts/build_kb.py               # PubMed corpus + BM25/BGE index
python scripts/train.py                  # models, SHAP stats, Tables II & VI numbers
streamlit run app.py                     # dashboard
```
Optional and evaluation steps:
```bash
python scripts/finetune_biobert_grader.py            # BioBERT grader (30-60 min CPU)
python scripts/evaluate_retrieval.py pool --n 100    # then label artifacts/annotation/retrieval_pool.csv
python scripts/evaluate_retrieval.py score           # Table VI (retrieval)
python scripts/calibrate_thresholds.py               # tau_low / tau_high from your labels
python scripts/evaluate_generation.py --n 100        # Table VII (generation) + routing split
python scripts/run_case.py                           # one case in the terminal
pytest -q
```

## Where each paper number comes from
| Paper item | File |
|---|---|
| Table II (dataset) | `artifacts/results/dataset_summary.csv` |
| Table VI (prediction), selected model, CIs | `artifacts/results/predictive_metrics.csv`, `selected_model.json` |
| Sec. V-D drivers and bridge trigger rate | `artifacts/results/explanation_stats.json`, `shap_summary.png` |
| Corpus size (Sec. III-C) | `artifacts/results/corpus_stats.json` |
| Retrieval table | `artifacts/results/table_retrieval.csv` |
| Calibrated thresholds | `artifacts/results/thresholds.json` |
| Generation table and routing split | `artifacts/results/table_generation.csv`, `routing_distribution.csv` |

Files whose names end in `_SILVER` use automatic labels from a second cross-encoder.
They are for development only. **Use human relevance labels for any number you
publish.**

## Results so far (full dataset, run 2026-11-11)
| Item | Result |
|---|---|
| Cohort | 40,336 patients; 2,932 septic (7.27%); 1.55 M hourly rows |
| Selected model | XGBoost + SMOTE: AUROC **0.800** (95% CI 0.789–0.811), AUPRC 0.077, held-out Hospital B |
| Other models | LightGBM 0.793–0.796, XGBoost + weights 0.792, TabNet 0.757 |
| Sensitivity drift | threshold set for 0.80 sensitivity on Hospital A gives 0.64 on Hospital B |
| Top SHAP drivers | respiratory rate 46%, WBC 30%, BUN 30% of alerts; bridge triggers for 82% |
| Corpus | 1,315 documents, 2,319 passages |
| BioBERT grader | 86.7% validation accuracy (weak supervision, 1,800 pairs, 1 epoch) |
| Retrieval (SILVER labels only) | SHAP-guided nDCG@5 0.51 vs generic 0.36 |
| Routing split (30 alerts, BioBERT) | 37% correct / 57% ambiguous / 7% incorrect |

Still needed for the paper:
1. **Human relevance labels.** Run `evaluate_retrieval.py pool --n 100` *without* `--silver`,
   label the CSV, then run `score` and `calibrate_thresholds.py`.
2. **A Groq API key in `.env`.** Without one, the generation table cannot measure
   faithfulness, and citation precision is trivially 100% because the offline summariser
   copies evidence sentences. With a key, rerun `evaluate_generation.py --n 100`.

## Privacy
The LLM only receives the risk band, a calibrated probability, binned driver values
(for example "lactate 4–5 mmol/L") and public evidence text. If you use MIMIC-IV, run
generation locally: its data use agreement forbids sending records to third-party APIs.
