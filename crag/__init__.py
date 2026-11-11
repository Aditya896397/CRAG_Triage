"""SHAP-guided Corrective RAG for explainable clinical triage."""

# On Windows, importing pyarrow (used by Streamlit tables) BEFORE onnxruntime (used by
# ChromaDB) segfaults. Loading onnxruntime first avoids the crash.
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass
