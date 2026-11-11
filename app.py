"""Streamlit clinician dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import crag  # noqa: F401  (must load before streamlit/pyarrow, see crag/__init__.py)
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crag import config as C
from crag.data import row_from_manual
from crag.pipeline import TriagePipeline

st.set_page_config(page_title="CRAG Clinical Triage", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading models and index ...")
def get_pipeline(grader: str) -> TriagePipeline:
    return TriagePipeline(grader=grader)


@st.cache_data
def demo_patients() -> pd.DataFrame:
    return pd.read_pickle(C.DATA_PROCESSED / "demo_patients.pkl")


@st.cache_resource
def imputer():
    return joblib.load(C.MODEL_DIR / "imputer.joblib")


ROUTE_STYLE = {"correct": ("#1a7f37", "Correct: local evidence kept"),
               "ambiguous": ("#b35900", "Ambiguous: local evidence + PubMed"),
               "incorrect": ("#cf222e", "Incorrect: MeSH rewrite, PubMed only"),
               "filter": ("#57606a", "Grader filter only"),
               "standard": ("#57606a", "Standard RAG (no grading)")}

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 CRAG Triage")
st.sidebar.caption("SHAP-guided corrective RAG · research prototype, not a medical device")
source = st.sidebar.radio("Patient source", ["Test-set patient (Hospital B)", "Manual entry"])
mode = st.sidebar.selectbox("Retrieval strategy", ["crag", "filter", "standard"],
                            format_func={"crag": "Full CRAG (proposed)", "filter": "Grader filter only",
                                         "standard": "Standard RAG"}.get)
retrieval_mode = st.sidebar.selectbox("Retriever", ["hybrid", "bm25", "dense"])
grader = st.sidebar.selectbox("Grader", ["auto", "medcpt", "biobert", "msmarco"],
                              help="auto = fine-tuned BioBERT if trained, else MedCPT")
force = st.sidebar.checkbox("Retrieve even below alert threshold", value=False)
st.sidebar.markdown(f"LLM: **{'Groq · ' + C.GROQ_MODEL if C.GROQ_API_KEY else 'offline extractive (set GROQ_API_KEY)'}**")

try:
    pipe = get_pipeline(grader)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

row, trajectory = None, None
if source.startswith("Test"):
    demo = demo_patients()
    septic = demo.groupby("patient_id")[C.LABEL].max()
    pid = st.sidebar.selectbox("Patient", septic.index,
                               format_func=lambda p: f"{p} · {'develops sepsis' if septic[p] else 'no sepsis'}")
    pdf = demo[demo["patient_id"] == pid].reset_index(drop=True)
    X = pdf[pipe.model.features].to_numpy(np.float32)
    trajectory = pd.DataFrame({"hour": pdf["hour"], "risk": pipe.model.predict_proba(X),
                               "label": pdf[C.LABEL]})
    default_hour = int(trajectory["risk"].idxmax())
    hour = st.sidebar.slider("ICU hour", 0, len(pdf) - 1, default_hour)
    row = pdf.iloc[hour]
else:
    st.sidebar.markdown("**Current values** (leave 0 for not measured)")
    fields = {"HR": 125.0, "MAP": 60.0, "SBP": 84.0, "Temp": 39.0, "Resp": 28.0, "O2Sat": 94.0,
              "Lactate": 4.3, "WBC": 15.0, "Platelets": 140.0, "Creatinine": 1.6}
    vals = {k: st.sidebar.number_input(k, value=v, step=0.1) for k, v in fields.items()}
    vals = {k: v for k, v in vals.items() if v}
    age = st.sidebar.number_input("Age", value=65.0)
    row = row_from_manual(vals, imputer(), pipe.model.features,
                          {"Age": age, "Gender": 1, "ICULOS": 24, "HospAdmTime": -10})

run = st.sidebar.button("Run pipeline", type="primary", use_container_width=True)

# ------------------------------------------------------------------ main
st.title("Predictive Clinical Triage with SHAP-Guided Corrective RAG")

if trajectory is not None:
    fig = go.Figure()
    fig.add_scatter(x=trajectory["hour"], y=trajectory["risk"], mode="lines", name="calibrated risk")
    fig.add_hline(y=pipe.model.threshold, line_dash="dash", line_color="red",
                  annotation_text="alert threshold", annotation_position="bottom right")
    pos = trajectory[trajectory["label"] == 1]
    if len(pos):
        fig.add_vrect(x0=pos["hour"].min(), x1=pos["hour"].max(), fillcolor="orange", opacity=0.15,
                      line_width=0, annotation_text="label: sepsis ≤ 6 h",
                      annotation_position="top left")
    fig.add_vline(x=int(row["hour"]), line_color="gray")
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="risk",
                      xaxis_title="ICU hour", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

if not run:
    st.info("Choose a patient and hour in the sidebar, then press **Run pipeline**.")
    st.stop()

with st.spinner("Predicting, explaining, retrieving and generating ..."):
    res = pipe.run(row, mode=mode, retrieval_mode=retrieval_mode, force=force)

c1, c2, c3 = st.columns([1, 1.4, 1.6])
with c1:
    st.subheader("1 · Risk")
    # Hourly sepsis prevalence is ~2%, so calibrated risks are small; scale the gauge to 4x threshold.
    top = max(4 * 100 * res["threshold"], 100 * res["prob"] * 1.1)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * res["prob"], number={"suffix": "%", "valueformat": ".2f"},
        gauge={"axis": {"range": [0, top]}, "bar": {"color": "#cf222e" if res["alert"] else "#1a7f37"},
               "threshold": {"line": {"color": "white", "width": 3}, "value": 100 * res["threshold"]}}))
    gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=0))
    st.plotly_chart(gauge, use_container_width=True)
    ratio = res["prob"] / res["threshold"]
    st.metric("Risk band", res["band"].upper(), f"{ratio:.1f}x alert threshold",
              delta_color="inverse" if res["alert"] else "off")
    st.caption("Calibrated probability of sepsis within 6 h. Baseline hourly rate ≈ 1.4–2%.")
with c2:
    st.subheader("2 · Why (SHAP)")
    phi = res["phi"].head(10)[::-1]
    bar = go.Figure(go.Bar(x=phi.values, y=phi.index, orientation="h",
                           marker_color=["#cf222e" if v > 0 else "#0969da" for v in phi.values]))
    bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="aggregated SHAP (log-odds)")
    st.plotly_chart(bar, use_container_width=True)
with c3:
    st.subheader("3 · SHAP-to-RAG query")
    if res["status"] != "ok":
        st.write(res["status"])
        st.stop()
    drv = pd.DataFrame([{"variable": d.variable, "value": d.value, "SHAP": round(d.shap, 3)}
                        for d in res["drivers"]])
    st.dataframe(drv, hide_index=True, use_container_width=True)
    b = res["bridge"]
    st.markdown("**Local query**")
    st.code(b.query, language=None)
    with st.expander("PubMed MeSH query"):
        st.code(b.mesh_query, language=None)

st.divider()
st.subheader("4 · Corrective retrieval")
color, text = ROUTE_STYLE.get(res["route"], ("#57606a", res["route"]))
best = res["best"]
st.markdown(f"<span style='background:{color};color:white;padding:4px 10px;border-radius:6px'>"
            f"{text}</span> &nbsp; best grader score S = "
            f"{'n/a' if best != best else f'{best:.3f}'}", unsafe_allow_html=True)
with st.expander("Routing log"):
    for line in res["log"]:
        st.write("•", line)
for i, p in enumerate(res["evidence"], start=1):
    with st.container(border=True):
        st.markdown(f"**[P{i}] {p.title}**  \n<small>{p.source} · {p.origin} · score {p.score:.3f}"
                    + (f" · [link]({p.url})" if p.url else "") + "</small>", unsafe_allow_html=True)
        st.write(p.text)
if not res["evidence"]:
    st.warning("No passage passed the relevance check.")

st.divider()
st.subheader("5 · Grounded summary")
g = res["generation"]
st.markdown(g["answer"])
m1, m2, m3 = st.columns(3)
m1.metric("Citation precision", "n/a" if g["citation_precision"] != g["citation_precision"]
          else f"{100 * g['citation_precision']:.0f}%")
m2.metric("Cited sentences", g["cited_sentences"])
m3.metric("Total latency", f"{res['seconds']:.1f} s")
st.caption(f"Generator: {g['backend']}. Only binned values and public evidence are sent to the LLM. "
           "Decision support only; does not replace clinical judgement.")
