"""Stage 4: deterministic SHAP-to-RAG bridge (paper Table III)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .explain import Driver


@dataclass(frozen=True)
class Rule:
    concept: str
    mesh: str
    test: callable  # value -> bool


# Thresholds adapted from SIRS / SOFA conventions. Each variable may have
# a high and a low rule; the first that fires is used.
LEXICON: dict[str, list[Rule]] = {
    "MAP": [Rule("hypotension", "Hypotension", lambda v: v < 65)],
    "SBP": [Rule("hypotension", "Hypotension", lambda v: v < 90)],
    "DBP": [Rule("hypotension", "Hypotension", lambda v: v < 50)],
    "Lactate": [Rule("raised lactate", "Acidosis, Lactic", lambda v: v > 2)],
    "HR": [Rule("tachycardia", "Tachycardia", lambda v: v > 90),
           Rule("bradycardia", "Bradycardia", lambda v: v < 50)],
    "Resp": [Rule("tachypnoea", "Tachypnea", lambda v: v > 20)],
    "Temp": [Rule("fever", "Fever", lambda v: v > 38),
             Rule("hypothermia", "Hypothermia", lambda v: v < 36)],
    "O2Sat": [Rule("hypoxaemia", "Hypoxia", lambda v: v < 92)],
    "SaO2": [Rule("hypoxaemia", "Hypoxia", lambda v: v < 92)],
    "WBC": [Rule("leukocytosis", "Leukocytosis", lambda v: v > 12),
            Rule("leukopenia", "Leukopenia", lambda v: v < 4)],
    "Platelets": [Rule("thrombocytopenia", "Thrombocytopenia", lambda v: v < 150)],
    "Creatinine": [Rule("renal dysfunction", "Acute Kidney Injury", lambda v: v > 1.2)],
    "BUN": [Rule("renal dysfunction", "Acute Kidney Injury", lambda v: v > 25)],
    "Bilirubin_total": [Rule("hepatic dysfunction", "Hyperbilirubinemia", lambda v: v > 1.2)],
    "pH": [Rule("acidaemia", "Acidosis", lambda v: v < 7.35)],
    "HCO3": [Rule("metabolic acidosis", "Acidosis", lambda v: v < 22)],
    "BaseExcess": [Rule("metabolic acidosis", "Acidosis", lambda v: v < -4)],
    "Glucose": [Rule("hyperglycaemia", "Hyperglycemia", lambda v: v > 180)],
    "Potassium": [Rule("hyperkalaemia", "Hyperkalemia", lambda v: v > 5.5)],
    "FiO2": [Rule("increased oxygen requirement", "Respiratory Insufficiency", lambda v: v > 0.4)],
    "PTT": [Rule("coagulopathy", "Blood Coagulation Disorders", lambda v: v > 45)],
}

UNITS = {"HR": "/min", "Resp": "/min", "MAP": "mmHg", "SBP": "mmHg", "DBP": "mmHg",
         "Temp": "°C", "O2Sat": "%", "SaO2": "%", "Lactate": "mmol/L", "WBC": "x10^3/uL",
         "Platelets": "x10^3/uL", "Creatinine": "mg/dL", "BUN": "mg/dL",
         "Bilirubin_total": "mg/dL", "Glucose": "mg/dL", "Potassium": "mmol/L",
         "HCO3": "mmol/L", "PTT": "s"}

QUERY_TEMPLATE = ("Evidence-based assessment and management of {concepts} "
                  "in an adult ICU patient at {band} risk of sepsis.")
GENERIC_QUERY = "Evidence-based assessment and management of sepsis in adult ICU patients."


@dataclass
class Concept:
    variable: str
    value: float
    concept: str
    mesh: str
    shap: float


@dataclass
class BridgeOutput:
    concepts: list[Concept]
    query: str
    mesh_query: str


def _fires(var: str, value: float | None) -> Rule | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for rule in LEXICON.get(var, []):
        if rule.test(value):
            return rule
    return None


def join_concepts(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def build_mesh_query(meshes: list[str], years: int = 10) -> str:
    core = " AND ".join(f'"{m}"[MeSH Terms]' for m in meshes) if meshes else ""
    sepsis = '("Sepsis"[MeSH Terms] OR "Shock, Septic"[MeSH Terms])'
    types = '(guideline[pt] OR practice guideline[pt] OR review[pt] OR systematic review[pt])'
    date = f'("last {years} years"[dp])'
    parts = [f"({core})"] if core else []
    adult = 'NOT ("Infant"[MeSH Terms] OR "Child"[MeSH Terms] OR "Pediatrics"[MeSH Terms])'
    return " AND ".join(parts + [sepsis, types, date]) + " " + adult


def build_query(drivers: list[Driver], band: str) -> BridgeOutput:
    """Turn SHAP drivers into a deterministic local query and a MeSH PubMed query."""
    concepts: list[Concept] = []
    seen = set()
    for d in drivers:
        rule = _fires(d.variable, d.value)
        if rule and rule.concept not in seen:
            seen.add(rule.concept)
            concepts.append(Concept(d.variable, float(d.value), rule.concept, rule.mesh, d.shap))
    if not concepts:
        return BridgeOutput([], GENERIC_QUERY, build_mesh_query([]))
    names = [c.concept for c in concepts]
    query = QUERY_TEMPLATE.format(concepts=join_concepts(names), band=band)
    meshes = list(dict.fromkeys(c.mesh for c in concepts))
    return BridgeOutput(concepts, query, build_mesh_query(meshes))


def binned(var: str, value: float) -> str:
    """Coarse value description sent to the LLM instead of the raw number."""
    step = {"HR": 10, "Resp": 4, "MAP": 5, "SBP": 10, "DBP": 5, "Temp": 0.5, "O2Sat": 2,
            "Lactate": 1, "WBC": 2, "Platelets": 25, "Creatinine": 0.5, "BUN": 10,
            "Bilirubin_total": 0.5, "Glucose": 20, "Potassium": 0.5}.get(var, None)
    unit = UNITS.get(var, "")
    if not step:
        return f"{value:.3g} {unit}".strip()
    lo = math.floor(value / step) * step
    fmt = (lambda x: f"{x:.1f}") if step < 1 else (lambda x: f"{x:.0f}")
    return f"{fmt(lo)}-{fmt(lo + step)} {unit}".strip()


def all_abnormal_query(observed: dict[str, float | None], band: str) -> str:
    """Ablation baseline: every abnormal variable, no SHAP ranking."""
    names = []
    for var, val in observed.items():
        rule = _fires(var, val)
        if rule and rule.concept not in names:
            names.append(rule.concept)
    if not names:
        return GENERIC_QUERY
    return QUERY_TEMPLATE.format(concepts=join_concepts(names), band=band)
