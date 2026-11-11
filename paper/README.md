# CRAG research paper — Overleaf project

## Upload to Overleaf
1. Overleaf → **New Project → Upload Project** → choose `crag-paper.zip`.
2. Menu → Compiler: **pdfLaTeX**. Main document: `main.tex`.
3. Click **Recompile**. Both figures are drawn in TikZ, so no image files are needed.

## Before you submit — mandatory
- **Replace every red `TBD`** with your own measured results (search `\tbd` in `main.tex`).
  Do not leave or invent numbers.
- Rewrite the lines marked `% Describe...`, `% State...`, `% Interpret...`,
  `% Support or revise...` so they describe what your experiments actually show.
- Section VI (Discussion) paragraphs 2–3 state expected effects. If your results disagree,
  change the text. Negative results are fine to report.
- Check Table II: the patient counts (20,336 / 20,000) and "roughly 7% septic" come from
  the PhysioNet 2019 challenge paper (Reyna et al., 2020). Confirm them against the paper.
- Check every entry in `references.bib` (volume, pages) on Google Scholar.
- If you add a SHAP summary plot, put it in a `figures/` folder (a placeholder comment
  is in Section V-D).

## Page length
About 5,100 words + 2 figures + 7 tables + 1 algorithm ≈ 8 IEEE pages.
Once results are in:
- **Too long:** shorten Section II, or remove Table V (configuration).
- **Too short:** add a SHAP beeswarm plot and a ROC/PR curve figure.

## Plagiarism
The text is written from scratch, but no one can promise a Turnitin score in advance.
- Run it through your college's checker yourself.
- Equations, reference lists and dataset descriptions (e.g., "40,336 patients")
  always produce some matches. Most checkers let you exclude the bibliography and quotes.
- Rewrite any flagged sentence in your own words. That also helps with AI-detection checks.
