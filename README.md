# Beyond the Encounter-Level Benchmark

Research-methodology investigation of 30-day hospital readmission prediction on the
*Diabetes 130-US Hospitals (1999--2008)* database.

## Submission contents

| Item | Path |
|---|---|
| (a) Final paper, LNCS format | `paper/main.pdf` |
| (b) Overleaf project (zip) | `submission/overleaf_project.zip` |
| (c) Notebook + PDF/HTML export | `notebook/readmission_analysis.ipynb`, `.pdf`, `.html` |
| (d) Prompt history | `PROMPTS.md` |
| Dataset details | `data/DATASET.md` |

## Dataset

**Diabetes 130-US hospitals for years 1999-2008** — UCI Machine Learning Repository,
ID 296, donated by Strack et al. (2014), licensed CC BY 4.0.
101,766 encounters, 71,518 patients, 130 hospitals, 50 variables.
Canonical page: <https://archive.ics.uci.edu/dataset/296/>
DOI of the source publication: 10.1155/2014/781670

The raw file `data/diabetic_data.csv` (19 MB, 101,766 rows) is unmodified.

## Reproducing

```bash
pip install numpy pandas scikit-learn matplotlib scipy xgboost lightgbm shap imbalanced-learn
python3 src/analysis.py       # ~12 min; writes figures, tables and artifacts/results.json
python3 src/screening.py      # regenerates the literature screening tables
python3 src/make_tables.py    # wraps generated tables as LaTeX floats
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

All randomness is seeded (`RANDOM_STATE = 42`); a rerun reproduces every number in the
paper.

## Layout

```
data/       raw dataset + provenance note
src/        analysis.py, screening.py, make_tables.py
notebook/   executed notebook and its PDF/HTML exports
paper/      main.tex, refs.bib, llncs.cls, figures/, tables/
lit/        the literature-search harness used to retrieve candidates
artifacts/  results.json, fitted models, run log
```

## Headline findings

1. Patient-level LightGBM: AUROC 0.677, AUPRC 0.223 (prevalence 0.113); the margin over
   regularised logistic regression is reliable but small (+0.017 AUPRC).
2. Encounter-level splitting shares 36% of test patients with training yet inflates
   AUROC by only ~0.003 — a reported negative result.
3. Imbalance corrections do not improve ranking and destroy calibration (ECE 0.003 to
   >0.34); SMOTE-NC costs 30% of AUPRC.
4. Demographics alone are non-informative (AUROC 0.510); prior utilisation contributes
   more than all other feature blocks combined.
