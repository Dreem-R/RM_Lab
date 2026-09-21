# Declaration of AI Assistance and Complete Prompt History

Submission item (d): complete prompt history for any AI-generated content.

| | |
|---|---|
| **Paper** | Beyond the Encounter-Level Benchmark: Patient-Level Validation, Imbalance Corrections and the Provenance of Signal in 30-Day Hospital Readmission Prediction |
| **Author** | Ruchak Khatri |
| **Affiliation** | Department of Computer Science, CHRIST (Deemed to be University), Bengaluru, India |
| **Email** | ruchak.khatri@bcah.christuniversity.in |
| **Date** | 21 September 2026 |

---

## 1. Tools used and scope of assistance

| Tool | Role |
|---|---|
| Claude Code (Anthropic, Opus 5) | Analysis pipeline, figures, tables, manuscript drafting, LaTeX build |
| Google Gemini (in Google Colab) | Assisted the author's literature-search harness; later gave an independent review of the draft |
| OpenAlex API | Programmatic retrieval of literature candidates |

**Session transcript:** https://claude.ai/code/session_01JxATpExgJe5k1Xv8HErCtR

AI assistance was used for code, analysis and drafting under the author's direction.
The research question, the acceptance of every result, and the final text are the
author's responsibility. No result was written by hand: every number in the paper is
produced by `src/analysis.py` from the raw dataset and read from
`artifacts/results.json`, so the manuscript cannot drift from the analysis.

## 2. The author's independent contributions

1. **Literature search harness.** The author had previously written and maintained a
   reproducible search notebook (`lit/lit_search.ipynb`) which queries the OpenAlex
   API, logs every query with its date and hit count, de-duplicates results and emits
   a LaTeX search log. This was run independently and supplied as input.
2. **Pre-registered screening criteria.** Inclusion and exclusion criteria were fixed
   in the notebook *before* any search was executed, precisely so that criteria could
   not be rationalised after seeing the results.
3. **Template sourcing.** The author obtained the official Springer Nature LNCS
   Overleaf template and supplied it for conformance checking; it proved to be
   version 2.25, newer than the copy in the build environment, and was adopted.
4. **Independent parallel analysis.** The author produced a separate draft analysis
   using their own Colab code. Its results (XGBoost AUROC 0.681, AUPRC 0.227) closely
   corroborate the final reported figures, and its calibration plots independently
   showed the same probability-overestimation pattern.
5. **Independent review.** The draft was submitted to a second AI system (Gemini) for
   review, and the author conducted a page-by-page visual inspection of the compiled
   PDF.
6. **Defect identification.** That inspection found three genuine layout defects that
   were then corrected (Section 4).

## 3. Complete chronological prompt record

All prompts are reproduced verbatim as entered by the author.

### Prompt 1 — Task specification

> im in a research methodology in machine learning hackathon and we are given this thing question
>
> you are provided with a domain-specific background and problem statement on the following page based on the given case, conduct an independent research investigation and communicate your findings as a research paper
>
> conduct a literature review to understand the existing research, methodologies and research gaps related to the given problem. based on the existing literature and the background, formulate a clear research problem and objective(s).
>
> identify and select suitable dataset(s) relevant to the research problem and justify the choice of dataset(s). perform appropriate data preparation and exploratory data analysis (EDA) including visualisation, statistical analysis and feature engineering wherever relevant. Develop, evaluate and compare suitable machine learning models and justify the meaningful findings, relate them to the existing literature and discuss relevant limitations and implications
>
> present the complete investigation as a research paper using overleaf, including appropriate sections, figures, tables, citations, references and submit the paper along with the code/notebook and dataset source/details
>
> "HOSPITAL records contain information about patient demographics, diagnoses, laboratory measurements, procedures, medications and previous admissions. readmission following discharge is influenced by a combination of clinical and non-clinical factors. large publicly available healthcare datasets contain de-identified records that can be used to investigate relationship between patient characteristics and subsequent outcomes. however, healthcare data often contain missing values, imbalanced outcomes and numerous interacting variables."
>
> format: springer nature template (lecture notes in computer science)

Followed by the submission checklist: (a) final PDF in journal format, (b) zip of the
Overleaf project, (c) PDF and HTML export of the notebook, (d) complete prompt history.

### Prompt 2 — Literature search results

> i ran this old script i had in google colab and got this in cell 5 output

Attached: `lit/lit_search.ipynb` and the console output listing 77 de-duplicated
candidate papers with titles, years, venues, DOIs and abstracts.

### Prompt 3 — Literature artefacts

> these are some files i got from that same google colab code i had kept ready, maybe it comes of use for research paper making

Attached: `search_log.tex`, `extraction.tex`, `refs.bib`, `screening_log.tex`.

### Prompt 4 — Progress check

> whats the status? where are we at right now?

### Prompt 5 — Independent parallel draft

> this is a side document i made btw till you were doing your work. please dont keep this as your own. im just sending to let you know what i was doing in background (i made with help of my google colab previous code and there build in gemini ai). maybe it comes to use idk not necessary to use this just in case im giving

Attached: `Draft_1.pdf`. At the author's explicit instruction this document was **not**
incorporated into the paper; it was used only as an independent cross-check of results.

### Prompt 6 — Request to publish work in progress

> if some part of work is done can you push also went you can without effecting your progress so i can see from my side also in github? or you cant push?

### Prompt 7 — Template conformance and integrity checks

> this is the correct format i found inside overleaf template and im pretty sure this is the right one. so you can check this for reference if needed [...] also let me know if you need any details to be added in research paper like personal details like my name and institution and stuff and make sure there is no ai filler like "PUT IMAGE HERE", "REPLACE THIS WITH REAL DATA" and ofc as mentioned no synthetic data to be used a actual dataset should be used so please crosscheck these things and format also

Attached: the official Springer Nature LNCS Overleaf template.

### Prompt 8 — Author details

> name: Ruchak Khatri\
> Department: Department of Computer Science\
> Institution: CHRIST (Deemed to be University), Bengaluru, India\
> email: ruchak.khatri@bcah.christuniversity.in

### Prompt 9 — Progress check

> whats the status

### Prompt 10 — Defect report and external review findings

The author reported three layout defects found by page-by-page inspection of the
compiled PDF (legend text overlapping the bars in Figure 6; colliding axis labels in
Figure 8; a table floating into the reference list), and forwarded four items from an
independent Gemini review of the manuscript for verification.

### Prompt 11 — This document

> where is the prompt history? can i get that in doc format to submit that as well

## 4. Verification and quality control

Because AI-generated material was used, each of the following was checked against a
primary source rather than accepted as produced.

### 4.1 Defects found by the author's inspection

| Defect | Resolution |
|---|---|
| Figure 6: legend box overlapped the bars; paired value labels collided | Shared legend moved below the panels; value labels rotated |
| Figure 8: cumulative-block axis labels collided | Labels rotated to 25 degrees |
| Table 9 floated into the reference list | Float barrier inserted before the bibliography |

A subsequent systematic page-by-page review, prompted by these findings, identified and
corrected six further issues of the same class in Figures 1, 4, 5, 7 and 9, the appendix
table header, and table typography.

### 4.2 Integrity problem found in the supplied literature artefacts

The `screening_log.tex` and `extraction.tex` files produced by the Colab harness were
found to be unusable: their decision reasons belonged to an unrelated study on
dishonesty and observability experiments, and the decisions were inconsistent with this
topic — Strack et al. (2014), the paper describing the dataset used here, had been
excluded as a "conceptual paper on algorithmic nudging". Submitting them would have
misrepresented the screening process. All 77 candidates were therefore re-screened
against the pre-registered criteria (`src/screening.py`), yielding 23 included studies,
and the extraction table was rewritten from the retrieved abstracts. The search log
itself, and the bibliography entries retrieved from OpenAlex, were verified as genuine
and retained.

### 4.3 External review findings, verified

Four items raised by the independent Gemini review were checked against the compiled
PDF and found to be **false positives** arising from PDF text-layer extraction, which
does not reliably recover mathematical glyphs. The rendered document correctly shows
the delta symbol in the abstract and Sections 6.1 and 6.4, the "not an element of"
symbol in Section 3, the full model ordering in Section 7, and the disputed comma in
the abstract. No change was required. Verifying rather than acting on these reports
avoided introducing four errors into a correct document.

### 4.4 Data authenticity

No synthetic or simulated data was used. The pipeline performs exactly one data read,
of the raw UCI file. Its integrity was verified against the published record:

| Property | Local file | UCI published |
|---|---|---|
| Rows x columns | 101,766 x 50 | 101,766 x 50 |
| Unique patients | 71,518 | 71,518 |
| Class distribution | NO 54,864 / >30 35,545 / <30 11,357 | identical |

SHA-256: `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97`
(recorded in `data/CHECKSUM.txt`).

The word "synthetic" appears twice in the paper. Both occurrences refer to SMOTE —
Synthetic Minority Over-sampling Technique — which is one of the published methods
evaluated, and found harmful, in Section 6.3. It does not refer to fabricated data.

### 4.5 Error found and corrected during the analysis

An error in the evaluation code was identified and corrected before submission. The
top-decile alert metric selected cases by thresholding (`p >= threshold`), which for
the isotonic-recalibrated model flagged 4,115 encounters instead of the intended 2,476,
because isotonic regression collapses predictions onto roughly 100 distinct values.
Selection is now by rank. This corrected that model's reported sensitivity from 0.341
to 0.232; the value was then verified independently. Models with continuous outputs
were unaffected.

## 5. Reproducibility

All randomness is seeded (`RANDOM_STATE = 42`). The pipeline was executed end to end
several times during development; the final re-execution reproduced every reported
figure exactly — LightGBM AUROC 0.6770, AUPRC 0.2228, SMOTE-NC AUPRC 0.1554,
net-benefit zero crossing 0.113 — confirming the run is deterministic.

Reproduce with:

```
pip install numpy pandas scikit-learn matplotlib scipy xgboost lightgbm shap imbalanced-learn
python3 src/analysis.py
python3 src/screening.py
python3 src/make_tables.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 6. Declaration

I declare that this submission was prepared with the AI assistance described above,
that the assistance is disclosed in full, and that I have verified the data, the
results and the literature record against primary sources. The interpretation of the
findings and the final text are my own responsibility.

Signed: ............................................  Date: ......................

Ruchak Khatri
