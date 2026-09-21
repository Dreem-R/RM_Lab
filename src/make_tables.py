"""Split generated tables into tabular + note, and wrap each in a LaTeX float.

Keeping the caption/label out of the analysis code means the manuscript controls
presentation while the numbers remain machine-generated.
"""
from pathlib import Path

TAB = Path("/home/user/RM_Lab/paper/tables")
MARK = "\\\\[2pt]"

CAPTIONS = {
 "cohort": ("Cohort characteristics by 30-day readmission status.", "tab:cohort"),
 "results": ("Held-out performance on the patient-disjoint test set. Bracketed values are "
             "bootstrap 95\\% confidence intervals over 400 resamples.", "tab:results"),
 "split": ("Optimism induced by encounter-level splitting. Model configurations are held "
           "fixed; only the split changes.", "tab:split"),
 "imbalance": ("Effect of class-imbalance treatments on a fixed LightGBM configuration.",
               "tab:imbalance"),
 "ablation": ("Feature-block ablation on the patient-disjoint test set.", "tab:ablation"),
 "predictors": ("Strongest predictors: logistic-regression odds ratios and SHAP ranking.",
                "tab:predictors"),
 "missing": ("Missing data and the informativeness of missingness. $p$-values are from "
             "$\\chi^2$ tests of the missingness indicator against the outcome.", "tab:missing"),
 "extraction": ("Included studies most relevant to this investigation, with the limitation "
                "each leaves open.", "tab:related"),
}
WIDE = {"results", "split", "imbalance", "cohort", "missing", "ablation"}

for name, (cap, lab) in CAPTIONS.items():
    src = TAB / f"tab_{name}.tex"
    if not src.exists():
        print("  missing, skipped:", src.name); continue
    body = src.read_text()
    note = ""
    if MARK in body:
        body, note = body.split(MARK, 1)
        note = note.strip()
        src.write_text(body.rstrip() + "\n")          # tabular only
        (TAB / f"tab_{name}_note.tex").write_text(note + "\n")
    elif (TAB / f"tab_{name}_note.tex").exists():
        note = (TAB / f"tab_{name}_note.tex").read_text().strip()

    inner = ("\\resizebox{\\textwidth}{!}{\\input{tables/tab_%s}}" % name
             if name in WIDE else "\\input{tables/tab_%s}" % name)
    note_line = ("\n\\\\[3pt]\n\\begin{minipage}{\\textwidth}\\centering %s\\end{minipage}"
                 % note if note else "")
    (TAB / f"tab_{name}_wrapper.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n"
        + "\\caption{%s}\\label{%s}\n" % (cap, lab)
        + inner + note_line + "\n\\end{table}\n")
    print("  wrapped", name, "(note)" if note else "")
print("done")
