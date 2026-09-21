"""Split generated tables into tabular + note, and wrap each in a LaTeX float.

Keeping the caption/label out of the analysis code means the manuscript controls
presentation while the numbers remain machine-generated.
"""
import re
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

# Raw one-hot column names are unreadable in a manuscript table; the analysis keeps
# them verbatim so the code stays traceable, and presentation is handled here.
PREFIX = [(r"discharge\\_group\\_", "discharge: "), (r"payer\\_code\\_", "payer: "),
          (r"medical\\_specialty\\_", "specialty: "), (r"diag\\_1\\_group\\_", "diagnosis 1: "),
          (r"diag\\_2\\_group\\_", "diagnosis 2: "), (r"diag\\_3\\_group\\_", "diagnosis 3: "),
          (r"admission\\_source\\_", "adm. source: "), (r"admission\\_type\\_", "adm. type: ")]

def prettify_features(body):
    for pat, rep in PREFIX:
        body = re.sub(pat, rep, body)
    body = body.replace(r"number\_inpatient", "prior inpatient visits")
    body = body.replace(r"diabetesMed\_No", "no diabetes medication")
    body = body.replace("ObstetricsandGynecology", "Obstetrics \\& Gynaecology")
    body = body.replace("Surgery-Cardiovascular/Thoracic", "Cardiothoracic surgery")
    body = body.replace("Orthopedics-Reconstructive", "Reconstructive orthopaedics")
    return body

for name, (cap, lab) in CAPTIONS.items():
    src = TAB / f"tab_{name}.tex"
    if not src.exists():
        print("  missing, skipped:", src.name); continue
    body = src.read_text()
    # "-0.0469" typesets as a hyphen in text mode; wrap signed numbers in maths
    body = re.sub(r"(?<=[&\s])([+-])(\d+\.\d+)(?=\s*(?:&|\\\\))", r"$\1\2$", body)
    if name == "predictors":
        body = prettify_features(body)
    src.write_text(body)
    note = ""
    if MARK in body:
        body, note = body.split(MARK, 1)
        note = note.strip()
        src.write_text(body.rstrip() + "\n")          # tabular only
        note = re.sub(r"(?<=\\d)(\\d{3})\\b(?!,)",
                      lambda m: "," + m.group(1),
                      note) if False else note
        note = re.sub(r"n=(\\d{4,})", lambda m: "n={:,}".format(int(m.group(1))), note)
        (TAB / f"tab_{name}_note.tex").write_text(note + "\n")
    elif (TAB / f"tab_{name}_note.tex").exists():
        note = (TAB / f"tab_{name}_note.tex").read_text().strip()

    if note:
        # thousands separators, and wording that matches the rank-based alert rule
        note = re.sub(r"n=(\d{4,})", lambda m: "n={:,}".format(int(m.group(1))), note)
        note = note.replace("at the threshold that flags the highest-risk decile",
                            "by flagging the highest-risk decile of encounters")
        (TAB / f"tab_{name}_note.tex").write_text(note + "\n")

    # cap at \textwidth but never scale a narrow table UP above body-text size
    inner = ("\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}"
             "{\\input{tables/tab_%s}}" % name
             if name in WIDE else "\\input{tables/tab_%s}" % name)
    note_line = ("\n\\\\[3pt]\n\\begin{minipage}{\\textwidth}\\centering %s\\end{minipage}"
                 % note if note else "")
    (TAB / f"tab_{name}_wrapper.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n"
        + "\\caption{%s}\\label{%s}\n" % (cap, lab)
        + inner + note_line + "\n\\end{table}\n")
    print("  wrapped", name, "(note)" if note else "")
print("done")
