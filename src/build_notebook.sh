#!/usr/bin/env bash
# Convert the analysis script to a notebook, execute it, and export PDF + HTML.
set -euo pipefail
cd /home/user/RM_Lab
mkdir -p notebook

jupytext --to notebook --output notebook/readmission_analysis.ipynb src/analysis.py

python3 - <<'PY'
import json
p = "notebook/readmission_analysis.ipynb"
nb = json.load(open(p))
nb["metadata"]["title"] = "Patient-Level Prediction of 30-Day Hospital Readmission"
json.dump(nb, open(p, "w"), indent=1)
print("cells:", len(nb["cells"]))
PY

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 notebook/readmission_analysis.ipynb

jupyter nbconvert --to html notebook/readmission_analysis.ipynb
jupyter nbconvert --to pdf  notebook/readmission_analysis.ipynb || {
  echo "xelatex route failed; falling back to headless Chromium"
  /opt/pw-browsers/chromium-1194/chrome-linux/chrome --headless --disable-gpu --no-sandbox \
    --print-to-pdf=/home/user/RM_Lab/notebook/readmission_analysis.pdf \
    --no-pdf-header-footer file:///home/user/RM_Lab/notebook/readmission_analysis.html
}
ls -la notebook/
