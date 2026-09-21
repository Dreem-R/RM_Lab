#!/usr/bin/env bash
# Assemble the four submission artefacts.
set -euo pipefail
cd /home/user/RM_Lab
mkdir -p submission
rm -f submission/*.zip

# (b) Overleaf project: everything needed to compile on Overleaf, nothing else.
rm -rf /tmp/overleaf && mkdir -p /tmp/overleaf
cp paper/main.tex paper/refs.bib paper/llncs.cls paper/splncs04.bst /tmp/overleaf/
cp -r paper/figures /tmp/overleaf/figures
cp -r paper/tables  /tmp/overleaf/tables
rm -f /tmp/overleaf/figures/*.png            # PDFs are what main.tex includes
cat > /tmp/overleaf/README.txt <<'TXT'
Overleaf project — upload this zip via "New Project > Upload Project".

Compiler: pdfLaTeX. Main document: main.tex. Bibliography: refs.bib (BibTeX).
llncs.cls and splncs04.bst are the official Springer LNCS class and style,
included so the project compiles without extra setup.

Before submitting, edit the author block near the top of main.tex
(the line marked ">>> EDIT THESE THREE LINES BEFORE SUBMITTING <<<").
TXT
(cd /tmp/overleaf && zip -qr /home/user/RM_Lab/submission/overleaf_project.zip .)

# (a) final PDF, (c) notebook exports
cp paper/main.pdf submission/paper_LNCS.pdf
cp notebook/readmission_analysis.pdf  submission/ 2>/dev/null || echo "  (notebook PDF missing)"
cp notebook/readmission_analysis.html submission/ 2>/dev/null || echo "  (notebook HTML missing)"
cp notebook/readmission_analysis.ipynb submission/ 2>/dev/null || true
cp PROMPTS.md submission/prompt_history.md
cp data/DATASET.md submission/dataset_source.md

# a single archive with everything, for convenience
rm -rf /tmp/allsub && mkdir -p /tmp/allsub
cp submission/* /tmp/allsub/ 2>/dev/null || true
(cd /tmp/allsub && zip -qr /home/user/RM_Lab/submission/full_submission.zip .)

ls -la submission/
