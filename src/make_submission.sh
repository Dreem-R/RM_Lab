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
llncs.cls (version 2.25, 03-Sep-2026) and splncs04.bst are the official
Springer LNCS files, taken from the Springer Nature LNCS Overleaf template
and included so the project compiles without extra setup.

The preamble follows the template: T1 font encoding, newtxtext/newtxmath
(Times Roman), and the Springer URL style.

Before submitting, edit the author block near the top of main.tex
The author block is already filled in.
TXT
(cd /tmp/overleaf && zip -qr /home/user/RM_Lab/submission/overleaf_project.zip .)

# (a) final PDF, (c) notebook exports
cp paper/main.pdf submission/paper_LNCS.pdf
cp notebook/readmission_analysis.pdf  submission/ 2>/dev/null || echo "  (notebook PDF missing)"
cp notebook/readmission_analysis.html submission/ 2>/dev/null || echo "  (notebook HTML missing)"
cp notebook/readmission_analysis.ipynb submission/ 2>/dev/null || true
cp PROMPTS.md submission/prompt_history.md
# (d) prompt history also as a Word document and a PDF
if [ -f submission/prompt_history.docx ]; then echo "  prompt_history.docx present"; fi
pandoc PROMPTS.md -o submission/prompt_history.pdf --pdf-engine=xelatex \
  -V geometry:a4paper -V geometry:margin=2.4cm -V fontsize=10pt -V colorlinks=true \
  --toc --toc-depth=2
cp data/DATASET.md submission/dataset_source.md

# a single archive with everything, for convenience
rm -rf /tmp/allsub && mkdir -p /tmp/allsub
cp submission/* /tmp/allsub/ 2>/dev/null || true
(cd /tmp/allsub && zip -qr /home/user/RM_Lab/submission/full_submission.zip .)

ls -la submission/
