#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."   # project root
cd report
for i in 1 2 3; do
    pdflatex -interaction=nonstopmode main.tex > /dev/null
    bibtex main > /dev/null 2>&1 || true
done
echo "built report/main.pdf"
