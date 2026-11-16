#!/bin/bash
# Build the manuscript with XeLaTeX (Wiley USG template).
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode main.tex > /dev/null
bibtex main
xelatex -interaction=nonstopmode main.tex > /dev/null
xelatex -interaction=nonstopmode main.tex > /dev/null
