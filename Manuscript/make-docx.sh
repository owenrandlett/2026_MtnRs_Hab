#!/bin/sh
# Build an editable .docx from the LaTeX manuscript.
#   
#   sh make-docx.sh                      -> 2026_MtnRHab.docx
#   sh make-docx.sh --csl=elife.csl      -> same, with a specific citation style
#   sh make-docx.sh --reference-doc=x.docx  -> same, using your own Word styles
# 
# run from "Git Bash" or similar shell in Windows, or from a terminal in Linux/Mac. : 
# sh ./Manuscript/make-docx.sh 

# The PDF build is untouched; this reads the same .tex and .bib.

set -e
cd "$(dirname "$0")"

TEX=2026_MtnRHab.tex
BIB=References_MtnRHab.bib
OUT=2026_MtnRHab.docx
WORK=.docx-build

mkdir -p "$WORK"

# Word keeps an exclusive lock on an open .docx, and pandoc's failure for that
# ("withBinaryFile: permission denied") is not obvious.  Say so plainly.
if [ -e "$OUT" ] && ! ( : >> "$OUT" ) 2>/dev/null; then
    echo "error: cannot write $OUT -- close it in Word first." >&2
    exit 1
fi

# 1. Rewrite the .tex: resolve figure/table numbers and cross-references,
#    lift title/authors into metadata, drop the preamble and \bibliography.
gawk -v meta="$WORK/meta.yaml" -f tools/preprocess.awk "$TEX" > "$WORK/body.tex"

# 2. Prepend the macro shim so pandoc can expand the class's macros.
cat tools/pandoc-shim.tex "$WORK/body.tex" > "$WORK/input.tex"

# 3. Convert.  --citeproc formats \citep/\citet from the .bib and builds the
#    reference list; --resource-path lets \includegraphics resolve.
pandoc "$WORK/input.tex" \
    --from=latex \
    --metadata-file="$WORK/meta.yaml" \
    --citeproc \
    --bibliography="$BIB" \
    --resource-path=. \
    "$@" \
    -o "$OUT"

echo "wrote $OUT"
