#!/bin/bash
base=$(basename $1 /)

cd "$1" || continue

echo it

tex_files=(*.tex)

if [[ ! -f ../../html/$base.html ]]; then
    if [[ ${#tex_files[@]} -eq 1 ]]; then
        latexmlc "${tex_files[0]}" --dest ../../html/$base.html --timeout=120
    fi

    if [[ ${#tex_files[@]} -gt 1 && -f "main.tex" ]]; then
        latexmlc "main.tex" --dest ../../html/$base.html --timeout=120
    fi
else
    echo skipping
fi

cd ..