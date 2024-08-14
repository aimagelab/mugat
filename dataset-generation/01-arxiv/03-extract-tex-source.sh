#!/bin/sh

for file in *.tar.gz; do
    basename=$(basename "$file" .tar.gz)
    echo file $basename
    if test -d $basename; then
        echo skipping
        continue
    fi

    echo going ahead
    mkdir "$basename"

    tar -xf "$file" -C "$basename"
done
