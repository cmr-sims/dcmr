#!/bin/bash
#
# Downsample images for the stimulus pool.

src=$1
dest=$2

for cat in cel loc obj; do
    for input in "$src/$cat/"*.jpg; do
        filename=$(basename "$input")
        output=$dest/$cat/$filename
        echo "$input -> $output"
        convert "$input" -resize 300x300 "$output"
    done
done
