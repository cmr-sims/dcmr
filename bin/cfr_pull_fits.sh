#!/bin/bash
#
# Pull fit results from TACC.

src=$1
dest=$2
shift 2

rsync -azvu "$src/cfr/fits/" \
  "$dest/cfr/fits" \
  --include="*/" \
  --include="*/cmr*/" \
  --include="*/cmr*/figs/" \
  --include="*/cmr*/.css" \
  --include="*.csv" \
  --include="*.json" \
  --include="*.pdf" \
  --include="*.txt" \
  --include="*.css" \
  --include="*.html" \
  --include="*.svg" \
  --exclude="*" \
  "$@"
