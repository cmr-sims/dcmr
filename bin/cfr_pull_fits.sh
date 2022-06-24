#!/bin/bash
#
# Pull fit results from TACC.

if [[ $# -lt 2 ]]; then
    echo "Usage:   cfr_pull_fits.sh src dest [rsync flags]"
    echo "Example: cfr_pull_fits.sh lonestar:work/cmr_cfr/cfr/fits ~/Dropbox/work/cmr_cfr/cfr/fits"
    exit 1
fi

src=$1
dest=$2
shift 2

rsync -azvu "$src/" \
    "$dest" \
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
