#!/bin/bash
#
# Pull fit results from TACC.

rsync -azvu stampede:work/cmr_cfr/cfr/fits/ \
  "$STUDYDIR"/cfr/fits \
  --include="cmr*/" \
  --include="*.csv" \
  --include="*.json" \
  --exclude="*" \
  "$@"
