# Set up Bash environment for running project scripts.

if [[ $USER = morton ]]; then
    conda activate cfr
    export STUDYDIR=$HOME/Dropbox/work/cmr_cfr
    export CFR_RESULTS=$HOME/Dropbox/work/cmr_cfr/cfr
    export CFR_FITS=$HOME/Dropbox/work/cmr_cfr/cfr/fits/v5
    export CFR_FIGURES=$HOME/Dropbox/work/cmr_cfr/cfr/figs/v1
else
    . /work/03206/mortonne/lonestar/venv/cfr/bin/activate
    unset PYTHONPATH
    export STUDYDIR=/work/03206/mortonne/lonestar/cmr_cfr
fi

export STUDY=cmr_cfr
export BATCHDIR=$STUDYDIR/batch
