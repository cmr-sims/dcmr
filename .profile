# Set up Bash environment for running project scripts.

if [[ $USER = morton ]]; then
    conda activate cfr
    export STUDYDIR=$HOME/Dropbox/work/cmr_cfr/cfr
else
    . /work/03206/mortonne/lonestar/venv/cfr/bin/activate
    unset PYTHONPATH
    export STUDYDIR=/work/03206/mortonne/lonestar/cmr_cfr/cfr
fi

export STUDY=cmr_cfr
export BATCHDIR=$STUDYDIR/batch
