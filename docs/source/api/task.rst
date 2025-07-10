Task
====

.. currentmodule:: dcmr.task

The :code:`task` module provides functions for working with task data.

Free-recall data
~~~~~~~~~~~~~~~~

Load and score free-recall data. 
Functions include options for working with categorized free recall data,
labeling category blocks and getting information related to the position,
length, and context of those blocks.

.. autosummary::
    :toctree: api/

    read_study_recall
    merge_free_recall
    read_free_recall

Analyses
~~~~~~~~

Most analyses of free recall data use the Psifr package. 
Additional analyses are included here.

.. autosummary::
    :toctree: api/

    crp_recency
