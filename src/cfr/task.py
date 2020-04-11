"""Analyze free recall data."""

import os
import pandas as pd
from psifr import fr


def read_free_recall(csv_file):
    """Read and score free recall data."""

    if not os.path.exists(csv_file):
        raise ValueError(f'Data file does not exist: {csv_file}')

    data = pd.read_csv(csv_file, dtype={'category': 'category'})
    data.category.cat.as_ordered(inplace=True)

    study = data.query('trial_type == "study"').copy()
    recall = data.query('trial_type == "recall"').copy()

    # additional fields
    list_keys = ['session']
    fields = ['list_type', 'list_category', 'distractor']
    for field in fields:
        if field in data:
            list_keys += [field]
    merged = fr.merge_lists(study, recall, list_keys=list_keys,
                            study_keys=['category'])
    return merged
