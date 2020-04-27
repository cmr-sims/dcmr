"""Analyze free recall data."""

import os
import numpy as np
from scipy import io
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


def unpack_array(x):
    if isinstance(x, np.ndarray):
        x = unpack_array(x[0])
    return x


def read_similarity(sim_file):
    """Read pairwise similarity values from a standard MAT-file."""

    mat = io.loadmat(sim_file)
    items = np.array([unpack_array(i) for i in mat['items']])
    similarity = mat['sem_mat']
    vectors = mat['vectors']
    sim = {'items': items, 'vectors': vectors, 'similarity': similarity}
    return sim


def set_item_index(data, items):
    """Set item index based on a pool."""

    data_index = np.empty(data.shape[0])
    data_index.fill(np.nan)
    for idx, item in enumerate(items):
        match = data['item'] == item
        if match.any():
            data_index[match.to_numpy()] = idx
    data.loc[:, 'item_index'] = data_index
    return data
