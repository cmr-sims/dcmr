"""Analyze free recall data."""

import os
import glob
import re
import shutil
import numpy as np
from scipy import io
from scipy import stats
import matplotlib.pyplot as plt
from skimage import transform
import pandas as pd
from psifr import fr
from cymr import network
from wikivector import vector


def get_subjects():
    subject_nos = [
        1, 2, 3, 5, 8,
        11, 16, 18,
        22, 23, 24, 25, 27, 28, 29,
        31, 32, 33, 34, 35, 37, 38,
        40, 41, 42, 43, 44, 45, 46
    ]
    subject_ids = [f'LTP{no:03d}' for no in subject_nos]
    return subject_ids, subject_nos


def set_list_columns(data, columns):
    """Set columns that are defined for each list."""
    modified = data.copy()
    for column in columns:
        unique_value = data[column][data[column].notna()].unique()
        if len(unique_value) > 1:
            raise ValueError(f"Column {column} has multiple values.")
        modified[column] = unique_value[0]
    return modified


def read_free_recall(csv_file):
    """Read and score free recall data."""
    if not os.path.exists(csv_file):
        raise ValueError(f'Data file does not exist: {csv_file}')

    data = pd.read_csv(csv_file, dtype={'category': 'category'})
    data.category.cat.as_ordered(inplace=True)

    # additional list fields
    list_keys = ['session']
    fields = ['list_type', 'list_category', 'distractor']
    for field in fields:
        if field in data:
            list_keys += [field]
    data = data.groupby(['subject', 'list']).apply(set_list_columns, list_keys)

    # split, add block fields to study
    study = data.query('trial_type == "study"').copy()
    study = label_block(study)
    recall = data.query('trial_type == "recall"').copy()

    # merge study and recalle events
    study_keys = ['item_index', 'category', 'block', 'n_block', 'block_pos',
                  'block_len']
    merged = fr.merge_lists(study, recall, list_keys=list_keys,
                            study_keys=study_keys)
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


def save_patterns_w2v(mat_file, use_file, h5_file):
    """Read wiki2vec data and write patterns and similarity matrices."""
    mat = read_similarity(mat_file)
    use_patterns, use_items = vector.load_vectors(use_file)
    np.testing.assert_array_equal(mat['items'], use_items)

    # localist patterns
    loc_patterns = np.eye(mat['items'].shape[0])

    # category patterns
    category = np.repeat(['cel', 'loc', 'obj'], 256)
    cat_patterns = np.zeros((mat['items'].shape[0], 3))
    cat_names = np.unique(category)
    for i in range(3):
        cat_patterns[category == cat_names[i], i] = 1

    # semantic patterns based on wiki2vec
    sem_patterns = mat['vectors']

    # normalize so that dot product is equal to correlation
    sem_z = stats.zscore(sem_patterns, axis=1) / np.sqrt(sem_patterns.shape[1])

    # use vectors
    use_z = stats.zscore(use_patterns, axis=1) / np.sqrt(use_patterns.shape[1])

    # write to standard format hdf5 file
    network.save_patterns(h5_file, mat['items'], loc=loc_patterns,
                          cat=cat_patterns, w2v=sem_z, use=use_z)


def read_pool_cfr(image_dir):
    """Read CFR pool information."""
    sub_dirs = ['celebrities', 'locations', 'objects']
    categories = ['cel', 'loc', 'obj']

    names = []
    tags = []
    filepaths = []
    item_category = []
    pattern = re.compile("([A-Za-z0-9_'.]*){([A-Za-z]*)}")
    for sub_dir, category in zip(sub_dirs, categories):
        cat_dir = os.path.join(image_dir, sub_dir)
        if not os.path.exists(cat_dir):
            raise IOError(f'Category directory not found: {cat_dir}')

        matches = sorted(glob.glob(os.path.join(cat_dir, '*.jpg')))
        for match in matches:
            filename = os.path.basename(match)
            m = re.match(pattern, filename)
            if m is None:
                raise IOError(f'Failed to parse filename: {filename}')
            names.append(m.group(1).replace('_', ' ').strip())
            tags.append(m.group(2))
            filepaths.append(match)
            item_category.append(category)
    pool = pd.DataFrame({'item': names, 'category': item_category,
                         'tag': tags, 'filepath': filepaths})
    return pool


def save_pool_images(pool, image_dir):
    """Save pool images in standard format."""
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    categories = pool['category'].unique()
    sub_dir = {}
    for category in categories:
        sub_dir[category] = os.path.join(image_dir, category)
        if not os.path.exists(sub_dir[category]):
            os.makedirs(sub_dir[category])

    for i, item in pool.iterrows():
        old_path = item['filepath']
        new_path = os.path.join(sub_dir[item['category']], item['item'] + '.jpg')
        shutil.copy2(old_path, new_path)


def load_pool_images(pool, image_dir, rescale=None):
    """Load pool images."""
    if not os.path.exists(image_dir):
        raise IOError(f'Image directory does not exist: {image_dir}')

    images = {}
    for i, item in pool.iterrows():
        image_file = os.path.join(image_dir, item['category'],
                                  item['item'] + '.jpg')
        image = plt.imread(image_file)
        if len(image.shape) == 2:
            image = np.tile(image[:, :, None], 3)
        image = np.asarray(image, dtype=float) / 255

        if rescale is not None:
            new_shape = np.array(image.shape).copy()
            new_shape[:2] = new_shape[:2] * rescale
            rescaled = transform.resize(image, new_shape)
        else:
            rescaled = np.asarray(image, dtype=float)

        images[item['item']] = rescaled
    return images


def label_clean_trials(data):
    """Label study and recall trials as clean or not."""
    # score data
    merged = fr.merge_free_recall(data)

    # make scored data comparable to raw data
    merged_recall = merged.query('recall').copy()
    merged_recall['trial_type'] = 'recall'
    merged_recall['position'] = merged_recall['output'].astype('int')

    # merge to copy scoring to raw data
    merge_keys = ['subject', 'list', 'item', 'trial_type', 'position']
    rmerged = pd.merge(
        data, merged_recall, left_on=merge_keys, right_on=merge_keys, how='outer'
    )

    # filter to label clean recalls
    rmerged['intrusion'] = rmerged['intrusion'].fillna(False)
    clean = rmerged.query('trial_type == "study" | (~intrusion & repeat == 0)')
    label = np.zeros(data.shape[0], dtype=bool)
    label[clean.index] = True
    labeled = data.copy()
    labeled['clean'] = label
    return labeled


def get_train_category(category):
    """Given current category, get previous and baseline categories."""
    trial_prev = ''
    trial_base = ''
    prev = []
    base = []
    for i, trial_curr in enumerate(category):
        prev.append(trial_prev)
        base.append(trial_base)
        if i < len(category) - 1 and trial_curr != category[i + 1]:
            trial_base = trial_prev
            trial_prev = trial_curr
    return np.array(prev), np.array(base)


def label_block_category(data):
    """Label block category."""
    study_data = fr.filter_data(data, trial_type='study')
    labeled = data.copy()
    for _, list_data in study_data.groupby(['subject', 'list']):
        prev, base = get_train_category(list_data['category'].to_numpy())
        labeled.loc[list_data.index, 'prev'] = prev
        labeled.loc[list_data.index, 'base'] = base
    labeled['prev'] = labeled['prev'].astype('category')
    labeled['base'] = labeled['base'].astype('category')
    labeled['prev'][labeled['prev'] == ''] = np.nan
    labeled['base'][labeled['base'] == ''] = np.nan
    return labeled


def label_block(data):
    labeled = data.copy()
    labeled['block'] = (
        data.groupby(['subject', 'list'])['category']
            .transform(fr.block_index)
    )
    # get the number of blocks for each study list
    n_block = labeled.groupby(['subject', 'list'])['block'].max()
    n_block.name = 'n_block'

    # merge the n_block field
    labeled = pd.merge(labeled, n_block, left_on=['subject', 'list'],
                       right_on=['subject', 'list'], how='outer')

    # position within block
    labeled.loc[:, 'block_pos'] = labeled.groupby(
        ['subject', 'list', 'block']
    )['position'].cumcount() + 1
    block_len = labeled.groupby(['subject', 'list', 'block'])['block_pos'].max()
    block_len.name = 'block_len'
    labeled = pd.merge(labeled, block_len, left_on=['subject', 'list', 'block'],
                       right_on=['subject', 'list', 'block'], how='outer')
    return labeled
