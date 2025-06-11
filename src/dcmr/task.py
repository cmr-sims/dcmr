"""Analyze free recall data."""

import os
import glob
import re
import shutil
import click
import numpy as np
from scipy import io
from scipy import stats
import matplotlib.pyplot as plt
from skimage import transform
import pandas as pd
import polars as pl
from psifr import fr
from cymr import cmr
from wikivector import vector


def get_subjects():
    """Get subject IDs and numbers for the CFR study."""
    subject_nos = np.hstack(
        [
            [1, 2, 3, 5, 8],
            [11, 16, 18],
            [22, 23, 24, 25, 27, 28, 29],
            [31, 32, 33, 34, 35, 37, 38],
            [40, 41, 42, 43, 44, 45, 46],
        ]
    )
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


def get_prev_category(category):
    """Given current category for a list, get previous category."""
    prev = category.copy()
    prev[:] = None
    category = np.asarray(category)
    trial_prev = None
    for i in range(1, len(category)):
        if category[i - 1] != category[i]:
            # just shifted category; update previous category
            trial_prev = category[i - 1]
        prev.iloc[i] = trial_prev
    return prev


def label_block_category(data):
    """Label block category."""
    study_data = fr.filter_data(data, trial_type='study')
    labeled = data.copy()
    labeled['curr'] = labeled['category']
    labeled['prev'] = study_data.groupby(['subject', 'list'])['category'].transform(
        get_prev_category
    )
    labeled['base'] = labeled['category'].copy()
    labeled.loc[:, 'base'] = None
    ucat = study_data['category'].unique()
    for (curr, prev), df in labeled.groupby(['curr', 'prev'], observed=True):
        if not prev:
            continue
        labeled.loc[df.index, 'base'] = np.setdiff1d(ucat, [curr, prev])[0]
    return labeled


def block_index(list_labels):
    """
    Get index of each block in a list.

    Parameters
    ----------
    list_labels : pandas.Series
        Position labels that define the blocks.

    Returns
    -------
    block : pandas.Series
        Block index of each position.
    """
    prev_label = ''
    curr_block = 0
    block = pd.Series(len(list_labels), dtype="int64[pyarrow]")
    for i, label in enumerate(list_labels):
        if pd.isnull(label):
            # null values of label have no block and do not advance block
            block[i] = None
            continue

        if prev_label != label:
            curr_block += 1
        block[i] = curr_block
        prev_label = label
    block.index = list_labels.index
    return block


def label_block(data):
    """Label features of category blocks."""
    # get the index of each contiguous block of same-category items
    labeled = data.copy()
    list_keys = ['subject', 'list', 'trial_type']
    block_keys = list_keys + ['block']
    labeled['block'] = data.groupby(list_keys)['category'].transform(block_index)

    # position within block
    labeled['block_pos'] = (labeled.groupby(block_keys)['block'].cumcount() + 1).astype('int64[pyarrow]')

    # block length
    labeled['block_len'] = labeled.groupby(block_keys)['block_pos'].transform('max')

    # get the number of blocks for each study list
    labeled['n_block'] = labeled.groupby(list_keys)['block'].transform('max')
    return labeled


def read_study_recall(csv_file, block=True, block_category=True):
    """Read study and recall data."""
    if not os.path.exists(csv_file):
        raise ValueError(f'Data file does not exist: {csv_file}')

    data = pd.read_csv(csv_file, engine='pyarrow', dtype_backend='pyarrow')
    if 'category' in data.columns:
        data = data.astype({'category': 'category'})
        data.category = data.category.cat.as_ordered()
    else:
        block = False
        block_category = False

    if block:
        data = label_block(data)

    if block_category:
        data = label_block_category(data)

    # additional list fields
    list_keys = ['session']
    fields = ['list_type', 'list_category', 'distractor']
    for field in fields:
        if field in data:
            list_keys += [field]
    data = data.groupby(['subject', 'list']).apply(set_list_columns, list_keys)
    data = data.reset_index(drop=True)
    return data


def read_free_recall(csv_file, block=True, block_category=True):
    """Read and score free recall data."""
    data = read_study_recall(csv_file, block=block, block_category=block_category)

    # split, add block fields to study
    study = data.query('trial_type == "study"').copy()
    recall = data.query('trial_type == "recall"').copy()

    # merge study and recall events
    study_keys = [i for i in ['item_index', 'category'] if i in data]
    if block:
        study_keys += ['block', 'n_block', 'block_pos', 'block_len']
    if block_category:
        study_keys += ['curr', 'prev', 'base']
    list_keys = [
        i for i in ['session', 'list_type', 'list_category', 'distractor'] if i in data
    ]
    merged = fr.merge_lists(study, recall, list_keys=list_keys, study_keys=study_keys)
    return merged


def crp_recency(data, op_thresh=3, edges=None, labels=None, sp_edges=(5, 19)):
    """Calculate a persistant recency effect lag-CRP analysis."""
    if edges is None:
        edges = [-19.5, -16.5, -5.5, -1.5, 0, 1.5, 5.5, 16.5, 19.5]
    if labels is None:
        labels = [-18, -10.5, -3.5, -1, 1, 3.5, 10.5, 18]

    crp_early = fr.lag_crp(
        data,
        test_key="input", 
        test=lambda x, y: (x >= sp_edges[0]) & (x <= sp_edges[1]),
        item_query=f"output <= {op_thresh} or not recall"
    )
    crp_early["Output"] = f"OP <= {op_thresh}"

    crp_late = fr.lag_crp(
        data,
        test_key="input", 
        test=lambda x, y: (x >= sp_edges[0]) & (x <= sp_edges[1]),
        item_query=f"output > {op_thresh} or not recall"
    )
    crp_late["Output"] = f"OP > {op_thresh}"
    crp = pd.concat([crp_early, crp_late], ignore_index=True)
    
    crp["Lag"] = pd.cut(crp["lag"], edges, labels=labels)
    m = crp.groupby(["subject", "Output", "Lag"], observed=True)["prob"].mean()
    res = m.reset_index()
    res["Lag"] = res["Lag"].astype(float)
    return res


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
    clean = rmerged.query('(trial_type == "study") | (~intrusion & (repeat == 0))')
    label = np.zeros(data.shape[0], dtype=bool)
    label[clean.index.to_numpy()] = True
    labeled = data.copy()
    labeled['clean'] = label
    return labeled


def unpack_array(x):
    """Recursively unpack an array with one element."""
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


def save_patterns_sem(use_file, h5_file):
    """Read wiki2USE data and write semantic patterns."""
    patterns, items = vector.load_vectors(use_file)

    # localist patterns
    loc_patterns = np.eye(len(items))

    # category patterns
    category = np.repeat(['cel', 'loc', 'obj'], 256)
    cat_patterns = np.zeros((len(items), 3))
    cat_names = np.unique(category)
    for i in range(3):
        cat_patterns[category == cat_names[i], i] = 1

    # use vectors
    use_z = stats.zscore(patterns, axis=1) / np.sqrt(patterns.shape[1])

    # write to standard format hdf5 file
    cmr.save_patterns(h5_file, items, loc=loc_patterns, cat=cat_patterns, use=use_z)


@click.command()
@click.argument("use_file", type=click.Path(exists=True))
@click.argument("h5_file", type=click.Path())
def save_patterns_cdcatfr2(use_file, h5_file):
    """Read wiki2USE data and write patterns for cdcatfr2."""
    patterns, items = vector.load_vectors(use_file)

    # localist patterns
    loc_patterns = np.eye(len(items))

    # use vectors
    use_z = stats.zscore(patterns, axis=1) / np.sqrt(patterns.shape[1])

    # write to standard format hdf5 file
    cmr.save_patterns(h5_file, items, loc=loc_patterns, use=use_z)


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
    pool = pd.DataFrame(
        {'item': names, 'category': item_category, 'tag': tags, 'filepath': filepaths}
    )
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
        image_file = os.path.join(image_dir, item['category'], item['item'] + '.jpg')
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


@click.command()
@click.argument("matrix_file", type=click.Path(exists=True))
@click.argument("frdata_file", type=click.Path())
def convert_matrix(matrix_file, frdata_file):
    """Convert recalls matrix to standard format."""
    LIST_LENGTH = 12
    raw = pl.read_csv(matrix_file).drop("")
    long = raw.melt(
        value_vars=raw.columns[3:],
        id_vars=['subject', 'condition', 'included_subjects'], 
        variable_name="output",
        value_name="code",
    )
    conditions = {
        "0": "Intentional DFR", 
        "1": "Incidental DFR",
        "2": "Intentional CDFR",
        "3": "Incidental CDFR",
    }
    encoding = {
        "0": "intentional", "1": "incidental", "2": "intentional", "3": "incidental"
    }
    distract = {"0": 0, "1": 0, "2": 16, "3": 16}
    retention = {"0": 16, "1": 16, "2": 16, "3": 16}

    # recode variables and filter to get included subjects
    clean = (
        long.with_columns(
            condition_code=pl.col("condition").cast(pl.Int64).cast(pl.String)
        )
        .select(
            "subject",
            "condition_code",
            pl.col("condition_code").replace(conditions).alias("condition"),
            pl.col("condition_code").replace(encoding).alias("encoding"),
            pl.col("included_subjects").cast(bool).alias("included"),
            pl.col("output").cast(pl.Int64) + 1,
            pl.col("code").cast(pl.Int64).replace(-1, None),
        )
        .with_columns(
            list=1,
            trial_type=pl.lit("recall"),
            position=pl.col("output"),
            item=pl.when(pl.col("code") >= 0).then(pl.col("code") + 1).otherwise(-1),
            distract=pl.col("condition_code").replace(distract),
            retention=pl.col("condition_code").replace(retention),
        )
        .drop_nulls("code")
        .sort("subject", "output")
        .filter(pl.col("included"))
    )

    # define study events
    n_subject = clean["subject"].unique().len()
    pos = np.tile(np.arange(1, LIST_LENGTH + 1), n_subject)
    subj = np.repeat(clean["subject"].unique(), LIST_LENGTH)
    all_items = pl.DataFrame(
        {"subject": subj, "position": pos, "item": pos, "list": 1, "trial_type": "study"}
    )
    conds = (
        clean.group_by("subject")
        .agg(
            pl.col("condition").first(),
            pl.col("encoding").first(),
            pl.col("distract").first(),
            pl.col("retention").first(),
        )
    )
    columns = ["subject", "condition", "encoding", "distract", "retention", "list", "trial_type", "position", "item"]
    study = (
        all_items.join(conds, on="subject")
        .select(columns)
    )
    recall = clean.select(columns)
    full = (
        pl.concat([study, recall])
        .sort("subject", "trial_type", "position", descending=[False, True, False])
    )
    full.write_csv(frdata_file)


@click.command()
@click.argument("peers_patterns_file", type=click.Path(exists=True))
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("out_patterns_file", type=click.Path())
@click.argument("out_data_file", type=click.Path())
def prepare_incidental(
    peers_patterns_file, data_file, out_patterns_file, out_data_file
):
    # read data and PEERS patterns
    data = pl.read_csv(data_file, null_values="-1")
    patterns = cmr.load_patterns(peers_patterns_file)

    # randomly assign items from the pool
    n_items = len(patterns['items'])
    indexed = (
        data.with_columns(
            pl.int_range(n_items)
            .sample(pl.col('item').max(), shuffle=True)
            .gather(pl.col('item') - 1)
            .over('subject', 'list')
            .alias('item_index')
        )
    )
    items = indexed.drop_nulls('item')
    items = items.with_columns(
        item=pl.lit(patterns['items'][items['item_index'].to_numpy().astype(int)])
    )
    labeled = indexed.update(items, on=['subject', 'list', 'trial_type', 'position'])

    # save data with assigned items
    labeled.write_csv(out_data_file)

    # copy patterns
    shutil.copyfile(peers_patterns_file, out_patterns_file)
