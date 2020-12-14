"""Functions to run classification of EEG data or network representation."""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import model_selection as ms
import sklearn.linear_model as lm
from sklearn import preprocessing


def impute_samples(patterns):
    """Impute missing samples for variables in patterns."""
    m = np.nanmean(patterns, 0)
    fixed = patterns.copy()
    for i in range(fixed.shape[1]):
        isnan = np.isnan(fixed[:, i])
        if not np.any(isnan):
            continue
        fixed[isnan, i] = m[i]
    return fixed


def classify_patterns(trials, patterns, clf='svm'):
    """Run cross-validation and return evidence for each category."""
    trials = trials.reset_index()
    labels = trials['category'].to_numpy()
    groups = trials['list'].to_numpy()
    categories = trials['category'].unique()
    evidence = pd.DataFrame(index=trials.index, columns=categories, dtype='float')
    logo = ms.LeaveOneGroupOut()

    if clf == 'svm':
        clf = svm.SVC(probability=True)
    elif clf == 'logreg':
        clf = lm.LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f'Unknown classifier: {clf}')

    patterns = preprocessing.scale(patterns)
    for train, test in logo.split(patterns, labels, groups):
        clf.fit(patterns[train], labels[train])
        prob = clf.predict_proba(patterns[test])
        xval = pd.DataFrame(prob, index=test, columns=clf.classes_)
        evidence.loc[test, :] = xval
    return evidence


def label_evidence(data, prefix):
    """Label evidence by block category."""
    cats = data['curr'].dropna().unique()
    evidence = ['curr', 'prev', 'base']
    res = data.copy()
    for cat in cats:
        for evid in evidence:
            include = data[evid] == cat
            evid_key = prefix + evid
            cat_key = prefix + cat
            res.loc[include, evid_key] = data.loc[include, cat_key]
    return res


def regress_evidence_block_pos(data, prefix):
    """Regress evidence on block position."""
    data = data.reset_index()
    evidence = ['curr', 'prev', 'base']
    n = data['n'].to_numpy()
    x = data['block_pos'].to_numpy().reshape(-1, 1)
    d = {}
    for evid in evidence:
        y = data[prefix + evid].to_numpy()
        model = lm.LinearRegression()
        model.fit(x, y, sample_weight=n)
        d[prefix + evid + '_slope'] = model.coef_[0]
    res = pd.Series(d)
    return res
