"""Functions to run classification of EEG data or network representation."""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import model_selection as ms


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


def classify_patterns(trials, patterns):
    """Run cross-validation and return evidence for each category."""
    trials = trials.reset_index()
    labels = trials['category'].to_numpy()
    groups = trials['list'].to_numpy()
    categories = trials['category'].unique()
    evidence = pd.DataFrame(index=trials.index, columns=categories, dtype='float')
    logo = ms.LeaveOneGroupOut()
    clf = svm.SVC(probability=True)
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
