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


def classify_patterns(trials, patterns, clf='svm', multi_class='auto', C=1.0):
    """Run cross-validation and return evidence for each category."""
    trials = trials.reset_index()
    labels = trials['category'].to_numpy()
    groups = trials['list'].to_numpy()
    categories = trials['category'].unique()
    evidence = pd.DataFrame(index=trials.index, columns=categories, dtype='float')
    logo = ms.LeaveOneGroupOut()

    if clf == 'svm':
        clf = svm.SVC(probability=True, C=C)
    elif clf == 'logreg':
        clf = lm.LogisticRegression(max_iter=1000, multi_class=multi_class, C=C)
    else:
        raise ValueError(f'Unknown classifier: {clf}')

    patterns = preprocessing.scale(patterns)
    for train, test in logo.split(patterns, labels, groups):
        clf.fit(patterns[train], labels[train])
        prob = clf.predict_proba(patterns[test])
        xval = pd.DataFrame(prob, index=test, columns=clf.classes_)
        evidence.loc[test, :] = xval
    return evidence


def label_evidence(data, evidence_keys=None):
    """Label evidence by block category."""
    if evidence_keys is None:
        evidence_keys = ['curr', 'prev', 'base']
    categories = data[evidence_keys].melt().dropna()['value'].unique()
    d = {}
    for evidence in evidence_keys:
        d[evidence] = np.empty(data.shape[0])
        d[evidence][:] = np.nan
        for category in categories:
            include = data[evidence] == category
            d[evidence][include] = data.loc[include, category]
    results = pd.DataFrame(d, index=data.index)
    return results


def _regress_subject(data, evidence_keys):
    """"Regress evidence for one subject."""
    n = data['n'].to_numpy()
    x = data['block_pos'].to_numpy()[:, np.newaxis]
    d = {}
    for evidence in evidence_keys:
        y = data[evidence].to_numpy()
        model = lm.LinearRegression()
        model.fit(x, y, sample_weight=n)
        d[evidence] = model.coef_[0]
    slopes = pd.Series(d)
    return slopes


def regress_evidence_block_pos(data, max_pos=3):
    """Regress evidence on block position."""
    data = data.reset_index()
    if max_pos is not None:
        data = data.query(f'block_pos <= {max_pos}')
    evidence_keys = ['curr', 'prev', 'base']
    slopes = data.groupby('subject').apply(_regress_subject, evidence_keys)
    return slopes
