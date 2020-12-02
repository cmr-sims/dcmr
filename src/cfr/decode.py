"""Functions to run classification of EEG data or network representation."""

import pandas as pd
from sklearn import svm
from sklearn import model_selection as ms


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
