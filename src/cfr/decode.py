"""Functions to run classification of EEG data or network representation."""

import numpy as np
from numpy import linalg
import pandas as pd
from sklearn import svm
from sklearn import model_selection as ms
import sklearn.linear_model as lm
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


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


def _likelihood(x, y, w, l):
    """Likelihood of data for ridge regression weights."""
    a = y @ (w.T @ x).T
    b = np.sum(np.log(1 + np.exp(w.T @ x)))
    c = l / 2 * w.T @ w
    ll = a - b - c
    return ll


def logistic_regression(x, y, l, tol, max_rounds):
    """Logistic regression from Princeton MVPA toolbox."""
    n_feature, n_sample = x.shape
    w_old = np.zeros((n_feature, 1))
    delta_ll = 1
    rounds = 0
    old_ll = _likelihood(x, y, w_old, l)
    C2 = l * np.eye(n_feature)
    ll = []
    while delta_ll > tol and rounds < max_rounds:
        f = np.exp(w_old.T @ x)
        p = f / (1 + f)
        A = np.diag(p[0] * (1 - p[0]))
        C1 = x @ A @ x.T
        B = C1 + C2
        w_grad = linalg.lstsq(B, (x @ (y - p).T - l * w_old), rcond=None)[0]
        w_new = w_old + w_grad
        new_ll = _likelihood(x, y, w_new, l)
        w_old = w_new
        delta_ll = np.abs((old_ll - new_ll) / old_ll)
        rounds += 1
        old_ll = new_ll
        ll.append(new_ll)
    ll = np.array(ll)
    w = w_old
    return w, ll, rounds


class LogReg(BaseEstimator, ClassifierMixin):
    """Logistic ridge regression from Princeton MVPA toolbox."""

    def __init__(self, l=10.0, tol=0.0001, max_iter=5000):
        self.l = l
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Estimate regression coefficients.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # Estimate coefficients for each class
        self.coef_ = np.zeros((X.shape[1], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            t = np.zeros((1, X.shape[0]))
            t[:, y == c] = 1
            w, ll, n = logistic_regression(
                X.T, t, self.l, self.tol, self.max_iter
            )
            self.coef_[:, i] = w.T

        # Return the classifier
        return self

    def _proba(self, X):
        """Calculate class probabilities."""
        prob = np.zeros((len(self.classes_), X.shape[0]))
        for i in range(len(self.classes_)):
            w = self.coef_[:, i]
            prob[i, :] = np.exp(w.T @ X.T) / (1 + np.exp(w.T @ X.T))
        return prob

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        p = self._proba(X)
        winner = np.argmax(p, axis=0)
        return self.classes_[winner]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        proba = self._proba(X).T
        return proba


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
