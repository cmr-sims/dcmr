"""Test implementation of logistic ridge regression classifier."""

import numpy as np
import pandas as pd
from cfr import decode
import pytest


@pytest.fixture()
def patterns():
    """Create test patterns and labels."""
    vectors = np.array(
        [
            [-0.1022, -1.2141, 1.5442],
            [-0.2414, -1.1135, 0.0859],
            [0.3192, -0.0068, -1.4916],
            [0.3129, 1.5326, -0.7423],
            [-0.8649, -0.7697, -1.0616],
            [-0.0301, 0.3714, 2.3505],
            [-0.1649, -0.2256, -0.6156],
            [0.6277, 1.1174, 0.7481],
            [1.0933, -1.0891, -0.1924],
            [1.1093, 0.0326, 0.8886],
            [-0.8637, 0.5525, -0.7648],
            [0.0774, 1.1006, -1.4023],
        ]
    )
    labels = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3])
    chunks = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    data = {'vectors': vectors, 'labels': labels, 'chunks': chunks}
    return data


def test_logreg1(patterns):
    """Test logistic regression of one class vs others."""
    train = patterns['chunks'] == 1
    labels = patterns['labels'][train]
    x = patterns['vectors'][train, :].T

    # taken from Matlab implementation of Princeton MVPA toolbox
    expected = np.array([[0.0016], [-0.1482], [0.0955]])
    y = np.zeros((1, len(labels)))
    y[:, labels == 1] = 1
    l = 10
    tol = 0.0001
    max_rounds = 5000
    w, ll, n = decode.logistic_regression(x, y, l, tol, max_rounds)
    np.testing.assert_allclose(w, expected, atol=0.0001)


def test_prob(patterns):
    """Test probability based on logistic regression."""
    # taken from Matlab implementation of Princeton MVPA toolbox
    expected = [
        [0.4936, 0.5165, 0.4898],
        [0.4767, 0.5264, 0.5107],
        [0.5361, 0.4842, 0.4746],
        [0.5205, 0.4816, 0.5003],
        [0.4610, 0.5434, 0.5016],
        [0.4263, 0.6165, 0.4791],
    ]
    train = patterns['chunks'] == 1
    y = patterns['labels'][train]
    X = patterns['vectors'][train, :]

    clf = decode.LogReg(l=10)
    clf.fit(X, y)

    test = patterns['chunks'] == 2
    X_ = patterns['vectors'][test, :]
    prob = clf.predict_proba(X_)
    np.testing.assert_allclose(prob, expected, atol=0.0001)


def test_preprocessing(patterns):
    """Test pattern preprocessing."""
    flattened = patterns['vectors']
    # set samples to impute
    flattened[[0, 2, 2, 6], [1, 0, 2, 0]] = np.nan
    imputed = decode.impute_samples(patterns['vectors'])
    normalized = decode.normalize(imputed, 'range')
    expected = np.array(
        [
            [0.3863, 0.4724, 0.7851],
            [0.3158, 0.0000, 0.3966],
            [0.4947, 0.4182, 0.3940],
            [0.5966, 1.0000, 0.1759],
            [0.0000, 0.1299, 0.0908],
            [0.4229, 0.5612, 1.0000],
            [0.4947, 0.3356, 0.2096],
            [0.7561, 0.8431, 0.5730],
            [0.9919, 0.0092, 0.3224],
            [1.0000, 0.4331, 0.6105],
            [0.0006, 0.6296, 0.1699],
            [0.4773, 0.8367, 0.0000],
        ]
    )
    np.testing.assert_allclose(normalized, expected, atol=0.0001)


def test_class(patterns):
    """Test full classification procedure."""
    trials = pd.DataFrame({'list': patterns['chunks'], 'category': patterns['labels']})
    evidence = decode.classify_patterns(
        trials, patterns['vectors'], normalization='range', clf='plogreg', C=0.1
    )
    expected = np.array(
        [
            [0.4912, 0.5019, 0.4743],
            [0.4933, 0.5007, 0.4832],
            [0.4870, 0.4919, 0.4816],
            [0.4816, 0.4771, 0.4776],
            [0.4982, 0.4957, 0.4978],
            [0.4850, 0.4869, 0.4689],
            [0.4870, 0.4977, 0.4901],
            [0.4702, 0.4932, 0.4780],
            [0.4883, 0.4930, 0.4812],
            [0.4777, 0.4893, 0.4768],
            [0.4835, 0.5006, 0.4941],
            [0.4736, 0.5071, 0.4845],
        ]
    )
    np.testing.assert_allclose(evidence, expected, atol=0.0001)
