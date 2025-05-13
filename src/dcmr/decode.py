"""Functions to run classification of EEG data or network representation."""

import sys
import logging
from pathlib import Path
import click
import numpy as np
from numpy import linalg
import pandas as pd
from joblib import Parallel, delayed
from sklearn import svm
from sklearn import model_selection as ms
import sklearn.linear_model as lm
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from psifr import fr
from cymr import cmr
from dcmr import framework
from dcmr import task


def impute_samples(patterns):
    """Impute missing samples for variables in patterns."""
    fixed = patterns.copy()

    # replace missing variables
    missing = np.all(np.isnan(patterns), 0)
    n = np.nanmean(patterns, 1)
    fixed[:, missing] = n[:, np.newaxis]

    # replace missing observations
    m = np.nanmean(fixed, 0)
    ind = np.where(np.isnan(fixed))
    fixed[ind] = m[ind[1]]
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
        w_grad = linalg.solve(B, (x @ (y - p).T - l * w_old))
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
            w, ll, n = logistic_regression(X.T, t, self.l, self.tol, self.max_iter)
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


def normalize(patterns, normalization):
    """Normalize variable ranges across observations."""
    if normalization == 'range':
        p_min = np.min(patterns, 0)
        p_max = np.max(patterns, 0)
        i = p_min != p_max
        normalized = np.ones(patterns.shape)
        normalized[:, i] = (patterns[:, i] - p_min[i]) / (p_max[i] - p_min[i])
    elif normalization == 'z':
        normalized = preprocessing.scale(patterns)
    else:
        raise ValueError(f'Invalid normalization: {normalization}')
    return normalized


def classify_patterns(
    trials,
    patterns,
    normalization='range',
    clf='svm',
    multi_class='auto',
    C=1.0,
    logger=None,
):
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
    elif clf == 'plogreg':
        clf = LogReg(l=1 / C, max_iter=1000)
    else:
        raise ValueError(f'Unknown classifier: {clf}')

    if np.any(np.all(np.isnan(patterns), 1)):
        raise ValueError('One or more observations has only undefined features.')

    if logger is not None:
        logger.info(f'Using {normalization} normalization.')
        logger.info(
            f'Using {clf} classifier with {C=} and multiclass strategy {multi_class}.'
        )
    for i, (train, test) in enumerate(logo.split(patterns, labels, groups)):
        if logger is not None:
            logger.info(f'Running cross-validation fold {i + 1}.')

        # deal with undefined features and scale feature ranges
        train_patterns = impute_samples(patterns[train])
        test_patterns = impute_samples(patterns[test])
        normalized = normalize(
            np.vstack((train_patterns, test_patterns)), normalization
        )
        n = train_patterns.shape[0]
        train_patterns = normalized[:n]
        test_patterns = normalized[n:]

        # calculate class probabilities in test data based on training data
        clf.fit(train_patterns, labels[train])
        prob = clf.predict_proba(test_patterns)
        xval = pd.DataFrame(prob, index=test, columns=clf.classes_)
        evidence.loc[test, :] = xval
    return evidence


def read_evidence(class_dir, subjects):
    """Read classifier evidence for multiple subjects."""
    evidence = pd.concat(
        [
            pd.read_csv(class_dir / f'sub-{subject}_decode.csv', index_col=0)
            for subject in subjects
        ],
        axis=0,
        ignore_index=True,
    )
    return evidence


def mark_included_eeg_events(data, eeg_dir, subjects=None):
    """Mark events that were included in the EEG data."""
    # load EEG events and use to label whether included or not
    if subjects is None:
        subjects, _ = task.get_subjects()
    events = pd.concat(
        [pd.read_csv(eeg_dir / f'sub-{subject}_decode.csv') for subject in subjects]
    )
    events['include'] = True

    # merge to get an array of included trials
    columns = ['subject', 'list', 'position', 'trial_type']
    included_columns = columns + ['include']
    merged = pd.merge(data, events[included_columns], how='outer', on=columns)
    merged['include'] = merged['include'].fillna(0).astype(bool)
    return merged


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


def evidence_block_pos(data):
    """Get mean evidence by block position."""
    # get study events, excluding the first blocks where previous
    # and baseline categories are undefined
    included = data.query('block > 1 and trial_type == "study"')

    # reorganize classifier evidence by block category (curr, prev, base)
    labeled = label_evidence(included)
    labeled['subject'] = included['subject']
    labeled['block_pos'] = included['block_pos']

    # get average evidence and event counts
    block_pos = labeled.groupby(['subject', 'block_pos'])
    mean_evidence = block_pos[['curr', 'prev', 'base']].mean()
    mean_evidence['n'] = block_pos['curr'].count()
    return mean_evidence


def _regress_subject(data, evidence_keys):
    """Regress evidence for one subject."""
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


def _decode_eeg_subject(patterns_dir, out_dir, subject, **kwargs):
    """Decode category from EEG patterns for one suject."""
    log_dir = out_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'sub-{subject}_log.txt'

    logger = logging.getLogger(subject)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)

    pattern_file = patterns_dir / f'sub-{subject}_pattern.txt'
    logger.info(f'Loading pattern from {pattern_file}.')
    pattern = np.loadtxt(pattern_file.as_posix())

    events_file = patterns_dir / f'sub-{subject}_events.csv'
    logger.info(f'Loading events from {events_file}.')
    events = pd.read_csv(events_file)

    logger.info(f'Running classification.')
    evidence = classify_patterns(events, pattern, logger=logger, **kwargs)

    out_file = out_dir / f'sub-{subject}_decode.csv'
    logger.info(f'Writing results to {out_file}.')
    df = pd.concat([events, evidence], axis=1)
    df.to_csv(out_file.as_posix())


@click.command()
@click.argument("patterns_dir")
@click.argument("out_dir")
@click.option(
    "--n-jobs", "-n", type=int, default=1, help="Number of processes to run in parallel"
)
@click.option("--subjects", "-s", help="Comma-separated list of subjects")
@click.option(
    "--normalization",
    "-l",
    default="range",
    help='Normalization to apply before classification {"z", ["range"]}',
)
@click.option(
    "--classifier",
    "-c",
    default="svm",
    help='classifier type {["svm"], "logreg", "plogreg"}',
)
@click.option(
    "--multi-class",
    "-m",
    default="auto",
    help='multi-class method {["auto"], "ovr", "multinomial"}',
)
@click.option("--regularization", "-C", type=float, default=1, help="Regularization parameter (1.0)")
def decode_eeg(
    patterns_dir, out_dir, n_jobs, subjects, normalization, classifier, multi_class, regularization
):
    "Decode category from EEG patterns measured during the CFR study."
    if subjects is None:
        subjects, _ = task.get_subjects()
    else:
        subjects = [f'LTP{subject:0>3}' for subject in subjects.split(',')]

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    Parallel(n_jobs=n_jobs)(
        delayed(_decode_eeg_subject)(
            Path(patterns_dir),
            Path(out_dir),
            subject,
            normalization=normalization,
            clf=classifier,
            multi_class=multi_class,
            C=regularization,
        )
        for subject in subjects
    )


def _decode_context_subject(
    data_file,
    patterns_file,
    fit_dir,
    eeg_class_dir,
    sublayer,
    out_dir,
    subject,
    sigmas=None,
    n_reps=1,
    **kwargs,
):
    """Decode category from simulated context patterns for one suject."""
    log_dir = out_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    subject_id = f'LTP{subject:03n}'
    log_file = log_dir / f'sub-{subject_id}_log.txt'

    logger = logging.getLogger(subject_id)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)

    logger.info(f'Loading data from {data_file}.')
    data = task.read_study_recall(data_file)
    study = fr.filter_data(data, subjects=subject, trial_type='study').reset_index()
    study = mark_included_eeg_events(study, eeg_class_dir, subjects=[subject_id])

    param_file = fit_dir / 'fit.csv'
    logger.info(f'Loading parameters from {param_file}.')
    subj_param = framework.read_fit_param(param_file)

    config_file = fit_dir / 'parameters.json'
    logger.info(f'Loading model configuration from {config_file}.')
    param_def = cmr.read_config(config_file)

    logger.info(f'Loading model patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)

    logger.info('Recording network states.')
    model = cmr.CMR()
    state = model.record(
        study, {}, subj_param, param_def=param_def, patterns=patterns, include=['c']
    )
    net = state[0]
    context = np.vstack([s.c[net.get_slice('c', sublayer, 'item')] for s in state])

    logger.info(f'Running classification.')
    include = study['include'].to_numpy()
    study_include = study.loc[include]
    c = context[include]
    if sigmas is not None:
        # set random seed using a hash of the subject code
        seed = hash(subject) % ((sys.maxsize + 1) * 2)
        rng = np.random.default_rng(seed)
        d_list = []
        for sigma in sigmas:
            for i in range(n_reps):
                # add noise to the context we are classifying
                c_noise = c + rng.normal(scale=sigma, size=c.shape)
                evidence = classify_patterns(
                    study_include, c_noise, logger=logger, **kwargs
                )
                evidence.index = study_include.index
                d = pd.concat([study, evidence], axis=1)
                d["sigma"] = sigma
                d["rep"] = i + 1
                d_list.append(d)
        df = pd.concat(d_list, axis=0)
    else:
        # classify the raw context
        evidence = classify_patterns(
            study_include, c, logger=logger, **kwargs
        )
        evidence.index = study_include.index
        df = pd.concat([study, evidence], axis=1)
    out_file = out_dir / f'sub-{subject_id}_decode.csv'
    logger.info(f'Writing results to {out_file}.')
    df.to_csv(out_file.as_posix())


@click.command()
@click.argument("data_file")
@click.argument("patterns_file")
@click.argument("fit_dir")
@click.argument("eeg_class_dir")
@click.argument("sublayer")
@click.argument("res_name")
@click.option(
    "--n-jobs", "-n", type=int, default=1, help="Number of processes to run in parallel"
)
@click.option("--subjects", "-s", help="Comma-separated list of subjects")
@click.option(
    "--normalization",
    "-l",
    default="range",
    help='Normalization to apply before classification {"z", ["range"]}',
)
@click.option(
    "--classifier",
    "-c",
    default="svm",
    help='classifier type {["svm"], "logreg", "plogreg"}',
)
@click.option(
    "--multi-class",
    "-m",
    default="auto",
    help='multi-class method {["auto"], "ovr", "multinomial"}',
)
@click.option("--regularization", "-C", type=float, default=1, help="Regularization parameter (1.0)")
@click.option("--sigmas", help="Noise levels to add (default: no noise)")
@click.option("--n-reps", type=int, default=1, help="Number of replications of each noise level")
def decode_context(
    data_file,
    patterns_file,
    fit_dir,
    eeg_class_dir,
    sublayer,
    res_name,
    n_jobs,
    subjects,
    normalization,
    classifier,
    multi_class,
    regularization,
    sigmas,
    n_reps,
):
    "Decode category from simulated context states."
    if subjects is None:
        _, subjects = task.get_subjects()
    else:
        subjects = [int(s) for s in subjects.split(",")]
    
    if sigmas is not None:
        sigmas = [float(s) for s in sigmas.split(",")]

    out_dir = Path(fit_dir) / f'decode_{sublayer}' / res_name
    out_dir.mkdir(exist_ok=True, parents=True)
    Parallel(n_jobs=n_jobs)(
        delayed(_decode_context_subject)(
            Path(data_file),
            Path(patterns_file),
            Path(fit_dir),
            Path(eeg_class_dir),
            sublayer,
            out_dir,
            subject,
            sigmas=sigmas,
            n_reps=n_reps,
            normalization=normalization,
            clf=classifier,
            multi_class=multi_class,
            C=regularization,
        )
        for subject in subjects
    )
