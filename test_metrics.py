import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

import metrics as mt


def test_balanced_accuracy_matches_sklearn():
    rng = np.random.RandomState(0)
    y, p = rng.randint(0, 5, 300), rng.randint(0, 5, 300)
    assert mt.balanced_accuracy(y, p) == pytest.approx(balanced_accuracy_score(y, p))


def test_per_class_recall_and_confusions():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 1, 2])
    rec = mt.per_class_recall(y, p, 3)
    assert rec.tolist() == [0.5, 1.0, 0.5]
    top = mt.top_confusions(y, p, ["a", "b", "c"], k=5)
    assert top == [("a", "b", 1), ("c", "b", 1)]


def test_bootstrap_ci_contains_value_and_repeats():
    rng = np.random.RandomState(1)
    groups = np.repeat(np.arange(30), 18)
    y = np.tile(np.arange(18), 30)
    p = np.where(rng.rand(len(y)) < 0.6, y, rng.randint(0, 18, len(y)))
    ba, lo, hi = mt.bootstrap_ci(y, p, groups, n_boot=300, seed=0)
    assert lo <= ba <= hi
    assert (ba, lo, hi) == mt.bootstrap_ci(y, p, groups, n_boot=300, seed=0)


def test_paired_bootstrap_diff_signs():
    groups = np.repeat(np.arange(30), 18)
    y = np.tile(np.arange(18), 30)
    worse = np.where(np.arange(len(y)) % 3 == 0, (y + 1) % 18, y)
    d, lo, hi = mt.paired_bootstrap_diff(y, worse, y, groups, n_boot=300, seed=0)
    assert d > 0 and lo > 0
    d, lo, hi = mt.paired_bootstrap_diff(y, y, y, groups, n_boot=300, seed=0)
    assert d == lo == hi == 0
