"""Scores for the holdout: balanced accuracy, per-label recall, top confusions, bootstrap intervals over prompt_id."""
import numpy as np


def per_class_recall(y, p, n_classes):
    """Share of each label's passages that were called that label (nan for a label with no passages)."""
    y, p = np.asarray(y), np.asarray(p)
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = y == c
        if m.any():
            out[c] = float((p[m] == c).mean())
    return out


def balanced_accuracy(y, p):
    """The competition metric: the mean of per-label recall over the labels present."""
    y = np.asarray(y)
    return float(np.nanmean(per_class_recall(y, p, int(max(np.max(y), np.max(p))) + 1)))


def top_confusions(y, p, labels, k=10):
    """The k most common (true label, predicted label) mistakes, biggest first."""
    pairs = {}
    for a, b in zip(np.asarray(y), np.asarray(p)):
        if a != b:
            pairs[(labels[a], labels[b])] = pairs.get((labels[a], labels[b]), 0) + 1
    return sorted(((a, b, n) for (a, b), n in pairs.items()), key=lambda t: (-t[2], t[0], t[1]))[:k]


def _resample_index(groups, n_boot, seed):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        yield np.concatenate([rows[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])


def bootstrap_ci(y, p, groups, n_boot=2000, seed=0):
    """Balanced accuracy and its 95% interval, resampling whole prompts (a prompt's passages move together)."""
    y, p = np.asarray(y), np.asarray(p)
    vals = [balanced_accuracy(y[i], p[i]) for i in _resample_index(groups, n_boot, seed)]
    return balanced_accuracy(y, p), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(y, p_a, p_b, groups, n_boot=2000, seed=0):
    """B minus A in balanced accuracy, with a 95% interval from the same resampled prompts for both runs.
    If the interval doesn't cross 0, the difference is probably real, not noise."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    vals = [balanced_accuracy(y[i], p_b[i]) - balanced_accuracy(y[i], p_a[i]) for i in _resample_index(groups, n_boot, seed)]
    return (balanced_accuracy(y, p_b) - balanced_accuracy(y, p_a),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
