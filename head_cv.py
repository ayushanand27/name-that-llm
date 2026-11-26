"""The measuring stick: 5-fold CV, grouped by prompt_id, of Rohan's PB115 head on the 2,699 official rows.

The head is his: StandardScaler, then 55% LogisticRegression(C=0.025) + 45% mean of three LightGBM seeds
(42, 1042, 2042) with his settings. Features are his 115 (54 GLM-likelihood stats + 61 token-aligned).
add_features() puts a new block beside them; compare() says whether the block earns its place.

KEEP RULE: keep a block only if the gain is at least +0.005 and the 95% interval stays above 0.
Submit only when the gain is at least +0.008 (public LB noise is about +-0.01).
Prints numbers and label names only, never ids or passage text.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as mt

LGB_SEEDS = (42, 1042, 2042)
LR_SHARE = 0.55
KEEP_GAIN, SUBMIT_GAIN = 0.005, 0.008
LGB_PARAMS = dict(objective="multiclass", n_estimators=220, learning_rate=0.03, num_leaves=15, min_child_samples=25,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0, verbose=-1, n_jobs=4)


def build_X(pb_dir, ids, expect=115):
    """Rohan's 115-column matrix, rows in the order of ids (same code path as his build_54_matrix)."""
    pb = Path(pb_dir)
    spec = json.loads((pb / "spec.json").read_text())
    feats = {t: pd.read_csv(pb / f"{t}_features.csv", dtype={"id": str}).set_index("id") for t in ("flash", "glm53")}
    F = pd.concat([feats[t].add_prefix(f"{t}:") for t in ("flash", "glm53")], axis=1).reindex(ids)
    for c in spec["diff_features"]:
        F[f"d:{c}"] = F[f"flash:{c}"] - F[f"glm53:{c}"]
    paired = pd.read_parquet(pb / "paired_token_features.parquet")
    paired.index = paired.index.astype(str)
    X = np.hstack([F[spec["features_54"]].to_numpy(np.float64),
                   paired.reindex(ids)[spec["features_61"]].to_numpy(np.float64)])
    names = list(spec["features_54"]) + list(spec["features_61"])
    assert X.shape == (len(ids), expect), f"expected {expect} columns (115 for PB115), got {X.shape[1]}"
    assert np.isfinite(X).all(), "non-finite features (an id missing from the feature files?)"
    return X, names


def grouped_folds(groups, n_folds=5):
    from sklearn.model_selection import GroupKFold
    folds = np.full(len(groups), -1, dtype=np.int64)
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(np.zeros(len(groups)), groups=groups)):
        folds[va] = k
    return folds


def _full(p, classes, n_classes):
    """Columns for every class, zeros for a class this fold never saw."""
    out = np.zeros((len(p), n_classes))
    out[:, np.asarray(classes, dtype=int)] = p
    return out


def head_fit_predict(Xtr, ytr, Xva, n_classes, lgb_seeds=LGB_SEEDS):
    """Rohan's head. With no lgb_seeds it is the LR part alone (used by fast tests)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    lr = LogisticRegression(C=0.025, max_iter=1000).fit(A, ytr)
    p = _full(lr.predict_proba(B), lr.classes_, n_classes)
    if lgb_seeds:
        import lightgbm as lgb
        gs = []
        for sd in lgb_seeds:
            m = lgb.LGBMClassifier(num_class=len(np.unique(ytr)), random_state=sd, **LGB_PARAMS).fit(A, ytr)
            gs.append(_full(m.predict_proba(B), m.classes_, n_classes))
        p = LR_SHARE * p + (1 - LR_SHARE) * np.mean(gs, axis=0)
    return p / p.sum(1, keepdims=True)


def oof_probs(X, y, groups, n_classes, n_folds=5, lgb_seeds=LGB_SEEDS):
    """Out-of-fold probabilities: each row is predicted by a head that never saw its prompt."""
    folds = grouped_folds(groups, n_folds)
    out = np.zeros((len(y), n_classes))
    for k in range(n_folds):
        va = folds == k
        out[va] = head_fit_predict(X[~va], y[~va], X[va], n_classes, lgb_seeds)
    return out


def add_features(X, block):
    """X with a new block of columns beside it. The block must have one finite row per row of X, same order."""
    B = np.asarray(block, dtype=np.float64)
    B = B[:, None] if B.ndim == 1 else B
    if len(B) != len(X):
        raise ValueError(f"block has {len(B)} rows, X has {len(X)}")
    if not np.isfinite(B).all():
        raise ValueError("block has non-finite values")
    return np.hstack([X, B])


def keep_rule(gain, lo):
    """(keep, submit) from the gain and the lower end of its 95% interval."""
    ok = lo > 0
    return bool(ok and gain >= KEEP_GAIN - 1e-12), bool(ok and gain >= SUBMIT_GAIN - 1e-12)


def compare(base, new, y, groups, labels, n_boot=2000):
    """base and new are OOF probability matrices on the same rows. Gain = new - base with a paired bootstrap."""
    pb, pn = np.asarray(base).argmax(1), np.asarray(new).argmax(1)
    gain, lo, hi = mt.paired_bootstrap_diff(y, pb, pn, groups, n_boot=n_boot)
    keep, submit = keep_rule(gain, lo)
    k = len(labels)
    return {"ba_base": mt.balanced_accuracy(y, pb), "ba_new": mt.balanced_accuracy(y, pn),
            "ci_new": mt.bootstrap_ci(y, pn, groups, n_boot=n_boot)[1:], "gain": gain, "lo": lo, "hi": hi,
            "keep": keep, "submit": submit, "recall_base": mt.per_class_recall(y, pb, k),
            "recall_new": mt.per_class_recall(y, pn, k), "rows": np.bincount(y, minlength=k),
            "confusions": mt.top_confusions(y, pn, labels, k=10), "n": len(y), "prompts": len(np.unique(groups))}


def report_lines(r, labels):
    lo_ci, hi_ci = r["ci_new"]
    verdict = "KEEP" if r["keep"] else "DROP"
    lines = [f"rows {r['n']}  prompts {r['prompts']}",
             f"balanced accuracy  base {r['ba_base']:.4f} -> new {r['ba_new']:.4f}  (new 95% interval {lo_ci:.4f} to {hi_ci:.4f})",
             f"gain {r['gain']:+.4f}  paired 95% interval {r['lo']:+.4f} to {r['hi']:+.4f}  -> {verdict}"
             + ("  (and big enough to submit)" if r["submit"] else ""),
             "per-label recall (rows, base -> new), weakest new first:"]
    order = np.argsort(np.nan_to_num(r["recall_new"], nan=9))
    for i in order:
        lines.append(f"  {labels[i]:<26} {int(r['rows'][i]):4d}  {r['recall_base'][i]:.3f} -> {r['recall_new'][i]:.3f}")
    lines.append("top mix-ups (true -> called, count):")
    lines += [f"  {a} -> {b}: {n}" for a, b, n in r["confusions"]]
    return lines


def save_oof(path, ids, groups, y, probs, labels, run):
    np.savez(path, ids=np.asarray(ids, dtype="U"), groups=np.asarray(groups, dtype="U"), y=np.asarray(y),
             probs=np.asarray(probs, dtype=np.float32), labels=np.asarray(labels, dtype="U"), run=np.asarray(run))


def load_block(path, ids, labels):
    """A saved OOF file's probabilities, in the row order of ids; every official row must have one."""
    with np.load(path, allow_pickle=False) as z:
        if [str(x) for x in z["labels"]] != labels:
            raise SystemExit(f"{path}: labels differ from train.csv's sorted labels")
        idx = pd.Series(np.arange(len(z["ids"])), index=z["ids"].astype(str)).reindex(ids)
        if idx.isna().any():
            raise SystemExit(f"{path}: {int(idx.isna().sum())} official rows have no prediction")
        return z["probs"][idx.to_numpy(int)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pb_dir", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--base_block", nargs="*", default=[], help="OOF .npz files added as features on both sides (the yardstick)")
    ap.add_argument("--block", nargs="*", default=[], help="OOF .npz files whose probs are added as a feature block")
    ap.add_argument("--out", default="oof_head_pb115.npz")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args(argv)
    tr = pd.read_csv(a.train_csv, dtype={"id": str}, usecols=["id", "prompt_id", "model"])
    labels = sorted(tr.model.unique())
    y = tr.model.map({l: i for i, l in enumerate(labels)}).to_numpy(np.int64)
    g = tr.prompt_id.astype(str).to_numpy()
    X, _ = build_X(a.pb_dir, tr.id.tolist())
    for path in a.base_block:
        X = add_features(X, load_block(path, tr.id, labels))
    base = oof_probs(X, y, g, len(labels))
    save_oof(a.out, tr.id, g, y, base, labels, "pb115_head" + "".join("+base_block" for _ in a.base_block))
    Xn = X
    for path in a.block:
        Xn = add_features(Xn, load_block(path, tr.id, labels))
    new = oof_probs(Xn, y, g, len(labels)) if a.block else base
    print("\n".join(report_lines(compare(base, new, y, g, labels, n_boot=a.n_boot), labels)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
