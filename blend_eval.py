"""Blend saved holdout probabilities from any models and score the blend with head_cv.compare().

Input: two or more OOF .npz files with ids, y, probs, labels, and groups (prompt_id) in at least one of them.
Rows are matched by id (shared ids only); labels must be the same list in the same order.
Prints numbers and label names only, never ids or passage text.

Weights are found greedily in steps of 0.05: start from the best single model, then for each other model
try blend = (1 - w) * current + w * model for w in 0..1 and keep the best. With two files this is the plain
w-grid. The weights are picked on the same rows that are scored, so the blend score is a little optimistic;
the cross-fit line (weights picked on half the prompts, scored on the other half, then swapped) is the honest one.
"""
import argparse
import sys

import numpy as np

import head_cv as hc
import metrics as mt


def load(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    for k in ("ids", "y", "probs", "labels"):
        if k not in d:
            raise SystemExit(f"{path}: missing '{k}'")
    d["path"] = path
    return d


def norm(p):
    """Rows sum to 1, so a model that outputs bigger numbers doesn't win the blend by scale alone."""
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum(1, keepdims=True)


def check_probs(d):
    p = d["probs"]
    if p.ndim != 2 or p.shape[1] != len(d["labels"]):
        raise SystemExit(f"{d['path']}: probs has {p.shape[-1]} columns but {len(d['labels'])} labels")
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
        raise SystemExit(f"{d['path']}: probs must be finite, non-negative, and not all zero in a row")


def align(runs):
    """Shared ids in the first file's order; fails loudly on any mismatch in labels, true labels or groups."""
    first = [str(x) for x in runs[0]["labels"]]
    for r in runs:
        if [str(x) for x in r["labels"]] != first:
            raise SystemExit(f"{r['path']}: labels differ (same names in the same order needed)")
        check_probs(r)
    index = [{str(x): i for i, x in enumerate(r["ids"])} for r in runs]
    keep = [x for x in (str(v) for v in runs[0]["ids"]) if all(x in ix for ix in index)]
    if not keep:
        raise SystemExit("no shared ids between the files")
    rows = [np.array([ix[x] for x in keep]) for ix in index]
    y = runs[0]["y"][rows[0]]
    groups = None
    for r, s in zip(runs, rows):
        if (r["y"][s] != y).any():
            raise SystemExit(f"{r['path']}: true label differs on {(r['y'][s] != y).sum()} shared rows")
        if "groups" in r:
            g = np.array([x[4:] if x.startswith("off:") else x for x in r["groups"][s].astype(str)])  # LoRA/TF-IDF save "off:<prompt_id>"
            if groups is not None and (g != groups).any():
                raise SystemExit(f"{r['path']}: prompt groups differ from another file on shared rows")
            groups = g
    if groups is None:
        raise SystemExit("need prompt groups (key 'groups' = prompt_id) in at least one file for the bootstrap")
    return {"ids": np.asarray(keep), "y": y, "groups": groups, "P": [norm(r["probs"][s]) for r, s in zip(runs, rows)],
            "dropped": [len(ix) - len(keep) for ix in index]}


def weight_grid():
    return np.round(np.arange(0, 21) * 0.05, 2)


def blend(pa, pb, w):
    return w * pa + (1 - w) * pb


def _ba(y, p):
    return mt.balanced_accuracy(y, p.argmax(1))


def best_weight(y, pa, pb):
    """w (weight on pa) with the best balanced accuracy; ties go to the w nearest 0.5, the safest middle."""
    ba, _, w = max((_ba(y, blend(pa, pb, w)), -abs(w - 0.5), w) for w in weight_grid())
    return float(w), float(ba)


def greedy_weights(y, P):
    """Weights (summing to 1) found by adding one model at a time to the best single model."""
    start = int(np.argmax([_ba(y, p) for p in P]))
    w = np.zeros(len(P)); w[start] = 1.0
    for j in [k for k in range(len(P)) if k != start]:
        cur = sum(wk * p for wk, p in zip(w, P))
        a, _ = best_weight(y, P[j], cur)
        w = w * (1 - a); w[j] += a
    return w / w.sum()


def mix(w, P):
    return sum(wk * p for wk, p in zip(w, P))


def cross_fit(y, P, groups, seed=0):
    """Weights picked on one half of the prompts, used on the other half, and swapped. Returns BA and both weights."""
    uniq = np.unique(groups)
    half = set(np.random.RandomState(seed).permutation(uniq)[: len(uniq) // 2])
    m = np.array([g in half for g in groups])
    pred = np.empty(len(y), dtype=int); ws = []
    for fit, score in ((m, ~m), (~m, m)):
        w = greedy_weights(y[fit], [p[fit] for p in P])
        pred[score] = mix(w, [p[score] for p in P]).argmax(1); ws.append(w)
    return mt.balanced_accuracy(y, pred), ws


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="two or more OOF .npz files")
    ap.add_argument("--names", nargs="*", help="a short name per file (default: file names)")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if len(a.files) < 2:
        raise SystemExit("give at least two files")
    names = a.names or [f.rsplit("/", 1)[-1].removesuffix(".npz") for f in a.files]
    if len(names) != len(a.files):
        raise SystemExit("--names needs one name per file")
    runs = [load(f) for f in a.files]
    labels = [str(x) for x in runs[0]["labels"]]
    d = align(runs)
    y, g, P = d["y"], d["groups"], d["P"]
    w = greedy_weights(y, P)
    pbl = mix(w, P)
    print(f"rows {len(y)}  prompts {len(np.unique(g))}  (not shared, per file: {d['dropped']})")
    print("balanced accuracy, 95% interval over prompts:")
    for n, p in zip(names, P):
        ba, lo, hi = mt.bootstrap_ci(y, p.argmax(1), g, n_boot=a.n_boot)
        print(f"  {n + ' alone':<20} {ba:.4f}  ({lo:.4f} to {hi:.4f})")
    print("blend weights: " + ", ".join(f"{n} {wk:.2f}" for n, wk in zip(names, w)))
    print("paired difference, same resampled prompts for both (interval not crossing 0 = probably real):")
    for n, p in zip(names, P):
        diff, lo, hi = mt.paired_bootstrap_diff(y, p.argmax(1), pbl.argmax(1), g, n_boot=a.n_boot)
        print(f"  {'blend - ' + n:<20} {diff:+.4f}  ({lo:+.4f} to {hi:+.4f})")
    best = int(np.argmax([_ba(y, p) for p in P]))
    print(f"compare(best single = {names[best]}, blend):")
    print("\n".join("  " + s for s in hc.report_lines(hc.compare(P[best], pbl, y, g, labels, n_boot=a.n_boot), labels)))
    cf, ws = cross_fit(y, P, g)
    print("the weights were picked on these same rows, so the blend score is a little optimistic.")
    print(f"cross-fit blend (weights picked on half the prompts, scored on the other half): {cf:.4f}  "
          f"(weights {[round(float(x), 2) for x in ws[0]]} and {[round(float(x), 2) for x in ws[1]]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
