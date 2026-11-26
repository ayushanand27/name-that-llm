"""Blend two runs' holdout probabilities (e.g. our LoRA and Rohan's PB115) and say whether the blend really helps.

Input: two OOF .npz files with ids, y, probs, labels, and groups (prompt_id) in at least one of them.
Rows are matched by id; only shared ids are used. Labels must be the same list in the same order.
Prints numbers only, never ids or passage text, so the output is safe for public Actions logs.

The blend is  w * LoRA + (1 - w) * PB115  with w from 0 to 1 in steps of 0.05, picked on this holdout.
Picking w on the same rows we score makes the blend score a little optimistic, so a cross-fit line
(w picked on half the prompts, scored on the other half, then swapped) gives the honest number.
"""
import argparse
import sys

import numpy as np

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
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
        raise SystemExit(f"{d['path']}: probs must be finite, non-negative, and not all zero in a row")
    if p.ndim != 2 or p.shape[1] != len(d["labels"]):
        raise SystemExit(f"{d['path']}: probs has {p.shape[-1]} columns but {len(d['labels'])} labels")


def align(a, b):
    """Shared ids in a's order; fails loudly on any mismatch in labels, true labels or prompt groups."""
    la, lb = [str(x) for x in a["labels"]], [str(x) for x in b["labels"]]
    if la != lb:
        diff = [i for i in range(max(len(la), len(lb))) if la[i:i + 1] != lb[i:i + 1]]
        raise SystemExit(f"labels differ (same names in the same order needed); first different position: {diff[0]}")
    for d in (a, b):
        check_probs(d)
    ia, ib = [str(x) for x in a["ids"]], {str(x): i for i, x in enumerate(b["ids"])}
    keep = [(i, ib[x]) for i, x in enumerate(ia) if x in ib]
    if not keep:
        raise SystemExit("no shared ids between the two files")
    sa = np.array([i for i, _ in keep]); sb = np.array([j for _, j in keep])
    if (a["y"][sa] != b["y"][sb]).any():
        raise SystemExit(f"true label differs on {(a['y'][sa] != b['y'][sb]).sum()} shared rows")
    ga = a["groups"][sa].astype(str) if "groups" in a else None
    gb = b["groups"][sb].astype(str) if "groups" in b else None
    if ga is None and gb is None:
        raise SystemExit("need prompt groups (key 'groups' = prompt_id) in at least one file for the bootstrap")
    if ga is not None and gb is not None and (ga != gb).any():
        raise SystemExit("prompt groups differ between the two files on shared rows")
    return {"ids": a["ids"][sa], "y": a["y"][sa], "groups": ga if ga is not None else gb,
            "pa": norm(a["probs"][sa]), "pb": norm(b["probs"][sb]), "dropped_a": len(ia) - len(keep),
            "dropped_b": len(ib) - len(keep)}


def weight_grid():
    return np.round(np.arange(0, 21) * 0.05, 2)


def blend(pa, pb, w):
    return w * pa + (1 - w) * pb


def best_weight(y, pa, pb):
    """w (weight on pa) with the best balanced accuracy; ties go to the w nearest 0.5, the safest middle."""
    scores = [(mt.balanced_accuracy(y, blend(pa, pb, w).argmax(1)), -abs(w - 0.5), w) for w in weight_grid()]
    ba, _, w = max(scores)
    return float(w), float(ba)


def cross_fit(y, pa, pb, groups, seed=0):
    """Pick w on one half of the prompts, predict the other half with it, and swap; returns BA and both ws."""
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    half = set(rng.permutation(uniq)[: len(uniq) // 2])
    m = np.array([g in half for g in groups])
    pred = np.empty(len(y), dtype=int); ws = []
    for fit, score in ((m, ~m), (~m, m)):
        w, _ = best_weight(y[fit], pa[fit], pb[fit])
        pred[score] = blend(pa[score], pb[score], w).argmax(1); ws.append(w)
    return mt.balanced_accuracy(y, pred), ws


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lora", help="OOF .npz of the LoRA run (weight w)")
    ap.add_argument("pb115", help="OOF .npz of the PB115 run (weight 1 - w)")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args(argv)
    d = align(load(a.lora), load(a.pb115))
    y, g, pa, pb = d["y"], d["groups"], d["pa"], d["pb"]
    w, _ = best_weight(y, pa, pb)
    pl, pp, pblend = pa.argmax(1), pb.argmax(1), blend(pa, pb, w).argmax(1)
    print(f"rows {len(y)}  prompts {len(np.unique(g))}  (not shared: {d['dropped_a']} LoRA rows, {d['dropped_b']} PB115 rows)")
    print("balanced accuracy, 95% interval over prompts:")
    for name, p in (("LoRA alone", pl), ("PB115 alone", pp), (f"blend w = {w:.2f}", pblend)):
        ba, lo, hi = mt.bootstrap_ci(y, p, g, n_boot=a.n_boot)
        print(f"  {name:<16} {ba:.4f}  ({lo:.4f} to {hi:.4f})")
    print("paired difference, same resampled prompts for both (interval not crossing 0 = probably real):")
    for name, base in (("blend - PB115", pp), ("blend - LoRA", pl)):
        diff, lo, hi = mt.paired_bootstrap_diff(y, base, pblend, g, n_boot=a.n_boot)
        print(f"  {name:<16} {diff:+.4f}  ({lo:+.4f} to {hi:+.4f})")
    cf, ws = cross_fit(y, pa, pb, g)
    print(f"w was picked on these same rows, so the blend score is a little optimistic.")
    print(f"cross-fit blend (w picked on half the prompts, scored on the other half): {cf:.4f}  (w = {ws[0]:.2f}, {ws[1]:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
