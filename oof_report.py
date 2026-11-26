"""Where a run is weak: recall per label with its row count and a 95% interval, weakest first, plus top mix-ups.
Reads an oof_<run>.npz saved by baseline_tfidf.save_oof. Prints numbers only, never passage text."""
import argparse
import math

import numpy as np

import metrics as mt

FEW_ROWS = 20   # below this a label's recall moves by 5+ points per passage; treat it as a rough guess


def wilson(k, n, z=1.96):
    """95% interval for k hits out of n (Wilson score), which stays sensible when n is small."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def label_table(d):
    """One row per label: holdout rows, recall, 95% interval, few-rows flag; weakest recall first."""
    y, p, labels = d["y"], d["probs"].argmax(1), [str(x) for x in d["labels"]]
    rows = []
    for i, name in enumerate(labels):
        n = int((y == i).sum()); k = int(((y == i) & (p == i)).sum())
        lo, hi = wilson(k, n)
        rows.append({"label": name, "n": n, "recall": k / n if n else float("nan"), "lo": lo, "hi": hi,
                     "few_rows": n < FEW_ROWS})
    return sorted(rows, key=lambda r: (r["recall"], r["label"]))


def report_lines(d, k=10):
    """Text report: balanced accuracy with prompt-level 95% interval, the label table, top k mix-ups."""
    y, p, g = d["y"], d["probs"].argmax(1), d["groups"]
    labels = [str(x) for x in d["labels"]]
    ba, lo, hi = mt.bootstrap_ci(y, p, g)
    out = [f"run {d['run']}: balanced accuracy {ba:.4f} (95% interval {lo:.4f} to {hi:.4f}, {len(set(g))} prompts, {len(y)} rows)",
           "recall per label, weakest first (rows, recall, 95% interval):"]
    for r in label_table(d):
        flag = "  <- few rows, rough" if r["few_rows"] else ""
        out.append(f"  {r['label']:<24} n={r['n']:<3} {r['recall']:.3f}  [{r['lo']:.2f}, {r['hi']:.2f}]{flag}")
    out.append(f"top {k} mix-ups (true -> predicted: count):")
    out += [f"  {a} -> {b}: {n}" for a, b, n in mt.top_confusions(y, p, labels, k=k)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="+")
    for path in ap.parse_args(argv).npz:
        print("\n".join(report_lines(np.load(path))))


if __name__ == "__main__":
    main()
