"""CPU baseline on the same split as the LoRA run (TF-IDF + linear classifier), plus helpers every run shares:
save the holdout probabilities, print the report, compare saved runs with a paired bootstrap."""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import metrics as mt
import pool_data as pdm


def fit_predict(train, hold, n_labels, seed):
    """Word and character n-gram TF-IDF, then a balanced logistic-loss linear model; returns holdout probabilities."""
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300_000)
    xtr = hstack([vw.fit_transform(train.text), vc.fit_transform(train.text)]).tocsr()
    xho = hstack([vw.transform(hold.text), vc.transform(hold.text)]).tocsr()
    clf = SGDClassifier(loss="log_loss", alpha=2e-6, class_weight="balanced", max_iter=30, tol=None, random_state=seed)
    clf.fit(xtr, train.label.values)
    probs = np.zeros((len(hold), n_labels))
    probs[:, clf.classes_] = clf.predict_proba(xho)
    return probs


def report_lines(hold, probs, labels, n_boot=2000):
    """Holdout balanced accuracy with its 95% interval, recall per label, and the most common mix-ups."""
    y, p, g = hold.label.values, probs.argmax(1), hold.group.values
    ba, lo, hi = mt.bootstrap_ci(y, p, g, n_boot=n_boot)
    out = [f"holdout balanced accuracy {ba:.4f}  (95% interval {lo:.4f} to {hi:.4f}, {len(set(g))} prompts, {len(y)} rows)"]
    out += [f"  recall {labels[i]:<24} {r:.3f}" for i, r in enumerate(mt.per_class_recall(y, p, len(labels)))]
    out += [f"  confused {a} -> {b}: {n}" for a, b, n in mt.top_confusions(y, p, labels, k=10)]
    return out


def save_oof(path, hold, probs, labels, run_name):
    """Saves holdout ids, prompt groups, true labels and probabilities, so runs can be compared or blended later."""
    np.savez(path, ids=hold.id.astype(str).values.astype("U"), groups=hold.group.astype(str).values.astype("U"),
             y=hold.label.values, probs=probs.astype(np.float32), labels=np.asarray(labels, dtype="U"), run=np.asarray(run_name))


def compare_oof(paths, n_boot=2000):
    """For saved runs on the same holdout: each run's score, then every pair's difference with a 95% interval."""
    runs = [np.load(p, allow_pickle=False) for p in paths]
    for r in runs[1:]:
        assert (r["ids"] == runs[0]["ids"]).all(), "runs must share the same holdout rows in the same order"
    y, g = runs[0]["y"], runs[0]["groups"]
    out = []
    for r in runs:
        ba, lo, hi = mt.bootstrap_ci(y, r["probs"].argmax(1), g, n_boot=n_boot)
        out.append(f"{str(r['run']):<12} {ba:.4f}  ({lo:.4f} to {hi:.4f})")
    for a, b in itertools.combinations(runs, 2):
        d, lo, hi = mt.paired_bootstrap_diff(y, a["probs"].argmax(1), b["probs"].argmax(1), g, n_boot=n_boot)
        out.append(f"{str(b['run'])} minus {str(a['run'])}: {d:+.4f}  (95% interval {lo:+.4f} to {hi:+.4f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default="baseline_oof.npz")
    ap.add_argument("--use_pool", type=int, default=1)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--kimi_high_only", type=int, default=0)
    ap.add_argument("--drop_flag_kb_proxy", type=int, default=1)
    ap.add_argument("--drop_gem37_after_oct8", type=int, default=0, help="drop pool gemini-3.7 rows dated 8 Oct 2026 or later")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = pdm.Config(use_pool=bool(a.use_pool), cap=a.cap, kimi_high_only=bool(a.kimi_high_only),
                     drop_flag_kb_proxy=bool(a.drop_flag_kb_proxy), drop_gem37_after_oct8=bool(a.drop_gem37_after_oct8), seed=a.seed)
    official = pd.read_csv(a.train_csv, dtype={"id": str, "prompt_id": str})
    pool = pd.read_parquet(a.pool) if cfg.use_pool else official.iloc[:0]
    train, hold, labels = pdm.build_dataset(official, pool, cfg)
    print(f"train rows {len(train)} (official {int(train.is_official.sum())}), holdout rows {len(hold)}", flush=True)
    probs = fit_predict(train, hold, len(labels), a.seed)
    print("\n".join(report_lines(hold, probs, labels)))
    save_oof(a.out, hold, probs, labels, run_name=f"tfidf_pool{a.use_pool}_kimi{a.kimi_high_only}_g37drop{a.drop_gem37_after_oct8}")


if __name__ == "__main__":
    main()
