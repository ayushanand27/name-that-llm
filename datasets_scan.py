"""Phase 3 part A: find public Kaggle datasets for this competition and size them up. Report only.

Runs in GitHub Actions (Kaggle token from secrets). For each candidate dataset: owner, files, rows with one of
our 18 labels, labels covered, licence, how much is an exact duplicate of the Open Pool or the official train
(after normalize_text), and how often the train-only TF-IDF yardstick recognises each label's rows.
Prints dataset refs, counts, label names and licence names only, never passage text.
"""
import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_tfidf as bt
import pool_data as pdm

SEARCHES = ["name that llm", "name-that-llm", "ntl llm", "ntl generations", "llm attribution 18 models"]
SKIP = {"crtproduction/name-that-llm-open-pool"}
TEXT_HINTS = ("text", "passage", "generation", "output", "response", "completion", "content")
LABEL_HINTS = ("model", "label", "source", "generator", "llm", "target")


def parse_list(text):
    return [r for r in csv.DictReader(io.StringIO(text)) if r.get("ref")]


def map_label(v, labels):
    """Exact label after lowercasing, trimming and dropping a 'vendor/' prefix; anything else is None."""
    s = str(v).strip().lower().split("/")[-1].removesuffix(":free")
    return s if s in labels else None


def find_columns(df, labels):
    """(text column, label column): text = longest string column, label = a column whose values are mostly our labels."""
    obj = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    if not obj:
        return None, None
    lens = {c: df[c].astype(str).str.len().mean() for c in obj}
    hinted = [c for c in obj if any(h in c.lower() for h in TEXT_HINTS)]
    text = max(hinted or obj, key=lambda c: lens[c])
    best, best_share = None, 0.0
    for c in obj:
        if c == text:
            continue
        share = df[c].head(5000).map(lambda v: map_label(v, labels) is not None).mean()
        if share > best_share:
            best, best_share = c, share
    return text, (best if best_share >= 0.3 else None)


def norm_key(s):
    return hashlib.md5(pdm.normalize_text(s).encode()).hexdigest()


def overlap(texts, keys):
    return int(texts.map(norm_key).isin(keys).sum())


def sh(cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout


def candidates(extra):
    seen = {}
    for q in SEARCHES:
        for r in parse_list(sh(["kaggle", "datasets", "list", "-s", q, "--csv", "--sort-by", "updated"], check=False)):
            seen.setdefault(r["ref"], r)
    for ref in extra:
        seen.setdefault(ref, {"ref": ref})
    return {k: v for k, v in seen.items() if k not in SKIP}


def read_table(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    return pd.read_csv(path, low_memory=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--max_mb", type=float, default=400)
    ap.add_argument("--per_label", type=int, default=300)
    a = ap.parse_args(argv)
    off = pd.read_csv(a.train_csv, dtype={"id": str})
    labels = pdm.check_labels(off)
    ids = {m: i for i, m in enumerate(labels)}
    pool_keys = set(pd.read_parquet(a.pool, columns=["text"]).text.map(norm_key))
    off_keys = set(off.text.map(norm_key))
    print("pool version:", sh(["kaggle", "datasets", "files", "crtproduction/name-that-llm-open-pool", "--csv"], check=False).splitlines()[1:4])
    # yardstick: fit on the official train split only; its holdout recall is the "real rows" rate per label
    tr, ho, _ = pdm.build_dataset(off, off.iloc[:0], pdm.Config(use_pool=False))
    train = tr[["text", "label"]]
    real = pd.Series((bt.fit_predict(train, ho, len(labels), seed=0).argmax(1) == ho.label.values)).groupby(
        np.asarray(labels)[ho.label.values]).mean()
    print("yardstick on real official holdout rows (share called its own label):", {k: round(float(v), 3) for k, v in real.items()})
    cands = candidates(a.extra)
    print(f"{len(cands)} candidate datasets")
    for ref, row in sorted(cands.items()):
        print(f"\n== {ref}  updated {row.get('lastUpdated', '?')}  size {row.get('size', '?')}  downloads {row.get('downloadCount', '?')}")
        d = Path("scan") / ref.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        sh(["kaggle", "datasets", "metadata", ref, "-p", str(d)], check=False)
        mp = d / "dataset-metadata.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            lic = [l.get("name") for l in m.get("licenses", [])]
            print(f"  licence {lic}; subtitle: {str(m.get('subtitle') or '')[:120]}")
        files = parse_list(sh(["kaggle", "datasets", "files", ref, "--csv", "--page-size", "200"], check=False).replace("name,", "ref,", 1))
        tables = [f for f in files if f["ref"].endswith((".csv", ".parquet", ".jsonl", ".json"))]
        big = [f["ref"] for f in tables if int(f.get("size") or 0) > a.max_mb * 2**20]
        print(f"  {len(files)} files, {len(tables)} tables" + (f", skipped over {a.max_mb:.0f} MB: {big}" if big else ""))
        for f in tables:
            if f["ref"] in big:
                continue
            sh(["kaggle", "datasets", "download", ref, "-f", f["ref"], "-p", str(d), "-q"], check=False)
        for z in d.glob("*.zip"):
            subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(d)], check=False)
        frames = []
        for p in sorted(d.rglob("*")):
            if p.suffix not in (".csv", ".parquet", ".jsonl", ".json") or p.name == "dataset-metadata.json":
                continue
            try:
                df = read_table(p)
            except Exception as e:  # noqa: BLE001 - report and move on
                print(f"  {p.name}: can't read ({type(e).__name__})"); continue
            tc, lc = find_columns(df, labels)
            print(f"  {p.name}: {len(df)} rows, text col {tc}, label col {lc}")
            if tc and lc:
                lab = df[lc].map(lambda v: map_label(v, labels))
                frames.append(pd.DataFrame({"text": df[tc].astype(str), "model": lab}).dropna())
        if not frames:
            print("  no rows with one of our 18 labels"); continue
        x = pd.concat(frames, ignore_index=True)
        x = x[x.text.str.split().str.len().between(40, 250)]
        keys = x.text.map(norm_key)
        print(f"  rows with our labels (40-250 words): {len(x)}; dup of pool {int(keys.isin(pool_keys).sum())}, "
              f"dup of official train {int(keys.isin(off_keys).sum())}, new {int((~keys.isin(pool_keys | off_keys)).sum())}")
        new = x[~keys.isin(pool_keys | off_keys)]
        print("  new rows per label:", new.model.value_counts().to_dict())
        if len(new):
            s = new.groupby("model", group_keys=False).apply(lambda g: g.sample(min(len(g), a.per_label), random_state=0))
            probs = bt.fit_predict(train, s.reset_index(drop=True), len(labels), seed=0)
            hit = (np.asarray(labels)[probs.argmax(1)] == s.model.values)
            rate = pd.Series(hit).groupby(s.model.values).mean()
            print("  yardstick on these rows vs real rows (fidelity gate: at least 0.8 x real):")
            for k, v in rate.items():
                gate = "PASS" if v >= 0.8 * real.get(k, np.nan) else "fail"
                print(f"    {k:<24} {v:.3f} vs {real.get(k, float('nan')):.3f}  {gate}  (n={int((s.model == k).sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
