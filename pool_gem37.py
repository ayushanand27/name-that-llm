"""Pool fix check: are pool gemini-3.7-flash rows dated 8 Oct 2026 or later really 3.8?

Google has routed gemini-3.7-flash requests to 3.8 since 8 Oct 2026. This counts those rows, shows which model
the pool says served them, and asks the train-only TF-IDF yardstick (fit on the 2,699 official rows only) what it
calls 3.7 rows before and after that date, next to real 3.8 rows. Prints counts and shares only, never text.
"""
import argparse

import numpy as np
import pandas as pd

import baseline_tfidf as bt
import pool_data as pdm

FLASH = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]


def date_counts(pool):
    g37 = pool[pool.model == pdm.GEM37_LABEL]
    d = pd.to_datetime(g37.gen_date, errors="coerce", utc=True, format="mixed")
    after = d >= pd.Timestamp(pdm.GEM37_CUTOFF, tz="UTC")
    return {"before 8 Oct": int((d.notna() & ~after).sum()), "8 Oct or later": int(after.sum()),
            "no usable date": int(d.isna().sum())}


def yardstick_groups(pool):
    """'3.7 before', '3.7 after', '3.7 no date', '3.8' or None for each row."""
    d = pd.to_datetime(pool.gen_date, errors="coerce", utc=True, format="mixed")
    after = d >= pd.Timestamp(pdm.GEM37_CUTOFF, tz="UTC")
    g = pd.Series([None] * len(pool), index=pool.index, dtype=object)
    is37 = pool.model == pdm.GEM37_LABEL
    g[is37 & d.notna() & ~after] = "3.7 before"
    g[is37 & d.isna()] = "3.7 no date"
    g[is37 & after] = "3.7 after"
    g[pool.model == "gemini-3.8-flash"] = "3.8"
    return g


def share_table(groups, pred, columns):
    t = pd.crosstab(pd.Series(groups.values, name="group"), pd.Series(pred, name="called"), normalize="index")
    return t.reindex(columns=columns, fill_value=0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--per_group", type=int, default=1500)
    a = ap.parse_args(argv)
    off = pd.read_csv(a.train_csv, dtype={"id": str})
    cols = ["id", "model", "text", "gen_date", "gen_served_model"]
    pool = pd.read_parquet(a.pool, columns=cols)
    print("pool gemini-3.7-flash rows by gen_date:", date_counts(pool))
    print("distinct gen_date values on 3.7 rows:", pool[pool.model == pdm.GEM37_LABEL].gen_date.value_counts(dropna=False).to_dict())
    g37 = pool[pool.model == pdm.GEM37_LABEL]
    print("which model the pool says served the 3.7 rows (gen_served_model), before vs after:")
    print(pd.crosstab(g37.gen_served_model.fillna("(none)"), pdm.gem37_after_mask(g37).map({True: "8 Oct or later", False: "before or no date"})).to_string())
    labels = pdm.check_labels(off)
    ids = {m: i for i, m in enumerate(labels)}
    train = pd.DataFrame({"text": off.text.values, "label": off.model.map(ids).values})
    groups = yardstick_groups(pool)
    rows = []
    for name in ("3.7 before", "3.7 after", "3.7 no date", "3.8"):
        idx = groups.index[groups == name]
        if len(idx):
            rows.append(pool.loc[np.random.RandomState(0).choice(idx, min(len(idx), a.per_group), replace=False)].assign(grp=name))
    if not rows:
        print("no rows to score"); return 0
    hold = pd.concat(rows, ignore_index=True)
    probs = bt.fit_predict(train, hold, len(labels), seed=0)
    pred = np.asarray(labels)[probs.argmax(1)]
    print(f"train-only TF-IDF yardstick (fit on official rows only), share called each Flash label; rows per group: "
          f"{hold.grp.value_counts().to_dict()}")
    print(share_table(hold.grp, pred, FLASH).round(3).to_string())
    print("share called any other label:", (~pd.Series(pred).isin(FLASH)).groupby(hold.grp.values).mean().round(3).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
