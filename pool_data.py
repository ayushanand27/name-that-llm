"""Builds the training set for the Open Pool run: official train.csv rows plus Open Pool rows.
Pure pandas, so it is tested on a laptop and runs the same way on Kaggle."""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_LABELS = 18


def normalize_text(s):
    """Lowercase and squash all whitespace, so two copies of a passage that differ only in spacing or case match."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def check_labels(official):
    """Returns the sorted label names from official train and stops if there are not exactly 18."""
    labels = sorted(official.model.unique())
    if len(labels) != N_LABELS:
        raise ValueError(f"expected {N_LABELS} labels in official train, got {len(labels)}")
    return labels


def drop_duplicate_texts(pool, official):
    """Drops pool rows whose text (after normalize_text) is already in official train, or appears earlier in the pool.
    This stops a holdout passage from sneaking into training through the pool."""
    seen = set(official.text.map(normalize_text))
    norm = pool.text.map(normalize_text)
    keep = ~norm.isin(seen) & ~norm.duplicated(keep="first")
    return pool[keep].reset_index(drop=True)


def apply_switches(pool, kimi_high_only, drop_flag_kb_proxy, kimi_label="kimi-k3"):
    """Optional filters, each an on/off switch so it can be A/B tested.
    kimi_high_only keeps pool kimi rows only when gen_reasoning == "high"; drop_flag_kb_proxy drops rows flagged
    flag_kb_proxy_after_ruling (made through the Kaggle Benchmarks proxy after the staff ruling)."""
    keep = pd.Series(True, index=pool.index)
    if kimi_high_only:
        keep &= (pool.model != kimi_label) | (pool.gen_reasoning == "high")
    if drop_flag_kb_proxy:
        keep &= ~pool.flag_kb_proxy_after_ruling.astype(bool)
    return pool[keep].reset_index(drop=True)


GEM37_LABEL, GEM37_CUTOFF = "gemini-3.7-flash", "2026-10-08"   # Google has served 3.8 for 3.7 requests since 8 Oct 2026


def gem37_after_mask(pool, label=GEM37_LABEL, cutoff=GEM37_CUTOFF):
    """True for pool rows labelled gemini-3.7-flash with gen_date on or after the cutoff (probably really 3.8).
    Rows with no date, or a date that doesn't parse, are False (kept): there is nothing to judge them by."""
    if "gen_date" not in pool.columns:
        return pd.Series(False, index=pool.index)
    d = pd.to_datetime(pool.gen_date, errors="coerce", utc=True, format="mixed")
    return (pool.model == label) & (d >= pd.Timestamp(cutoff, tz="UTC"))


def drop_gem37_after(pool, label=GEM37_LABEL, cutoff=GEM37_CUTOFF):
    return pool[~gem37_after_mask(pool, label, cutoff)].reset_index(drop=True)


def cap_per_label(pool, cap, seed):
    """Keeps at most `cap` random rows per label (same seed, same rows), so big labels don't drown small ones."""
    parts = [g.sample(n=min(cap, len(g)), random_state=seed) for _, g in pool.groupby("model", sort=True)]
    return pd.concat(parts).reset_index(drop=True) if parts else pool.iloc[:0]


def split_holdout(official, frac, seed):
    """Puts round(frac * number of prompts) whole prompt_ids into the holdout, so no prompt is in both halves."""
    prompts = np.array(sorted(official.prompt_id.astype(str).unique()))
    n_hold = int(round(frac * len(prompts)))
    hold_ids = set(np.random.RandomState(seed).choice(prompts, size=n_hold, replace=False))
    is_hold = official.prompt_id.astype(str).isin(hold_ids)
    return official[~is_hold].reset_index(drop=True), official[is_hold].reset_index(drop=True)


@dataclass
class Config:
    use_pool: bool = True
    cap: int = 1500
    kimi_high_only: bool = False
    drop_flag_kb_proxy: bool = False
    drop_gem37_after_oct8: bool = True
    holdout_frac: float = 0.2
    seed: int = 42
    kimi_label: str = "kimi-k3"


def _frame(df, label_ids, group, is_official):
    return pd.DataFrame({"id": df.id.astype(str).values, "text": df.text.values, "label": df.model.map(label_ids).values,
                         "group": group, "is_official": is_official})


def build_dataset(official, pool, cfg):
    """Returns (train, holdout, labels). Holdout = only official rows of the held-out prompts.
    Train = the other official rows, plus (if use_pool) the filtered, de-duplicated, capped pool rows."""
    labels = check_labels(official)
    ids = {m: i for i, m in enumerate(labels)}
    off_tr, off_ho = split_holdout(official, cfg.holdout_frac, cfg.seed)
    parts = [_frame(off_tr, ids, "off:" + off_tr.prompt_id.astype(str).values, True)]
    if cfg.use_pool:
        p = pool[pool.model.isin(ids)]
        p = apply_switches(p, cfg.kimi_high_only, cfg.drop_flag_kb_proxy, cfg.kimi_label)
        if cfg.drop_gem37_after_oct8:
            p = drop_gem37_after(p)
        p = drop_duplicate_texts(p, official)
        p = cap_per_label(p, cfg.cap, cfg.seed)
        parts.append(_frame(p, ids, "pool:" + p.prompt_group.astype(str).values, False))
    train = pd.concat(parts, ignore_index=True)
    hold = _frame(off_ho, ids, "off:" + off_ho.prompt_id.astype(str).values, True)
    return train, hold, labels
