import numpy as np
import pandas as pd
import pytest

import pool_data as pdm

LABELS = [f"m{i:02d}" for i in range(18)]


def fake_official(n_prompts=10, seed=0):
    rows = []
    for p in range(n_prompts):
        for j, lab in enumerate(LABELS):
            rows.append({"id": f"o{p}_{j}", "prompt_id": f"P{p}", "model": lab, "text": f"official text {p} {j}"})
    return pd.DataFrame(rows)


def fake_pool():
    rows = []
    for j, lab in enumerate(LABELS):
        for k in range(30):
            rows.append({"id": f"q{j}_{k}", "model": lab, "text": f"pool text {j} {k}", "prompt_group": f"g{k % 7}",
                         "gen_reasoning": "high" if k % 3 == 0 else "default", "flag_kb_proxy_after_ruling": k % 5 == 0})
    df = pd.DataFrame(rows)
    # duplicates: one copies an official text (different spacing/case), one copies another pool row
    df.loc[0, "text"] = "  OFFICIAL   text 0 0 "
    df.loc[1, "text"] = "Pool text 0 2"
    return df


def test_normalize_text():
    assert pdm.normalize_text("  Hello\n\tWORLD  ") == "hello world"


def test_check_labels_exact_18():
    assert pdm.check_labels(fake_official()) == LABELS
    bad = fake_official()
    bad = bad[bad.model != "m03"]
    with pytest.raises(ValueError):
        pdm.check_labels(bad)


def test_drop_duplicates_vs_official_and_pool():
    out = pdm.drop_duplicate_texts(fake_pool(), fake_official())
    assert "q0_0" not in set(out.id)            # copied an official text
    assert len(out) == 18 * 30 - 2              # q0_1 duplicates q0_2, only one of them stays
    assert out.text.map(pdm.normalize_text).is_unique


def test_switches():
    pool = fake_pool()
    kimi = LABELS[5]
    out = pdm.apply_switches(pool, kimi_high_only=True, drop_flag_kb_proxy=False, kimi_label=kimi)
    assert (out[out.model == kimi].gen_reasoning == "high").all()
    assert (out.model != kimi).sum() == (pool.model != kimi).sum()       # other labels untouched
    out2 = pdm.apply_switches(pool, kimi_high_only=False, drop_flag_kb_proxy=True, kimi_label=kimi)
    assert not out2.flag_kb_proxy_after_ruling.any()
    assert len(pdm.apply_switches(pool, False, False, kimi)) == len(pool)


def test_cap_per_label():
    out = pdm.cap_per_label(fake_pool(), cap=7, seed=1)
    assert out.model.value_counts().max() == 7 and out.model.nunique() == 18
    assert pdm.cap_per_label(fake_pool(), cap=7, seed=1).id.tolist() == out.id.tolist()   # repeatable
    assert len(pdm.cap_per_label(fake_pool(), cap=0, seed=1)) == 0


def test_split_holdout_grouped():
    off = fake_official(n_prompts=10)
    tr, ho = pdm.split_holdout(off, frac=0.2, seed=3)
    assert set(tr.prompt_id).isdisjoint(set(ho.prompt_id))
    assert ho.prompt_id.nunique() == 2 and len(tr) + len(ho) == len(off)


def test_build_dataset_no_leak_and_labels():
    off, pool = fake_official(), fake_pool()
    cfg = pdm.Config(use_pool=True, cap=10, kimi_high_only=True, drop_flag_kb_proxy=False, holdout_frac=0.2, seed=0,
                     kimi_label=LABELS[5])
    train, hold, labels = pdm.build_dataset(off, pool, cfg)
    assert labels == LABELS
    assert hold.is_official.all()
    off_train_groups = set(train[train.is_official].group)
    assert off_train_groups.isdisjoint(set(hold.group))
    assert not set(train.group).intersection(set(hold.group))
    assert set(train.text.map(pdm.normalize_text)).isdisjoint(set(hold.text.map(pdm.normalize_text)))
    assert train[~train.is_official].groupby("label").size().max() <= 10
    assert train.label.between(0, 17).all() and hold.label.between(0, 17).all()
    assert all(g.startswith("pool:") for g in train[~train.is_official].group)


def test_build_dataset_official_only():
    train, hold, _ = pdm.build_dataset(fake_official(), fake_pool(), pdm.Config(use_pool=False, seed=0))
    assert train.is_official.all()


def test_holdout_frac_zero_builds_no_holdout():
    train, hold, _ = pdm.build_dataset(fake_official(), fake_pool(), pdm.Config(use_pool=True, cap=5, holdout_frac=0.0, seed=0))
    assert len(hold) == 0 and int(train.is_official.sum()) == len(fake_official())
