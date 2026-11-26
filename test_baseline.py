import numpy as np
import pandas as pd

import baseline_tfidf as bt
import pool_data as pdm
from test_pool_data import fake_official, fake_pool, LABELS


def test_save_and_compare_oof(tmp_path):
    hold = pd.DataFrame({"id": ["a", "b", "c", "d"], "group": ["off:1", "off:1", "off:2", "off:2"], "label": [0, 1, 0, 1]})
    probs = np.array([[.9, .1], [.2, .8], [.6, .4], [.7, .3]])
    f = tmp_path / "oof.npz"
    bt.save_oof(f, hold, probs, ["x", "y"], run_name="A")
    d = np.load(f, allow_pickle=False)
    assert d["ids"].tolist() == ["a", "b", "c", "d"] and d["probs"].shape == (4, 2) and str(d["run"]) == "A"
    lines = bt.compare_oof([f, f], n_boot=50)
    assert any("A" in l for l in lines) and any("+0.0000" in l for l in lines)


def test_baseline_runs_on_fake_data():
    off = fake_official(n_prompts=10)
    off["text"] = [f"{m} style words {m} {i}" for i, m in enumerate(off.model)]   # learnable fake signal
    train, hold, labels = pdm.build_dataset(off, fake_pool(), pdm.Config(use_pool=False, seed=0))
    probs = bt.fit_predict(train, hold, n_labels=len(labels), seed=0)
    assert probs.shape == (len(hold), 18)
    assert np.allclose(probs.sum(1), 1)
    assert (probs.argmax(1) == hold.label.values).mean() > 0.9


def test_pool_only_tfidf_scores_every_official_row(tmp_path):
    off = fake_official(n_prompts=10)
    off["text"] = [f"{m} style words {m} {i}" for i, m in enumerate(off.model)]
    pool = fake_pool()
    pool["text"] = [f"{m} style words {m} p{i}" for i, m in enumerate(pool.model)]
    tr, pq, out = tmp_path / "train.csv", tmp_path / "pool.parquet", tmp_path / "oof.npz"
    off.to_csv(tr, index=False)
    pool.to_parquet(pq)
    bt.main(["--train_csv", str(tr), "--pool", str(pq), "--official_in_train", "0", "--out", str(out)])
    d = np.load(out, allow_pickle=False)
    assert sorted(d["ids"].tolist()) == sorted(off.id.astype(str).tolist())
