import json

import numpy as np
import pandas as pd
import pytest

import head_cv as hc

LABELS = ["a", "b", "c"]


def toy(n_prompts=30, per=3, n_feat=6, signal=2.0, seed=0):
    """Rows with prompt groups; feature 0 carries the label, the rest is noise."""
    rng = np.random.RandomState(seed)
    n = n_prompts * per
    y = rng.randint(0, 3, n)
    X = rng.randn(n, n_feat)
    X[:, 0] += signal * y
    groups = np.repeat([f"p{i}" for i in range(n_prompts)], per)
    return X, y, groups


def test_build_X_matches_rohans_layout(tmp_path):
    ids = ["1", "2"]
    for t, base in (("flash", 1.0), ("glm53", 0.25)):
        pd.DataFrame({"id": ids, "lp_mean": [base, base * 2], "n_tok": [10, 20]}).to_csv(tmp_path / f"{t}_features.csv", index=False)
    pd.DataFrame({"p:x": [5.0, 6.0]}, index=pd.Index(ids, name="id")).to_parquet(tmp_path / "paired_token_features.parquet")
    spec = {"format": "ntl-prefixb-115-v1", "features_54": ["flash:lp_mean", "d:lp_mean"], "diff_features": ["lp_mean"],
            "features_61": ["p:x"]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    X, names = hc.build_X(tmp_path, ["2", "1"], expect=3)
    assert names == ["flash:lp_mean", "d:lp_mean", "p:x"]
    assert np.allclose(X, [[2.0, 1.5, 6.0], [1.0, 0.75, 5.0]])


def test_build_X_checks_width(tmp_path):
    test_build_X_matches_rohans_layout(tmp_path)
    with pytest.raises(AssertionError, match="115"):
        hc.build_X(tmp_path, ["1"])


def test_folds_keep_prompts_together():
    _, y, g = toy()
    f = hc.grouped_folds(g, 5)
    assert set(f) == set(range(5))
    for p in np.unique(g):
        assert len(set(f[g == p])) == 1


def test_oof_is_out_of_fold_and_learns():
    X, y, g = toy()
    p = hc.oof_probs(X, y, g, n_classes=3, lgb_seeds=())
    assert p.shape == (len(y), 3) and np.allclose(p.sum(1), 1)
    assert (p.argmax(1) == y).mean() > 0.5     # chance is 0.33; C=0.025 is strong shrinkage


def test_oof_rows_never_see_their_own_fold(monkeypatch):
    X, y, g = toy()
    seen = []
    real = hc.head_fit_predict
    def spy(Xtr, ytr, Xva, n_classes, lgb_seeds):
        seen.append((len(Xtr), len(Xva)))
        return real(Xtr, ytr, Xva, n_classes, lgb_seeds)
    monkeypatch.setattr(hc, "head_fit_predict", spy)
    hc.oof_probs(X, y, g, n_classes=3, lgb_seeds=())
    assert len(seen) == 5 and all(a + b == len(y) for a, b in seen)


def test_head_with_lgb_runs_and_mixes():
    X, y, g = toy(n_prompts=20)
    p = hc.head_fit_predict(X[:40], y[:40], X[40:], 3, lgb_seeds=(42,))
    assert p.shape == (len(y) - 40, 3) and np.allclose(p.sum(1), 1, atol=1e-6)


def test_head_maps_missing_class_to_zero():
    X, y, g = toy()
    keep = y != 2
    p = hc.head_fit_predict(X[keep], y[keep], X[:5], 3, lgb_seeds=())
    assert p.shape == (5, 3) and np.allclose(p[:, 2], 0)


def test_add_features_checks_rows_and_values():
    X = np.zeros((4, 2))
    assert hc.add_features(X, np.ones((4, 3))).shape == (4, 5)
    with pytest.raises(ValueError, match="rows"):
        hc.add_features(X, np.ones((3, 1)))
    bad = np.ones((4, 1)); bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        hc.add_features(X, bad)
    assert hc.add_features(X, np.ones(4)).shape == (4, 3)     # a single column is fine


def test_compare_keeps_a_real_gain():
    X, y, g = toy(n_prompts=60, signal=1.5)
    base = hc.oof_probs(X[:, 1:], y, g, n_classes=3, lgb_seeds=())   # noise only
    new = hc.oof_probs(X, y, g, n_classes=3, lgb_seeds=())           # with the signal column
    r = hc.compare(base, new, y, g, LABELS, n_boot=300)
    assert r["gain"] > 0.2 and r["lo"] > 0 and r["keep"] and r["submit"]
    assert len(r["recall_base"]) == len(r["recall_new"]) == 3
    assert len(r["confusions"]) <= 10


def test_compare_drops_noise():
    X, y, g = toy(n_prompts=60, signal=1.5)
    base = hc.oof_probs(X, y, g, n_classes=3, lgb_seeds=())
    r = hc.compare(base, base, y, g, LABELS, n_boot=200)
    assert r["gain"] == 0 and not r["keep"] and not r["submit"]


def test_keep_rule_thresholds():
    assert hc.keep_rule(0.005, 0.001) == (True, False)
    assert hc.keep_rule(0.008, 0.001) == (True, True)
    assert hc.keep_rule(0.010, -0.001) == (False, False)    # interval crosses 0
    assert hc.keep_rule(0.004, 0.001) == (False, False)


def test_report_lines_are_numbers_and_labels_only():
    X, y, g = toy(n_prompts=40)
    p = hc.oof_probs(X, y, g, n_classes=3, lgb_seeds=())
    lines = hc.report_lines(hc.compare(p, p, y, g, LABELS, n_boot=100), LABELS)
    text = "\n".join(lines)
    assert "balanced accuracy" in text and "DROP" in text and "mix-ups" in text


def test_base_block_goes_into_both_sides_and_block_only_into_new(tmp_path, monkeypatch):
    tr = pd.DataFrame({"id": ["1", "2", "3", "4"], "prompt_id": ["p1", "p1", "p2", "p2"], "model": ["a", "b", "c", "a"]})
    tr.to_csv(tmp_path / "train.csv", index=False)
    for name in ("t", "p"):
        np.savez(tmp_path / f"{name}.npz", ids=np.array(["4", "3", "2", "1"]), labels=np.array(LABELS),
                 probs=np.full((4, 3), 1 / 3), y=np.array([0, 2, 1, 0]), groups=np.array(["off:p2"] * 2 + ["off:p1"] * 2))
    monkeypatch.setattr(hc, "build_X", lambda pb, ids: (np.zeros((len(ids), 5)), None))
    widths = []
    monkeypatch.setattr(hc, "oof_probs", lambda X, y, g, k: widths.append(X.shape[1]) or np.eye(k)[y])
    hc.main(["--pb_dir", "x", "--train_csv", str(tmp_path / "train.csv"), "--base_block", str(tmp_path / "t.npz"),
             "--block", str(tmp_path / "p.npz"), "--out", str(tmp_path / "o.npz"), "--n_boot", "20"])
    assert widths == [8, 11]
