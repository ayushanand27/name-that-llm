import numpy as np
import pytest

import blend_eval as be

LABELS = ["a", "b", "c"]


def make(path, ids, y, probs, labels=LABELS, groups=None):
    kw = dict(ids=np.asarray(ids, dtype="U"), y=np.asarray(y), probs=np.asarray(probs, dtype=np.float32),
              labels=np.asarray(labels, dtype="U"))
    if groups is not None:
        kw["groups"] = np.asarray(groups, dtype="U")
    np.savez(path, **kw)
    return str(path)


def onehot(y, n=3, hit=0.9):
    p = np.full((len(y), n), (1 - hit) / (n - 1))
    p[np.arange(len(y)), y] = hit
    return p


def test_align_keeps_intersection_in_same_order(tmp_path):
    a = make(tmp_path / "a.npz", ["1", "2", "3"], [0, 1, 2], onehot([0, 1, 2]), groups=["p1", "p1", "p2"])
    b = make(tmp_path / "b.npz", ["3", "4", "1"], [2, 0, 0], onehot([2, 0, 0]))
    d = be.align([be.load(a), be.load(b)])
    assert list(d["ids"]) == ["1", "3"]
    assert list(d["y"]) == [0, 2]
    assert list(d["groups"]) == ["p1", "p2"]
    assert d["P"][0].shape == d["P"][1].shape == (2, 3)


def test_labels_must_match_exactly(tmp_path):
    a = make(tmp_path / "a.npz", ["1"], [0], onehot([0]), groups=["p"])
    b = make(tmp_path / "b.npz", ["1"], [0], onehot([0]), labels=["a", "c", "b"])
    with pytest.raises(SystemExit, match="labels"):
        be.align([be.load(a), be.load(b)])


def test_true_label_must_agree(tmp_path):
    a = make(tmp_path / "a.npz", ["1"], [0], onehot([0]), groups=["p"])
    b = make(tmp_path / "b.npz", ["1"], [1], onehot([1]))
    with pytest.raises(SystemExit, match="true label"):
        be.align([be.load(a), be.load(b)])


def test_no_overlap_fails(tmp_path):
    a = make(tmp_path / "a.npz", ["1"], [0], onehot([0]), groups=["p"])
    b = make(tmp_path / "b.npz", ["2"], [0], onehot([0]))
    with pytest.raises(SystemExit, match="no shared ids"):
        be.align([be.load(a), be.load(b)])


def test_needs_prompt_groups(tmp_path):
    a = make(tmp_path / "a.npz", ["1"], [0], onehot([0]))
    b = make(tmp_path / "b.npz", ["1"], [0], onehot([0]))
    with pytest.raises(SystemExit, match="prompt"):
        be.align([be.load(a), be.load(b)])


def test_bad_probs_fail(tmp_path):
    bad = onehot([0]); bad[0, 1] = np.nan
    a = make(tmp_path / "a.npz", ["1"], [0], bad, groups=["p"])
    b = make(tmp_path / "b.npz", ["1"], [0], onehot([0]))
    with pytest.raises(SystemExit, match="finite"):
        be.align([be.load(a), be.load(b)])


def test_rows_are_renormalised():
    p = be.norm(np.array([[2.0, 2.0, 0.0]]))
    assert np.allclose(p, [[0.5, 0.5, 0.0]])


def test_weight_grid_is_0_to_1_by_005():
    g = be.weight_grid()
    assert len(g) == 21 and g[0] == 0.0 and g[-1] == 1.0
    assert np.allclose(np.diff(g), 0.05)


def test_best_weight_picks_the_model_that_is_right():
    y = np.array([0, 1, 2, 0, 1, 2])
    good, bad = onehot(y), onehot((y + 1) % 3)
    w, ba = be.best_weight(y, good, bad)        # w is the weight on the first (LoRA) model
    assert w > 0.5 and ba == 1.0                # every w above 0.5 ties at 1.0; ties go to the w nearest 0.5
    w, _ = be.best_weight(y, bad, good)
    assert w < 0.5


def test_a_sure_wrong_model_pushes_w_high():
    y = np.array([0, 1, 2, 0])
    a = onehot(y, hit=0.5)
    b = onehot((y + 1) % 3, hit=0.95)           # b is very sure and always wrong: any b weight hurts
    assert be.best_weight(y, a, b) == (0.8, 1.0)   # a wins only once w > 0.79


def test_blend_beats_both_when_they_err_on_different_rows():
    # A is sure on rows 0-2 and unsure-wrong on 3-5; B the other way round. Half-half gets all six.
    y = np.array([0, 1, 2, 0, 1, 2])
    a = np.vstack([onehot(y[:3], hit=0.9), onehot((y[3:] + 1) % 3, hit=0.4)])
    b = np.vstack([onehot((y[:3] + 1) % 3, hit=0.4), onehot(y[3:], hit=0.9)])
    w, ba = be.best_weight(y, a, b)
    assert 0.0 < w < 1.0 and ba == 1.0


def test_report_has_three_scores_and_two_paired_diffs(tmp_path, capsys):
    rng = np.random.RandomState(0)
    n = 60
    y = rng.randint(0, 3, n)
    ids = [str(i) for i in range(n)]
    groups = [f"p{i // 3}" for i in range(n)]
    a = make(tmp_path / "a.npz", ids, y, rng.dirichlet([1, 1, 1], n) * 0.5 + onehot(y) * 0.5, groups=groups)
    b = make(tmp_path / "b.npz", ids, y, rng.dirichlet([1, 1, 1], n), groups=groups)
    rc = be.main([a, b, "--names", "LoRA", "PB115", "--n_boot", "200"])
    out = capsys.readouterr().out
    assert rc == 0
    for word in ["LoRA alone", "PB115 alone", "blend", "blend - PB115", "blend - LoRA", "weights", "rows 60", "prompts 20",
                 "KEEP" if "KEEP" in out else "DROP", "per-label recall"]:
        assert word in out
    assert "optimistic" in out and "cross-fit" in out


def test_three_files_greedy_blend(tmp_path):
    y = np.array([0, 1, 2] * 6)
    good = onehot(y, hit=0.6)
    noisy = np.random.RandomState(1).dirichlet([1, 1, 1], len(y))
    w = be.greedy_weights(y, [noisy, good, noisy])
    assert np.isclose(w.sum(), 1) and w[1] >= 0.5


def test_names_must_match_file_count(tmp_path):
    y = [0, 1, 2]
    a = make(tmp_path / "a.npz", ["1", "2", "3"], y, onehot(y), groups=["p1", "p2", "p3"])
    with pytest.raises(SystemExit):
        be.main([a, a, "--names", "only-one"])


def test_report_prints_no_text_or_ids(tmp_path, capsys):
    y = [0, 1, 2]
    a = make(tmp_path / "a.npz", ["secret-id-1", "secret-id-2", "secret-id-3"], y, onehot(y), groups=["p1", "p2", "p3"])
    b = make(tmp_path / "b.npz", ["secret-id-1", "secret-id-2", "secret-id-3"], y, onehot(y))
    be.main([a, b, "--n_boot", "50"])
    assert "secret-id" not in capsys.readouterr().out
