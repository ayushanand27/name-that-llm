import numpy as np
import oof_report as r


def fake_npz(tmp_path):
    labels = np.array(["a", "b", "c"])
    y = np.array([0] * 30 + [1] * 30 + [2] * 5)
    probs = np.eye(3)[y].astype(np.float32)
    probs[:12] = np.eye(3)[1]                      # 12 of label a predicted as b
    path = tmp_path / "oof_B.npz"
    np.savez(path, ids=np.arange(len(y)).astype("U"), groups=np.array(["g%d" % (i % 10) for i in range(len(y))]),
             y=y, probs=probs, labels=labels, run=np.asarray("B"))
    return path


def test_wilson_interval():
    lo, hi = r.wilson(15, 30)
    assert 0.31 < lo < 0.34 and 0.66 < hi < 0.69
    assert r.wilson(0, 0) == (0.0, 1.0)


def test_label_table_sorted_weakest_first_with_counts(tmp_path):
    rows = r.label_table(np.load(fake_npz(tmp_path)))
    assert [x["label"] for x in rows] == ["a", "b", "c"]
    assert rows[0]["n"] == 30 and abs(rows[0]["recall"] - 0.6) < 1e-9
    assert rows[2]["n"] == 5 and rows[2]["few_rows"] and not rows[0]["few_rows"]


def test_report_lines_show_confusions_and_flags(tmp_path):
    text = "\n".join(r.report_lines(np.load(fake_npz(tmp_path))))
    assert "a -> b: 12" in text and "few rows" in text and "balanced accuracy" in text
