import numpy as np
import pandas as pd

import pool_gem37 as pg


def pool():
    return pd.DataFrame({"model": ["gemini-3.7-flash"] * 5 + ["gemini-3.8-flash"] * 2,
                         "gen_date": ["2026-09-01", "2026-10-08", "2026-10-20", None, "bad", "2026-11-01", "2026-09-01"],
                         "gen_served_model": ["g37", "g38", "g38", None, "g37", "g38", "g38"]})


def test_date_counts():
    assert pg.date_counts(pool()) == {"before 8 Oct": 1, "8 Oct or later": 2, "no usable date": 2}


def test_groups_for_yardstick():
    g = pg.yardstick_groups(pool())
    assert list(g) == ["3.7 before", "3.7 after", "3.7 after", None, None, "3.8", "3.8"]


def test_share_table_rows_sum_to_one():
    groups = pd.Series(["a", "a", "b"])
    pred = np.array(["x", "y", "x"])
    t = pg.share_table(groups, pred, ["x", "y"])
    assert np.allclose(t.sum(1), 1) and t.loc["a", "x"] == 0.5 and t.loc["b", "x"] == 1.0
