import pytest

import train_lora as tl


def test_estimate_hours_and_abort():
    # 50 steps took 100 s -> 2 s/step; 20,000 steps left -> 40,000 s = 11.1 h
    est, abort = tl.probe_decision(elapsed_s=100, probe_steps=50, steps_left=20_000, limit_h=8)
    assert est == pytest.approx(100 / 50 * 20_000 / 3600) and abort
    est, abort = tl.probe_decision(elapsed_s=100, probe_steps=50, steps_left=10_000, limit_h=8)
    assert est == pytest.approx(5.5556, rel=1e-3) and not abort


def test_time_guard():
    assert not tl.out_of_time(elapsed_s=3600 * 10, budget_h=10.5)
    assert tl.out_of_time(elapsed_s=3600 * 10.6, budget_h=10.5)


def test_last_checkpoint(tmp_path):
    assert tl.last_checkpoint(tmp_path) is None
    for n in (200, 1000, 400):
        (tmp_path / f"checkpoint-{n}").mkdir()
    (tmp_path / "checkpoint-junk").mkdir()
    assert tl.last_checkpoint(tmp_path).endswith("checkpoint-1000")


def test_steps_per_epoch():
    assert tl.steps_per_epoch(n_rows=1000, batch=8, grad_accum=2, n_gpus=1) == 63
    assert tl.steps_per_epoch(n_rows=1000, batch=8, grad_accum=2, n_gpus=2) == 32


def test_softmax_rows():
    import numpy as np
    p = tl.softmax(np.array([[0.0, 0.0], [1000.0, 0.0]]))
    assert p[0].tolist() == [0.5, 0.5] and p[1, 0] == pytest.approx(1.0)
