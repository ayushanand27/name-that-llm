"""End-to-end CPU run of train_lora.main on a tiny random Qwen3 and tiny fake data (no 4-bit, no GPU)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

import train_lora as tl
from test_pool_data import fake_official, fake_pool


def tiny_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForSequenceClassification
    words = ["[PAD]", "[UNK]"] + [f"m{i:02d}" for i in range(18)] + "official pool text style words".split() + [str(i) for i in range(400)]
    t = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]")
    cfg = Qwen3Config(vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_labels=18, pad_token_id=0)
    Qwen3ForSequenceClassification(cfg).save_pretrained(path)
    tok.save_pretrained(path)


def write_data(tmp_path):
    off = fake_official(n_prompts=10)
    off["text"] = [f"{m} style words {m} {i % 50}" for i, m in enumerate(off.model)]
    off.to_csv(tmp_path / "train.csv", index=False)
    fake_pool().to_parquet(tmp_path / "pool.parquet")


def test_main_end_to_end(tmp_path):
    tiny_model(tmp_path / "model"); write_data(tmp_path)
    args = ["--train_csv", str(tmp_path / "train.csv"), "--pool", str(tmp_path / "pool.parquet"),
            "--model_path", str(tmp_path / "model"), "--out_dir", str(tmp_path / "out"), "--work_root", str(tmp_path), "--run_name", "B",
            "--cap", "5", "--batch", "4", "--grad_accum", "1", "--save_steps", "20", "--probe_steps", "3",
            "--lr", "5e-3", "--epochs", "1", "--quant_4bit", "0"]
    assert tl.main(args) == 0
    d = np.load(tmp_path / "out" / "oof_B.npz")
    assert d["probs"].shape == (36, 18) and np.allclose(d["probs"].sum(1), 1, atol=1e-5)
    assert (tmp_path / "out" / "adapter" / "adapter_config.json").exists()
    assert tl.last_checkpoint(tmp_path / "out" / "ckpt") is not None


def test_main_aborts_when_estimate_too_high(tmp_path):
    tiny_model(tmp_path / "model"); write_data(tmp_path)
    args = ["--train_csv", str(tmp_path / "train.csv"), "--pool", "", "--use_pool", "0",
            "--model_path", str(tmp_path / "model"), "--out_dir", str(tmp_path / "out"), "--work_root", str(tmp_path), "--run_name", "A",
            "--batch", "2", "--grad_accum", "1", "--probe_steps", "2", "--max_hours", "0.0000001",
            "--epochs", "3", "--quant_4bit", "0"]
    assert tl.main(args) == 2
    assert not (tmp_path / "out" / "oof_A.npz").exists()


def base_args(tmp_path, run):
    return ["--train_csv", str(tmp_path / "train.csv"), "--pool", str(tmp_path / "pool.parquet"),
            "--model_path", str(tmp_path / "model"), "--out_dir", str(tmp_path / "out"), "--work_root", str(tmp_path), "--run_name", run,
            "--cap", "5", "--batch", "4", "--grad_accum", "1", "--save_steps", "20", "--probe_steps", "3",
            "--lr", "5e-3", "--epochs", "1", "--quant_4bit", "0"]


def test_final_retrain_has_no_holdout(tmp_path, capsys):
    tiny_model(tmp_path / "model"); write_data(tmp_path)
    assert tl.main(base_args(tmp_path, "final") + ["--holdout_frac", "0"]) == 0
    out = capsys.readouterr().out
    assert "holdout rows 0" in out and "official 180" in out and "no holdout (final retrain)" in out
    assert (tmp_path / "out" / "adapter" / "adapter_config.json").exists()
    assert not (tmp_path / "out" / "oof_final.npz").exists()


def test_probe_only_reports_speed_losses_and_scoring_time(tmp_path, capsys):
    tiny_model(tmp_path / "model"); write_data(tmp_path)
    assert tl.main(base_args(tmp_path, "probe") + ["--probe_only", "1"]) == 0
    out = capsys.readouterr().out
    assert "speed probe:" in out and "first losses" in out and "min for 4,000" in out
    assert not (tmp_path / "out" / "adapter").exists()


def test_zero_head_starts_at_equal_odds(tmp_path):
    from transformers import AutoModelForSequenceClassification
    tiny_model(tmp_path / "model")
    m = AutoModelForSequenceClassification.from_pretrained(tmp_path / "model")
    tl.zero_head(m)
    logits = m(input_ids=torch.tensor([[3, 4, 5]])).logits
    assert torch.all(logits == 0)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    assert abs(loss.item() - np.log(18)) < 1e-6


def test_logged_loss_is_not_scaled_by_grad_accum(tmp_path, capsys):
    tiny_model(tmp_path / "model"); write_data(tmp_path)
    args = base_args(tmp_path, "x") + ["--grad_accum", "2", "--batch", "2", "--lr", "1e-12",
                                       "--probe_only", "1", "--probe_steps", "10"]
    assert tl.main(args) == 2                   # flat loss at lr 1e-12, so the not-falling guard stops it
    out = capsys.readouterr().out
    assert "ABORT: loss not falling" in out
    line = [l for l in out.splitlines() if l.startswith("first losses")][0]
    first = float(line.split("[")[1].split(",")[0])
    assert abs(first - np.log(18)) < 0.01      # zero head: every class equal, loss = ln 18, not 2 x ln 18


def learnable_data(tmp_path, n_prompts=10):
    """Official texts carry their label word, so a working model must score well above chance on the holdout."""
    off = fake_official(n_prompts=n_prompts)
    off["text"] = [f"{m} style words {m} " + " ".join(["words"] * (i % 7)) for i, m in enumerate(off.model)]
    off.to_csv(tmp_path / "train.csv", index=False)
    fake_pool().to_parquet(tmp_path / "pool.parquet")
    return off


def test_holdout_probabilities_line_up_with_their_ids(tmp_path):
    # transformers 5 length-groups the predict loader too, which shuffled run B's probabilities (BA 0.046)
    tiny_model(tmp_path / "model"); learnable_data(tmp_path)
    args = base_args(tmp_path, "B") + ["--use_pool", "0", "--epochs", "8", "--lr", "1e-2", "--lora_r", "8"]
    assert tl.main(args) == 0
    d = np.load(tmp_path / "out" / "oof_B.npz")
    assert (d["probs"].argmax(1) == d["y"]).mean() > 0.6


def test_pool_only_run_scores_all_official_rows_and_the_test_set(tmp_path):
    tiny_model(tmp_path / "model"); off = learnable_data(tmp_path)
    test = off.iloc[:7][["id", "text"]].assign(id=lambda d: "t" + d.id)
    test.to_csv(tmp_path / "test.csv", index=False)
    args = base_args(tmp_path, "P") + ["--official_in_train", "0", "--test_csv", str(tmp_path / "test.csv")]
    assert tl.main(args) == 0
    d = np.load(tmp_path / "out" / "oof_P.npz")
    assert d["probs"].shape == (len(off), 18) and sorted(d["ids"]) == sorted(off.id)
    t = np.load(tmp_path / "out" / "test_P.npz")
    assert t["probs"].shape == (7, 18) and list(t["ids"]) == list(test.id) and len(t["labels"]) == 18


def test_rows_agree_and_load_test(tmp_path):
    assert tl.rows_agree(np.eye(3), np.eye(3)) == 1.0
    assert tl.rows_agree(np.eye(3), np.eye(3)[[1, 2, 0]]) == 0.0
    import pandas as pd
    pd.DataFrame({"id": ["a"], "text": ["x"], "extra": [1]}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"id": ["a"], "body": ["x"]}).to_csv(tmp_path / "bad.csv", index=False)
    assert list(tl.load_test(tmp_path / "t.csv").columns) == ["id", "text"]
    msgs = []
    assert tl.load_test(tmp_path / "bad.csv", msgs.append) is None and "skipping test scoring" in msgs[0]
