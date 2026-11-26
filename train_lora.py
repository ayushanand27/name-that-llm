"""Qwen3-8B QLoRA (4-bit, fp16 for T4) 18-way classifier on the pool_data split.
Pure helpers at the top are tested on CPU; torch/transformers/peft load only inside main()."""
import argparse
import math
import os
import re
import time
from pathlib import Path

import numpy as np


def probe_decision(elapsed_s, probe_steps, steps_left, limit_h):
    """Seconds per step from the probe, times the steps still to go, in hours; abort if over the limit."""
    est_h = elapsed_s / probe_steps * steps_left / 3600
    return est_h, est_h > limit_h


def out_of_time(elapsed_s, budget_h):
    """True once the run has used its wall-clock budget, so it saves and stops before Kaggle's 12 h cut."""
    return elapsed_s > budget_h * 3600


def last_checkpoint(folder):
    """Path of the newest checkpoint-N folder (biggest N), or None, so a rerun resumes instead of starting over."""
    best = None
    for p in Path(folder).glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir() and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), str(p))
    return best[1] if best else None


def steps_per_epoch(n_rows, batch, grad_accum, n_gpus):
    """Optimizer steps in one pass over n_rows."""
    return math.ceil(n_rows / (batch * grad_accum * n_gpus))


def softmax(logits):
    """Turns raw scores into probabilities that sum to 1 per row (stable for big numbers)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def length_grouping(training_args_cls):
    """Batches passages of similar length together (less padding, faster). The argument name changed in
    transformers 5, so pick whichever this version has."""
    import inspect
    params = inspect.signature(training_args_cls.__init__).parameters
    if "train_sampling_strategy" in params:
        return {"train_sampling_strategy": "group_by_length"}
    return {"group_by_length": True} if "group_by_length" in params else {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--pool", default="")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", default="/kaggle/working/lora_run")
    ap.add_argument("--resume_from", default="", help="folder holding checkpoint-N from an earlier run's output")
    ap.add_argument("--run_name", default="run")
    ap.add_argument("--use_pool", type=int, default=1)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--kimi_high_only", type=int, default=0)
    ap.add_argument("--drop_flag_kb_proxy", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--probe_steps", type=int, default=50)
    ap.add_argument("--max_hours", type=float, default=8.0, help="abort after the probe if the estimate is above this")
    ap.add_argument("--time_budget_h", type=float, default=10.5, help="save and stop training after this much wall clock")
    ap.add_argument("--probe_only", type=int, default=0, help="stop right after the speed probe")
    ap.add_argument("--quant_4bit", type=int, default=1, help="0 only for the CPU smoke test")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    import pandas as pd
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments)
    import baseline_tfidf as bt
    import pool_data as pdm

    t_start = time.time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0
    say = (lambda *x: print(*x, flush=True)) if is_main else (lambda *x: None)

    cfg = pdm.Config(use_pool=bool(a.use_pool), cap=a.cap, kimi_high_only=bool(a.kimi_high_only),
                     drop_flag_kb_proxy=bool(a.drop_flag_kb_proxy), seed=a.seed)
    official = pd.read_csv(a.train_csv, dtype={"id": str, "prompt_id": str})
    pool = pd.read_parquet(a.pool) if cfg.use_pool else official.iloc[:0]
    train, hold, labels = pdm.build_dataset(official, pool, cfg)
    say(f"[{a.run_name}] train rows {len(train)} (official {int(train.is_official.sum())}, pool {int((~train.is_official).sum())}); "
        f"holdout rows {len(hold)}, prompts {hold.group.nunique()}; GPUs {world}")

    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class DS(torch.utils.data.Dataset):
        def __init__(self, df, with_labels=True):
            self.enc = tok(df.text.tolist(), truncation=True, max_length=a.max_len)
            self.y = df.label.values if with_labels else None
        def __len__(self):
            return len(self.enc["input_ids"])
        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            if self.y is not None:
                item["labels"] = int(self.y[i])
            return item

    ds_tr, ds_ho = DS(train), DS(hold)
    n_tok = sum(min(len(x), a.max_len) for x in ds_tr.enc["input_ids"])
    say(f"tokens per epoch {n_tok:,} (mean {n_tok / len(ds_tr):.0f} per passage)")

    gpu = torch.cuda.is_available()
    if a.quant_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model_path, num_labels=len(labels), quantization_config=bnb, torch_dtype=torch.float16,
            device_map={"": local_rank})
        model.config.pad_token_id = tok.pad_token_id
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = AutoModelForSequenceClassification.from_pretrained(a.model_path, num_labels=len(labels))
        model.config.pad_token_id = tok.pad_token_id
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    for n, p in model.named_parameters():        # trainable LoRA + head weights in fp32 so fp16 grad scaling works
        if p.requires_grad:
            p.data = p.data.float()

    total_steps = math.ceil(steps_per_epoch(len(ds_tr), a.batch, a.grad_accum, world) * a.epochs)
    state = {"aborted": False, "t_probe": None}

    class Guard(TrainerCallback):
        """Speed probe after `probe_steps` steps, then a wall-clock guard on every step."""
        def on_step_end(self, args, st, control, **kw):
            if st.global_step == 5:
                state["t_probe"] = time.time()
            if state["t_probe"] and st.global_step == 5 + a.probe_steps:
                el = time.time() - state["t_probe"]
                est, abort = probe_decision(el, a.probe_steps, total_steps - st.global_step, a.max_hours)
                tps = n_tok * a.epochs / total_steps * a.probe_steps / el
                say(f"speed probe: {el / a.probe_steps:.2f} s/step, ~{tps:,.0f} tokens/s, "
                    f"{total_steps - st.global_step} steps left, estimated {est:.2f} h more (limit {a.max_hours} h)")
                if abort or a.probe_only:
                    say("ABORT: estimate is over the limit" if abort else "probe only: stopping")
                    state["aborted"] = True
                    control.should_training_stop = True
                    control.should_save = False
            if out_of_time(time.time() - t_start, a.time_budget_h):
                say("time budget reached: saving a checkpoint and stopping training")
                control.should_save = True
                control.should_training_stop = True
            return control

    targs = TrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total_steps)), per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch * 2,
        gradient_accumulation_steps=a.grad_accum, fp16=gpu, logging_steps=25, save_strategy="steps",
        save_steps=a.save_steps, save_total_limit=2, report_to="none", seed=a.seed,
        dataloader_num_workers=2 if gpu else 0, ddp_find_unused_parameters=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False, **length_grouping(TrainingArguments))
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, data_collator=DataCollatorWithPadding(tok),
                      callbacks=[Guard()])
    resume = last_checkpoint(a.resume_from) if a.resume_from else last_checkpoint(out / "ckpt")
    if resume:
        say(f"resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    if state["aborted"]:
        say("stopped after the probe; no adapter or holdout probabilities saved")
        return 2
    if is_main:
        model.save_pretrained(str(out / "adapter"))
        tok.save_pretrained(str(out / "adapter"))
    logits = trainer.predict(ds_ho).predictions
    if is_main:
        probs = softmax(np.asarray(logits, dtype=np.float64))
        say("\n".join(bt.report_lines(hold, probs, labels)))
        bt.save_oof(out / f"oof_{a.run_name}.npz", hold, probs, labels, a.run_name)
        say(f"saved adapter and oof_{a.run_name}.npz in {out}; total {(time.time() - t_start) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
