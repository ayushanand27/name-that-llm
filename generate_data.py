"""Generate extra passages for chosen labels from OpenAI-compatible chat APIs, with a hard spend cap.

Dry run by default: prints how many calls and what they should cost, and makes no network calls.
Add --execute to really call the APIs. The API key is read only from the environment variable named in the
config ("key_env"); it is never printed, logged or written. Each passage is appended to a JSONL as it arrives,
so a rerun skips the (prompt_id, label, sample_idx) rows that are already done.

Config (JSON):
{"key_env": "MY_KEY_ENV",                              # default env var for every label
 "labels": {"kimi-k3": {"base_url": "https://.../v1", "model": "...", "reasoning": "high",
                        "price_in_per_mtok": 0.6, "price_out_per_mtok": 2.5,   # USD per million tokens
                        "est_in_tokens": 120, "est_out_tokens": 600,          # for the dry-run estimate
                        "key_env": "OPTIONAL_PER_LABEL_ENV", "max_tokens": 2000}}}
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MIN_WORDS, MAX_WORDS = 40, 250          # the competition's passage length
RETRY_CODES = {429, 500, 502, 503, 504}


def word_ok(text):
    """True if the passage has 40-250 words, like the competition passages."""
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS


def cost_of(usage, cfg):
    """USD for one call, from the token counts the API returned and the config's per-million prices."""
    return (usage.get("prompt_tokens", 0) * cfg["price_in_per_mtok"]
            + usage.get("completion_tokens", 0) * cfg["price_out_per_mtok"]) / 1e6


def estimate_call(cfg):
    """Expected USD for one call before any response is seen, from est_in_tokens / est_out_tokens."""
    return cost_of({"prompt_tokens": cfg.get("est_in_tokens", 150),
                    "completion_tokens": cfg.get("est_out_tokens", 600)}, cfg)


def load_prompts(path):
    """[(prompt_id, prompt)] from a CSV with columns prompt_id, prompt."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"prompt_id", "prompt"} <= set(rows[0]):
        raise SystemExit(f"{path} needs columns prompt_id and prompt")
    return [(r["prompt_id"], r["prompt"]) for r in rows]


def plan(prompts, labels, n):
    """Every (prompt_id, prompt, label, sample_idx) wanted: n samples per prompt per label."""
    return [(pid, text, lab, i) for lab in labels for pid, text in prompts for i in range(n)]


def read_done(out_path):
    """Keys already in the output file, and the USD already spent on them."""
    done, spent = set(), 0.0
    if Path(out_path).exists():
        for line in Path(out_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["prompt_id"], r["label"], r["sample_idx"]))
                spent += r.get("cost_usd", 0.0)
    return done, spent


def scrub(msg, key):
    """Error text with the key masked, in case a server echoes it back."""
    return str(msg).replace(key, "***") if key else str(msg)


def call_chat(cfg, key, prompt, retries=5, timeout=120):
    """One chat completion. Retries 429 and 5xx with growing waits (honours Retry-After). Returns (text, usage)."""
    body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}]}
    if cfg.get("reasoning"):
        body["reasoning_effort"] = cfg["reasoning"]
    if cfg.get("max_tokens"):
        body["max_tokens"] = cfg["max_tokens"]
    req = urllib.request.Request(cfg["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"] or "", out.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or attempt == retries:
                raise RuntimeError(f"HTTP {e.code} from {cfg['base_url']}") from None
            wait = max(float(e.headers.get("Retry-After") or 0), 2.0 * 2 ** attempt)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise RuntimeError(scrub(f"network error: {e}", key)) from None
            wait = 2.0 * 2 ** attempt
        time.sleep(wait)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=1, help="samples per prompt")
    ap.add_argument("--labels", nargs="*", help="subset of config labels (default: all)")
    ap.add_argument("--max_usd", type=float, default=5.0, help="hard cap on total spend in the output file")
    ap.add_argument("--execute", action="store_true", help="really call the APIs (default is a dry run)")
    a = ap.parse_args(argv)

    conf = json.loads(Path(a.config).read_text())
    labels = a.labels or list(conf["labels"])
    unknown = set(labels) - set(conf["labels"])
    if unknown:
        raise SystemExit(f"labels not in config: {sorted(unknown)}")
    jobs = plan(load_prompts(a.prompts), labels, a.n)
    done, spent = read_done(a.out)
    todo = [j for j in jobs if (j[0], j[2], j[3]) not in done]
    est = sum(estimate_call(conf["labels"][j[2]]) for j in todo)
    print(f"{len(todo)} calls to make ({len(jobs) - len(todo)} already done), estimated ${est:.4f}; "
          f"already spent ${spent:.4f}; cap ${a.max_usd:.2f}")
    if not a.execute:
        print("dry run: no calls made. Add --execute to run.")
        return 0

    keys = {}
    for lab in labels:
        env = conf["labels"][lab].get("key_env") or conf.get("key_env")
        if not env or not os.environ.get(env):
            raise SystemExit(f"API key missing: set the environment variable {env or '(key_env not in config)'}")
        keys[lab] = os.environ[env]

    made = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for pid, prompt, lab, i in todo:
            cfg = conf["labels"][lab]
            if spent + estimate_call(cfg) > a.max_usd:
                print(f"spend cap: stopping at ${spent:.4f}; the next call (about ${estimate_call(cfg):.4f}) "
                      f"would pass ${a.max_usd:.2f}")
                break
            try:
                text, usage = call_chat(cfg, keys[lab], prompt)
            except RuntimeError as e:
                print(f"skipped {pid}/{lab}/{i}: {scrub(e, keys[lab])}", file=sys.stderr)
                continue
            cost = cost_of(usage, cfg)
            spent += cost
            row = {"prompt_id": pid, "label": lab, "sample_idx": i, "model": cfg["model"],
                   "reasoning": cfg.get("reasoning"), "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                   "cost_usd": round(cost, 6), "words": len(text.split()), "ok": word_ok(text), "text": text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            made += 1
    print(f"made {made} calls; total spent in {a.out}: ${spent:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
