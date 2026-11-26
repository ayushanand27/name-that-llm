"""Generate extra passages for chosen labels from OpenRouter's free models only ($0), within the daily free limit.

Dry run by default: prints how many calls it would make, and makes no network calls at all.
Add --execute to really call OpenRouter. Only model ids ending in ":free" are accepted, so no call can cost money.
The API key is read only from the environment variable named in the config ("key_env", e.g.
OPENROUTER_API_KEY); it is never printed, logged or written. With --execute:
- the daily free budget is read from GET {base_url}/key (free_model_daily_requests.remaining) and the run
  stops when it reaches --keep (default 0); it is re-read every 10 calls;
- calls are paced to at most 15 per minute; 429 and 5xx are retried with growing waits;
- each reply is appended to the JSONL as it arrives, so a rerun (the next day) skips the
  (prompt_id, label, sample_idx) rows already done;
- the model that actually served the reply is logged, and a row whose served model is not the label's model
  is kept but marked ok=false (reject="served by another model"), so it never enters training.

Config (JSON):
{"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
 "labels": {"<one of the 18 labels>": {"model": "<vendor>/<model>:free", "max_tokens": 600}}}
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

MIN_WORDS, MAX_WORDS = 40, 250
PER_MINUTE = 15
RECHECK_EVERY = 10
RETRY_CODES = {429, 500, 502, 503, 504}


def word_ok(text):
    return MIN_WORDS <= len(str(text).split()) <= MAX_WORDS


def same_model(requested, served):
    """Served id must be the requested id, give or take the ':free' suffix."""
    if not served:
        return False
    strip = lambda m: m.removesuffix(":free").strip().lower()
    return strip(requested) == strip(served)


def load_prompts(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"prompt_id", "prompt"} <= set(rows[0]):
        raise SystemExit(f"{path}: needs columns prompt_id, prompt")
    return rows


def plan(prompts, labels, n):
    return [(p["prompt_id"], p["prompt"], lab, k) for p in prompts for lab in labels for k in range(n)]


def read_done(out):
    if not Path(out).exists():
        return set()
    done = set()
    for line in Path(out).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["prompt_id"], r["label"], r["sample_idx"]))
    return done


def _request(url, key, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    return req


def http_json(url, key, body=None, retries=5, timeout=120):
    """GET (body None) or POST; retries 429/5xx with max(Retry-After, 2*2^attempt) seconds. Never shows the key."""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(_request(url, key, body), timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in RETRY_CODES and attempt < retries:
                wait = max(float(e.headers.get("Retry-After") or 0), 2.0 * 2 ** attempt)
                print(f"  HTTP {e.code}, waiting {wait:.0f}s (try {attempt + 1}/{retries})", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}") from None
    raise RuntimeError("unreachable")


def daily_remaining(base_url, key):
    d = http_json(base_url.rstrip("/") + "/key", key)
    return int(d["data"]["free_model_daily_requests"]["remaining"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per prompt and label")
    ap.add_argument("--labels", nargs="*", help="subset of config labels (default: all)")
    ap.add_argument("--keep", type=int, default=0, help="leave this many free requests unused today")
    ap.add_argument("--execute", action="store_true", help="really call OpenRouter (default is a dry run)")
    a = ap.parse_args(argv)
    conf = json.loads(Path(a.config).read_text())
    labels = a.labels or list(conf["labels"])
    for lab in labels:
        if not conf["labels"][lab]["model"].endswith(":free"):
            raise SystemExit(f"{lab}: model must end in ':free' (money is $0, paid models are refused)")
    todo = [t for t in plan(load_prompts(a.prompts), labels, a.n) if (t[0], t[2], t[3]) not in read_done(a.out)]
    print(f"{len(todo)} calls left to make ({len(labels)} labels), free models only, cost $0")
    if not a.execute:
        print("dry run: no network calls made. Add --execute to run (at most the day's free requests, 15 per minute).")
        return 0
    env = conf.get("key_env", "OPENROUTER_API_KEY")
    key = os.environ.get(env)
    if not key:
        raise SystemExit(f"API key missing: set the environment variable {env}")
    base = conf.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")
    left = daily_remaining(base, key)
    print(f"free requests left today: {left} (keeping {a.keep})", flush=True)
    made, last = 0, None
    with open(a.out, "a", encoding="utf-8") as f:
        for pid, prompt, lab, k in todo:
            if made and made % RECHECK_EVERY == 0:
                left = daily_remaining(base, key)
            if left <= a.keep:
                print(f"daily free budget reached after {made} calls; run again tomorrow to resume", flush=True)
                break
            if last is not None:
                time.sleep(max(0.0, 60.0 / PER_MINUTE - (time.monotonic() - last)))
            cfg = conf["labels"][lab]
            body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": cfg.get("max_tokens", 600)}
            try:
                last = time.monotonic()
                r = http_json(base + "/chat/completions", key, body)
            except RuntimeError as e:
                print(f"  {pid} {lab} #{k}: {e}; stopping", flush=True)
                break
            made += 1; left -= 1
            text = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            served = r.get("model")
            reject = None if same_model(cfg["model"], served) else "served by another model"
            if reject is None and not word_ok(text):
                reject = f"words outside {MIN_WORDS}-{MAX_WORDS}"
            u = r.get("usage") or {}
            row = {"prompt_id": pid, "label": lab, "sample_idx": k, "model": cfg["model"], "served_model": served,
                   "provider": r.get("provider"), "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                   "words": len(text.split()), "ok": reject is None, "reject": reject, "text": text}
            f.write(json.dumps(row) + "\n"); f.flush()
    print(f"made {made} calls this run", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
