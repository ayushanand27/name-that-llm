"""Which of our 18 labels have a free OpenRouter model that is clearly the SAME model (same family and version)?

Reads the public models list (no key). A free id maps to a label only when its name, without the vendor and
':free', equals the label exactly. Anything that only looks close (an -air, -lite, -preview, -instruct or a
dated snapshot) is shown as 'near' and never mapped, because a wrong label would poison training.
Free variants can still be quantised or a different snapshot; generate_data.py checks the served model per reply.
"""
import argparse
import json
import urllib.request

URL = "https://openrouter.ai/api/v1/models"
LABELS = ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3",
          "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3",
          "qwen3.8-max"]


def name_of(model_id):
    return model_id.split("/", 1)[-1].removesuffix(":free").strip().lower().replace("_", "-")


def match(model_id, labels=LABELS):
    """(label, 'same') for an exact name match; (None, 'near <label>') when it only starts like a label."""
    n = name_of(model_id)
    if n in labels:
        return n, "same"
    near = [l for l in labels if n.startswith(l)]
    return None, (f"near {max(near, key=len)}" if near else "")


def free_only(models):
    return [m for m in models if str(m.get("id", "")).endswith(":free")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="read a saved models list instead of the web (for tests)")
    a = ap.parse_args(argv)
    if a.json:
        data = json.load(open(a.json))
    else:
        with urllib.request.urlopen(URL, timeout=60) as r:
            data = json.loads(r.read())
    free = free_only(data["data"])
    print(f"{len(data['data'])} models listed, {len(free)} free")
    hits = {}
    for m in sorted(free, key=lambda m: m["id"]):
        lab, why = match(m["id"])
        print(f"  {m['id']:<55} -> {lab or '-':<22} {why}")
        if lab:
            hits.setdefault(lab, []).append(m["id"])
    print(f"labels with a same-model free id: {len(hits)} of {len(LABELS)}")
    for lab in LABELS:
        print(f"  {lab:<24} {', '.join(hits.get(lab, [])) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
