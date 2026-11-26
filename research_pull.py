"""Read-only look at public Kaggle notebooks for the Name That LLM competition.

Runs in GitHub Actions (kaggle.com is reachable there) and only PRINTS to the job log.
It never runs, saves or commits anyone else's code.

  python research_pull.py list                 # top competition notebooks by public score
  python research_pull.py pull owner/slug ...  # metadata + every code cell of each notebook,
                                               # plus one level of utility-script sources
  python research_pull.py meta kind:owner/name  # licence + files of a dataset (ds:) or model (model:)
"""
import json
import os
import subprocess
import sys
import tempfile
import time

COMP = "name-that-llm"


def sh(args, tries=4):
    """Run a kaggle CLI command, retry on 429 / transient errors. Returns stdout text."""
    for i in range(tries):
        p = subprocess.run(args, capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0:
            return p.stdout
        if "429" in out or "Too Many" in out or "timed out" in out.lower():
            time.sleep(20 * (i + 1))
            continue
        return "ERROR rc=%d: %s" % (p.returncode, out.strip()[-500:])
    return "ERROR after retries: " + out.strip()[-500:]


def cells_text(path):
    """Code of a pulled kernel file: ipynb -> code cells joined; script -> as is."""
    if path.endswith(".ipynb"):
        nb = json.load(open(path, encoding="utf-8"))
        parts = []
        for i, c in enumerate(nb.get("cells", [])):
            src = c.get("source", "")
            src = "".join(src) if isinstance(src, list) else src
            if c.get("cell_type") == "markdown":
                parts.append("# [md %d]\n# " % i + src.replace("\n", "\n# "))
            elif c.get("cell_type") == "code":
                parts.append("# [code %d]\n%s" % (i, src))
        return "\n\n".join(parts)
    return open(path, encoding="utf-8", errors="replace").read()


def pull_one(ref, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if ref in seen:
        return
    seen.add(ref)
    d = tempfile.mkdtemp()
    res = sh(["kaggle", "kernels", "pull", ref, "-p", d, "-m"])
    print("\n" + "=" * 100 + "\n=== KERNEL %s (depth %d)\n" % (ref, depth) + "=" * 100)
    if res.startswith("ERROR"):
        print(res)
        return
    meta = {}
    mp = os.path.join(d, "kernel-metadata.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
        keep = {k: meta.get(k) for k in ("id", "title", "code_file", "language", "kernel_type",
                                          "is_private", "enable_gpu", "enable_internet",
                                          "dataset_sources", "competition_sources",
                                          "kernel_sources", "model_sources")}
        print("--- METADATA\n" + json.dumps(keep, indent=1))
    for f in sorted(os.listdir(d)):
        if f == "kernel-metadata.json":
            continue
        text = cells_text(os.path.join(d, f))
        print("--- FILE %s (%d chars)\n%s" % (f, len(text), text))
    if depth < 1:
        for k in meta.get("kernel_sources") or []:
            time.sleep(3)
            pull_one(k, depth + 1, seen)
    time.sleep(3)


def meta_one(spec):
    kind, ref = spec.split(":", 1)
    print("\n=== META %s" % spec)
    if kind == "ds":
        d = tempfile.mkdtemp()
        r = sh(["kaggle", "datasets", "metadata", ref, "-p", d])
        p = os.path.join(d, "dataset-metadata.json")
        if os.path.exists(p):
            m = json.load(open(p))
            print(json.dumps({k: m.get(k) for k in ("title", "id", "licenses", "subtitle")}, indent=1))
            desc = (m.get("description") or "")[:1500]
            print("description: " + desc.replace("\n", " "))
        else:
            print(r)
        print(sh(["kaggle", "datasets", "files", ref, "--page-size", "40"])[:3000])
    elif kind == "model":
        print(sh(["kaggle", "models", "get", ref])[:3000])
    time.sleep(3)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        for sort in ("scoreDescending", "voteCount"):
            print("\n=== LIST sort=%s" % sort)
            for page in (1, 2):
                print(sh(["kaggle", "kernels", "list", "--competition", COMP, "--sort-by", sort,
                          "--page-size", "100", "--page", str(page), "-v"]))
                time.sleep(3)
    elif cmd == "pull":
        seen = set()
        for r in rest:
            pull_one(r, 0, seen)
    elif cmd == "meta":
        for s in rest:
            meta_one(s)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
