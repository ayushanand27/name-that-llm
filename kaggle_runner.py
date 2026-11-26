"""Free pipeline: pull a Kaggle notebook, append your CV cells, push it on GPU, wait, download the log.
Runs on YOUR laptop with YOUR Kaggle login (kaggle CLI). Nothing here needs sharing a token with anyone.

Setup once:  pip install kaggle ; put kaggle.json in ~/.kaggle/ (Kaggle > Settings > API > Create token)
Usage:
  python kaggle_runner.py run --src mathleteutd/name-that-llm-pb115-coverage-sibling-odds \
      --slug ayush-cut256-cv --cells cut256_cv_cell.txt decode_cell.txt
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path


def code_cell(text):
    lines = text.splitlines(keepends=True)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def append_cells(nb, cell_texts):
    """Return a copy of notebook dict nb with the given code cells added at the end."""
    out = json.loads(json.dumps(nb))
    if "cells" not in out:
        raise ValueError("not an ipynb notebook")
    out["cells"] += [code_cell(t) for t in cell_texts]
    return out


def make_metadata(meta, owner, slug, title=None):
    """New kernel metadata from the pulled one: same inputs, GPU on, internet off, private."""
    m = dict(meta)
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError("slug must be lowercase letters, digits, dashes")
    m.update(id=f"{owner}/{slug}", title=title or slug.replace("-", " "), enable_gpu=True,
             enable_internet=False, is_private=True, code_file=f"{slug}.ipynb", language="python", kernel_type="notebook")
    return m


def parse_status(text):
    """'... has status "KernelWorkerStatus.COMPLETE"' -> 'complete'. Unknown -> 'unknown'."""
    m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)"?', text)
    return m.group(1).lower() if m else "unknown"


def sh(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def wait(ref, poll_s=300, max_s=12 * 3600, run=sh):
    t0 = time.time()
    while time.time() - t0 < max_s:
        st = parse_status(run(["kaggle", "kernels", "status", ref], check=False))
        print(f"[{time.strftime('%H:%M:%S')}] {ref}: {st}", flush=True)
        if st in ("complete", "error", "cancelacknowledged", "cancelled"):
            return st
        time.sleep(poll_s)
    return "timeout"


def cmd_run(a):
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "pull", a.src, "-p", str(work), "-m"])
    nb_file = next(work.glob("*.ipynb"), None)
    if nb_file is None:
        sys.exit("pulled notebook is not .ipynb (I am not sure this notebook can be pulled that way); open it in Kaggle instead")
    meta = json.loads((work / "kernel-metadata.json").read_text())
    owner = sh(["kaggle", "config", "view"]).split("username:")[-1].split()[0]
    new = append_cells(json.loads(nb_file.read_text()), [Path(c).read_text() for c in a.cells])
    md = make_metadata(meta, owner, a.slug)
    out = Path(a.work + "_push"); out.mkdir(exist_ok=True)
    (out / md["code_file"]).write_text(json.dumps(new))
    (out / "kernel-metadata.json").write_text(json.dumps(md, indent=1))
    print("inputs kept:", {k: md.get(k) for k in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources")})
    print(sh(["kaggle", "kernels", "push", "-p", str(out)]))
    if a.no_wait:
        print("pushed; Kaggle runs it on its own. Later run the 'fetch' action with the same slug.")
        return
    cmd_fetch(argparse.Namespace(ref=md["id"], work=a.work, wait=True))


def cmd_fetch(a):
    ref = a.ref if "/" in a.ref else sh(["kaggle", "config", "view"]).split("username:")[-1].split()[0] + "/" + a.ref
    st = wait(ref) if a.wait else parse_status(sh(["kaggle", "kernels", "status", ref], check=False))
    print("status:", st)
    if st in ("running", "queued", "unknown"):
        print("not finished yet; run fetch again later"); return
    res = Path(a.work + "_out"); res.mkdir(exist_ok=True)
    sh(["kaggle", "kernels", "output", ref, "-p", str(res)], check=False)
    for f in res.glob("*.log"):
        print("--- last lines of", f.name); print("".join(f.read_text().splitlines(True)[-60:]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest="c", required=True)
    r = s.add_parser("run"); r.add_argument("--src", required=True); r.add_argument("--slug", required=True)
    r.add_argument("--cells", nargs="+", required=True); r.add_argument("--work", default="kr_work"); r.add_argument("--no-wait", action="store_true"); r.set_defaults(f=cmd_run)
    f = s.add_parser("fetch"); f.add_argument("--ref", required=True); f.add_argument("--work", default="kr_work")
    f.add_argument("--wait", action="store_true"); f.set_defaults(f=cmd_fetch)
    a = p.parse_args(); a.f(a)
