"""Free pipeline: pull a Kaggle notebook, append your CV cells, push it on GPU, wait, download the log.
Runs on YOUR laptop with YOUR Kaggle login (kaggle CLI). Nothing here needs sharing a token with anyone.

Setup once:  pip install kaggle ; put kaggle.json in ~/.kaggle/ (Kaggle > Settings > API > Create token)
Usage:
  python kaggle_runner.py run --src mathleteutd/name-that-llm-pb115-coverage-sibling-odds \
      --slug ayush-cut256-cv --cells cut256_cv_cell.txt decode_cell.txt
"""
import argparse, json, os, re, subprocess, sys, time
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


def kaggle_owner(run=sh, env=os.environ):
    """Kaggle username. The CLI's own login (the token's real owner) wins, since a kernel id with any other
    owner is refused with 403. KAGGLE_USERNAME is only a fallback when the CLI can't say."""
    want = (env.get("KAGGLE_USERNAME") or "").strip()
    try:
        out = run(["kaggle", "config", "view"])
        cli = out.split("username:")[-1].split()[0] if "username:" in out else ""
    except Exception:
        cli = ""
    if cli and cli.lower() != "none":
        if want and want != cli:
            print(f"note: KAGGLE_USERNAME differs from the logged-in Kaggle user; using {cli}", flush=True)
        return cli
    if want:
        return want
    raise RuntimeError("could not tell the Kaggle username; check the KAGGLE_API_TOKEN secret")


SOURCE_CHECKS = {
    "competition_sources": ["kaggle", "competitions", "files"],
    "dataset_sources": ["kaggle", "datasets", "files"],
    "kernel_sources": ["kaggle", "kernels", "files"],
    "model_sources": ["kaggle", "models", "instances", "versions", "files"],
}


def check_sources(md, run=sh):
    """Read-only probe of every input the kernel uses; returns the ones this account can't read."""
    bad = []
    for key, cmd in SOURCE_CHECKS.items():
        for ref in md.get(key) or []:
            try:
                run(cmd + [ref])
            except Exception as e:
                last = [l for l in str(e).splitlines() if l.strip()]
                bad.append((key, ref, " | ".join(last[1:3])[:200]))
    return bad


def cmd_run(a):
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "pull", a.src, "-p", str(work), "-m"])
    nb_file = next(work.glob("*.ipynb"), None)
    if nb_file is None:
        sys.exit("pulled notebook is not .ipynb (I am not sure this notebook can be pulled that way); open it in Kaggle instead")
    meta = json.loads((work / "kernel-metadata.json").read_text())
    owner = kaggle_owner()
    new = append_cells(json.loads(nb_file.read_text()), [Path(c).read_text() for c in a.cells])
    md = make_metadata(meta, owner, a.slug)
    out = Path(a.work + "_push"); out.mkdir(exist_ok=True)
    (out / md["code_file"]).write_text(json.dumps(new))
    (out / "kernel-metadata.json").write_text(json.dumps(md, indent=1))
    print("inputs kept:", {k: md.get(k) for k in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources")})
    print("pushing kernel id:", md["id"], flush=True)
    try:
        print(sh(["kaggle", "kernels", "push", "-p", str(out)]))
    except RuntimeError:
        print("push refused; checking which inputs this account can't read...", flush=True)
        bad = check_sources(md)
        for key, ref, why in bad:
            print(f"  CAN'T READ {key}: {ref}  ({why})", flush=True)
        if not bad:
            print("  every input is readable, so the refusal is about the account or token (e.g. token scope, phone verification)")
        raise
    if a.no_wait:
        print("pushed; Kaggle runs it on its own. Later run the 'fetch' action with the same slug.")
        return
    cmd_fetch(argparse.Namespace(ref=md["id"], work=a.work, wait=True))


def cmd_fetch(a):
    ref = a.ref if "/" in a.ref else kaggle_owner() + "/" + a.ref
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
