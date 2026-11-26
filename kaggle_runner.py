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


WORK = "kr_work"            # fixed local work folder (it is in .gitignore as *_work/)
NB_FILE = "notebook.ipynb"   # fixed name, so nothing a user types ends up in a file path


def check_slug(slug):
    """Stop unless the slug looks like a Kaggle notebook name (lowercase letters, digits, dashes), so it
    can't carry a path like ../x into the file we write."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,49}", slug or ""):
        raise SystemExit(f"bad slug {slug!r}: use 3-50 lowercase letters, digits or dashes")
    return slug


def make_metadata(meta, owner, slug, title=None):
    """New kernel metadata from the pulled one: same inputs, GPU on, internet off, private."""
    m = dict(meta)
    m.pop("id_no", None)  # pulled id_no is the source kernel's number; keeping it makes push try to edit their kernel (403)
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise ValueError("slug must be lowercase letters, digits, dashes")
    m.update(id=f"{owner}/{slug}", title=title or slug.replace("-", " "), enable_gpu=True,
             enable_internet=False, is_private=True, code_file=NB_FILE, language="python", kernel_type="notebook")
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
    work = Path(WORK); work.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "pull", a.src, "-p", str(work), "-m"])
    nb_file = next(work.glob("*.ipynb"), None)
    if nb_file is None:
        sys.exit("pulled notebook is not .ipynb (I am not sure this notebook can be pulled that way); open it in Kaggle instead")
    meta = json.loads((work / "kernel-metadata.json").read_text())
    owner = kaggle_owner()
    new = append_cells(json.loads(nb_file.read_text()), [Path(c).read_text() for c in a.cells])
    md = make_metadata(meta, owner, a.slug)
    out = Path(WORK + "_push"); out.mkdir(exist_ok=True)
    (out / NB_FILE).write_text(json.dumps(new))
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
    cmd_fetch(argparse.Namespace(ref=md["id"], wait=True, tail=80))


FETCH_PATTERN = r"(\.log|(oof|test)_[^/]*\.npz)$"   # log and probabilities only; adapters stay on Kaggle


def cmd_fetch(a):
    ref = a.ref if "/" in a.ref else kaggle_owner() + "/" + a.ref
    st = wait(ref) if a.wait else parse_status(sh(["kaggle", "kernels", "status", ref], check=False))
    print("status:", st)
    if st in ("running", "queued", "unknown"):
        print("not finished yet; run fetch again later"); return
    res = Path(WORK + "_out"); res.mkdir(exist_ok=True)
    # only the log and holdout probabilities: adapters and checkpoints are hundreds of MB and stay on Kaggle
    sh(["kaggle", "kernels", "output", ref, "-p", str(res), "--file-pattern", FETCH_PATTERN], check=False)
    for f in res.glob("*.log"):
        print("--- last printed lines of", f.name); print("\n".join(stdout_lines(f.read_text())[-a.tail:]))
        if st == "error":
            print("--- last error lines"); print("\n".join(error_lines(f.read_text())[-15:]))
    for f in sorted(res.rglob("oof_*.npz")):
        print("--- saved", f.relative_to(res))
        try:
            import numpy as np, oof_report
            print("\n".join(oof_report.report_lines(np.load(f))))
        except ImportError:
            print("(install numpy to see the per-label report)")
    print(sh(["kaggle", "quota"], check=False))


POOL_FILES = ["pool_data.py", "metrics.py", "baseline_tfidf.py", "train_lora.py"]
POOL_PIP = "transformers==5.24.1 peft==0.22.0 bitsandbytes==0.50.2 accelerate==1.16.0"
POOL_RUNS = {  # A = official only, B = official + capped pool, P = pool only, final = no holdout
    # B and probe pin the recipe run B was trained with (proxy rows kept, gemini-3.7 rows kept). B is diagnostics only.
    # Ayush 26 Nov: every retrain drops the after-ruling proxy rows; the gemini-3.7 date switch stays off.
    "A": "--use_pool 0 --epochs 3",
    "B": "--use_pool 1 --cap 1500 --kimi_high_only 0 --drop_flag_kb_proxy 0 --drop_gem37_after_oct8 0 --epochs 1",
    "P": "--use_pool 1 --official_in_train 0 --cap 1500 --kimi_high_only 0 --drop_flag_kb_proxy 1 --drop_gem37_after_oct8 0 --epochs 1",
    "final": "--use_pool 1 --cap 1500 --kimi_high_only 0 --drop_flag_kb_proxy 1 --drop_gem37_after_oct8 0 --epochs 1 --holdout_frac 0",
    "probe": "--use_pool 1 --cap 1500 --kimi_high_only 0 --drop_flag_kb_proxy 0 --drop_gem37_after_oct8 0 --epochs 1 --probe_only 1",
}
POOL_FIND = """import glob, os, subprocess, sys
def find(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True))
    assert hits, 'not found: ' + pattern
    return hits[0]
TRAIN = find('train.csv')
POOL = find('ntl_open_pool.parquet')
TEST = find('test.csv')
MODEL = os.path.dirname(find('qwen*/**/config.json'))
print('train', TRAIN, '| pool', POOL, '| model', MODEL, flush=True)
subprocess.run(['nvidia-smi', '-L'])
"""


def pool_run_args(run):
    """Command-line flags for run A, B, final or the speed probe (KeyError for anything else)."""
    return POOL_RUNS[run]


def build_pool_notebook(files, run, n_gpus):
    """Notebook: install pinned packages, write each repo .py file with %%writefile, then launch train_lora.py
    with torchrun on n_gpus GPUs for the chosen run."""
    cells = [code_cell(f"!pip install -q {POOL_PIP}")]
    cells += [code_cell(f"%%writefile {Path(f).name}\n" + Path(f).read_text()) for f in files]
    launch = (POOL_FIND +
              f"cmd = ['torchrun', '--nproc_per_node', '{n_gpus}', 'train_lora.py', '--train_csv', TRAIN, '--pool', POOL,\n"
              f"       '--model_path', MODEL, '--test_csv', TEST, '--run_name', '{run}'] + '{pool_run_args(run)}'.split()\n"
              "print(' '.join(cmd), flush=True)\n"
              "rc = subprocess.run(cmd).returncode\n"
              "print('exit code', rc, flush=True)\n"
              "assert rc == 0, 'training failed or stopped (look for an ABORT line above)'\n")
    cells.append(code_cell(launch))
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def pool_metadata(owner, slug, model="qwen-lm/qwen-3/Transformers/8b-base/1"):
    """Private GPU notebook with internet (for the pip install), the Open Pool, the competition data and Qwen3-8B."""
    return {"id": f"{owner}/{slug}", "title": slug.replace("-", " "), "code_file": NB_FILE, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["crtproduction/name-that-llm-open-pool"],
            "competition_sources": ["name-that-llm"], "model_sources": [model], "kernel_sources": []}


def error_lines(log_text):
    """Lines from the stderr part of a Kaggle log that look like a traceback or an error, to see why a run failed."""
    try:
        chunks = json.loads(log_text)
    except ValueError:
        return []
    text = "".join(c.get("data", "") for c in chunks if c.get("stream_name") == "stderr")
    keep = ("Traceback", "Error", "error:", "File \"", "raise ", "Killed", "OOM")
    return [ln for ln in text.splitlines() if any(k in ln for k in keep)]


def stdout_lines(log_text):
    """Kaggle's .log is a JSON list of stdout/stderr chunks; return only the stdout lines (plain text passes through)."""
    try:
        chunks = json.loads(log_text)
    except ValueError:
        return log_text.splitlines()
    text = "".join(c.get("data", "") for c in chunks if c.get("stream_name") == "stdout")
    return text.splitlines()


def cmd_pool_train(a):
    owner = kaggle_owner()
    out = Path(WORK + "_pool"); out.mkdir(exist_ok=True)
    md = pool_metadata(owner, check_slug(a.slug))
    (out / NB_FILE).write_text(json.dumps(build_pool_notebook(POOL_FILES, a.run, a.gpus)))
    (out / "kernel-metadata.json").write_text(json.dumps(md, indent=1))
    print("pushing kernel id:", md["id"], "run", a.run, "flags:", pool_run_args(a.run), flush=True)
    print(sh(["kaggle", "kernels", "push", "-p", str(out)]))
    print(sh(["kaggle", "quota"], check=False))


def cmd_probe(a):
    """Push a one-line private kernel without and then with GPU, report which the account may do, delete both."""
    owner = kaggle_owner()
    for gpu in (False, True):
        slug = "ci-probe-gpu" if gpu else "ci-probe-cpu"
        d = Path(WORK + "_" + slug); d.mkdir(exist_ok=True)
        (d / "probe.ipynb").write_text(json.dumps({"cells": [code_cell("print('ok')")], "metadata": {},
                                                    "nbformat": 4, "nbformat_minor": 5}))
        md = {"id": f"{owner}/{slug}", "title": slug.replace("-", " "), "code_file": "probe.ipynb",
              "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu,
              "enable_internet": False, "competition_sources": ["name-that-llm"]}
        (d / "kernel-metadata.json").write_text(json.dumps(md))
        try:
            sh(["kaggle", "kernels", "push", "-p", str(d)])
            print(f"probe {'GPU' if gpu else 'CPU'} push: OK", flush=True)
            sh(["kaggle", "kernels", "delete", "-y", md["id"]], check=False)
        except RuntimeError as e:
            why = [l for l in str(e).splitlines() if l.strip()][1:3]
            print(f"probe {'GPU' if gpu else 'CPU'} push: REFUSED ({' | '.join(why)[:200]})", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest="c", required=True)
    r = s.add_parser("run"); r.add_argument("--src", required=True); r.add_argument("--slug", required=True)
    r.add_argument("--cells", nargs="+", required=True); r.add_argument("--no-wait", action="store_true"); r.set_defaults(f=cmd_run)
    f = s.add_parser("fetch"); f.add_argument("--ref", required=True)
    f.add_argument("--tail", type=int, default=80)
    f.add_argument("--wait", action="store_true"); f.set_defaults(f=cmd_fetch)
    pt = s.add_parser("pool_train"); pt.add_argument("--slug", required=True); pt.add_argument("--run", required=True, choices=sorted(POOL_RUNS))
    pt.add_argument("--gpus", type=int, default=2); pt.set_defaults(f=cmd_pool_train)
    pr = s.add_parser("probe"); pr.set_defaults(f=cmd_probe)
    a = p.parse_args(); a.f(a)
