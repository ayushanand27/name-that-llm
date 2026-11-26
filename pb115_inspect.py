"""Find where Rohan's PB115 notebook gets its 115 head features, so head_cv.py can rebuild the head on CPU.

Runs in GitHub Actions (Kaggle token from secrets). Pulls the public notebook, saves its code cells to
an artifact folder, lists the files of the feature datasets (name and size only), and copies the small
ones into the artifact. Prints file names and sizes only, never passage text.
"""
import argparse
import csv
import io
import json
import os
import re
import subprocess
from pathlib import Path

SRC = "mathleteutd/name-that-llm-pb115-coverage-sibling-odds"
DATASETS = ["lumibee/ntl-prefixb-full-115"]


def parse_files(text):
    """Kaggle 'datasets files --csv' output -> [(name, size_bytes)]."""
    rows = list(csv.DictReader(io.StringIO(text)))
    return [(r["name"], int(r["size"])) for r in rows if r.get("name")]


def small(files, cap_mb=50):
    return [n for n, s in files if s <= cap_mb * 2**20]


def code_cells(nb):
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in nb["cells"] if c["cell_type"] == "code"]


KEYS = (r"spec|features_|115|oof|out.of.fold|fold|GroupKFold|cross_val|LogisticRegression|lgb|LGBM|"
        r"fit\(|predict_proba|in.sample|train_feat|_features\.csv|paired_token|parquet|np\.load|read_csv")


def matches(cells, pattern=KEYS, width=160):
    """(cell, line number, line) for code lines that touch features, folds or the head; long lines cut."""
    rx = re.compile(pattern)
    return [(i, j + 1, line.strip()[:width]) for i, c in enumerate(cells)
            for j, line in enumerate(c.splitlines()) if rx.search(line)]


def describe(path):
    """Shape and column names of a feature file (numbers and names only, no text cells)."""
    import pandas as pd
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, nrows=None)
    cols = list(df.columns)
    return f"{path.name}: {len(df)} rows x {len(cols)} cols; first cols {cols[:8]}; last cols {cols[-5:]}"


def show(cells, ranges):
    """Print chosen lines of the public notebook, e.g. '24:37-45' = cell 24, lines 37 to 45."""
    out = []
    for r in ranges:
        c, span = r.split(":"); lo, hi = (int(x) for x in span.split("-"))
        lines = cells[int(c)].splitlines()
        out += [f"[{c}:{j}] {lines[j - 1]}" for j in range(lo, min(hi, len(lines)) + 1)]
    return out


def sh(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pb115_art")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--cap_mb", type=int, default=50)
    ap.add_argument("--show", nargs="*", default=[], help="cell:first-last line ranges to print")
    a = ap.parse_args(argv)
    a.show = a.show or [r for r in os.environ.get("SHOW", "").split() if re.fullmatch(r"\d+:\d+-\d+", r)]
    out = Path(a.out); out.mkdir(exist_ok=True)
    src = Path("pb115_src"); src.mkdir(exist_ok=True)
    sh(["kaggle", "kernels", "pull", SRC, "-p", str(src), "-m"])
    nb = json.loads(next(src.glob("*.ipynb")).read_text())
    cells = code_cells(nb)
    (out / "pb115_code.py").write_text("\n\n# ---- cell ----\n".join(cells))
    print(f"notebook: {len(cells)} code cells, {sum(len(c.splitlines()) for c in cells)} lines")
    for i, c in enumerate(cells):
        print(f"cell {i}: {len(c.splitlines())} lines; first line: {c.strip().splitlines()[0][:120] if c.strip() else ''}")
    if a.show:
        print("\n".join(show(cells, a.show)))
        return
    print("lines about features, folds and the head:")
    for i, j, line in matches(cells):
        print(f"  [{i}:{j}] {line}")
    for ref in a.datasets:
        files = parse_files(sh(["kaggle", "datasets", "files", ref, "--csv", "--page-size", "200"]))
        print(f"{ref}: {len(files)} files")
        for n, s in files:
            print(f"  {s / 2**20:9.2f} MB  {n}")
        d = out / ref.replace("/", "__"); d.mkdir(exist_ok=True)
        for n in small(files, a.cap_mb):
            sh(["kaggle", "datasets", "download", ref, "-f", n, "-p", str(d), "-q"])
        for p in sorted(d.iterdir()):
            if p.suffix == ".zip":
                sh(["unzip", "-o", "-q", str(p), "-d", str(d)])
        for p in sorted(d.iterdir()):
            if p.suffix in (".csv", ".parquet"):
                print("  " + describe(p))
            elif p.suffix == ".json":
                spec = json.loads(p.read_text())
                print(f"  {p.name}: " + ", ".join(f"{k} ({len(v) if isinstance(v, (list, dict)) else v})" for k, v in spec.items()))


if __name__ == "__main__":
    main()
