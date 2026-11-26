"""Find where Rohan's PB115 notebook gets its 115 head features, so head_cv.py can rebuild the head on CPU.

Runs in GitHub Actions (Kaggle token from secrets). Pulls the public notebook, saves its code cells to
an artifact folder, lists the files of the feature datasets (name and size only), and copies the small
ones into the artifact. Prints file names and sizes only, never passage text.
"""
import argparse
import csv
import io
import json
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


def sh(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pb115_art")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--cap_mb", type=int, default=50)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(exist_ok=True)
    src = Path("pb115_src"); src.mkdir(exist_ok=True)
    sh(["kaggle", "kernels", "pull", SRC, "-p", str(src), "-m"])
    nb = json.loads(next(src.glob("*.ipynb")).read_text())
    cells = code_cells(nb)
    (out / "pb115_code.py").write_text("\n\n# ---- cell ----\n".join(cells))
    print(f"notebook: {len(cells)} code cells saved to the artifact")
    for ref in a.datasets:
        files = parse_files(sh(["kaggle", "datasets", "files", ref, "--csv", "--page-size", "200"]))
        print(f"{ref}: {len(files)} files")
        for n, s in files:
            print(f"  {s / 2**20:9.2f} MB  {n}")
        d = out / ref.replace("/", "__"); d.mkdir(exist_ok=True)
        for n in small(files, a.cap_mb):
            sh(["kaggle", "datasets", "download", ref, "-f", n, "-p", str(d), "-q"])
        print(f"  copied {len(small(files, a.cap_mb))} files of {a.cap_mb} MB or less into the artifact")


if __name__ == "__main__":
    main()
