"""Print the shape of the Open Pool dataset and the official train.csv: files, columns, label counts.
Prints NO passage text (this repo's Actions logs are public; competition data must not be shared)."""
import glob, os, subprocess, sys
import pandas as pd

POOL = "crtproduction/name-that-llm-open-pool"


def load(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)


def describe(df, name):
    print(f"== {name}: {len(df)} rows")
    for c in df.columns:
        s = df[c]
        n_uni = s.nunique(dropna=True)
        avg_len = s.astype(str).str.len().mean()
        print(f"  {c}: {s.dtype}, {n_uni} unique, mean length {avg_len:.0f}")
        if n_uni <= 80 and avg_len < 60:
            print("    counts:", s.value_counts(dropna=False).head(80).to_dict())


def main():
    subprocess.run(["kaggle", "datasets", "files", POOL], check=False)
    subprocess.run(["kaggle", "datasets", "download", POOL, "-p", "pool", "--unzip", "-q"], check=True)
    subprocess.run(["kaggle", "competitions", "download", "name-that-llm", "-f", "train.csv", "-p", "comp", "-q"], check=False)
    for z in glob.glob("comp/*.zip"):
        subprocess.run(["unzip", "-o", "-q", z, "-d", "comp"], check=False)
    files = sorted(f for f in glob.glob("pool/**/*", recursive=True) if f.endswith((".csv", ".parquet")))
    print("pool files:", [(f, os.path.getsize(f)) for f in files])
    for f in files:
        if os.path.basename(f) in ("ntl_open_pool.parquet", "prompts.parquet", "per_model.csv"):
            describe(load(f), f)
    if os.path.exists("comp/train.csv"):
        describe(load("comp/train.csv"), "official train.csv")


if __name__ == "__main__":
    sys.exit(main())
