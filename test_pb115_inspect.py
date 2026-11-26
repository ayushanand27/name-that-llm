import pb115_inspect as pi


def test_parse_file_list_reads_kaggle_csv():
    csv = "name,size,creationDate\ntrain_115.parquet,1234567,2026-11-01\nREADME.md,90,2026-11-01\n"
    assert pi.parse_files(csv) == [("train_115.parquet", 1234567), ("README.md", 90)]


def test_small_files_only():
    files = [("a.parquet", 10), ("big.bin", 900 * 2**20), ("c.csv", 5 * 2**20)]
    assert pi.small(files, cap_mb=50) == ["a.parquet", "c.csv"]


def test_code_cells_only_source_text():
    nb = {"cells": [{"cell_type": "markdown", "source": ["# hi"]},
                    {"cell_type": "code", "source": ["x = 1\n", "y = 2"]}]}
    assert pi.code_cells(nb) == ["x = 1\ny = 2"]


def test_matches_reports_cell_and_line():
    cells = ["a = 1\nfit = LogisticRegression(C=0.025)\n", "print('x')\nfolds = GroupKFold(5)"]
    got = pi.matches(cells, r"LogisticRegression|GroupKFold")
    assert got == [(0, 2, "fit = LogisticRegression(C=0.025)"), (1, 2, "folds = GroupKFold(5)")]


def test_matches_cuts_long_lines():
    got = pi.matches(["x = 'oof" + "a" * 500 + "'"], "oof")
    assert len(got[0][2]) <= 160


def test_show_ranges():
    cells = ["a\nb\nc\nd", "x\ny"]
    assert pi.show(cells, ["0:2-3", "1:1-1"]) == ["[0:2] b", "[0:3] c", "[1:1] x"]
