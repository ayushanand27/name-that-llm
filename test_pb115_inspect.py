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
