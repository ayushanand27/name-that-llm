import json,pytest
from kaggle_runner import *
def test_append_keeps_original():
    nb={"cells":[code_cell("a=1")],"metadata":{}}
    out=append_cells(nb,["b=2\nc=3"]);assert len(nb["cells"])==1 and len(out["cells"])==2
    assert "".join(out["cells"][1]["source"])=="b=2\nc=3"
def test_append_rejects_non_notebook():
    with pytest.raises(ValueError):append_cells({"x":1},["a"])
def test_metadata_gpu_private_no_internet():
    m=make_metadata({"dataset_sources":["a/b"],"enable_internet":True},"me","cut256-cv")
    assert m["id"]=="me/cut256-cv" and m["enable_gpu"] and not m["enable_internet"] and m["is_private"] and m["dataset_sources"]==["a/b"]
def test_metadata_bad_slug():
    with pytest.raises(ValueError):make_metadata({}, "me","Bad Slug")
@pytest.mark.parametrize("t,s",[('x has status "KernelWorkerStatus.COMPLETE"',"complete"),('has status "running"',"running"),("garbage","unknown"),("",'unknown')])
def test_status(t,s):assert parse_status(t)==s
def test_wait_stops_on_complete():
    seq=iter(['status "KernelWorkerStatus.RUNNING"','status "KernelWorkerStatus.COMPLETE"'])
    assert wait("a/b",poll_s=0,run=lambda c,check=False:next(seq))=="complete"
def test_wait_timeout():
    assert wait("a/b",poll_s=0,max_s=0,run=lambda c,check=False:"")=="timeout"
def test_cli_parses_new_modes():
    import subprocess,sys
    for args in (["run","--src","a/b","--slug","c","--cells","x","--no-wait"],["fetch","--ref","c"]):
        r=subprocess.run([sys.executable,"kaggle_runner.py",*args,"--help"] if False else [sys.executable,"-c","import kaggle_runner"],capture_output=True);assert r.returncode==0
def test_fetch_not_finished(monkeypatch,capsys):
    import kaggle_runner as k,argparse
    monkeypatch.setattr(k,"sh",lambda c,check=True:'status "KernelWorkerStatus.RUNNING"')
    k.cmd_fetch(argparse.Namespace(ref="me/x",work="w",wait=False));assert "not finished" in capsys.readouterr().out
def test_owner_from_env_when_cli_fails():
    import kaggle_runner as k
    assert k.kaggle_owner(run=lambda c:1/0,env={"KAGGLE_USERNAME":" me "})=="me"
def test_owner_cli_beats_env():
    import kaggle_runner as k
    assert k.kaggle_owner(run=lambda c:"- username: ayushanand2327\n",env={"KAGGLE_USERNAME":"AyushAnand2327"})=="ayushanand2327"
def test_owner_cli_none_uses_env():
    import kaggle_runner as k
    assert k.kaggle_owner(run=lambda c:"- username: None\n",env={"KAGGLE_USERNAME":"me"})=="me"
def test_owner_nothing_raises():
    import kaggle_runner as k, pytest
    with pytest.raises(RuntimeError):
        k.kaggle_owner(run=lambda c:"",env={})
def test_owner_from_cli():
    import kaggle_runner as k
    assert k.kaggle_owner(run=lambda c:"username: bob\nx: y",env={})=="bob"

def test_check_sources_lists_unreadable():
    import kaggle_runner as k
    def run(cmd):
        if cmd[-1] == "x/private":
            raise RuntimeError("cmd failed:\n\n403 Client Error: Forbidden")
        return "ok"
    md = {"dataset_sources": ["a/ok", "x/private"], "competition_sources": ["name-that-llm"], "model_sources": []}
    bad = k.check_sources(md, run=run)
    assert [(b[0], b[1]) for b in bad] == [("dataset_sources", "x/private")]
    assert "403" in bad[0][2]

def test_metadata_drops_source_id_no():
    import kaggle_runner as k
    m = k.make_metadata({"id": "rohan/x", "id_no": 12345, "dataset_sources": ["a/b"]}, "me", "my-run")
    assert "id_no" not in m and m["id"] == "me/my-run" and m["dataset_sources"] == ["a/b"]


def test_pool_run_args():
    import kaggle_runner as k
    assert "--use_pool 0" in k.pool_run_args("A") and "--epochs 3" in k.pool_run_args("A")
    b = k.pool_run_args("B")
    assert "--use_pool 1" in b and "--cap 1500" in b and "--kimi_high_only 0" in b
    assert "--holdout_frac 0" in k.pool_run_args("final") and "--use_pool 1" in k.pool_run_args("final")
    assert "C" not in k.POOL_RUNS
    assert "--probe_only 1" in k.pool_run_args("probe") and "--use_pool 1" in k.pool_run_args("probe")
    import pytest
    with pytest.raises(KeyError):
        k.pool_run_args("Z")


def test_build_pool_notebook(tmp_path):
    import kaggle_runner as k
    for n in ("pool_data.py", "metrics.py"):
        (tmp_path / n).write_text(f"# {n}\nx = 1\n")
    nb = k.build_pool_notebook([tmp_path / "pool_data.py", tmp_path / "metrics.py"], "B", n_gpus=2)
    src = ["".join(c["source"]) for c in nb["cells"]]
    assert src[1].startswith("%%writefile pool_data.py\n# pool_data.py") and src[2].startswith("%%writefile metrics.py")
    assert "pip install" in src[0]
    assert "torchrun" in src[-1] and "'--nproc_per_node', '2'" in src[-1] and "'--run_name', 'B'" in src[-1]
    assert nb["nbformat"] == 4


def test_pool_metadata():
    import kaggle_runner as k
    m = k.pool_metadata("me", "ayush-pool-b")
    assert m["id"] == "me/ayush-pool-b" and m["enable_gpu"] and m["enable_internet"] and m["is_private"]
    assert "crtproduction/name-that-llm-open-pool" in m["dataset_sources"]
    assert m["competition_sources"] == ["name-that-llm"] and m["model_sources"] and "id_no" not in m


def test_stdout_lines_from_kaggle_log():
    import kaggle_runner as k
    log = '[{"stream_name":"stdout","time":1,"data":"hello\\n"}\n,{"stream_name":"stderr","time":2,"data":"warn\\n"}\n,{"stream_name":"stdout","time":3,"data":"a\\nb\\n"}]'
    assert k.stdout_lines(log) == ["hello", "a", "b"]
    assert k.stdout_lines("not json\nplain") == ["not json", "plain"]


def test_pool_notebook_cells_are_valid_python(tmp_path):
    import ast, kaggle_runner as k
    nb = k.build_pool_notebook([], "final", n_gpus=1)
    ast.parse("".join(nb["cells"][-1]["source"]))
    assert "--holdout_frac" in "".join(nb["cells"][-1]["source"])


def test_error_lines_keep_traceback_only():
    import kaggle_runner as k
    log = ('[{"stream_name":"stdout","time":1,"data":"Error in stdout is ignored\\n"}\n,'
           '{"stream_name":"stderr","time":2,"data":"warn: slow\\nTraceback (most recent call last):\\n  File \\"x.py\\", line 1\\nValueError: bad\\n"}]')
    assert k.error_lines(log) == ["Traceback (most recent call last):", '  File "x.py", line 1', "ValueError: bad"]
    assert k.error_lines("not json") == []


def test_launch_cell_needs_exit_code_zero():
    import kaggle_runner as k
    src = "".join(k.build_pool_notebook([], "B", n_gpus=2)["cells"][-1]["source"])
    assert "assert rc == 0" in src


def test_check_slug_and_inside_refuse_paths(tmp_path):
    import pytest, kaggle_runner as k
    assert k.check_slug("ayush-pool-b") == "ayush-pool-b"
    for bad in ("../x", "a/b", "AB", "", "x"):
        with pytest.raises(SystemExit):
            k.check_slug(bad)
    assert k.pool_metadata("me", "ayush-pool-b")["code_file"] == "notebook.ipynb"


def test_run_p_is_pool_only_without_proxy_rows_and_b_keeps_its_recipe():
    import kaggle_runner as k
    p = k.pool_run_args("P")
    assert "--official_in_train 0" in p and "--drop_flag_kb_proxy 1" in p and "--drop_gem37_after_oct8 0" in p
    assert "--cap 1500" in p and "--epochs 1" in p and "--holdout_frac" not in p
    assert "--drop_flag_kb_proxy 1" in k.pool_run_args("final")
    assert "--drop_flag_kb_proxy 0" in k.pool_run_args("B") and "--drop_flag_kb_proxy 0" in k.pool_run_args("probe")


def test_pool_notebook_scores_the_test_set(tmp_path):
    import kaggle_runner as k
    src = "".join(k.build_pool_notebook([], "P", n_gpus=2)["cells"][-1]["source"])
    assert "find('test.csv')" in src and "'--test_csv', TEST" in src


def test_fetch_pattern_keeps_oof_and_test_files_only():
    import re, kaggle_runner as k
    pat = re.compile(k.FETCH_PATTERN)
    for f in ("x.log", "lora_run/oof_P.npz", "lora_run/test_P.npz"):
        assert pat.search(f), f
    for f in ("lora_run/adapter/adapter_model.safetensors", "lora_run/ckpt/checkpoint-200/optimizer.pt"):
        assert not pat.search(f), f
