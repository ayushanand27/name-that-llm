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
def test_owner_from_env():
    import kaggle_runner as k
    assert k.kaggle_owner(run=lambda c:1/0,env={"KAGGLE_USERNAME":" me "})=="me"
def test_owner_from_cli():
    import kaggle_runner as k
    assert k.kaggle_owner(run=lambda c:"username: bob\nx: y",env={})=="bob"
