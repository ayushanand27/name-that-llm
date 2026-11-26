"""generate_data.py (OpenRouter :free) against a fake local server. No real network, no real keys, $0."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import generate_data as gd

KEY = "sk-or-test-SECRET-123456"
TEXT = " ".join(["word"] * 60)          # 60 words, inside 40-250
MODEL = "z-ai/glm-5.3:free"


class Fake:
    """Local OpenRouter stand-in. `script` = status codes to return first; `remaining` = daily free requests left;
    `served` = model id the server says served the reply."""
    def __init__(self):
        self.calls, self.auth, self.script, self.text = [], [], [], TEXT
        self.remaining, self.served, self.key_reads = 50, "z-ai/glm-5.3", 0
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, extra=()):
                data = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json")
                for k, v in extra:
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            def do_GET(self):
                fake.auth.append(self.headers.get("Authorization")); fake.key_reads += 1
                self._send(200, {"data": {"free_model_daily_requests": {"used": 50 - fake.remaining, "limit": 50,
                                                                         "remaining": fake.remaining}}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.calls.append(body); fake.auth.append(self.headers.get("Authorization"))
                code = fake.script.pop(0) if fake.script else 200
                if code != 200:
                    return self._send(code, {"error": "slow down"}, [("Retry-After", "0")])
                fake.remaining -= 1
                self._send(200, {"model": fake.served, "provider": "SomeHost",
                                 "choices": [{"message": {"content": fake.text}}],
                                 "usage": {"prompt_tokens": 100, "completion_tokens": 200}})

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/api/v1"

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def fake():
    f = Fake(); yield f; f.close()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(gd.time, "sleep", lambda s: waits.append(s))
    return waits


def setup(tmp_path, url, prompts=2, model=MODEL):
    (tmp_path / "prompts.csv").write_text("prompt_id,prompt\n" + "".join(f"p{i},Write about topic {i}\n" for i in range(prompts)))
    cfg = {"base_url": url, "key_env": "FAKE_OR_KEY", "labels": {"glm-5.3": {"model": model}}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    return ["--prompts", str(tmp_path / "prompts.csv"), "--config", str(tmp_path / "cfg.json"),
            "--out", str(tmp_path / "out.jsonl")]


def rows(tmp_path):
    p = tmp_path / "out.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def test_word_count_rule():
    assert gd.word_ok(TEXT) and not gd.word_ok("too short") and not gd.word_ok(" ".join(["w"] * 251))


def test_same_model_rule():
    assert gd.same_model("z-ai/glm-5.3:free", "z-ai/glm-5.3")
    assert gd.same_model("z-ai/glm-5.3:free", "z-ai/glm-5.3:free")
    assert not gd.same_model("z-ai/glm-5.3:free", "z-ai/glm-5.3-flash")
    assert not gd.same_model("z-ai/glm-5.3:free", None)


def test_only_free_models_are_allowed(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    with pytest.raises(SystemExit, match=":free"):
        gd.main(setup(tmp_path, fake.url, model="z-ai/glm-5.3") + ["--execute"])
    assert fake.calls == []


def test_dry_run_makes_no_network_calls(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.delenv("FAKE_OR_KEY", raising=False)          # dry run needs no key
    assert gd.main(setup(tmp_path, fake.url) + ["--n", "3"]) == 0
    out = capsys.readouterr().out
    assert fake.calls == [] and fake.key_reads == 0 and not (tmp_path / "out.jsonl").exists()
    assert "6 calls" in out and "dry run" in out.lower() and "$0" in out


def test_execute_without_key_fails_clearly(tmp_path, fake, monkeypatch):
    monkeypatch.delenv("FAKE_OR_KEY", raising=False)
    with pytest.raises(SystemExit, match="FAKE_OR_KEY"):
        gd.main(setup(tmp_path, fake.url) + ["--execute"])
    assert fake.calls == []


def test_execute_writes_rows_with_served_model(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    assert gd.main(setup(tmp_path, fake.url) + ["--execute"]) == 0
    r = rows(tmp_path)
    assert len(r) == 2 and all(x["ok"] for x in r)
    assert {x["served_model"] for x in r} == {"z-ai/glm-5.3"} and r[0]["provider"] == "SomeHost"
    assert {"prompt_id", "label", "sample_idx", "model", "date", "prompt_tokens", "completion_tokens", "words"} <= set(r[0])


def test_stops_when_daily_budget_is_zero(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    fake.remaining = 0
    assert gd.main(setup(tmp_path, fake.url) + ["--execute"]) == 0
    assert fake.calls == [] and "daily" in capsys.readouterr().out


def test_stops_when_budget_runs_out_mid_run(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    fake.remaining = 2
    gd.main(setup(tmp_path, fake.url, prompts=5) + ["--execute"])
    assert len(fake.calls) == 2


def test_keep_spare_requests(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    fake.remaining = 3
    gd.main(setup(tmp_path, fake.url, prompts=5) + ["--execute", "--keep", "2"])
    assert len(fake.calls) == 1


def test_paces_at_most_15_per_minute(tmp_path, fake, monkeypatch, no_sleep):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    clock = [1000.0]
    monkeypatch.setattr(gd.time, "monotonic", lambda: clock[0])
    gd.main(setup(tmp_path, fake.url, prompts=3) + ["--execute"])
    assert len(fake.calls) == 3
    assert sum(1 for w in no_sleep if w >= 4.0 - 1e-9) >= 2     # 60/15 = 4 s between calls


def test_wrong_served_model_is_rejected(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    fake.served = "z-ai/glm-5.3-flash"
    gd.main(setup(tmp_path, fake.url) + ["--execute"])
    r = rows(tmp_path)
    assert len(r) == 2 and not any(x["ok"] for x in r) and r[0]["reject"] == "served by another model"


def test_resume_skips_done_rows(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    args = setup(tmp_path, fake.url, prompts=3)
    fake.remaining = 2
    gd.main(args + ["--execute"])
    assert len(rows(tmp_path)) == 2
    fake.remaining = 50; n = len(fake.calls)
    gd.main(args + ["--execute"])
    assert len(fake.calls) == n + 1 and len(rows(tmp_path)) == 3
    keys = {(x["prompt_id"], x["label"], x["sample_idx"]) for x in rows(tmp_path)}
    assert len(keys) == 3


def test_429_is_retried_with_growing_waits(tmp_path, fake, monkeypatch, no_sleep):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    fake.script = [429, 429]
    gd.main(setup(tmp_path, fake.url, prompts=1) + ["--execute"])
    assert len(fake.calls) == 3 and len(rows(tmp_path)) == 1
    assert len(no_sleep) == 2 and no_sleep[1] > no_sleep[0] > 0   # one prompt, so these are both backoff waits


def test_out_of_range_text_is_kept_but_marked(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    fake.text = "far too short"
    gd.main(setup(tmp_path, fake.url, prompts=1) + ["--execute"])
    r = rows(tmp_path)[0]
    assert r["ok"] is False and r["words"] == 3 and r["reject"] == "words outside 40-250"


def test_key_never_appears_in_outputs(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_OR_KEY", KEY)
    fake.script = [500]
    gd.main(setup(tmp_path, fake.url) + ["--execute"])
    out = capsys.readouterr()
    files = "".join(p.read_text() for p in tmp_path.iterdir() if p.is_file())
    for blob in (out.out, out.err, files):
        assert KEY not in blob and "Bearer" not in blob and "Authorization" not in blob
    assert all(a == f"Bearer {KEY}" for a in fake.auth)          # it was sent, only to the server
