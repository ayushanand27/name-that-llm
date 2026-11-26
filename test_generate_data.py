"""generate_data.py against a fake local OpenAI-style server. No real network, no real keys."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import generate_data as gd

KEY = "sk-test-SECRET-123456"
TEXT = " ".join(["word"] * 60)          # 60 words, inside 40-250


class Fake:
    """Local server: records each request; `script` is a list of status codes to return first (e.g. [429])."""
    def __init__(self):
        self.calls, self.auth, self.script, self.text = [], [], [], TEXT
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.calls.append(body); fake.auth.append(self.headers.get("Authorization"))
                code = fake.script.pop(0) if fake.script else 200
                if code != 200:
                    self.send_response(code); self.send_header("Retry-After", "0"); self.end_headers()
                    self.wfile.write(b'{"error":"slow down"}'); return
                out = {"choices": [{"message": {"content": fake.text}}],
                       "usage": {"prompt_tokens": 100, "completion_tokens": 200}}
                data = json.dumps(out).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def fake():
    f = Fake(); yield f; f.close()


def setup(tmp_path, url, prompts=2, price_in=1.0, price_out=4.0):
    (tmp_path / "prompts.csv").write_text("prompt_id,prompt\n" + "".join(f"p{i},Write about topic {i}\n" for i in range(prompts)))
    cfg = {"key_env": "FAKE_API_KEY",
           "labels": {"kimi-k3": {"base_url": url, "model": "kimi-k3", "reasoning": "high",
                                  "price_in_per_mtok": price_in, "price_out_per_mtok": price_out,
                                  "est_in_tokens": 100, "est_out_tokens": 200}}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    return ["--prompts", str(tmp_path / "prompts.csv"), "--config", str(tmp_path / "cfg.json"),
            "--out", str(tmp_path / "out.jsonl")]


# cost per call at these prices: 100 * 1/1e6 + 200 * 4/1e6 = 0.0009 USD

def test_cost_of_usage():
    cfg = {"price_in_per_mtok": 1.0, "price_out_per_mtok": 4.0}
    assert gd.cost_of({"prompt_tokens": 100, "completion_tokens": 200}, cfg) == pytest.approx(0.0009)


def test_word_count_rule():
    assert gd.word_ok(TEXT) and not gd.word_ok("too short") and not gd.word_ok(" ".join(["w"] * 251))


def test_dry_run_makes_no_calls_and_prints_estimate(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.delenv("FAKE_API_KEY", raising=False)          # dry run needs no key
    assert gd.main(setup(tmp_path, fake.url) + ["--n", "3"]) == 0
    out = capsys.readouterr().out
    assert fake.calls == [] and not (tmp_path / "out.jsonl").exists()
    assert "6 calls" in out and "0.0054" in out and "dry run" in out.lower()


def test_execute_without_key_fails_clearly(tmp_path, fake, monkeypatch):
    monkeypatch.delenv("FAKE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="FAKE_API_KEY"):
        gd.main(setup(tmp_path, fake.url) + ["--n", "1", "--execute"])
    assert fake.calls == []


def test_execute_writes_rows_and_sends_reasoning(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", KEY)
    assert gd.main(setup(tmp_path, fake.url) + ["--n", "1", "--execute", "--max_usd", "1"]) == 0
    rows = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["label"] == "kimi-k3" and rows[0]["reasoning"] == "high"
    assert rows[0]["prompt_tokens"] == 100 and rows[0]["cost_usd"] == pytest.approx(0.0009) and rows[0]["ok"]
    assert {"prompt_id", "sample_idx", "model", "date", "text", "words"} <= set(rows[0])
    assert fake.calls[0]["reasoning_effort"] == "high" and fake.auth[0] == f"Bearer {KEY}"


def test_spend_cap_stops_before_going_over(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_API_KEY", KEY)
    # 5 prompts x 1 = 5 calls of 0.0009; cap 0.002 allows 2 (0.0018), a 3rd would make 0.0027
    assert gd.main(setup(tmp_path, fake.url, prompts=5) + ["--n", "1", "--execute", "--max_usd", "0.002"]) == 0
    assert len(fake.calls) == 2
    assert "spend cap" in capsys.readouterr().out


def test_resume_skips_finished_rows(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", KEY)
    args = setup(tmp_path, fake.url) + ["--n", "2", "--execute", "--max_usd", "1"]
    gd.main(args); assert len(fake.calls) == 4
    gd.main(args); assert len(fake.calls) == 4            # nothing new
    gd.main(setup(tmp_path, fake.url) + ["--n", "3", "--execute", "--max_usd", "1"])
    assert len(fake.calls) == 6                            # only sample_idx 2 for each prompt


def test_cap_counts_spend_already_in_the_file(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", KEY)
    gd.main(setup(tmp_path, fake.url, prompts=2) + ["--n", "1", "--execute", "--max_usd", "1"])
    gd.main(setup(tmp_path, fake.url, prompts=4) + ["--n", "1", "--execute", "--max_usd", "0.002"])
    assert len(fake.calls) == 2                            # 0.0018 already spent, the next call would pass 0.002


def test_429_is_retried_with_backoff(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", KEY)
    waits = []
    monkeypatch.setattr(gd.time, "sleep", lambda s: waits.append(s))
    fake.script = [429, 429]
    assert gd.main(setup(tmp_path, fake.url, prompts=1) + ["--n", "1", "--execute", "--max_usd", "1"]) == 0
    assert len(fake.calls) == 3 and len(waits) == 2 and waits[1] > waits[0]
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 1


def test_out_of_range_text_is_marked_not_ok(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", KEY)
    fake.text = "far too short"
    gd.main(setup(tmp_path, fake.url, prompts=1) + ["--n", "1", "--execute", "--max_usd", "1"])
    row = json.loads((tmp_path / "out.jsonl").read_text())
    assert row["ok"] is False and row["words"] == 3


def test_key_never_written_or_printed(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_API_KEY", KEY)
    monkeypatch.setattr(gd.time, "sleep", lambda s: None)
    fake.script = [500, 500, 500, 500, 500, 500]           # fails every retry: error path must not leak the key
    gd.main(setup(tmp_path, fake.url, prompts=1) + ["--n", "1", "--execute", "--max_usd", "1"])
    fake.script = []
    gd.main(setup(tmp_path, fake.url, prompts=1) + ["--n", "1", "--execute", "--max_usd", "1"])
    printed = capsys.readouterr()
    for f in tmp_path.iterdir():
        assert KEY not in f.read_text() and "Bearer" not in f.read_text()
    assert KEY not in printed.out + printed.err
