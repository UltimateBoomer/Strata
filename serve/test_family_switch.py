"""Model variants of two families behind one server (serve/models.py, serve/server.py's Family): a Qwen variant and a
DeepSeek-V4.1 one, each request read and written with its own family's tokenizer, template, parser and effort words,
against a fake engine (no GPU, no pack).

    python -m unittest serve.test_family_switch -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve import frontend, server  # noqa: E402
from serve import models as serve_models  # noqa: E402
from serve.test_dsv41 import EOS, mini_tokenizer  # noqa: E402
from serve.turns import turns_from_config  # noqa: E402

COMMON = {"exe": "engine/strata", "log": "/data/strata.log", "tokenizer": "/data/pack-q2/tokenizer",
          "env": {"STRATA_HIP_WMMA": "1"}, "vision": {"exe": "engine/strata-vision", "gpu": False},
          "args": ["--pack", "/data/pack-q2", "--spec", "4", "--kv", "int8", "--max-context", "200000"]}
DS = {"arch": "deepseek41", "exe": "engine/strata-dsv41", "tokenizer": "/data/pack-deepseek/tokenizer",
      "model_name": "deepseek-v4.1-flash", "aliases": ["deepseek"], "engine_ready_s": 3600,
      "args": ["--pack", "/data/pack-deepseek", "--max-context", "200000"], "env": {"DS4_FAST_ATTN": "1"}}


def config(**extra):
    return {**COMMON, "models": {"q2": {"model_name": "qwen-q2", "aliases": ["qwen"]}, "ds": dict(DS)}, **extra}


class ConfigTest(unittest.TestCase):
    def test_other_family_takes_no_common_engine_settings(self):
        m, cfg = serve_models.load(config(), None)
        ds = m.variants["ds"]
        self.assertEqual(ds.arch, "deepseek41")
        self.assertEqual(ds.cfg["args"], DS["args"], "another program's options: none of the common ones")
        self.assertEqual(ds.cfg["env"], DS["env"])
        self.assertNotIn("vision", ds.cfg, "DeepSeek reads pictures with its engine's own tower")
        self.assertEqual((ds.cfg["exe"], ds.cfg["tokenizer"], ds.cfg["log"]),
                         (DS["exe"], DS["tokenizer"], COMMON["log"]), "its own program and tokenizer, the common log")
        q2 = m.variants["q2"]
        self.assertEqual(q2.arch, "qwen4exp")
        self.assertEqual(q2.cfg["args"], COMMON["args"])
        self.assertIn("vision", q2.cfg)
        self.assertIs(cfg, q2.cfg)
        self.assertEqual(m.view()["arch"], {"q2": "qwen4exp", "ds": "deepseek41"})

    def test_same_family_arch_inherits(self):
        """A variant that names the common family's arch is one of its quantizations: it takes the common settings."""
        m, _ = serve_models.load({**COMMON, "models": {"q2": {}, "q3": {"arch": "qwen4exp", "args": ["--pack", "/p3"]}}},
                                 None)
        self.assertEqual(m.variants["q3"].cfg["env"], COMMON["env"])
        self.assertIn("--spec", m.variants["q3"].cfg["args"])

    def test_deepseek_keeps_its_own_slots(self):
        m, _ = serve_models.load(config(), None)
        self.assertTrue(m.variants["ds"].keeps_parks(), "its engine has no SAVE: no session file at a switch")
        self.assertFalse(m.variants["q2"].keeps_parks())

    def test_refused(self):
        for own, why in [({"arch": "deepseek41"}, "own \"tokenizer\""), ({"arch": 4, "tokenizer": "t"}, "arch"),
                         ({"tokenizer": ""}, "tokenizer"), ({"turn_slice_s": 5}, "cannot differ")]:
            with self.subTest(why=why), self.assertRaisesRegex(ValueError, why):
                serve_models.load({**COMMON, "models": {"q2": {}, "x": own}}, None)

    def test_pin(self):
        """A request that names no variant goes to the loaded one when that is all one family (resolve: None); with
        two families it is pinned to it, so its prompt is written in the family it will run on."""
        m, _ = serve_models.load(config(), None)
        self.assertEqual(m.pin("claude-sonnet"), "q2")
        self.assertEqual(m.pin(None), "q2")
        self.assertEqual(m.pin("deepseek"), "ds")
        m.committed = "ds"
        self.assertEqual(m.pin(None), "ds", "the one the scheduler switches to next")
        one, _ = serve_models.load({**COMMON, "models": {"q2": {}, "q3": {}}}, None)
        self.assertIsNone(one.pin("claude-sonnet"))


class Vision:
    """A picture reader that only says what was done to it."""
    elastic = False

    def __init__(self, events):
        self.events, self.up = events, True

    def alive(self):
        return self.up

    def unload(self):
        self.events.append(("vision", "unload"))
        self.up = False

    def restart(self):
        self.events.append(("vision", "start"))
        self.up = True


class ServerTest(unittest.TestCase):
    """Requests for a Qwen and a DeepSeek variant through the HTTP server."""

    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import IM_END, ByteTokenizer, Service, serve

        qtok, dtok = ByteTokenizer(), mini_tokenizer()
        self.toks = {"q2": qtok, "ds": dtok}
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        events = self.events = []
        prompts = self.prompts = []

        class Engine:
            max_context = 1 << 16

            def __init__(self):
                self.spawn, self.up, self.unloaded = spawn("q2"), True, False

            def alive(self):
                return self.up

            def unload(self):
                events.append(("unload", self.spawn[0]))
                self.up, self.unloaded = False, True

            def restart(self):
                events.append(("start", self.spawn[0]))
                self.up, self.unloaded = True, False

            def session_file(self, action, path):
                events.append((action, Path(path).name))
                return {"tokens": 7}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                name = self.spawn[0]
                prompts.append((name, ids))
                if name == "ds":
                    yield from dtok.encode("hmm</think>from ds", parse_special=True) + [EOS]
                else:
                    yield from qtok.encode("</think>\nfrom q2", parse_special=True)
                    yield from qtok.encode(IM_END, parse_special=True)

        m, _ = serve_models.load(config(model_session_dir=self.dir.name, model_min_resident_s=0), None)

        def spawn(name):
            return (name, m.variants[name].cfg["args"], None, None, None, False, None)
        for v in m.variants.values():
            v.spawn = spawn(v.name)
            v.ready_s = server.ready_timeout_of(v.cfg)
        self.engine = Engine()
        svc = self.svc = Service(self.engine, qtok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"),
                                 model_name="qwen-q2")
        svc.model_arch = "qwen4exp"
        svc.set_family(None)
        svc.vision = self.vision = Vision(events)
        svc.models = m
        m.loaded = svc.loaded
        svc.families = {"qwen4exp": svc.home,
                        "deepseek41": server.family_for({**m.variants["ds"].cfg, "tokenizer": self.dir.name}, dtok,
                                                        m.variants["ds"].spawn[1], offline=False)}
        svc.turns, _ = turns_from_config({}, svc.fifo, models=True)
        svc.turns.models = m
        ready = server.ENGINE_READY_S
        self.addCleanup(setattr, server, "ENGINE_READY_S", ready)
        self.addCleanup(frontend.use_family, None)
        self.httpd = serve(svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def chat(self, model, **extra):
        body = {"max_tokens": 50, "messages": [{"role": "user", "content": "hi"}], **extra}
        if model is not None:
            body["model"] = model
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                out = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
        return 200, out["choices"][0]["message"]["content"].strip()

    def last_prompt(self):
        name, ids = self.prompts[-1]
        return name, self.toks[name].decode(ids)

    def test_switch_between_families(self):
        self.assertEqual(self.chat("qwen"), (200, "from q2"))
        name, text = self.last_prompt()
        self.assertEqual(name, "q2")
        self.assertIn("<|im_start|>user\nhi", text, "Qwen's template")
        self.assertEqual(self.events, [])

        self.assertEqual(self.chat("deepseek", reasoning_effort=50), (200, "from ds"),
                         "DeepSeek's end token and parser; its numeric effort")
        name, text = self.last_prompt()
        self.assertEqual(name, "ds")
        self.assertIn("<｜User｜>hi", text, "DeepSeek's template, written with its tokenizer")
        self.assertIn("Reasoning Effort: 50", text)
        self.assertEqual(self.events, [("save", "variant-q2.session"), ("unload", "q2"), ("vision", "unload"),
                                       ("start", "ds")], "Qwen's picture reader stops with its engine")
        self.assertEqual(server.ENGINE_READY_S, 3600, "the DeepSeek variant's engine_ready_s")
        self.assertEqual(self.svc.model, "deepseek-v4.1-flash")
        self.assertEqual(self.svc.family, "deepseek41", "outside a request: the loaded variant's family")

        self.events.clear()
        self.assertEqual(self.chat(None, reasoning_effort="max"), (200, "from ds"), "no model: the loaded one")
        self.assertEqual(self.events, [])

        self.assertEqual(self.chat("qwen-q2", reasoning_effort=50)[0], 400,
                         "Qwen has no numeric effort (and nothing switched for a refused request)")
        self.assertEqual(self.chat("qwen-q2"), (200, "from q2"))
        self.assertEqual(self.events, [("unload", "ds"), ("vision", "start"), ("start", "q2"),
                                       ("restore", "variant-q2.session")],
                         "DeepSeek keeps its own slots (no SAVE); Qwen's reader starts before its engine")
        self.assertEqual(server.ENGINE_READY_S, server.ENGINE_READY_DEFAULT)
        self.assertIs(self.svc.fam, self.svc.home)
        self.assertEqual(self.svc.models.switches, 2)

    def test_reader_of_another_family(self):
        """A Qwen request's pictures, while DeepSeek's engine runs: Qwen's reader starts for them as it is (the encode
        comes before the request's turn and its switch); an unload stops every family's reader."""
        self.chat("deepseek")
        self.events.clear()
        self.svc.use_request_family("q2")
        try:
            self.assertIs(self.svc.vision, self.vision)
            self.assertIsNone(self.svc.loaded_vision, "DeepSeek's (no pack here: no pictures)")
            self.svc._vision_up()
            self.assertEqual(self.events, [("vision", "start")])
            self.assertFalse(self.svc._vision_down(), "the loaded family's reader decides a load")
        finally:
            self.svc.use_request_family(None)
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.events[-2:], [("unload", "ds"), ("vision", "unload")])

    def test_load_leaves_an_engine_a_switch_replaces(self):
        """svc.load() (before the request's turn) does not start the loaded variant's engine for a request that is for
        another: its switch loads that one."""
        self.chat("deepseek")
        self.svc.unload()
        self.events.clear()
        from serve.turns import Turn
        self.svc.request_turn.turn = Turn(model="q2")
        self.svc.load()
        self.assertEqual(self.events, [])
        self.svc.request_turn.turn = Turn(model="ds")
        self.svc.load()
        self.assertEqual(self.events, [("start", "ds")])

    def test_models_listing(self):
        """Both families' variants are listed, loaded or not, each with its own family's picture input."""
        with urllib.request.urlopen(self.base + "/v1/models", timeout=30) as r:
            data = {d["id"]: d for d in json.loads(r.read())["data"]}
        ds = data["deepseek-v4.1-flash"]
        self.assertEqual((ds["status"]["value"], ds["meta"]["variant"], ds["meta"]["n_ctx"]), ("unloaded", "ds", 200000))
        self.assertEqual(ds["architecture"]["input_modalities"], ["text"], "DeepSeek's reader (none without a pack)")
        self.assertEqual(data["qwen-q2"]["architecture"]["input_modalities"], ["text", "image"])
        self.assertEqual(data["deepseek"]["alias_of"], "deepseek-v4.1-flash")

    def test_props_follow_the_loaded_family(self):
        self.chat("deepseek")
        with urllib.request.urlopen(self.base + "/props", timeout=30) as r:
            props = json.loads(r.read())
        self.assertIn("｜DSML｜", props["chat_template"])


FAKE_ENGINE = """\
import json, os, sys
with open(os.environ["FAKE_ARGV_LOG"], "a") as f:
    f.write(json.dumps([os.path.basename(sys.argv[0])] + sys.argv[1:]) + "\\n")
print("READY 200000 stop", flush=True)
for line in sys.stdin:
    if line.strip() == "QUIT":
        break
"""


def tokenizer_folder(path: Path, specials: list[str], pre: str, special_ids: dict) -> Path:
    """A pack's tokenizer files over its special tokens, the 256 byte tokens and one merge ("h" "i")."""
    import strata_tokenizer as ST
    b2u = ST.bytes_to_unicode()
    tokens = specials + [b2u[b] for b in range(256)] + ["hi"]
    path.mkdir(parents=True)
    (path / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
    (path / "merges.txt").write_text("h i", encoding="utf-8")
    (path / "token_type.json").write_text(json.dumps([3] * len(specials) + [1] * 257))
    (path / "tokenizer.json").write_text(json.dumps({"pre": pre, "special_ids": special_ids}), encoding="utf-8")
    return path


@unittest.skipIf(sys.platform == "win32", "the fake engines are POSIX scripts")
class StartupTest(unittest.TestCase):
    """The server's own start (main) with a Qwen and a DeepSeek variant: fake engine programs that only say READY and
    write down how they were started, and each family's tokenizer folder."""

    def test_start_qwen_switch_to_deepseek(self):
        self.start_and_switch(None, "deepseek", ["strata", "strata-dsv41"], "｜DSML｜")

    def test_start_deepseek_switch_to_qwen(self):
        self.start_and_switch("ds", "qwen-q2", ["strata-dsv41", "strata"], "<|im_start|>")

    def start_and_switch(self, startup, to, programs, template_mark):
        import os
        import socket
        import subprocess
        import time
        d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, d, True)
        for name in ("strata", "strata-dsv41"):
            exe = d / name
            exe.write_text("#!" + sys.executable + "\n" + FAKE_ENGINE)
            exe.chmod(0o700)
        qtok = tokenizer_folder(d / "qtok", ["<|im_start|>", "<|im_end|>", "<|endoftext|>"], "qwen35", {})
        dtok = tokenizer_folder(d / "dtok", ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>",
                                             "<｜Assistant｜>"], "deepseek3", {"bos": 0, "eos": 1})
        ds = {**DS, "exe": str(d / "strata-dsv41"), "tokenizer": str(dtok)}
        cfg = {**COMMON, "exe": str(d / "strata"), "tokenizer": str(qtok), "log": str(d / "strata.log"),
               "models": {"q2": {"model_name": "qwen-q2"}, "ds": ds}, "model_min_resident_s": 0,
               "model_session_dir": False}
        del cfg["vision"]                               # (no reader program here)
        (d / "server.json").write_text(json.dumps(cfg))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        argv_log = d / "argv.log"
        env = {**os.environ, "FAKE_ARGV_LOG": str(argv_log), "PYTHONPATH": str(ROOT), "STRATA_VARIANT": startup or ""}
        proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config",
                                 str(d / "server.json"), "--port", str(port)], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = []

        def stop():
            proc.terminate()
            try:
                out.append(proc.communicate(timeout=30)[0])
            except subprocess.TimeoutExpired:
                proc.kill()
        self.addCleanup(stop)
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except OSError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                self.fail("the server did not start: " + (proc.communicate()[0] if proc.poll() is not None else ""))
            time.sleep(0.2)
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            ids = {m["id"] for m in json.loads(r.read())["data"]}
        self.assertLessEqual({"qwen-q2", "deepseek-v4.1-flash"}, ids)
        r = urllib.request.Request(base + "/v1/load", data=json.dumps({"model": to}).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            self.assertEqual(json.loads(resp.read())["status"], "loaded")
        with urllib.request.urlopen(base + "/props", timeout=10) as resp:
            self.assertIn(template_mark, json.loads(resp.read())["chat_template"], "the switched-to family's template")
        starts = {x[0]: x[1:] for x in map(json.loads, argv_log.read_text().splitlines())}
        self.assertEqual(list(starts), programs)
        self.assertNotIn("--spec", starts["strata-dsv41"], "none of the Qwen engine's options")
        self.assertIn("--spec", starts["strata"])
        self.assertEqual(starts["strata-dsv41"][starts["strata-dsv41"].index("--max-context") + 1], "200000")
        stop()
        self.assertIn("model families: ", out[-1])


if __name__ == "__main__":
    unittest.main()
