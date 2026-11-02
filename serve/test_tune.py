"""serve/test_tune.py - "strata_tune" (serve/tune.py): the keys a request sends on its GEN line, the config's "client_tune"
over a request's own keys, and model variants of one engine (serve/models.py: engine_identity), switched without a
restart.  A fake engine: no process, no GPU.

    python -m unittest serve.test_tune -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import models as serve_models  # noqa: E402
from serve import tune  # noqa: E402
from serve.turns import Turn, TurnScheduler, turns_from_config  # noqa: E402

CLIENT = ["tail_skip", "kva", "route_resident", "route_resident_ranks"]
COMMON = {"exe": "engine/strata", "log": "/data/strata.log",
          "args": ["--pack", "/data/pack", "--mtp", "/m", "--conversation-cache-dir", "/data/convcache"]}


def config(**extra):
    """q3s-slow and q3s-fast: one engine (the same args), two sampling blocks; q4 another engine."""
    return {**COMMON, "models": {
        "q3s-slow": {"model_name": "qwen-q3s-slow", "args": ["--pack", "/data/pack-q3s"],
                     "sampling": {"client_tune": []}},
        "q3s-fast": {"model_name": "qwen-q3s-fast", "args": ["--pack", "/data/pack-q3s"],
                     "sampling": {"strata_tune": {"route_prior": 1.5}, "client_tune": CLIENT}},
        "q4": {"model_name": "qwen-q4", "args": ["--pack", "/data/pack-q4"]}}, **extra}


class KeysTest(unittest.TestCase):
    def test_engine_keys(self):
        self.assertEqual(tune.engine_keys({"tail_skip": 7, "kva": True, "route_prior": 1.5, "route_resident": 0.5,
                                           "route_resident_ranks": [6, 9]}),
                         " tail_skip=7 kva=1 route_prior=1.5 route_resident=0.5 route_resident_ranks=6-9")
        self.assertEqual(tune.engine_keys({"pcie_frac": 0.2, "prefill_pipe": 1, "aux_cpus": False}),
                         " pcie_frac=0.2 pipe=1.0 aux_cpus=0")
        self.assertEqual(tune.engine_keys({"tail_skip": 0, "kva": False, "route_prior": 0}),
                         " tail_skip=0 kva=0 route_prior=0.0", "0 is sent: off for this request, whatever the start's")

    def test_unusable_values_are_not_sent(self):
        for bad in ({"tail_skip": 10}, {"tail_skip": 1.5}, {"tail_skip": True}, {"kva": 2}, {"route_prior": -1},
                    {"route_prior": "1.5"}, {"route_prior": float("nan")}, {"route_prior": 100},
                    {"route_resident_ranks": "9-6"}, {"route_resident_ranks": "6-10"}, {"pcie_frac": 1.5},
                    {"unknown": 1}):
            self.assertEqual(tune.engine_keys(bad), "", bad)
        self.assertEqual(tune.engine_keys(None), "")

    def test_clean(self):
        self.assertEqual(tune.clean({"route_prior": 1.5, "kva": None}, "w"), {"route_prior": 1.5})
        with self.assertRaisesRegex(ValueError, r"w\.route_priro: not a tuning key"):
            tune.clean({"route_priro": 1.5}, "w")
        with self.assertRaisesRegex(ValueError, r"w\.tail_skip=12"):
            tune.clean({"tail_skip": 12}, "w")
        self.assertEqual(tune.clean_client(["kva", "kva", "tail_skip"], "w"), ["kva", "tail_skip"])
        with self.assertRaisesRegex(ValueError, "spec not tuning keys"):
            tune.clean_client(["spec"], "w")

    def test_merge(self):
        self.assertEqual(tune.merge({"route_prior": 1.5, "pcie_frac": 0.2}, {"tail_skip": 7, "pcie_frac": 0.9},
                                    CLIENT),
                         ({"route_prior": 1.5, "pcie_frac": 0.2, "tail_skip": 7}, ["pcie_frac"]),
                         "a key the config does not let a request set is left out; the config's stays")
        self.assertEqual(tune.merge({"route_prior": 1.5}, {"route_prior": 0}, None), ({"route_prior": 0}, []),
                         "no client_tune: every key, over the config's one by one")
        self.assertEqual(tune.merge({"kva": 1}, {"kva": None}, None), (None, []), "null drops the config's key")
        self.assertEqual(tune.merge(None, {"tail_skip": 7}, []), (None, ["tail_skip"]))

    def test_sampling_keys(self):
        from serve.server import StrataEngine
        keys = StrataEngine.sampling_keys({"temperature": 0, "strata_tune": {"tail_skip": 7, "route_prior": 1.5}})
        self.assertIn(" tail_skip=7", keys)
        self.assertIn(" route_prior=1.5", keys)

    def test_config_block(self):
        from serve.server import sampling_defaults_from_config
        out = sampling_defaults_from_config({"sampling": {"strata_tune": {"route_prior": 1.5}, "client_tune": CLIENT}})
        self.assertEqual(out, {"strata_tune": {"route_prior": 1.5}, "client_tune": CLIENT})
        with self.assertRaises(SystemExit):
            sampling_defaults_from_config({"sampling": {"strata_tune": {"route_prior": "fast"}}})
        with self.assertRaises(SystemExit):
            sampling_defaults_from_config({"sampling": {"client_tune": ["nope"]}})


class EngineIdentityTest(unittest.TestCase):
    def test_siblings_are_one_engine(self):
        m, _ = serve_models.load(config(), None)
        self.assertTrue(m.same_engine("q3s-slow", "q3s-fast"))
        self.assertFalse(m.same_engine("q3s-fast", "q4"))
        self.assertTrue(m.same_engine("q4", "q4"))
        self.assertTrue(m.same_engine(None, None))
        self.assertFalse(m.same_engine("q4", None))
        self.assertFalse(m.same_engine("q4", "gone"))

    def test_siblings_share_a_cache_dir(self):
        """Variants of one engine park into one folder (named after the first of them); another engine has its own."""
        m, _ = serve_models.load(config(), None)

        def folder(name):
            args = m.variants[name].cfg["args"]
            return args[args.index("--conversation-cache-dir") + 1]
        self.assertEqual(folder("q3s-slow"), "/data/convcache/q3s-slow")
        self.assertEqual(folder("q3s-fast"), "/data/convcache/q3s-slow")
        self.assertEqual(folder("q4"), "/data/convcache/q4")
        self.assertTrue(m.same_engine("q3s-slow", "q3s-fast"), "still one engine after the folders are set")

    def test_engine_settings_make_another_engine(self):
        cfg = config()
        cfg["models"]["q3s-fast"]["env"] = {"STRATA_ROUTE_PRIOR": "1.5"}
        m, _ = serve_models.load(cfg, None)
        self.assertFalse(m.same_engine("q3s-slow", "q3s-fast"), "an engine setting (env) is a restart")

    def test_scheduler_does_not_wait_for_a_sibling(self):
        """A request for a variant of the loaded engine takes the turn as one for the loaded variant does: no
        model_min_resident_s, no switch to commit."""
        m, _ = serve_models.load(config(model_min_resident_s=30, model_max_wait_s=120), None)
        t = [10.0]
        m.clock = lambda: t[0]
        m.since = 0.0
        m.loaded = lambda: True
        s = TurnScheduler(threading.Lock(), 0.0, 0.0, 0.0, (), 0.0, clock=lambda: t[0])
        s.models = m
        fast = s.enter(Turn(model="q3s-fast"))
        self.assertTrue(s.acquire(fast, 0))
        self.assertIsNone(m.committed)
        s.release(1)
        other = s.enter(Turn(model="q4"))
        self.assertFalse(s.acquire(other, 0), "another engine still waits for model_min_resident_s")


class ServerTest(unittest.TestCase):
    """q3s-slow and q3s-fast through the HTTP server: the switch restarts nothing, and each request's strata_tune is
    its variant's with the client's allowed keys over it."""

    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import IM_END, ByteTokenizer, Service, serve

        tok = ByteTokenizer()
        self.dir = tempfile.TemporaryDirectory()
        events, seen = self.events, self.seen = [], []

        class Engine:
            max_context = 1 << 16

            def __init__(self):
                self.spawn, self.up = spawn("q3s-slow"), True

            def alive(self):
                return self.up

            def unload(self):
                events.append(("unload", self.spawn[0]))
                self.up = False

            def restart(self):
                events.append(("start", self.spawn[0]))
                self.up = True

            def session_file(self, action, path):
                events.append((action, Path(path).name))
                return {"tokens": 7}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                seen.append((sampling or {}).get("strata_tune"))
                yield from tok.encode(f"</think>\nfrom {self.spawn[0]}", parse_special=True)
                yield from tok.encode(IM_END, parse_special=True)

        m, _ = serve_models.load(config(model_session_dir=self.dir.name, model_min_resident_s=0), None)

        def spawn(name):
            return (name, m.variants[name].cfg["args"], None, None, None, False, None)
        for v in m.variants.values():
            v.spawn = spawn(v.name)
        self.engine = Engine()
        svc = self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"),
                                 model_name="qwen-q3s-slow")
        svc.models = m
        m.loaded = svc.loaded
        svc.apply_variant(m.variants["q3s-slow"])
        svc.turns, _ = turns_from_config({}, svc.fifo, models=True)
        svc.turns.models = m
        self.httpd = serve(svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, path, body=None):
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            return json.loads(resp.read())

    def chat(self, model, **extra):
        r = self.call("/v1/chat/completions", {"model": model, "max_tokens": 50, **extra,
                                               "messages": [{"role": "user", "content": "hi"}]})
        return r["model"], r["choices"][0]["message"]["content"].strip()

    def test_sibling_switch_restarts_nothing(self):
        self.assertEqual(self.chat("q3s-fast"), ("q3s-fast", "from q3s-fast"))
        self.assertEqual(self.chat("qwen-q3s-slow"), ("qwen-q3s-slow", "from q3s-slow"))
        self.assertEqual(self.events, [], "one engine: no save, unload or start")
        self.assertEqual(self.svc.model, "qwen-q3s-slow")
        self.assertEqual(self.seen, [{"route_prior": 1.5}, None], "each request runs with its variant's settings")
        self.assertEqual(self.chat("q4")[0], "q4")
        self.assertEqual(self.events, [("unload", "q3s-slow"), ("start", "q4")], "another engine restarts")

    def test_client_keys(self):
        self.chat("q3s-fast", strata_tune={"tail_skip": 7, "route_prior": 0, "pcie_frac": 0.9})
        self.assertEqual(self.seen[-1], {"route_prior": 1.5, "tail_skip": 7},
                         "q3s-fast lets a request set tail_skip, not route_prior or pcie_frac")
        self.chat("q3s-slow", strata_tune={"tail_skip": 7})
        self.assertIsNone(self.seen[-1], "q3s-slow lets a request set nothing")
        self.chat("q4", strata_tune={"pcie_frac": 0.3})
        self.assertEqual(self.seen[-1], {"pcie_frac": 0.3}, "no client_tune: as before, every key")

    def test_models_listing(self):
        ids = {d["id"]: d for d in self.call("/v1/models")["data"]}
        self.assertEqual(ids["qwen-q3s-fast"]["status"]["value"], "loaded", "a variant of the loaded engine")
        self.assertEqual(ids["qwen-q4"]["status"]["value"], "unloaded")
        self.assertEqual(ids["qwen-q3s-fast"]["meta"]["tune"], {"settings": {"route_prior": 1.5}, "client": CLIENT})
        self.assertEqual(ids["qwen-q3s-slow"]["meta"]["tune"], {"client": []})
        self.assertNotIn("tune", ids["qwen-q4"]["meta"])


if __name__ == "__main__":
    unittest.main()
