"""serve/test_models.py - model variants behind one server (serve/models.py): the config, the scheduler's switching
policy (serve/turns.py) and the switches through the server, with a fake engine (no process, no GPU).

    python -m unittest serve.test_models -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import models as serve_models  # noqa: E402
from serve.turns import Priorities, Turn, TurnScheduler, turns_from_config  # noqa: E402

COMMON = {"exe": "engine/strata", "log": "/data/strata.log", "turn_slice_s": 30, "sampling": {"top_k": 20},
          "env": {"A": "1"}, "args": ["--pack", "/data/pack", "--spec", "4", "--max-context", "200000"]}


def config(**extra):
    return {**COMMON, "models": {"q2": {"model_name": "qwen-q2", "aliases": ["fast"]},
                                 "q3": {"model_name": "qwen-q3", "args": ["--pack", "/data/pack-q3"],
                                        "env": {"B": "2"}, "sampling": {"top_k": 40}}}, **extra}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ConfigTest(unittest.TestCase):
    def test_variant_over_common(self):
        m, cfg = serve_models.load(config(), None)
        q3 = m.variants["q3"].cfg
        self.assertEqual(q3["args"], ["--spec", "4", "--max-context", "200000", "--pack", "/data/pack-q3"])
        self.assertEqual(q3["env"], {"A": "1", "B": "2"})
        self.assertEqual(q3["sampling"], {"top_k": 40})
        self.assertNotIn("models", q3)
        self.assertEqual(q3["turn_slice_s"], 30, "the common keys are every variant's")
        self.assertIs(cfg, m.variants["q2"].cfg, "it starts with the first (default) variant")
        self.assertEqual(cfg["args"], COMMON["args"])

    def test_files_next_to_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "server-q3.json").write_text(json.dumps({"model_name": "qwen-q3"}))
            path = Path(d, "server.json")
            m, _ = serve_models.load({**COMMON, "models": {"q2": {}, "q3": "server-q3.json"}, "default_model": "q3"},
                                     str(path))
        self.assertEqual(m.variants["q3"].model_name, "qwen-q3")
        self.assertEqual(m.variants["q2"].model_name, "q2", "a variant without a model name answers to its key")
        self.assertEqual((m.default, m.active), ("q3", "q3"))

    def test_start_with(self):
        m, cfg = serve_models.load(config(), None, "q3")
        self.assertEqual((m.default, m.active), ("q2", "q3"))
        self.assertEqual(cfg["args"][-1], "/data/pack-q3")

    def test_refused(self):
        for cfg, why in [
            (config(models={"q2": {"tokenizer": "x"}}), "cannot differ"),
            (config(default_model="q9"), "default_model"),
            (config(models={"q2": {"aliases": ["q3"]}, "q3": {}}), "both"),
            (config(parallel=2), "one request at a time"),
            (config(unknown_model="maybe"), "unknown_model"),
            (config(default_idle_s=-1), "default_idle_s"),
            (config(models={"bad name": {}}), "variant name"),
            (config(models={}), "models"),
        ]:
            with self.subTest(why=why), self.assertRaisesRegex(ValueError, why):
                serve_models.load(cfg, None)
        with self.assertRaisesRegex(ValueError, "q9"):
            serve_models.load(config(), None, "q9")

    def test_resolve(self):
        m, _ = serve_models.load(config(), None)
        self.assertEqual([m.resolve(x) for x in ("q2", "qwen-q2", "fast", "qwen-q3", "q3")],
                         ["q2", "q2", "q2", "q3", "q3"])
        self.assertIsNone(m.resolve("claude-sonnet"), "an unknown name: the loaded variant")
        self.assertIsNone(m.resolve(None))
        m.unknown = "default"
        self.assertEqual(m.resolve("claude-sonnet"), "q2")
        m.unknown = "error"
        with self.assertRaises(serve_models.UnknownModel):
            m.resolve("claude-sonnet")
        self.assertIsNone(m.resolve(None), "a request that names no model is not refused")


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.lock = threading.Lock()
        self.m, _ = serve_models.load(config(model_min_resident_s=30, model_max_wait_s=120), None)
        self.m.clock = self.clock
        self.m.since = 0.0
        self.engine_up = True
        self.m.loaded = lambda: self.engine_up

    def sched(self, prio=None, hold_s=0.0):
        s = TurnScheduler(self.lock, 0.0, 0.0, hold_s, (), 0.0, clock=self.clock, prio=prio)
        s.models = self.m
        return s

    def take(self, s, *tickets):
        for t in tickets:
            if s.acquire(t, 0):
                return t
        return None

    def switch(self, s, name):
        """What the server does once a request for another variant has the turn."""
        self.m.switched(name, 1.0)
        s.model_switched()

    def test_loaded_variant_first(self):
        self.clock.t = 100.0
        s = self.sched()
        other = s.enter(Turn(model="q3"))
        self.clock.t += 1
        same = s.enter(Turn(model="q2"))
        anyone = s.enter(Turn())
        self.assertIsNone(self.take(s, other), "q3 waits while requests for q2 do")
        self.assertIs(self.take(s, same), same)
        s.release(1)
        self.assertIs(self.take(s, anyone), anyone, "a request that names no variant takes the loaded one")
        s.release(1)
        self.assertIs(self.take(s, other), other, "nobody waits for q2: q3's turn")
        self.assertEqual(self.m.committed, "q3")

    def test_min_resident(self):
        s = self.sched()
        self.clock.t = 10.0
        other = s.enter(Turn(model="q3"))
        self.assertIsNone(self.take(s, other), "q2 was loaded 10 s ago: it keeps the engine for 30")
        self.clock.t = 30.0
        self.assertIs(self.take(s, other), other)

    def test_unloaded_engine_switches_at_once(self):
        s = self.sched()
        self.clock.t = 1.0
        self.engine_up = False
        other = s.enter(Turn(model="q3"))
        self.assertIs(self.take(s, other), other)

    def test_max_wait_commits(self):
        self.clock.t = 100.0
        s = self.sched()
        other = s.enter(Turn(model="q3"))
        busy = [s.enter(Turn(model="q2")) for _ in range(3)]
        self.assertIs(self.take(s, *busy), busy[0])
        s.release(1)
        self.clock.t += 120
        late = s.enter(Turn(model="q2"))
        self.assertIsNone(self.take(s, late, *busy[1:]), "q3 has waited 120 s: q2's requests wait for the switch")
        self.assertIs(self.take(s, other), other)
        self.switch(s, "q3")
        s.release(1)
        self.assertIsNone(self.take(s, late), "q3 is loaded now and has its minimum time")

    def test_queue_follows_the_switch(self):
        self.clock.t = 100.0
        s = self.sched()
        q3 = [s.enter(Turn(model="q3")) for _ in range(3)]
        self.assertIs(self.take(s, *q3), q3[0])
        self.switch(s, "q3")
        s.release(1)
        q2 = s.enter(Turn(model="q2"))
        self.clock.t += 40                               # past q3's minimum: only its queue keeps it
        for t in q3[1:]:
            self.assertIsNone(self.take(s, q2))
            self.assertIs(self.take(s, t), t)
            s.release(1)
        self.assertIs(self.take(s, q2), q2)

    def test_committed_target_left(self):
        self.clock.t = 100.0
        s = self.sched()
        mine = s.enter(Turn(model="q2"))
        self.assertIs(self.take(s, mine), mine)
        other = s.enter(Turn(model="q3"))
        self.clock.t += 120
        nxt = s.enter(Turn(model="q2"))
        s.release(1)
        self.assertIsNone(self.take(s, nxt))
        self.assertEqual(self.m.committed, "q3")
        s.leave(other)
        self.assertIs(self.take(s, nxt), nxt, "the switch is off once nobody wants it")
        self.assertIsNone(self.m.committed)

    def test_better_band_switches_without_preempting(self):
        self.clock.t = 100.0
        s = self.sched(Priorities(aging_s=0.0, reserve_s=0.0, min_run_s=0.0))
        low = s.enter(Turn(model="q2", priority=10, session_key="low"))
        self.assertIs(self.take(s, low), low)
        more = s.enter(Turn(model="q2", priority=10, session_key="low"))
        urgent = s.enter(Turn(model="q3", priority=0, session_key="urgent"))
        self.assertFalse(low.preempt.is_set(), "the running request is not stopped for another variant")
        s.release(1)
        self.assertIsNone(self.take(s, more), "a band better: q3 goes first although q2 has a request waiting")
        self.assertIs(self.take(s, urgent), urgent)

    def test_model_preempt(self):
        self.clock.t = 100.0
        self.m.preempt = True
        s = self.sched(Priorities(aging_s=0.0, reserve_s=0.0, min_run_s=0.0))
        low = s.enter(Turn(model="q2", priority=10, session_key="low"))
        self.assertIs(self.take(s, low), low)
        s.enter(Turn(model="q3", priority=0, session_key="urgent"))
        self.assertTrue(low.preempt.is_set())

    def test_hold_keeps_the_turn_for_the_follow_up(self):
        self.clock.t = 100.0
        s = self.sched(hold_s=3.0)
        s.slice_s = 30.0
        first = s.enter(Turn(model="q2"))
        self.assertIs(self.take(s, first), first)
        first.turn.issued.add("call_1")
        other = s.enter(Turn(model="q3"))
        s.release(1)
        self.assertIsNone(self.take(s, other), "the turn is held for q2's follow-up")
        follow = s.enter(Turn(frozenset({"call_1"}), model="q2"))
        self.assertIs(self.take(s, follow), follow)

    def test_scheduler_on_with_variants_alone(self):
        s, note = turns_from_config({}, self.lock, models=True)
        self.assertIsInstance(s, TurnScheduler)
        self.assertIsNone(s.prio, "no slice and no priorities asked: arrival order")
        self.assertIn("variants", note)
        self.assertEqual(turns_from_config({}, self.lock), (None, None))


class ServerTest(unittest.TestCase):
    """Requests for two variants through the HTTP server, with an engine that answers with its variant's name."""

    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import IM_END, ByteTokenizer, Service, serve

        tok = self.tok = ByteTokenizer()
        self.dir = tempfile.TemporaryDirectory()
        events = self.events = []

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
                if action == "save":
                    Path(path).write_text(self.spawn[0])
                return {"tokens": 7}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                yield from tok.encode(f"</think>\nfrom {self.spawn[0]}", parse_special=True)
                yield from tok.encode(IM_END, parse_special=True)

        m, _ = serve_models.load(config(model_session_dir=self.dir.name, model_min_resident_s=0), None)

        def spawn(name):                                # StrataEngine.spawn's shape: (exe, args, cwd, log, env, ...)
            return (name, m.variants[name].cfg["args"], None, None, None, False, None)
        for v in m.variants.values():
            v.spawn = spawn(v.name)
        self.engine = Engine()
        svc = self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"),
                                 model_name="qwen-q2")
        svc.models = m
        m.loaded = svc.loaded
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

    def chat(self, model):
        r = self.call("/v1/chat/completions", {"model": model, "max_tokens": 50,
                                               "messages": [{"role": "user", "content": "hi"}]})
        return r["model"], r["choices"][0]["message"]["content"].strip()

    def test_switch_and_back(self):
        self.assertEqual(self.chat("qwen-q2"), ("qwen-q2", "from q2"))
        self.assertEqual(self.events, [], "q2 is loaded: no switch")
        self.assertEqual(self.chat("q3"), ("q3", "from q3"))
        self.assertEqual(self.events, [("save", "variant-q2.session"), ("unload", "q2"), ("start", "q3")])
        self.assertEqual(self.svc.model, "qwen-q3")
        self.assertEqual(self.chat("claude-sonnet"), ("qwen-q3", "from q3"), "an unknown name: the loaded one")
        self.events.clear()
        self.assertEqual(self.chat("fast"), ("fast", "from q2"))
        self.assertEqual(self.events, [("save", "variant-q3.session"), ("unload", "q3"), ("start", "q2"),
                                       ("restore", "variant-q2.session")], "q2's conversation comes back")
        self.assertFalse(Path(self.dir.name, "variant-q2.session").exists(), "a restored session file goes")
        self.assertEqual(self.svc.models.switches, 2)

    def test_models_listing(self):
        data = self.call("/v1/models")["data"]
        ids = {d["id"]: d for d in data}
        self.assertEqual(ids["qwen-q2"]["status"]["value"], "loaded")
        self.assertEqual(ids["qwen-q3"]["status"]["value"], "unloaded")
        self.assertEqual(ids["fast"]["alias_of"], "qwen-q2")
        self.assertEqual(ids["q3"]["alias_of"], "qwen-q3")
        self.assertEqual(ids["qwen-q3"]["meta"]["n_ctx"], 200000)
        status = self.call("/v1/status")
        self.assertEqual(status["models"]["active"], "q2")

    def test_load_switches(self):
        r = self.call("/v1/load", {"model": "qwen-q3"})
        self.assertEqual(r["status"], "loaded")
        self.assertEqual(self.svc.models.active, "q3")
        self.assertEqual(self.call("/health")["variant"], "q3")

    def test_default_when_idle(self):
        m = self.svc.models
        m.idle_s = 60
        self.chat("q3")
        self.assertFalse(self.svc.default_if_idle(), "a request just ran")
        self.svc.last_request_at = time.time() - 61
        self.events.clear()
        self.assertTrue(self.svc.default_if_idle())
        self.assertEqual(m.active, "q2")
        self.assertEqual(self.events[-2:], [("start", "q2"), ("restore", "variant-q2.session")])
        self.assertFalse(self.svc.default_if_idle(), "already the default")

    def test_default_when_idle_and_unloaded(self):
        m = self.svc.models
        m.idle_s = 60
        self.chat("q3")
        self.engine.unload()
        self.svc.last_request_at = time.time() - 61
        self.events.clear()
        self.assertTrue(self.svc.default_if_idle())
        self.assertEqual((m.active, self.engine.spawn[0]), ("q2", "q2"))
        self.assertEqual(self.events, [], "an unloaded engine is only pointed at the default")
        self.assertEqual(self.chat("q2"), ("q2", "from q2"))
        self.assertEqual(self.events, [("start", "q2"), ("restore", "variant-q2.session")])


if __name__ == "__main__":
    unittest.main()
