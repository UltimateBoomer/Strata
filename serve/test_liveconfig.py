"""serve/test_liveconfig.py - the Config view: the run config's files read, checked, written and applied while the server
runs (GET / POST /api/config, Service.apply_config), with an engine that records its starts.

    python -m unittest serve.test_liveconfig -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import models as serve_models
from serve.turns import turns_from_config

MAIN = {"exe": "/bin/engine", "tokenizer": "/tok", "log": "LOG", "turn_slice_s": 300, "priority": True,
        "models": {"q2": "server-q2.json", "q3": "server-q3.json"}, "model_min_resident_s": 0,
        "sampling": {"temperature": 1.0, "top_k": 20}, "engine": {"spec": 4}}
Q2 = {"model_name": "qwen-q2", "aliases": ["qwen"], "engine": {"pack": "/data/pack-q2"},
      "sampling": {"temperature": 0.6}}
Q3 = {"model_name": "qwen-q3", "engine": {"pack": "/data/pack-q3"}}


def fake_spawn(cfg, **_):
    return (cfg["exe"], list(cfg.get("args") or []), None, None, dict(cfg.get("env") or {}), False, None)


class LiveConfigTest(unittest.TestCase):
    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, Service, serve
        from serve import engine_fields

        self.dir = tempfile.TemporaryDirectory()
        root = self.root = Path(self.dir.name)
        main = {**MAIN, "log": str(root / "strata.log")}
        for name, cfg in (("server.json", main), ("server-q2.json", Q2), ("server-q3.json", Q3)):
            (root / name).write_text(json.dumps(cfg, indent=2))
        events = self.events = []
        self.fail_start = False
        test = self

        class Engine:
            max_context = 1 << 16

            def __init__(self):
                self.up, self.unloaded, self.spawn = True, False, None

            def alive(self):
                return self.up

            def unload(self):
                events.append(("unload", self.spawn[1]))
                self.up, self.unloaded = False, True

            def restart(self):
                events.append(("start", self.spawn[1]))
                if test.fail_start and "--bad" in self.spawn[1]:
                    raise RuntimeError("the engine exited before READY")
                self.up, self.unloaded = True, False

        self.patch = mock.patch("serve.server.engine_spawn", fake_spawn)
        self.patch.start()
        m, cfg = serve_models.load(engine_fields.expand(main), str(root / "server.json"))
        for v in m.variants.values():
            v.spawn = fake_spawn(v.cfg)
        self.engine = Engine()
        self.engine.spawn = m.variants["q2"].spawn
        svc = self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"),
                                 model_name="qwen-q2")
        svc.models, m.loaded = m, svc.loaded
        svc.apply_variant(m.variants["q2"])
        svc.turns, _ = turns_from_config(cfg, svc.fifo, models=True)
        svc.turns.models = m
        svc.config_path = str(root / "server.json")
        svc.config_history = root / "history"
        svc.live_engine = True
        self.httpd = serve(svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.patch.stop()
        self.dir.cleanup()

    def call(self, path, body=None):
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def files(self):
        return self.call("/api/config")[1]["files"]

    def post(self, files, **kw):
        return self.call("/api/config", {"files": files, **kw})

    def test_view_lists_main_and_variants(self):
        code, v = self.call("/api/config")
        self.assertEqual(code, 200)
        self.assertEqual(v["main"], "server.json")
        self.assertEqual(set(v["files"]), {"server.json", "server-q2.json", "server-q3.json"})
        self.assertEqual(v["files"]["server-q2.json"]["engine"], {"pack": "/data/pack-q2"})   # as on disk
        self.assertEqual(v["active"], "q2")

    def test_sampling_applies_without_a_restart(self):
        f = self.files()
        f["server-q2.json"]["sampling"] = {"temperature": 0.3, "top_p": 0.8}
        code, r = self.post(f)
        self.assertEqual(code, 200, r)
        self.assertEqual((r["applied"], r["engine_restart"], r["restart_required"]), (True, False, []))
        self.assertEqual(r["changes"], {"server-q2.json": ["sampling"]})
        self.assertEqual(self.svc.sampling_defaults, {"temperature": 0.3, "top_p": 0.8, "top_k": 20})
        self.assertEqual(self.events, [])
        self.assertEqual(json.loads((self.root / "server-q2.json").read_text())["sampling"]["top_p"], 0.8)
        self.assertEqual(json.loads((Path(r["backup"]) / "server-q2.json").read_text())["sampling"], {"temperature": 0.6})

    def test_loaded_variants_engine_change_restarts_it(self):
        f = self.files()
        f["server-q2.json"]["engine"]["expert_cache"] = 4000
        code, r = self.post(f)
        self.assertEqual(code, 200, r)
        self.assertTrue(r["engine_restarted"])
        self.assertEqual([e[0] for e in self.events], ["unload", "start"])
        self.assertIn("--expert-cache", self.engine.spawn[1])

    def test_other_variants_change_waits_for_its_switch(self):
        f = self.files()
        f["server-q3.json"]["engine"]["expert_cache"] = 3000
        f["server.json"]["model_max_wait_s"] = 30
        code, r = self.post(f)
        self.assertEqual(code, 200, r)
        self.assertFalse(r["engine_restart"])
        self.assertEqual(self.events, [])
        self.assertIn("--expert-cache", self.svc.models.variants["q3"].spawn[1])
        self.assertEqual(self.svc.models.max_wait_s, 30)

    def test_scheduler_keys_apply_in_place(self):
        f = self.files()
        f["server.json"]["turn_slice_s"] = 60
        f["server.json"]["priority_preempt"] = "off"
        turns = self.svc.turns
        code, r = self.post(f)
        self.assertEqual(code, 200, r)
        self.assertIs(self.svc.turns, turns)
        self.assertEqual((turns.slice_s, turns.policy.slice_s, turns.prio.preempt), (60, 60, "off"))

    def test_other_keys_need_a_server_restart(self):
        f = self.files()
        f["server.json"]["tokenizer"] = "/other"
        code, r = self.post(f)
        self.assertEqual(code, 200, r)
        self.assertEqual(r["restart_required"], ["tokenizer"])
        self.assertEqual(json.loads((self.root / "server.json").read_text())["tokenizer"], "/other")

    def test_bad_config_changes_nothing(self):
        f = self.files()
        f["server-q2.json"]["sampling"] = {"top_k": 500}
        before = (self.root / "server-q2.json").read_text()
        code, r = self.post(f)
        self.assertEqual(code, 400)
        self.assertIn("top_k", r["error"]["message"])
        self.assertEqual((self.root / "server-q2.json").read_text(), before)
        f = self.files()
        f["server-q2.json"]["tokenizer"] = "/x"
        code, r = self.post(f)
        self.assertEqual(code, 400)
        self.assertIn("cannot differ between variants", r["error"]["message"])

    def test_failed_start_puts_the_earlier_config_back(self):
        self.fail_start = True
        f = self.files()
        f["server-q2.json"]["engine"]["bad"] = True
        f["server-q2.json"]["sampling"] = {"temperature": 0.1}
        before = (self.root / "server-q2.json").read_text()
        code, r = self.post(f)
        self.assertEqual(code, 500)
        self.assertEqual(r["error"]["code"], "apply_failed")
        self.assertEqual((self.root / "server-q2.json").read_text(), before)
        self.assertNotIn("--bad", self.engine.spawn[1])
        self.assertEqual(self.svc.sampling_defaults["temperature"], 0.6)
        self.assertEqual([e[0] for e in self.events], ["unload", "start", "start"])
        self.assertTrue(self.engine.up)

    def test_dry_run_and_new_variant(self):
        f = self.files()
        f["server.json"]["models"]["q1"] = "server-q1.json"
        f["server-q1.json"] = {"model_name": "qwen-q1", "engine": {"pack": "/data/pack-q1"}}
        code, r = self.post(f, dry_run=True)
        self.assertEqual((code, r["applied"]), (200, False))
        self.assertFalse((self.root / "server-q1.json").exists())
        code, r = self.post(f)
        self.assertEqual(code, 200, r)
        self.assertIn("q1", self.svc.models.variants)
        self.assertTrue((self.root / "server-q1.json").exists())
        code, r = self.post({"server.json": f["server.json"]})
        self.assertEqual(code, 400)                     # a variant file the main one names was not sent

    def test_root_serves_the_console_not_the_chat(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as resp:
            page = resp.read().decode()
        self.assertIn("web/console.js", page)
        self.assertNotIn('id="composer"', page)
        self.assertIn('id="view-config"', page)

    def test_busy_turn_answers_409(self):
        f = self.files()
        f["server-q2.json"]["sampling"] = {"temperature": 0.2}
        self.svc.fifo.acquire()
        try:
            code, r = self.post(f, wait_s=0.1)
        finally:
            self.svc.fifo.release()
        self.assertEqual(code, 409)
        self.assertEqual(json.loads((self.root / "server-q2.json").read_text())["sampling"], {"temperature": 0.6})


if __name__ == "__main__":
    unittest.main()
