"""serve/test_engine_fields.py - "engine" / "engine_env" fields of the run config, and in model variants' files.

    python -m unittest serve.test_engine_fields -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from serve import engine_fields
from serve import models


class ExpandTests(unittest.TestCase):
    def test_fields_become_args_and_env(self):
        cfg = engine_fields.expand({
            "log": "/data/x.log",
            "args": ["--kv", "int8"],
            "env": {"HIP_FOO": "bar"},
            "engine": {"conversation_cache_mib": 16384, "spec_min_p": 0.7, "resident_experts": True,
                       "vision": False, "prefill": None},
            "engine_env": {"load_image": "/data/load-image/q2", "unbuffered_load": True, "off": False}})
        self.assertEqual(cfg, {
            "log": "/data/x.log",
            "args": ["--kv", "int8", "--conversation-cache-mib", "16384", "--spec-min-p", "0.7", "--resident-experts"],
            "env": {"HIP_FOO": "bar", "STRATA_LOAD_IMAGE": "/data/load-image/q2", "STRATA_UNBUFFERED_LOAD": "1"}})

    def test_without_fields_the_config_is_untouched(self):
        cfg = {"args": ["--kv", "int8"]}
        self.assertIs(engine_fields.expand(cfg), cfg)

    def test_option_set_twice_is_refused(self):
        with self.assertRaisesRegex(ValueError, "--kv"):
            engine_fields.expand({"args": ["--kv", "int8"], "engine": {"kv": "int8"}})
        with self.assertRaisesRegex(ValueError, "STRATA_LOAD_IMAGE"):
            engine_fields.expand({"env": {"STRATA_LOAD_IMAGE": "a"}, "engine_env": {"load_image": "b"}})

    def test_list_value_is_refused(self):
        with self.assertRaisesRegex(ValueError, "engine.layer_split"):
            engine_fields.expand({"engine": {"layer_split": [1, 2]}})

    def test_variant_fields_replace_the_common_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "server-a.json").write_text(json.dumps(
                {"engine": {"pack": "/data/pack-a", "expert_cache": 100}, "engine_env": {"load_image": "/li/a"},
                 "sampling": {"temperature": 0.7}}))
            common = engine_fields.expand({"models": {"a": "server-a.json"}, "engine": {"expert_cache": 50, "spec": 4},
                                           "sampling": {"temperature": 1.0, "top_k": 20}})
            m, cfg = models.load(common, str(Path(tmp) / "server.json"))
        self.assertEqual(cfg["args"], ["--spec", "4", "--pack", "/data/pack-a", "--expert-cache", "100"])
        self.assertEqual(cfg["env"], {"STRATA_LOAD_IMAGE": "/li/a"})
        self.assertEqual(cfg["sampling"], {"temperature": 0.7, "top_k": 20})


if __name__ == "__main__":
    unittest.main()
