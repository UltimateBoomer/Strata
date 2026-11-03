import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import engine_config  # noqa: E402


class ExpandTests(unittest.TestCase):
    def test_fields_become_args_and_env(self):
        cfg = engine_config.expand({
            'log': '/data/x.log',
            'args': ['--kv', 'int8'],
            'env': {'HIP_FOO': 'bar'},
            'engine': {'conversation_cache_mib': 16384, 'spec_min_p': 0.7, 'resident_experts': True,
                       'vision': False, 'prefill': None},
            'engine_env': {'load_image': '/data/load-image/q2', 'unbuffered_load': True, 'off': False}})
        self.assertEqual(cfg, {
            'log': '/data/x.log',
            'args': ['--kv', 'int8', '--conversation-cache-mib', '16384', '--spec-min-p', '0.7', '--resident-experts'],
            'env': {'HIP_FOO': 'bar', 'STRATA_LOAD_IMAGE': '/data/load-image/q2', 'STRATA_UNBUFFERED_LOAD': '1'}})

    def test_option_set_twice_is_refused(self):
        with self.assertRaisesRegex(ValueError, '--kv'):
            engine_config.expand({'args': ['--kv', 'int8'], 'engine': {'kv': 'int8'}})
        with self.assertRaisesRegex(ValueError, 'STRATA_LOAD_IMAGE'):
            engine_config.expand({'env': {'STRATA_LOAD_IMAGE': 'a'}, 'engine_env': {'load_image': 'b'}})

    def test_list_value_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'engine.layer_split'):
            engine_config.expand({'engine': {'layer_split': [1, 2]}})

    def test_checked_in_configs_expand(self):
        config = Path(__file__).parents[1] / 'config'
        main = json.loads((config / 'server.json').read_text())
        self.assertIn('--max-context', engine_config.expand(main)['args'])
        for variant in main['models'].values():
            self.assertIn('--pack', engine_config.expand(json.loads((config / variant).read_text()))['args'])

    def test_each_variant_has_its_own_sampling(self):
        """Sampling is per model: each Qwen variant's file holds its whole set, the common file none. A variant of
        another family (its own "arch") sets none: the server applies that family's model-card sampling."""
        config = Path(__file__).parents[1] / 'config'
        main = json.loads((config / 'server.json').read_text())
        self.assertNotIn('sampling', main)
        for variant in main['models'].values():
            own = json.loads((config / variant).read_text())
            if 'arch' in own:
                self.assertNotIn('sampling', own, variant)
                continue
            self.assertTrue({'temperature', 'top_p', 'top_k'} <= set(own.get('sampling') or {}), variant)

    def test_fast_and_slow_are_one_engine(self):
        """q3s and q4 come as -slow and -fast: the same engine (they differ only in model_name and sampling, so the
        fork switches between them without a restart), -fast with residency-biased routing on and the other answer
        trades a client's, -slow letting a request change nothing."""
        config = Path(__file__).parents[1] / 'config'
        main = json.loads((config / 'server.json').read_text())
        for base in ('q3s', 'q4'):
            with self.subTest(base=base):
                slow = json.loads((config / main['models'][f'{base}-slow']).read_text())
                fast = json.loads((config / main['models'][f'{base}-fast']).read_text())
                server_only = {'model_name', 'aliases', 'sampling', 'log'}
                self.assertEqual({k: v for k, v in slow.items() if k not in server_only},
                                 {k: v for k, v in fast.items() if k not in server_only})
                self.assertEqual(engine_config.expand(slow)['env']['STRATA_ROUTE_LIVE'], '1')
                self.assertEqual(slow['model_name'] + '', fast['model_name'].replace('-fast', '-slow'))
                self.assertNotIn('strata_tune', slow['sampling'])
                self.assertEqual(slow['sampling']['client_tune'], [])
                self.assertEqual(fast['sampling']['strata_tune'], {'route_prior': 1.5})
                self.assertEqual(fast['sampling']['client_tune'],
                                 ['tail_skip', 'kva', 'route_resident', 'route_resident_ranks'])
                rest = {k: v for k, v in fast['sampling'].items() if k not in ('strata_tune', 'client_tune')}
                self.assertEqual(rest, {k: v for k, v in slow['sampling'].items() if k != 'client_tune'},
                                 'the same sampling otherwise')

    def test_deepseek_is_a_variant_of_its_own_family(self):
        """DeepSeek runs as a variant of the one server: its own engine, tokenizer and options, none of Qwen's."""
        config = Path(__file__).parents[1] / 'config'
        main = json.loads((config / 'server.json').read_text())
        self.assertEqual(main['models']['deepseek'], 'server-deepseek.json')
        self.assertEqual(next(iter(main['models'])), 'q2', 'the server starts with the first variant')
        ds = engine_config.expand(json.loads((config / 'server-deepseek.json').read_text()))
        self.assertEqual(ds['arch'], 'deepseek41')
        self.assertEqual(ds['exe'], '/opt/strata/engine/strata-dsv41')
        self.assertEqual(ds['args'][ds['args'].index('--max-context') + 1], '200000')
        self.assertFalse({'cwd', 'backend', 'vision'} & set(ds), 'the common config\'s keys')


if __name__ == '__main__':
    unittest.main()
