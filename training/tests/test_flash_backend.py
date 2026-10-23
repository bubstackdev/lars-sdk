import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lars.backends.flash import FlashBackend, FlashWorkerError
from lars.render import compose_prompt

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get('FLASH_TEST_BUILD', ROOT / 'target/flash-teacher'))
MOCK = BUILD / 'flash-teacher-mock'
RUNTIME = BUILD / 'flash-teacher'


@unittest.skipUnless(MOCK.exists(), 'build the CPU Flash teacher executable first')
class FlashBackendTests(unittest.TestCase):
    def backend(self, **kwargs):
        backend = FlashBackend(MOCK, mock=True, timeout=5, **kwargs)
        self.addCleanup(backend.close)
        return backend

    def test_mode_and_label_token_contract(self):
        b = self.backend()
        self.assertTrue(b.info['synthetic'])
        self.assertFalse(b.info['gpu_loaded'])
        self.assertTrue(b.is_single_token('A'))
        self.assertFalse(b.is_single_token('AA'))
        self.assertEqual(b.count_tokens('AB'), 1)
        self.assertEqual(b.tokenize('a\u2028b'), list('a\u2028b'.encode()))

    def test_cached_suffix_rows_equal_independent_full_passes(self):
        b = self.backend()
        cached = b.score('context:', ['question one', 'question two', ''], [('A', 'B')] * 3)
        self.assertEqual(b.last_stats['fallback_rows'], 0)
        self.assertTrue(b.score('context:', ['question one'], [('A', 'B')]))
        self.assertTrue(b.last_stats['prefix_reused'])
        full = [b.score('context:', [s], [('A', 'B')], use_cache=False)[0] for s in ['question one', 'question two', '']]
        for a, c in zip(cached, full, strict=True):
            np.testing.assert_array_equal(a, c)

    def test_token_boundary_merge_falls_back_and_restores_other_rows(self):
        b = self.backend()
        cached = b.score('A', ['B', 'x'], [('A', 'B')] * 2)
        self.assertEqual(b.last_stats['fallback_rows'], 1)
        full = b.score('A', ['B', 'x'], [('A', 'B')] * 2, use_cache=False)
        for a, c in zip(cached, full, strict=True):
            np.testing.assert_array_equal(a, c)

    def test_requested_label_order_is_preserved(self):
        b = self.backend()
        z = b.score('prefix', ['end'], [('A', 'B', 'C')])[0]
        reverse = b.score('prefix', ['end'], [('C', 'A', 'B')])[0]
        np.testing.assert_array_equal(reverse, z[[2, 0, 1]])

    def test_explicit_label_prefix_and_mixed_token_diagnostics(self):
        b = self.backend()
        spaced = b.score('prefix', ['end'], [('A', 'B')])[0]
        bare = b.score('prefix', ['end'], [('A', 'B')], label_prefix='')[0]
        mixed = b.score('prefix', ['end'], [('A', 'B', ' A', ' B')], label_prefix='')[0]
        np.testing.assert_array_equal(mixed[:2], bare)
        np.testing.assert_array_equal(mixed[2:], spaced)
        self.assertFalse(np.array_equal(bare, spaced))
        with self.assertRaisesRegex(ValueError, 'label_prefix'):
            b.score('prefix', ['end'], [('A', 'B')], label_prefix='invalid')
        with self.assertRaisesRegex(FlashWorkerError, 'label_prefix'):
            b._rpc('label_logits', prefix='prefix', suffixes=['end'], labels=[['A']], label_prefix='invalid')
        b.info.pop('label_prefixes')  # Older protocol-1 workers only implement space-prefixed labels.
        with self.assertRaisesRegex(FlashWorkerError, 'rebuild'):
            b.score('prefix', ['end'], [('A', 'B')], label_prefix='')
        np.testing.assert_array_equal(b.score('prefix', ['end'], [('A', 'B')])[0], spaced)

    def test_errors_recover_without_stale_cache(self):
        b = self.backend(n_ctx=32)
        b.score('prefix', ['x'], [('A', 'B')])
        for suffixes, labels in [(['x'], [('AA', 'B')]), (['x'], [('A', 'A')]), (['x'], []),
                                  (['x'], [()]), (['x' * 40], [('A', 'B')]), ([], [])]:
            with self.subTest(suffixes=suffixes), self.assertRaises(FlashWorkerError):
                b.score('prefix', suffixes, labels)
        z = b.score('prefix', ['valid'], [('A', 'B')])
        self.assertFalse(b.last_stats['prefix_reused'])
        np.testing.assert_array_equal(z[0], b.score('prefix', ['valid'], [('A', 'B')], use_cache=False)[0])

    def test_randomized_cache_and_full_pass_parity(self):
        b = self.backend()
        rng = random.Random(901)
        for _ in range(30):
            prefix = ''.join(rng.choices('AB xy\n', k=20))
            suffixes = [''.join(rng.choices('AB xy\n', k=10)) for _ in range(3)]
            labels = [('A', 'B', 'C')] * 3
            cached = b.score(prefix, suffixes, labels)
            full = b.score(prefix, suffixes, labels, use_cache=False)
            for a, c in zip(cached, full, strict=True):
                np.testing.assert_array_equal(a, c)

    def test_transport_recovers_from_malformed_json(self):
        r = subprocess.run([str(MOCK), '--mock'], input='broken json\n{"id":1,"op":"info"}\n{"id":2,"op":"quit"}\n',
                           capture_output=True, text=True, timeout=5, check=True)
        lines = [json.loads(l) for l in r.stdout.split('\n') if l]
        self.assertFalse(lines[1]['ok'])
        self.assertTrue(lines[2]['ok'])
        self.assertEqual(lines[3]['id'], 2)

    def test_capture_pipeline_keeps_gold_targets(self):
        from lars.train.accuracy import capture, load_capture, record_key
        from lars.train.data import Record
        rows = [Record('a', 'fixture', 'choice', {'text': 'same'}, 'Pick', ['x', 'y'], [.2, .8]),
                Record('b', 'fixture', 'noul', {'text': 'same'}, 'True?', ['yes', 'no'], [.7, .3]),
                Record('c', 'fixture', 'score', 'other', 'Rate', ['low', 'mid', 'high'], [.1, .6, .3])]
        b = self.backend()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.jsonl'
            capture(b, rows, path)
            scored = load_capture(path)
            self.assertEqual(len(scored), 3)
            for row in rows:
                value = scored[record_key(row)]
                self.assertEqual(value['target'], row.target)
                self.assertAlmostEqual(sum(value['p']), 1.)
            self.assertTrue(b.provenance()['worker']['synthetic'])

    def test_response_id_mismatch_and_timeout_close_only_own_worker(self):
        for behavior in ['wrong-id', 'timeout']:
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as directory:
                script = Path(directory) / 'fake-worker'
                body = ('import json,sys,time\n'
                        'print(json.dumps(dict(protocol=1,ok=True,mode="mock",synthetic=True,gpu_loaded=False)),flush=True)\n'
                        'for line in sys.stdin:\n')
                body += (' print(json.dumps(dict(id=999,ok=True,result={})),flush=True)\n' if behavior == 'wrong-id'
                         else ' time.sleep(.2)\n')
                script.write_text('#!' + sys.executable + '\n' + body)
                script.chmod(0o755)
                b = FlashBackend(script, mock=True, timeout=5)
                b._timeout = .1
                self.addCleanup(b.close)
                with self.assertRaisesRegex(FlashWorkerError, 'ID mismatch|timed out'):
                    b.tokenize('x')
                self.assertTrue(b._closed)
                self.assertIsNotNone(b._process.poll())

    def test_template_and_eof_close(self):
        b = self.backend()
        prefix, suffix = compose_prompt(b.template(), 'STATE', 'QUESTION')
        self.assertIn('STATE', prefix)
        self.assertIn('QUESTION', prefix + suffix)
        process = b._process
        b.close()
        self.assertEqual(process.returncode, 0)
        b.close()
        with self.assertRaisesRegex(FlashWorkerError, 'closed'):
            b.tokenize('x')

    def test_greedy_generation_limits_eos_and_clears_scoring_cache(self):
        b = self.backend()
        reference = b.score('prefix', ['question'], [('A', 'B')])[0]
        result = b.generate('generate:', max_new_tokens=64)
        self.assertEqual(result['text'], 'brief</think>\nFINAL: A')
        self.assertEqual(result['finish_reason'], 'eos')
        self.assertEqual(result['completion_tokens'], len(result['tokens']))
        short = b.generate('generate:', max_new_tokens=2)
        self.assertEqual(short['text'], 'br')
        self.assertEqual(short['finish_reason'], 'length')
        self.assertEqual(short['completion_tokens'], 2)
        for budget in [0, -1, 2049, 8192]:
            with self.assertRaisesRegex(FlashWorkerError, 'budget|context'):
                b.generate('generate:', max_new_tokens=budget)
        np.testing.assert_array_equal(b.score('prefix', ['question'], [('A', 'B')])[0], reference)
        self.assertFalse(b.last_stats['prefix_reused'])
        b.info.pop('generation')
        with self.assertRaisesRegex(FlashWorkerError, 'rebuild'):
            b.generate('generate:')


@unittest.skipUnless(os.environ.get('FLASH_TOKENIZER_MODEL') and RUNTIME.exists(), 'optional real CPU tokenizer check')
class RealFlashTokenizerTests(unittest.TestCase):
    def test_real_vocabulary_and_template_without_gpu(self):
        template = os.environ['FLASH_CHAT_TEMPLATE']
        with FlashBackend(RUNTIME, os.environ['FLASH_TOKENIZER_MODEL'], template, timeout=60) as b:
            self.assertEqual(b.info['mode'], 'tokenizer-only')
            self.assertFalse(b.info['gpu_loaded'])
            self.assertEqual(b.info['architecture'], 'qwen4exp')
            self.assertTrue(b.is_single_token('A'))
            self.assertTrue(b.is_single_token('B'))
            self.assertNotEqual(b.tokenize(' A', special=False), b.tokenize(' B', special=False))
            prefix, suffix = compose_prompt(b.template(), 'Test state', 'Choose A or B.\nAnswer:')
            self.assertGreater(b.count_tokens(prefix + suffix), 0)
            with self.assertRaisesRegex(FlashWorkerError, 'tokenizer-only'):
                b.label_logits(prefix, [suffix], [('A', 'B')])
            with self.assertRaisesRegex(FlashWorkerError, 'tokenizer-only'):
                b.generate(prefix + suffix)
            with self.assertRaisesRegex(FlashWorkerError, 'tokenizer-only'):
                b._rpc('generate', prompt=prefix + suffix, max_new_tokens=8)
            # The native worker also refuses scoring if a caller bypasses Python's gate.
            with self.assertRaisesRegex(FlashWorkerError, 'tokenizer-only'):
                b._rpc('label_logits', prefix=prefix, suffixes=[suffix], labels=[['A', 'B']])
            self.assertFalse(b._rpc('info')['gpu_loaded'])
            from tokenizers import Tokenizer
            reference = Tokenizer.from_file(str(Path(template).parent / 'tokenizer.json'))
            for text in [' A', ' B', 'hello world', 'a\u2028b', prefix + suffix]:
                self.assertEqual(b.tokenize(text), reference.encode(text, add_special_tokens=False).ids)



if __name__ == '__main__':
    unittest.main()
