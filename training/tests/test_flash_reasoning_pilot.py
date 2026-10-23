import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from sweep.flash_reasoning_pilot import choose, parse_answer, report


class FlashReasoningPilotTests(unittest.TestCase):
    def test_parse_requires_completed_reasoning_and_one_valid_final_label(self):
        def parse(text, finish='eos', thinking=True):
            return parse_answer({'text': text, 'finish_reason': finish}, ['A', 'AA'], thinking)
        self.assertEqual(parse('brief reasoning</think>\nFINAL: AA')['index'], 1)
        self.assertEqual(parse('FINAL: A', thinking=False)['index'], 0)
        for text in ['FINAL: A', '</think>\nFINAL: A', 'x</think>\nFINAL: B',
                     'x</think>\nFINAL: A\nFINAL: AA', 'x</think>\nA', 'x</think>\nFINAL: A\nextra']:
            self.assertFalse(parse(text)['valid'])
        self.assertFalse(parse('x</think>\nFINAL: A', finish='length')['valid'])

    def test_selection_ignores_gold_and_keeps_states_disjoint(self):
        rows = [{'key': str(i), 'source': str(i%2), 'state_key': str(i//2), 'hard_index': i%2} for i in range(8)]
        a = choose(rows, 3, set())
        b = choose([row | {'hard_index': 1-row['hard_index']} for row in reversed(rows)], 3, set())
        self.assertEqual([r['key'] for r in a], [r['key'] for r in b])
        self.assertEqual(len({r['state_key'] for r in a}), 3)

    def test_report_counts_truncation_wrong_and_retains_incomplete_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cases = [{'key': str(i), 'stratum': 'disagreement', 'hard_index': 1,
                      'student_index': 0, 'flash_direct_index': 1} for i in range(2)]
            rows = [{'key': '0', 'arm': arm, 'synthetic': False, 'seconds': 1,
                     'result': {'completion_tokens': 3}, 'parsed': parsed}
                    for arm, parsed in [('generated_direct', {'valid': True, 'index': 1}),
                                        ('reasoning', {'valid': False, 'index': None})]]
            (out/'cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases))
            (out/'capture.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                report(SimpleNamespace(out=out))
            value = json.loads((out/'comparison.json').read_text())
            self.assertEqual(value['incomplete_cases'], 1)
            self.assertEqual(value['groups']['all']['reasoning']['accuracy'], 0)
            self.assertEqual(value['groups']['all']['generated_direct']['accuracy'], 1)
            rows[0]['synthetic'] = True
            (out/'capture.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'synthetic'):
                report(SimpleNamespace(out=out))
