import json
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sweep.flash_teacher_capture import score_plan, report


class FlashTeacherPlanTests(unittest.TestCase):
    def test_capture_requires_explicit_gpu_opt_in_before_touching_files(self):
        with self.assertRaisesRegex(ValueError, '--allow-gpu'):
            score_plan(SimpleNamespace(allow_gpu=False, plan=Path('/missing')))

    def test_changed_examples_are_rejected_before_any_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'model.gguf').write_bytes(b'x')
            (p / 'pilot.records.jsonl').write_text('changed')
            (p / 'pilot.wire.jsonl').write_text('changed')
            meta = {'model': str(p / 'model.gguf'), 'model_bytes': 1,
                    'model_mtime_ns': (p / 'model.gguf').stat().st_mtime_ns,
                    'worker': '/not-launched', 'template': '/not-loaded',
                    'pilot_records_sha256': 'expected-original', 'pilot_wire_sha256': 'expected-original'}
            (p / 'plan.json').write_text(json.dumps(meta))
            with patch('sweep.flash_teacher_capture.FlashBackend') as backend:
                with self.assertRaisesRegex(ValueError, 'examples or labels changed'):
                    score_plan(SimpleNamespace(allow_gpu=True, plan=p))
                backend.assert_not_called()


    def test_report_preserves_hard_gold_independently_of_soft_target(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            row = {'key': 'one', 'state_key': 'state', 'source': 'source', 'kind': 'choice',
                   'keys': ['a', 'b'], 'target': [.7, .3], 'hard_index': 1, 'p': [.9, .1]}
            teacher = row | {'p': [.2, .8]}
            (p / 'student.jsonl').write_text(json.dumps(row) + '\n')
            (p / 'capture.jsonl').write_text(json.dumps(teacher) + '\n')
            (p / 'capture-metadata.json').write_text(json.dumps({'provenance': {'worker': {'synthetic': False}},
                'model_load_seconds': 1., 'capture_seconds': 2.}))
            with contextlib.redirect_stdout(io.StringIO()):
                report(SimpleNamespace(plan=p, student=p / 'student.jsonl'))
            result = json.loads((p / 'comparison.json').read_text())
            self.assertEqual(result['macro_accuracy_delta'], 1.)
            row['hard_index'] = 0
            (p / 'student.jsonl').write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'gold labels differ'):
                report(SimpleNamespace(plan=p, student=p / 'student.jsonl'))

    def test_report_refuses_mock_predictions_as_quality_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'capture-metadata.json').write_text(json.dumps({'provenance': {'worker': {'synthetic': True}}}))
            with self.assertRaisesRegex(ValueError, 'synthetic CPU predictions'):
                report(SimpleNamespace(plan=p, student=p / 'not-read'))


if __name__ == '__main__':
    unittest.main()
