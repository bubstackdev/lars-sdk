import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lars.backends.flash import FlashBackend
from lars.train.accuracy import capture, load_capture
from lars.train.data import Record, write_records
from sweep.accuracy_ladder import digest
from sweep.flash_prompt_investigation import investigate
from training.tests.test_flash_backend import MOCK


@unittest.skipUnless(MOCK.exists(), 'build CPU mock worker')
class FlashPromptInvestigationTests(unittest.TestCase):
    def test_worker_rotation_never_overlaps_or_changes_gold_and_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prior = root / 'prior'; prior.mkdir()
            model = root / 'model'; model.write_bytes(b'fixture')
            template = root / 'template'; template.write_text('{{ messages[0].content }}\n{{ messages[1].content }}\n')
            records = [Record(str(i), 'fixture', 'noul', 'fixture state', 'True?', ['yes', 'no'], [.7, .3]) for i in range(101)]
            write_records(records, prior / 'pilot.records.jsonl')
            (prior / 'pilot.wire.jsonl').write_text('fixture wire retained by hash')
            with FlashBackend(MOCK, mock=True, template_path=template) as backend, contextlib.redirect_stdout(io.StringIO()):
                capture(backend, records, prior / 'raw.jsonl')
            original = load_capture(prior / 'raw.jsonl')
            for row in original.values():
                row.update(hard_index=1, keys=['yes', 'no'])  # Hard gold intentionally differs from soft argmax.
            (prior / 'capture.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in original.values()))
            (prior / 'plan.json').write_text(json.dumps({'model': str(model), 'template': str(template),
                'model_bytes': model.stat().st_size, 'model_mtime_ns': model.stat().st_mtime_ns,
                'context': 8192, 'template_sha256': digest(template),
                'pilot_records_sha256': digest(prior / 'pilot.records.jsonl'),
                'pilot_wire_sha256': digest(prior / 'pilot.wire.jsonl')}))
            workers = []

            def factory(worker, model, template, **kwargs):
                if kwargs.get('allow_gpu'):
                    self.assertTrue(all(w._closed for w in workers), 'GPU worker instances overlap')
                b = FlashBackend(MOCK, mock=True, template_path=template, n_ctx=kwargs['n_ctx'])
                if kwargs.get('allow_gpu'):
                    workers.append(b)
                return b

            args = SimpleNamespace(allow_gpu=True, prior=prior, out=root / 'out', worker=MOCK,
                student=prior / 'capture.jsonl')
            with patch('sweep.flash_prompt_investigation.FlashBackend', side_effect=factory), contextlib.redirect_stdout(io.StringIO()):
                investigate(args)
            self.assertEqual(len(workers), 2)
            self.assertTrue(all(w._closed for w in workers))
            scored = load_capture(args.out / 'canonical_spaced.jsonl')
            self.assertEqual(scored, original)
            for name in ['native_bare', 'native_spaced']:
                values = load_capture(args.out / f'{name}.jsonl')
                self.assertEqual(values.keys(), original.keys())
                for row in values.values():
                    self.assertEqual(row['hard_index'], 1)
                    self.assertEqual(row['target'], [.7, .3])
            result = json.loads((args.out / 'comparison.json').read_text())
            self.assertEqual(result['metadata']['canonical_label_flips'], 0)
            self.assertEqual(result['metadata']['canonical_max_probability_delta'], 0)
