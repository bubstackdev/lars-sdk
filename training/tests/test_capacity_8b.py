import json
from pathlib import Path
import tempfile
import unittest

from sweep.capacity_8b import probe_projection, worker


class Capacity8BTests(unittest.TestCase):
    def test_projection_discards_initial_compilation_time(self):
        log = 'train 39838 presentations, 4800 steps per epoch\n' + '\n'.join(json.dumps(row) for row in [
            {"progress_step": 10, "minutes": 2.},
            {"progress_step": 30, "minutes": 4.},
            {"progress_step": 50, "minutes": 6.}])
        result = probe_projection(log)
        self.assertEqual(result["seconds_per_step"], 6.)
        self.assertEqual(result["projected_training_hours"], 8.)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            probe_projection('train 4800 steps per epoch')

    def test_rebuilt_pack_identifies_8b_and_q3_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'model.gguf').write_bytes(b'fixture model bytes')
            (p / 'cal.json').write_text(json.dumps({"temperatures": {"choice": 1.}}))
            worker({"operation": "emit-8b", "model": str(p / 'model.gguf'),
                    "calibration": str(p / 'cal.json'), "output": str(p / 'pack')})
            pack = json.loads((p / 'pack/pack.json').read_text())
            self.assertEqual(pack['model']['base'], 'Qwen/Qwen3-8B')
            self.assertEqual(pack['model']['quantization'], 'Q3_K_M')
            self.assertEqual(pack['template']['model'], 'Qwen/Qwen3-8B')
            cal = json.loads((p / 'pack/calibration.json').read_text())
            self.assertEqual(cal['pack']['model_sha256'], pack['model']['sha256'])


if __name__ == '__main__':
    unittest.main()
