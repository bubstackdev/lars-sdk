"""Failure-path tests for transactional, checksum-pinned model setup. No downloads."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class SetupTests(unittest.TestCase):
    def fixture(self, root):
        artifacts = root / "artifacts"; artifacts.mkdir()
        lock = {"format": 1, "pack_name": "test", "pack_version": "0", "repository": "unused/test",
                "revision": "pinned", "license": "MIT", "model_card": "https://example.invalid/model",
                "llama_cpp": "f"*40, "template": "chatml", "visual_max_tokens": 256, "artifacts": {}}
        for role in ["model", "projector"]:
            data = b"GGUF" + role.encode(); (artifacts / f"{role}.gguf").write_bytes(data)
            lock["artifacts"][role] = {"filename": f"{role}.gguf", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        path = root / "model.lock.json"; path.write_text(json.dumps(lock))
        return ["python3", str(Path(__file__).with_name("prepare-model.py")), "--lock", str(path),
                "--output", str(root / "pack"), "--artifacts-dir", str(artifacts)]

    def test_failed_second_artifact_never_publishes_partial_pack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = self.fixture(root)
            (root / "artifacts/projector.gguf").write_bytes(b"wrong")
            result = subprocess.run(args, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "pack").exists())
            self.assertFalse(list(root.glob(".lars-model-*")))
            self.assertEqual((root / "artifacts/model.gguf").read_bytes(), b"GGUFmodel")

    def test_existing_cache_is_verified_and_corruption_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = self.fixture(root)
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 0)
            (root / "pack/model.gguf").write_bytes(b"corrupt")
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertEqual((root / "pack/model.gguf").read_bytes(), b"corrupt")


if __name__ == "__main__": unittest.main()
