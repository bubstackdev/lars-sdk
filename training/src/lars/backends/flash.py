"""Persistent Flash worker bridge. GPU loading is explicitly opt-in; no model training."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time

import numpy as np

from lars.backends.base import Backend


class FlashWorkerError(RuntimeError):
    pass


class FlashBackend(Backend):
    name = "flash"

    def __init__(self, worker: str | Path, model: str | Path | None = None,
                 template_path: str | Path | None = None, *, allow_gpu: bool = False,
                 mock: bool = False, n_ctx: int = 8192, timeout: float = 120.) -> None:
        if mock and allow_gpu:
            raise ValueError("mock and GPU modes are mutually exclusive")
        if not mock and (model is None or template_path is None):
            raise ValueError("real Flash requires model and its chat template")
        self._lock = threading.RLock()
        self._timeout = timeout
        self._id = 0
        self._buffer = b""
        self._labels: dict[str, int | None] = {}
        self._closed = False
        self._process = None
        self._selector = selectors.DefaultSelector()
        self._stderr = tempfile.TemporaryFile()
        self._template_source = Path(template_path).read_text() if template_path else "{{ messages[0].content }}\n{{ messages[1].content }}"
        from jinja2 import Environment
        env = Environment(trim_blocks=True, lstrip_blocks=True)
        def raise_exception(message):
            raise ValueError(message)
        env.globals["raise_exception"] = raise_exception
        self._chat_template = env.from_string(self._template_source)
        self.model_name = "flash-mock-SYNTHETIC" if mock else str(Path(model).resolve())
        args = [str(worker), "--context", str(n_ctx)]
        if mock:
            args += ["--mock"]
        else:
            args += ["--model", str(model)]
        if allow_gpu:
            args += ["--allow-gpu"]
        try:
            self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr)
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            self.info = self._read()
            if self.info.get("protocol") != 1 or not self.info.get("ok"):
                raise FlashWorkerError("unsupported worker handshake")
            expected = "mock" if mock else "flash" if allow_gpu else "tokenizer-only"
            if self.info.get("mode") != expected or self.info.get("synthetic") != mock:
                raise FlashWorkerError("worker mode differs from requested mode")
            if not allow_gpu and self.info.get("gpu_loaded"):
                raise FlashWorkerError("worker unexpectedly enabled GPU")
            self.can_score = mock or allow_gpu
            self.last_stats: dict = {}
        except BaseException:
            self.close()
            raise

    def template(self):
        def render(system: str, user: str) -> str:
            return self.render_chat(system, user)
        return render

    def render_chat(self, system: str, user: str, *, thinking: bool = False,
                    reasoning_effort: str = "medium") -> str:
        return self._chat_template.render(messages=[{"role": "system", "content": system},
            {"role": "user", "content": user}], add_generation_prompt=True,
            enable_thinking=thinking, reasoning_effort=reasoning_effort)

    def generate(self, prompt: str, *, max_new_tokens: int = 1024):
        if not self.can_score:
            raise FlashWorkerError("tokenizer-only worker cannot generate; enable GPU explicitly")
        if self.info.get("generation") != "greedy-v1":
            raise FlashWorkerError("worker does not support greedy generation; rebuild the worker")
        result = self._rpc("generate", prompt=prompt, max_new_tokens=max_new_tokens)
        if (result.get("completion_tokens") != len(result["tokens"])
                or result["completion_tokens"] > max_new_tokens
                or result.get("finish_reason") not in ("eos", "length")):
            raise FlashWorkerError("invalid generation response")
        result["text"] = bytes(result.pop("bytes")).decode("utf-8", errors="replace")
        return result

    def _diagnostic(self) -> str:
        self._stderr.seek(0, os.SEEK_END)
        size = self._stderr.tell()
        self._stderr.seek(max(0, size - 4000))
        return self._stderr.read().decode("utf-8", errors="replace")

    def _read(self):
        deadline = time.monotonic() + self._timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                raise FlashWorkerError("Flash worker response timed out")
            chunk = os.read(self._process.stdout.fileno(), 65536)
            if not chunk:
                raise FlashWorkerError("Flash worker exited: " + self._diagnostic())
            self._buffer += chunk
            if len(self._buffer) > 16 * 1024 * 1024:
                raise FlashWorkerError("Flash worker response too large")
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            return json.loads(line)
        except ValueError as error:
            raise FlashWorkerError("non-JSON output on Flash protocol stdout") from error

    def _rpc(self, op, **params):
        with self._lock:
            if self._closed:
                raise FlashWorkerError("worker is closed")
            self._id += 1
            try:
                payload = json.dumps({"id": self._id, "op": op, **params}, allow_nan=False).encode() + b"\n"
                self._process.stdin.write(payload)
                self._process.stdin.flush()
                reply = self._read()
                if reply.get("id") != self._id:
                    raise FlashWorkerError("Flash response ID mismatch")
            except (OSError, FlashWorkerError) as error:
                self.close()
                raise FlashWorkerError(str(error)) from error
            if not reply.get("ok"):
                if reply.get("fatal"):
                    self.close()
                raise FlashWorkerError(reply.get("error", "Flash worker error"))
            return reply["result"]

    def tokenize(self, text: str, *, special: bool = True) -> list[int]:
        return self._rpc("tokenize", text=text, special=special)["tokens"]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def is_single_token(self, label: str) -> bool:
        if label not in self._labels:
            ids = self.tokenize(" " + label, special=False)
            self._labels[label] = ids[0] if len(ids) == 1 else None
        return self._labels[label] is not None

    def label_logits(self, prefix: str, suffixes: list[str], labels: list[tuple[str, ...]]) -> list[np.ndarray]:
        return self.score(prefix, suffixes, labels)

    def score(self, prefix: str, suffixes: list[str], labels: list[tuple[str, ...]], *, use_cache: bool = True,
              label_prefix: str = " "):
        """Read requested label tokens; empty prefix is an explicit diagnostic opt-in."""
        if not self.can_score:
            raise FlashWorkerError("tokenizer-only worker; enable GPU explicitly to score real model")
        if label_prefix not in ("", " "):
            raise ValueError("label_prefix must be empty or a single space")
        if label_prefix not in self.info.get("label_prefixes", [" "]):
            raise FlashWorkerError("worker does not support requested label prefix; rebuild the worker")
        result = self._rpc("label_logits", prefix=prefix, suffixes=suffixes, labels=labels,
                           use_cache=use_cache, label_prefix=label_prefix)
        if len(result["rows"]) != len(suffixes):
            raise FlashWorkerError("Flash scoring row count differs")
        values = []
        for row, row_labels in zip(result["rows"], labels, strict=True):
            z = np.asarray(row["logits"], dtype=np.float64)
            if z.shape != (len(row_labels),) or not np.isfinite(z).all():
                raise FlashWorkerError("Flash returned invalid label logits")
            values.append(z)
        self.last_stats = {k: v for k, v in result.items() if k != "rows"}
        return values

    def provenance(self):
        return {"worker": self.info, "template_sha256": hashlib.sha256(self._template_source.encode()).hexdigest(),
                "sampling": "none; raw last-position logits in requested label order", "thinking": False,
                "label_token_ids": {k: v for k, v in self._labels.items() if v is not None}}

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            # Controls only the subprocess created by this backend instance.
            try:
                if process.stdin:
                    process.stdin.close()
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            finally:
                if process.stdout:
                    process.stdout.close()
        self._selector.close()
        self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
