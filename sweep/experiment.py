"""Sequential subprocess stages with content-checked resumes and explicit ownership."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

from sweep.accuracy_ladder import digest


def fingerprint(path):
    path = Path(path)
    if path.is_file():
        return digest(path)
    if path.is_dir():
        files = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    raise FileNotFoundError(path)


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


class Runner:
    def __init__(self, out, env=None):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env = env or dict(os.environ)

    def stage(self, name, command=None, action=None, inputs=(), outputs=()):
        if (command is None) == (action is None):
            raise ValueError("provide one command or CPU action")
        identity = {"command": list(map(str, command)) if command else inspect.getsource(action),
                    "inputs": {str(p): fingerprint(p) for p in inputs}}
        done = self.out / f"{name}.done.json"
        previous = json.loads(done.read_text()) if done.exists() else {}
        intact = all(Path(p).exists() and previous.get("outputs", {}).get(str(p)) == fingerprint(p) for p in outputs)
        if previous.get("identity") == identity and intact:
            print(f"skip {name}: verified", flush=True)
            return
        status = {"stage": name, "state": "running", "owner_pid": os.getpid(), "started": time.time(),
                  "command": identity["command"] if command else "CPU action"}
        atomic_json(self.out / "status.json", status)
        print(f"start {name}", flush=True)
        try:
            if command:
                with (self.out / f"{name}.log").open("w") as log:
                    child = subprocess.Popen(list(map(str, command)), env=self.env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    atomic_json(self.out / "status.json", status | {"child_pid": child.pid})
                    result = child.wait()
                if result:
                    raise RuntimeError(f"{name}: exit {result}; see {name}.log")
            else:
                action()
            checks = {str(p): fingerprint(p) for p in outputs}
        except BaseException as exc:
            atomic_json(self.out / "status.json", status | {"state": "failed", "error": str(exc), "finished": time.time()})
            raise
        atomic_json(done, {"identity": identity, "outputs": checks, "started": status["started"], "finished": time.time()})
        atomic_json(self.out / "status.json", status | {"state": "stage_complete", "finished": time.time()})
        print(f"complete {name}", flush=True)
