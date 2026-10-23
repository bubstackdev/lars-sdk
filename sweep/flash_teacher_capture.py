"""Prepare a small CPU-validated Flash teacher pilot; GPU capture requires an explicit flag."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from lars.backends.flash import FlashBackend
from lars.train.accuracy import capture, load_capture, record_key, state_key
from lars.train.data import read_records
from lars.labels import assign_labels
from lars.render import compose_prompt, render_content
from lars.train.features import _render
from sweep.accuracy_ladder import digest
from sweep.confirmation import save_dataset, row_keys, summary
from sweep.distillation_ablation import freeze
from sweep.experiment import atomic_json


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = list(read_records(args.records))
    wire = {r['record_key']: r for r in [json.loads(l) for l in args.wire.read_text().split('\n') if l.strip()]}
    if len(wire) != len(records) or {record_key(r) for r in records} != wire.keys():
        raise ValueError('record/wire populations differ')
    if not 1 <= args.per_source <= 100:
        raise ValueError('per-source limit must be in1..100')
    groups = defaultdict(list)
    for r in records:
        groups[r.source].append(r)
    chosen = [r for source, group in sorted(groups.items()) for r in sorted(group, key=record_key)[:args.per_source]]
    # Share each input's prefix across its questions when possible, within one bounded cache slot.
    chosen.sort(key=lambda r: (state_key(r), record_key(r)))
    with FlashBackend(args.worker, args.model, args.template, n_ctx=args.context) as backend:
        if backend.info['mode'] != 'tokenizer-only' or backend.info['gpu_loaded']:
            raise ValueError('preparation must be CPU vocabulary-only')
        labels = assign_labels(max(len(r.options) for r in chosen), backend.is_single_token)
        lengths = []
        for r in chosen:
            body, _, _ = _render(r, labels, list(range(len(r.options))))
            prefix, suffix = compose_prompt(backend.template(), render_content(r.state), body)
            n = backend.count_tokens(prefix + suffix)
            if n < 1 or n > args.context:
                raise ValueError(f'pilot prompt exceeds context; no silent row drops: {r.id}')
            lengths.append(n)
        model = args.model.resolve()
        manifest = {'format': 1, 'cpu_only_validation': True, 'synthetic': False,
            'model': str(model), 'model_bytes': model.stat().st_size, 'model_mtime_ns': model.stat().st_mtime_ns,
            'worker': str(args.worker.resolve()), 'worker_sha256': digest(args.worker),
            'template': str(args.template.resolve()), 'template_sha256': digest(args.template),
            'records_input_sha256': digest(args.records), 'wire_input_sha256': digest(args.wire),
            'per_source': args.per_source, 'context': args.context, 'rows': len(chosen),
            'source_counts': {s: min(len(v), args.per_source) for s, v in sorted(groups.items())},
            'max_prompt_tokens': max(lengths), 'mean_prompt_tokens': sum(lengths) / len(lengths),
            'provenance': backend.provenance(),
            'limitations': 'CPU validates vocabulary/rendering/protocol only; no real model logits, GPU parity, speed or teacher quality measured. File size/mtime identify the local candidate, not a cryptographic weight checksum.'}
        pairs = [(r, wire[record_key(r)]['hard_index']) for r in chosen]
        if any(not 0 <= hard < len(r.options) for r, hard in pairs):
            raise ValueError('invalid hard label')
        save_dataset(out, 'pilot', pairs)
        manifest['pilot_records_sha256'] = digest(out / 'pilot.records.jsonl')
        manifest['pilot_wire_sha256'] = digest(out / 'pilot.wire.jsonl')
        freeze(out / 'plan.json', manifest)
        atomic_json(out / 'status.json', {'state': 'cpu_prepared', 'gpu_started': False, 'rows': len(chosen)})
    print(json.dumps({k: manifest[k] for k in ['rows', 'max_prompt_tokens', 'mean_prompt_tokens', 'source_counts']}, indent=2))


def score_plan(args):
    if not args.allow_gpu:
        raise ValueError('capture requires --allow-gpu; preparation and CPU tests do not load Flash weights')
    out = args.plan.resolve()
    manifest = json.loads((out / 'plan.json').read_text())
    model, worker, template = map(Path, [manifest['model'], manifest['worker'], manifest['template']])
    if (model.stat().st_size, model.stat().st_mtime_ns) != (manifest['model_bytes'], manifest['model_mtime_ns']):
        raise ValueError('model file changed; prepare a new pilot')
    if digest(out / 'pilot.records.jsonl') != manifest['pilot_records_sha256'] or digest(out / 'pilot.wire.jsonl') != manifest['pilot_wire_sha256']:
        raise ValueError('prepared examples or labels changed')
    if digest(worker) != manifest['worker_sha256'] or digest(template) != manifest['template_sha256']:
        raise ValueError('worker/template changed; prepare a new pilot')
    import fcntl
    with (out / 'capture.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'capture.jsonl').exists():
            raise ValueError('capture already exists; retain it and use a fresh pilot for another model')
        records = list(read_records(out / 'pilot.records.jsonl'))
        metadata = {r['record_key']: r for r in [json.loads(l) for l in (out / 'pilot.wire.jsonl').read_text().split('\n') if l.strip()]}
        if {record_key(r) for r in records} != metadata.keys() or len(records) != manifest['rows']:
            raise ValueError('prepared population changed')
        started = time.time()
        atomic_json(out / 'status.json', {'state': 'loading_flash', 'started': started})
        try:
            with FlashBackend(worker, model, template, allow_gpu=True, n_ctx=manifest['context']) as backend:
                loaded = time.time()
                partial = out / 'capture.partial.jsonl'
                capture(backend, records, partial, max_tokens=manifest['context'])
                rows = load_capture(partial)
                for r in records:
                    row = rows[record_key(r)]
                    if 'p' not in row:
                        raise ValueError('prepared row was silently skipped')
                    row['hard_index'] = metadata[record_key(r)]['hard_index']
                    row['keys'] = row_keys(r)
                partial.write_text(''.join(json.dumps(row) + '\n' for row in rows.values()))
                partial.replace(out / 'capture.jsonl')
                atomic_json(out / 'capture-metadata.json', {'provenance': backend.provenance(),
                    'rows': len(rows), 'started': started, 'finished': time.time(),
                    'model_load_seconds': loaded - started, 'capture_seconds': time.time() - loaded,
                    'interpretation': 'raw teacher predictions; original gold targets/labels retained; no training or certification'})
            atomic_json(out / 'status.json', {'state': 'capture_complete', 'rows': len(rows), 'finished': time.time()})
        except Exception as exc:
            atomic_json(out / 'status.json', {'state': 'failed', 'error': str(exc), 'finished': time.time()})
            raise



def report(args):
    out = args.plan.resolve()
    provenance = json.loads((out / 'capture-metadata.json').read_text())
    if provenance['provenance']['worker']['synthetic']:
        raise ValueError('synthetic CPU predictions cannot be used as teacher-quality evidence')
    teacher = load_capture(out / 'capture.jsonl')
    student = load_capture(args.student)
    selected = {}
    for key, row in teacher.items():
        if key not in student or 'p' not in row or 'p' not in student[key]:
            raise ValueError('missing matched scored decisions')
        other = student[key]
        if any(row.get(field) != other.get(field) for field in ('keys', 'target', 'hard_index', 'source', 'kind')):
            raise ValueError('teacher/student decision semantics or gold labels differ')
        selected[key] = other
    a, b = summary(selected), summary(teacher)
    result = {'rows': len(teacher), 'student': a, 'teacher': b,
        'macro_accuracy_delta': b['macro_accuracy'] - a['macro_accuracy'],
        'macro_brier_delta': b['macro_brier'] - a['macro_brier'],
        'model_load_seconds': provenance['model_load_seconds'], 'capture_seconds': provenance['capture_seconds'],
        'interpretation': 'small development pilot, raw uncalibrated distributions; not promotion or certification evidence; benchmark teacher before generating training targets'}
    atomic_json(out / 'comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('student', 'teacher')}, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--wire', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--per-source', type=int, default=10)
    p = sub.add_parser('capture')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--allow-gpu', action='store_true')
    p = sub.add_parser('report')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--student', type=Path, required=True, help='matching raw canonical student capture')
    args = parser.parse_args()
    {'prepare': prepare, 'capture': score_plan, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
