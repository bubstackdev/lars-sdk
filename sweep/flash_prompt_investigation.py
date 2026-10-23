"""Bounded Flash answer-format diagnostic; preserves the original pilot and gold labels."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np

from lars.backends.flash import FlashBackend
from lars.labels import assign_labels
from lars.render import compose_prompt, render_content
from lars.train.accuracy import load_capture, record_key, report
from lars.train.data import read_records
from lars.train.features import _render
from sweep.accuracy_ladder import digest
from sweep.experiment import atomic_json


def prompt_pair(backend, record, labels):
    body, _, _ = _render(record, labels, list(range(len(record.options))))
    prefix, suffix = compose_prompt(backend.template(), render_content(record.state), body)
    if not suffix.endswith('Answer:'):
        raise ValueError('unexpected canonical prefill')
    return prefix, suffix, suffix[:-len('Answer:')]


def make_row(reference, logits):
    z = np.asarray(logits, dtype=float)
    p = np.exp(z - z.max()); p /= p.sum()
    return reference | {'logits': z.tolist(), 'p': p.tolist()}


def investigate(args):
    if not args.allow_gpu:
        raise ValueError('real capture requires --allow-gpu')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'plan.json').exists():
            raise ValueError('retain previous diagnostic; choose a new output directory')
        prior = args.prior.resolve()
        m = json.loads((prior / 'plan.json').read_text())
        model, template = Path(m['model']), Path(m['template'])
        if (model.stat().st_size, model.stat().st_mtime_ns) != (m['model_bytes'], m['model_mtime_ns']):
            raise ValueError('original model candidate changed')
        for name, field in [('pilot.records.jsonl', 'pilot_records_sha256'), ('pilot.wire.jsonl', 'pilot_wire_sha256')]:
            if digest(prior / name) != m[field]:
                raise ValueError('original pilot inputs changed')
        if digest(template) != m['template_sha256']:
            raise ValueError('original template changed')
        records = list(read_records(prior / 'pilot.records.jsonl'))
        original = load_capture(prior / 'capture.jsonl')
        student = load_capture(args.student)
        if {record_key(r) for r in records} != original.keys():
            raise ValueError('original capture population differs')
        for key, row in original.items():
            if any(row.get(f) != student[key].get(f) for f in ('keys', 'target', 'hard_index', 'source', 'kind')):
                raise ValueError('original/student semantics differ')
        # Validate exact continuation token boundaries before any GPU allocation.
        prepared = []
        with FlashBackend(args.worker, model, template, n_ctx=m['context']) as cpu:
            labels = assign_labels(max(len(r.options) for r in records), cpu.is_single_token)
            bad_bare = [label for label in labels if len(cpu.tokenize(label, special=False)) != 1]
            for r in records:
                prefix, canonical, native = prompt_pair(cpu, r, labels)
                ls = tuple(labels[:len(r.options)])
                bare_ok = not any(label in bad_bare for label in ls)
                full_ids = cpu.tokenize(prefix + native)
                # No silently dropped alternatives or multi-token sequence shortcuts.
                for label in ls:
                    if cpu.tokenize(prefix + native + ' ' + label) != full_ids + cpu.tokenize(' ' + label, special=False):
                        raise ValueError('native spaced continuation changes prompt token boundary')
                    if bare_ok and cpu.tokenize(prefix + native + label) != full_ids + cpu.tokenize(label, special=False):
                        raise ValueError('native bare continuation changes prompt token boundary')
                mixed = (ls + tuple(' ' + label for label in ls)) if bare_ok else tuple(' ' + label for label in ls)
                ids = [cpu.tokenize(label, special=False)[0] for label in mixed]
                if len(set(ids)) != len(ids):
                    raise ValueError('diagnostic token classes overlap')
                prepared.append((r, prefix, canonical, native, ls, mixed, bare_ok))
        plan = {'format': 1, 'prior': str(prior), 'rows': len(records), 'context': m['context'],
            'model': str(model), 'model_bytes': m['model_bytes'], 'model_mtime_ns': m['model_mtime_ns'],
            'worker': str(args.worker.resolve()), 'worker_sha256': digest(args.worker),
            'template_sha256': digest(template), 'script_sha256': digest(Path(__file__)),
            'backend_sha256': digest(Path('training/src/lars/backends/flash.py')),
            'prior_capture_sha256': digest(prior / 'capture.jsonl'), 'student_sha256': digest(args.student),
            'pilot_records_sha256': m['pilot_records_sha256'], 'pilot_wire_sha256': m['pilot_wire_sha256'],
            'canonical_labels': labels, 'bare_multitoken_labels': bad_bare,
            'bare_eligible_rows': sum(v[-1] for v in prepared),
            'bare_excluded_sources': dict(Counter(v[0].source for v in prepared if not v[-1])),
            'variants': ['canonical_spaced', 'native_spaced', 'native_bare (only fully single-token option sets)'],
            'max_rows_per_worker': 100,
            'canonical_parity_gate': {'max_probability_difference': .001, 'max_label_flips': 0},
            'interpretation': 'Exploratory reused development data, no training/promotion; bare format compared only on identical eligible populations. No option relabeling, no omitted alternatives.'}
        atomic_json(out / 'plan.json', plan)
        print(json.dumps({k: plan[k] for k in ['rows', 'bare_eligible_rows', 'bare_excluded_sources', 'bare_multitoken_labels']}), flush=True)
        started = time.time()
        atomic_json(out / 'status.json', {'state': 'loading_flash', 'started': started})
        variants = {name: {} for name in ['canonical_spaced', 'native_spaced', 'native_bare']}
        max_delta = 0.; flips = 0
        try:
            # Bound the lifetime of lazy model allocations; never overlap model instances.
            with ExitStack() as workers:
                gpu = None
                load_seconds = 0.
                with (out / 'capture.partial.jsonl').open('w') as handle:
                    for i, (r, prefix, canonical, native, ls, mixed, bare_ok) in enumerate(prepared):
                        if i % plan['max_rows_per_worker'] == 0:
                            if gpu is not None:
                                gpu.close()
                            load_started = time.time()
                            gpu = workers.enter_context(FlashBackend(args.worker, model, template,
                                allow_gpu=True, n_ctx=m['context'], timeout=180))
                            load_seconds += time.time() - load_started
                        key = record_key(r); ref = original[key]
                        canonical_z = gpu.score(prefix, [canonical], [ls])[0]
                        current = make_row(ref, canonical_z)
                        delta = float(np.max(np.abs(np.asarray(current['p']) - ref['p'])))
                        flip = int(np.argmax(current['p']) != np.argmax(ref['p']))
                        max_delta = max(delta, max_delta); flips += flip
                        if delta > .001 or flip:
                            raise ValueError('new worker failed original canonical probability/label parity')
                        z = gpu.score(prefix, [native], [mixed], label_prefix='')[0]
                        variants['canonical_spaced'][key] = current
                        variants['native_spaced'][key] = make_row(ref, z[len(ls):] if bare_ok else z)
                        if bare_ok:
                            variants['native_bare'][key] = make_row(ref, z[:len(ls)])
                        handle.write(json.dumps({'key': key, 'variants': {name: rows[key] for name, rows in variants.items() if key in rows}}) + '\n')
                        handle.flush()
                        atomic_json(out / 'status.json', {'state': 'capturing', 'completed': i+1, 'rows': len(records), 'elapsed_seconds': time.time()-started})
                        if (i+1) % 25 == 0:
                            print(f'captured {i+1}/{len(records)} x 2 prompts', flush=True)
                metadata = {'started': started, 'model_load_seconds': load_seconds,
                    'capture_seconds': time.time()-started-load_seconds, 'canonical_max_probability_delta': max_delta,
                    'canonical_label_flips': flips, 'provenance': gpu.provenance()}
            for name, rows in variants.items():
                (out / f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows.values()))
            full = report({'student': {k: student[k] for k in original},
                'canonical_spaced': variants['canonical_spaced'], 'native_spaced': variants['native_spaced']})
            eligible = variants['native_bare'].keys()
            bare = report({'student': {k: student[k] for k in eligible},
                **{name: {k: rows[k] for k in eligible} for name, rows in variants.items()}})
            # Pair format effects against Flash control as well as student quality.
            native_effect = report({'canonical': variants['canonical_spaced'], 'native': variants['native_spaced']})
            bare_effect = report({'canonical': {k: variants['canonical_spaced'][k] for k in eligible},
                                  'bare': variants['native_bare']})
            atomic_json(out / 'comparison.json', {'full': full, 'bare_eligible': bare,
                'native_effect': native_effect, 'bare_effect': bare_effect, 'metadata': metadata,
                'interpretation': plan['interpretation']})
            atomic_json(out / 'status.json', {'state': 'complete', 'rows': len(records), 'finished': time.time()})
            print(json.dumps(metadata, indent=2), flush=True)
        except BaseException as exc:
            atomic_json(out / 'status.json', {'state': 'failed', 'error': str(exc), 'finished': time.time()})
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--prior', type=Path, default=Path('sweep/out/flash-teacher-pilot'))
    p.add_argument('--out', type=Path, default=Path('sweep/out/flash-prompt-investigation'))
    p.add_argument('--worker', type=Path, default=Path('target/flash-teacher-current/flash-teacher'))
    p.add_argument('--student', type=Path, default=Path('sweep/out/accuracy-full/baseline-s0-dev.jsonl'))
    p.add_argument('--allow-gpu', action='store_true')
    investigate(p.parse_args())
