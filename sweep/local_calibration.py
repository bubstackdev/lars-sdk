"""CPU learning curves from frozen, state-disjoint, per-predicate raw captures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from lars.train.accuracy import load_capture, record_key
from lars.train.data import read_records
from sweep.accuracy_ladder import digest
from sweep.analyze import wilson, wilson_upper
from sweep.experiment import atomic_json
from sweep.heads import Head, Rows, calibrated, refit

BUDGETS = [0, 10, 25, 50, 100, 200]
ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / 'sweep/out/accuracy-full'


def normalized(value):
    if isinstance(value, str):
        return re.sub(r'\s+', ' ', value).strip().casefold()
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows))


def validate(rows):
    if not rows:
        raise ValueError('empty capture set')
    seen, state_roles, family_roles, predicates = set(), {}, {}, defaultdict(list)
    models = {r['model_id'] for r in rows}
    if len(models) != 1:
        raise ValueError('captures mix model identities')
    for row in rows:
        if row['key'] in seen:
            raise ValueError('duplicate decision')
        seen.add(row['key'])
        if row['role'] not in ('fit', 'certificate', 'test'):
            raise ValueError('unknown data role')
        if row.get('gold_status') not in ('benchmark', 'reviewed'):
            raise ValueError('unreviewed/inferred labels cannot enter this learning curve')
        state = identity(normalized(row['state']))
        for mapping, key in [(state_roles, state), (family_roles, row.get('family_id', state))]:
            if key in mapping and mapping[key] != row['role']:
                raise ValueError('state or case family overlaps roles')
            mapping[key] = row['role']
        z, t = np.asarray(row['logits']), np.asarray(row['target'])
        if (row['kind'] not in ('noul', 'choice', 'score') or len(row['keys']) < 2
                or len(set(row['keys'])) != len(row['keys']) or z.shape != (len(row['keys']),)
                or not np.isfinite(z).all() or t.shape != z.shape or not np.isfinite(t).all()
                or np.any(t < 0) or not np.isclose(t.sum(), 1.)
                or not isinstance(row['hard_index'], int) or not 0 <= row['hard_index'] < len(z)):
            raise ValueError('invalid logits/options/gold')
        predicates[row['predicate_id']].append(row)
    for group in predicates.values():
        if len({(r['kind'], tuple(r['keys']), identity(r['question'])) for r in group}) != 1:
            raise ValueError('predicate changes question, kind or option semantics')
        if len({identity(normalized(r['state'])) for r in group}) != len(group):
            raise ValueError('repeated state within predicate')
        if {r['role'] for r in group} != {'fit', 'certificate', 'test'}:
            raise ValueError('each predicate needs all three data roles')
        hidden = [r.get('hidden') for r in group]
        if any(h is not None for h in hidden):
            dims = {len(h) for h in hidden if h is not None}
            if any(h is None for h in hidden) or len(dims) != 1 or not next(iter(dims)) or not np.isfinite(hidden).all():
                raise ValueError('hidden vectors must be finite and aligned across every role')
    return predicates


def certificate(p, gold, risk=.05):
    """SDK fixed-sequence Wilson rule, using an independent certification split."""
    confidence = p.max(1); correct = p.argmax(1) == gold
    order = np.argsort(-confidence, kind='stable'); ranked = confidence[order]
    ends = [k for k in range(1, len(p)+1) if k == len(p) or ranked[k] != ranked[k-1]]
    fewest = next((k for k in range(1, len(p)+1) if wilson(0, k)[1] <= risk), None)
    if fewest is None:
        return None
    predicted = np.cumsum(1-ranked)
    start = max([k for k in ends if k >= fewest and wilson_upper(predicted[k-1], k) <= .8*risk], default=fewest)
    errors = np.cumsum(~correct[order]); best = None
    for k in ends:
        if k < start:
            continue
        upper = wilson(int(errors[k-1]), k)[1]
        if upper > risk:
            break
        best = {'threshold': float(ranked[k-1]), 'accepted': k, 'errors': int(errors[k-1]), 'error_upper95': upper}
    return best


def metrics(p, rows, threshold):
    hard = np.asarray([r['hard_index'] for r in rows]); target = np.asarray([r['target'] for r in rows])
    hit = p.argmax(1) == hard
    accepted = np.zeros(len(rows), dtype=bool) if threshold is None else p.max(1) >= threshold
    n = int(accepted.sum()); wrong = int((~hit & accepted).sum())
    return {'n': len(rows), 'accuracy': float(hit.mean()), 'brier_soft': float(((p-target)**2).sum(1).mean()),
        'brier_hard': float(((p-np.eye(p.shape[1])[hard])**2).sum(1).mean()),
        'log_loss_hard': float(-np.log(np.maximum(p[np.arange(len(rows)), hard], 1e-12)).mean()),
        'accepted': n, 'errors': wrong, 'coverage': n/len(rows), 'accepted_error': wrong/n if n else None,
        'accepted_error_upper95': wilson(wrong, n)[1], 'threshold': threshold}


def arrays(rows):
    z = np.asarray([r['logits'] for r in rows]); k = z.shape[1]
    target = np.eye(k)[[r['hard_index'] for r in rows]]
    hidden = np.asarray([r['hidden'] for r in rows]) if rows[0].get('hidden') is not None else np.empty((len(rows), 0))
    return Rows(rows[0]['kind'], rows[0]['keys'], z, target, hidden)


def fit_predict(name, train, base_train, test, base_test):
    if name == 'pack':
        return base_test
    if name == 'fitted':
        return calibrated(refit(train), test)
    lam = math.inf if name == 'recal' else float(name.split(':')[1])
    return Head(lam).fit(base_train, train.h, train.target).predict(base_test, test.h)


def select(train, pack_cal, rng):
    base = calibrated(pack_cal, train)
    names = ['pack', 'fitted', 'recal']
    if train.h.shape[1] and len(train) >= 50:
        names += [f'head:{lam}' for lam in [30., 3., .3, .03, .003]]
    losses = {'pack': -(train.target*base).sum(1)}
    folds = np.array_split(rng.permutation(len(train)), min(5, len(train)))
    for name in names[1:]:
        loss = np.empty(len(train))
        for held in folds:
            fit = np.setdiff1d(np.arange(len(train)), held)
            q = fit_predict(name, train.take(fit), base[fit], train.take(held), base[held])
            loss[held] = -(train.target[held]*q).sum(1)
        losses[name] = loss
    current = 'pack'
    for name in names[1:]:
        delta = losses[name]-losses[current]
        if delta.mean()+1.96*delta.std(ddof=1)/math.sqrt(len(delta)) < 0:
            current = name
    return current, {name: float(value.mean()) for name, value in losses.items()}


def prepare(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('retain the old experiment and use a fresh directory')
    inputs, rows, exclusions = [], [], Counter()
    if args.input:
        rows = read_jsonl(args.input); inputs.append(args.input)
    else:
        # Rehearsal: same quantized model/runtime for both existing capture populations.
        blocked = {identity(normalized(r.state)) for r in read_records(FULL/'data/train.jsonl')}
        inputs.append(FULL/'data/train.jsonl')
        for split in ['dev', 'typed-regression']:
            records_path = FULL/f'data/{split}.records.jsonl'; capture_path = FULL/f'baseline-s0-{split}.jsonl'
            records = [r for r in read_records(records_path) if r.source.startswith('typed-decisions/')]
            capture = load_capture(capture_path); inputs += [records_path, capture_path]
            for r in records:
                row = capture[record_key(r)]
                state = identity(normalized(r.state))
                if state in blocked:
                    exclusions['base_training_state_overlap'] += 1; continue
                if row['target'] != r.target or row['kind'] != r.kind or row['keys'] != (r.options if r.kind != 'score' else [str(i) for i in range(len(r.options))]):
                    raise ValueError('capture/record semantics differ')
                rows.append(row | {'state': r.state, 'family_id': state, 'question': r.to_question(),
                    'predicate_id': r.source+'/'+identity(r.to_question())[:16], 'workflow': r.source,
                    'model_id': 'accuracy-full/baseline-s0-Q4_K_M', 'gold_status': 'benchmark'})
        by_workflow = defaultdict(set)
        for row in rows:
            by_workflow[row['workflow']].add(row['family_id'])
        roles = {}
        for workflow, states in sorted(by_workflow.items()):
            ordered = sorted(states, key=lambda s: identity(['local-calibration-v1', workflow, s]))
            if len(ordered) < 120:
                raise ValueError('need at least120 distinct states per saved workflow for25 fit/75 certificate/20 test')
            for i, state in enumerate(ordered):
                roles[state] = 'fit' if i < 25 else 'certificate' if i < 100 else 'test'
        for row in rows:
            row['role'] = roles[row['family_id']]
    groups = validate(rows)
    data = out/'captures.jsonl'; write_jsonl(data, rows)
    pack = json.loads(args.calibration.read_text()); atomic_json(out/'pack-calibration.json', pack)
    counts = {key: dict(Counter(r['role'] for r in group)) for key, group in sorted(groups.items())}
    plan = {'format': 1, 'budgets': BUDGETS, 'draws': args.draws, 'risk': .05,
        'data_sha256': digest(data), 'calibration_sha256': digest(out/'pack-calibration.json'),
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(), ROOT/'sweep/heads.py', ROOT/'sweep/analyze.py']},
        'input_sha256': {str(p.resolve()): digest(p) for p in inputs}, 'predicates': counts,
        'state_roles': dict(Counter({role: len({identity(normalized(r['state'])) for r in rows if r['role']==role}) for role in ['fit','certificate','test']})),
        'exclusions': dict(exclusions), 'fitting': 'hard labels only; five-fold paired-loss guard; pack -> fitted -> recal -> optional heads>=50 labels',
        'draws_rule': 'nested label subsets per draw; all budgets share fixed certificate/test states; repetitions are not independent trials',
        'certificate': 'separate fixed-sequence Wilson threshold at5%; test errors/coverage evaluated without threshold retuning',
        'label_cost': 'fit budget per predicate plus fixed certificate labels; test labels reported separately; input states may answer multiple predicates',
        'interpretation': 'Saved workflows are synthetic benchmark workflows already included in backbone training and earlier evaluation; this is a local-adaptation rehearsal, not unseen-company confirmation.' if not args.input else 'User-supplied per-predicate capture study; verify representativeness and review labels before deployment.',
        'heads': 'only evaluated when aligned hidden vectors are supplied and fitting budget>=50; current cached rehearsal has none',
        'parity': 'offline simulation using existing Python fitting helpers; not SDK byte-for-byte calibration parity or a deployable calibration export'}
    atomic_json(out/'plan.json', plan); atomic_json(out/'status.json', {'state':'cpu_prepared','rows':len(rows),'predicates':len(groups)})
    print(json.dumps({'rows':len(rows),'predicates':len(groups),'role_states':plan['state_roles'],'budgets':BUDGETS,'fit_rows_per_predicate':sorted({v['fit'] for v in counts.values()})}, indent=2))


def run(args):
    out = args.out.resolve(); plan = json.loads((out/'plan.json').read_text())
    checks = {out/'captures.jsonl':plan['data_sha256'],out/'pack-calibration.json':plan['calibration_sha256'],
        **{ROOT/p:sha for p,sha in plan['code_sha256'].items()}}
    if any(digest(path)!=expected for path,expected in checks.items()):
        raise ValueError('frozen captures/calibration/code changed')
    import fcntl
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'results.json').exists():
            raise ValueError('retain completed results; use a new experiment directory')
        groups = validate(read_jsonl(out/'captures.jsonl')); pack_cal = json.loads((out/'pack-calibration.json').read_text())
        started = time.time(); results = []
        atomic_json(out/'status.json', {'state':'running_cpu','started':started})
        try:
            for predicate, group in sorted(groups.items()):
                fit_rows = [r for r in group if r['role']=='fit']; cert_rows = [r for r in group if r['role']=='certificate']; test_rows = [r for r in group if r['role']=='test']
                cert_data, test_data = arrays(cert_rows), arrays(test_rows)
                base_cert, base_test = calibrated(pack_cal, cert_data), calibrated(pack_cal, test_data)
                for draw in range(plan['draws']):
                    order = sorted(fit_rows, key=lambda r: identity(['fit-order-v1',draw,r['key']]))
                    for budget in plan['budgets']:
                        row = {'predicate':predicate,'workflow':group[0].get('workflow',predicate),'draw':draw,'budget':budget,
                            'certificate_labels':len(cert_rows),'test_labels':len(test_rows),'fit_plus_certificate_labels':budget+len(cert_rows)}
                        if budget > len(order):
                            results.append(row | {'state':'unsupported','reason':'insufficient_fit_labels','available_fit_labels':len(order)}); continue
                        selected = order[:budget]
                        if not budget:
                            chosen, cv, cert_logq, test_logq = 'pack', {}, base_cert, base_test
                        else:
                            train = arrays(selected); rng = np.random.default_rng(int(identity([predicate,draw,budget])[:8],16))
                            chosen, cv = select(train, pack_cal, rng); base_train = calibrated(pack_cal,train)
                            cert_logq = fit_predict(chosen,train,base_train,cert_data,base_cert)
                            test_logq = fit_predict(chosen,train,base_train,test_data,base_test)
                        cert_p, test_p = np.exp(cert_logq), np.exp(test_logq)
                        certified = certificate(cert_p,np.asarray([r['hard_index'] for r in cert_rows]),plan['risk'])
                        threshold = certified['threshold'] if certified else None
                        row.update(state='complete',kept=chosen,cv_log_loss=cv,certificate=certified,
                            fit_keys=[r['key'] for r in selected],metrics=metrics(test_p,test_rows,threshold),
                            fixed_threshold_diagnostics={str(t):metrics(test_p,test_rows,t) for t in [.7,.8,.9,.95,.99]})
                        results.append(row)
                atomic_json(out/'partial-results.json', {'runs':results})
                print(f'completed {predicate}',flush=True)
            aggregate = {}
            for budget in plan['budgets']:
                runs=[r for r in results if r['budget']==budget and r['state']=='complete']
                if not runs:
                    aggregate[str(budget)]={'state':'unsupported','reason':'insufficient_fit_labels'};continue
                # Repeated draws reuse the same evaluation set; never inflate CI sample counts.
                m=[r['metrics'] for r in runs]
                aggregate[str(budget)]={'state':'complete','predicates':len({r['predicate'] for r in runs}),
                    'draws':plan['draws'],'macro_accuracy':float(np.mean([v['accuracy'] for v in m])),
                    'macro_brier_soft':float(np.mean([v['brier_soft'] for v in m])),
                    'macro_coverage':float(np.mean([v['coverage'] for v in m])),
                    'draws_are_not_independent':True,'kept_counts':dict(Counter(r['kept'] for r in runs))}
            atomic_json(out/'results.json',{'summary':aggregate,'runs':results,'elapsed_seconds':time.time()-started,
                'interpretation':plan['interpretation'],'risk':plan['risk']})
            atomic_json(out/'status.json',{'state':'complete','finished':time.time(),'elapsed_seconds':time.time()-started})
            print(json.dumps(aggregate,indent=2))
        except BaseException as exc:
            atomic_json(out/'status.json',{'state':'failed','error':str(exc),'finished':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--out',type=Path,default=Path('sweep/out/local-calibration'))
    p.add_argument('--input',type=Path,help='canonical captured rows with explicit fit/certificate/test roles; omit for saved-workflow rehearsal')
    p.add_argument('--calibration',type=Path,default=FULL/'baseline-s0-calibration.json')
    p.add_argument('--draws',type=int,default=3)
    args=p.parse_args()
    if not 1<=args.draws<=20:p.error('draws must be in1..20')
    globals()[args.command](args)
