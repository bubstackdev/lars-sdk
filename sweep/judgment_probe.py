"""Exploratory CPU adaptation with explicitly agent-assessed labels.

This separate diagnostic never relabels drafts as human-reviewed gold and never
exports a deployable calibration or certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from sweep.codegraph_judgment import file_hash, identity
from sweep.heads import calibrated
from sweep.judgment_capture import json_file, lines, write_lines
from sweep.local_calibration import arrays, fit_predict, select


def reject_retired(queue, retirement_path):
    if not retirement_path.exists():
        return
    retirement=json.loads(retirement_path.read_text())
    retired=set(retirement.get('retired_question_sha256',[]))
    if any(identity(case['question']) in retired for case in queue['cases']):
        raise ValueError('This judgment-category benchmark is retired for diagnostic hindsight. Use the corrected source-outcome replay audit; do not rerun its accuracy comparison.')


def prepare_rows(queue, captures, threshold=.8):
    cache = {r['original_key']:r for r in captures}
    roles, grouped = {}, defaultdict(list)
    for case in queue['cases']:
        review = case.get('review', {})
        status = review.get('status', 'pending')
        if status in {'approved', 'corrected'} and not review.get('family_concern'):
            state, label, gold_status = review['state'], review['label'], 'reviewed'
        elif status == 'pending' and case['confidence'] >= threshold:
            state, label, gold_status = case['state'], case['draft_label'], 'agent_assessed'
        else:
            continue
        row = cache[case['key']]
        if row['input_sha256'] != identity([state,case['question']]):
            raise ValueError(f'case facts changed after capture; recapture {case["key"]}')
        family = case['family_id']
        if family in roles and roles[family] != case['role']:
            raise ValueError('family crosses data roles')
        roles[family] = case['role']
        keys = row['keys']; hard = keys.index(label)
        grouped[(case['role'],family)].append({
            'key':case['key'],'state':state,'family_id':family,'role':case['role'],
            'kind':row['kind'],'keys':keys,'logits':row['logits'],'model_id':row['model_id'],
            'hard_index':hard,'target':[float(i==hard) for i in range(len(keys))],
            'gold_status':gold_status,'confidence':case['confidence']})
    # One state per dependence group in every role, selected without labels/logits.
    rows = [min(group,key=lambda r:identity(['group-representative-v1',r['key']]))
            for group in grouped.values()]
    if len({(r['kind'],tuple(r['keys']),r['model_id']) for r in rows}) != 1:
        raise ValueError('model/options changed')
    seen = set()
    for row in rows:
        key = identity(' '.join(json.dumps(row['state'],sort_keys=True).casefold().split()))
        if key in seen:
            raise ValueError('duplicate normalized state')
        seen.add(key)
    return rows


def score(logq, rows):
    p = np.exp(logq); gold = np.array([r['hard_index'] for r in rows])
    return {'n':len(rows),'correct':int((p.argmax(1)==gold).sum()),
            'accuracy':float((p.argmax(1)==gold).mean()),
            'brier_hard':float(((p-np.eye(p.shape[1])[gold])**2).sum(1).mean()),
            'log_loss_hard':float(-logq[np.arange(len(rows)),gold].mean())}


def probe(rows, calibration, budgets=(0,10,25,50,100,200), draws=3):
    fit = [r for r in rows if r['role']=='fit']
    test = [r for r in rows if r['role']=='test']
    certificate = [r for r in rows if r['role']=='certificate']
    if min(len(fit),len(test),len(certificate)) < 1:
        raise ValueError('all roles required')
    test_data = arrays(test); base = calibrated(calibration,test_data)
    results = []
    for draw in range(draws):
        order = sorted(fit,key=lambda r:identity(['judgment-probe-fit-v1',draw,r['key']]))
        for budget in budgets:
            if budget > len(order):
                results.append({'draw':draw,'budget':budget,'state':'unsupported','available_fit_families':len(order)});continue
            subset = order[:budget]
            if not budget:
                chosen,cv,predicted = 'pack',{},base
            else:
                train = arrays(subset)
                rng = np.random.default_rng(int(identity(['judgment-probe-cv-v1',draw,budget])[:8],16))
                chosen,cv = select(train,calibration,rng)
                predicted = fit_predict(chosen,train,calibrated(calibration,train),test_data,base)
            hard = np.array([r['hard_index'] for r in test])
            old = base.argmax(1)==hard; new = predicted.argmax(1)==hard
            results.append({'draw':draw,'budget':budget,'state':'complete','kept':chosen,
                            'fit_keys':[r['key'] for r in subset],'cv':cv,
                            'metrics':score(predicted,test),
                            'corrections':int((new & ~old).sum()),'regressions':int((old & ~new).sum()),
                            'test_keys':[r['key'] for r in test],'test_logq':predicted.tolist()})
    raw = np.array([r['logits'] for r in test]);raw-=raw.max(1,keepdims=True)
    raw-=np.log(np.exp(raw).sum(1,keepdims=True))
    summaries = {}
    for budget in budgets:
        runs = [r for r in results if r['budget']==budget and r['state']=='complete']
        if not runs:
            summaries[str(budget)]={'state':'unsupported','available_fit_families':len(fit)};continue
        summaries[str(budget)]={'state':'complete','mean_accuracy':float(np.mean([r['metrics']['accuracy'] for r in runs])),
                                'mean_brier_hard':float(np.mean([r['metrics']['brier_hard'] for r in runs])),
                                'kept':dict(Counter(r['kept'] for r in runs)),
                                'corrections_by_draw':[r['corrections'] for r in runs],
                                'regressions_by_draw':[r['regressions'] for r in runs]}
    return {'summary':summaries,'raw_baseline':score(raw,test),'runs':results,
            'role_families':dict(Counter(r['role'] for r in rows)),
            'gold_status':'agent_assessed' if all(r.get('gold_status')=='agent_assessed' for r in rows) else 'mixed_reviewed_and_agent_assessed',
            'label_provenance_counts':dict(Counter(r.get('gold_status','unknown') for r in rows)),
            'independently_reviewed_gold':False,
            'certificate':'not run: agent labels and small certification population',
            'deployed':False,'hidden_heads':False,'draws_are_not_independent':True,
            'interpretation':'Exploratory proxy-label result on confidence-selected retrospective cases. Not a validated company benchmark, prospective accuracy estimate, or deployment calibration.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('use a fresh diagnostic directory')
    queue=json.loads(args.queue.read_text())
    reject_retired(queue,Path(__file__).resolve().parents[1]/'sweep/out/codegraph-judgment/study-retirement.json')
    rows=prepare_rows(queue,lines(args.capture))
    args.out.mkdir(parents=True)
    json_file(args.out/'plan.json',{'confidence_threshold':.8,'budgets':[0,10,25,50,100,200],'draws':3,
        'representative':'one per family, hash chosen without labels/logits',
        'role_families':dict(Counter(r['role'] for r in rows)),
        'hashes':{str(path.resolve()):file_hash(path) for path in [args.queue,args.capture,args.calibration,Path(__file__),Path('sweep/local_calibration.py'),Path('sweep/heads.py')]},
        'gold_status':'agent_assessed','reviewed_runner_gate_unchanged':True,'deployment':False})
    write_lines(args.out/'agent-assessed-captures.jsonl',rows)
    (args.out/'frozen-driver.py').write_bytes(Path(__file__).read_bytes())
    result=probe(rows,json.loads(args.calibration.read_text()))
    json_file(args.out/'results.json',result)
    print(json.dumps({k:result[k] for k in ('summary','raw_baseline','role_families','gold_status')},indent=2))


if __name__=='__main__':main()
