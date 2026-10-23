"""Fitting-only exploration of explicit pre-action queue evidence. No validation claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np
from scipy.special import expit

from sweep.codegraph_judgment import identity
from sweep.judgment_capture import lines, json_file, write_lines, prepare
from sweep.queue_calibration import fit_affine, metrics, verify_files
from sweep.queue_evidence import digest

BASE = Path('sweep/out/codegraph-judgment/queue-pilot-20261023')


def extract(observations):
    """Literal matches only; every populated field carries its supporting quote."""
    patterns = {
        'error_codes': r'\b(?:2303|2304|2305)\b',
        'exception_types': r'\bSystem\.[\w.]*Exception\b',
        'domain_status': r'\b(?:serverUpdateProhibited|serverTransferProhibited|clientTransferProhibited|Deleted|expired)\b',
        'ownership_mismatch': r'\b(?:owner(?:ship)? mismatch|not owned by)\b',
        'contact_mismatch': r'contact ID we try to remove is not what the registry has',
        'work_already_completed': r'(?:domains were manually restored|already (?:completed|processed))',
        'known_replay_behavior': r'(?:safe to replay|idempotent|not a poison message)',
        'stale_messages': r'\bstale\b',
        'dependency_recovered': r'Failures stopped and all instances are healthy',
        'repair_already_completed': r'After de-duping.*fixing things up',
        'transient_fault': r'(?:Transient connection pool exhaustion|abruptly terminated outbound connections|System.Net.Sockets.SocketException)',
    }
    return {field: {'value': sorted({m.group(0) for q in observations for m in re.finditer(pattern,q,re.I)}),
                    'evidence': [q for q in observations if re.search(pattern,q,re.I)]}
            if any(re.search(pattern,q,re.I) for q in observations)
            else {'value': 'unknown', 'evidence': []}
            for field,pattern in patterns.items()}


def rule(fields):
    present = lambda key: fields[key]['value'] != 'unknown'
    if any(present(k) for k in ('stale_messages','work_already_completed')):
        return 'discard'
    if any(present(k) for k in ('transient_fault','dependency_recovered','repair_already_completed')):
        return 'reprocess'
    return None


def setup(out):
    if out.exists(): raise ValueError('fresh output directory required')
    cohort=BASE/'cohort-v3'
    verify_files(json.loads((cohort/'manifest.json').read_text()),cohort)
    # Filter roles before extraction; no old test inputs enter this experiment.
    rows=[r for r in lines(cohort/'primary.jsonl') if r['role']=='fit']
    assert len(rows)==25 and len({r['family_id'] for r in rows})==25
    out.mkdir(parents=True)
    write_lines(out/'fit-only.jsonl',rows)
    cases=[]
    for row in rows:
        fields=extract(row['state']['observations'])
        cases.append({'key':row['key'], 'state':{'observations':row['state']['observations'],
                      'structured_pre_action_facts':fields},'question':row['question'],
                      'draft_label':row['label']})
    json_file(out/'capture-data.json',{'cases':cases})
    # Stratified fixed folds: one discard plus four reprocess groups per fold.
    folds=[[] for _ in range(5)]
    for label in ['discard','reprocess']:
        ordered=sorted([r for r in rows if r['label']==label],key=lambda r:identity(['structured-fold-v1',r['key']]))
        for i,row in enumerate(ordered): folds[i%5].append(row['key'])
    json_file(out/'study-plan.json',{
        'scope':'25 original fitting groups only; old 16 test groups excluded',
        'comparison':'same original quotes versus quotes plus literal structured fields',
        'folds':folds,'affine_penalty':1.0,'temperature':'frozen pack choice temperature',
        'rule':'discard if explicitly stale/completed; reprocess if transient/recovered/repaired; otherwise abstain',
        'rule_missing_fallback':'out-of-fold Jeffreys class prior',
        'limitations':['Exploratory fitting data; field/rule design saw fitting examples and labels.',
                      'Cross-validation excludes calibration labels but cannot validate human feature/rule selection.',
                      'Labels are documented mechanisms, not optimal operational actions.',
                      'Unknown means unrecorded, never false; original evidence is retained.',
                      'Fresh independent holdout required for any improvement claim.'],
        'input_sha256':{str(p):digest(p) for p in [out/'fit-only.jsonl',out/'capture-data.json',Path(__file__)]}})
    shutil.copy2(__file__,out/'frozen-runner.py')
    prepare(out/'capture-data.json',out/'capture')


def score(out):
    plan=json.loads((out/'study-plan.json').read_text())
    for path,sha in plan['input_sha256'].items():
        if digest(path)!=sha: raise ValueError('frozen input changed')
    if (out/'results.json').exists(): raise ValueError('preserve existing results')
    rows=lines(out/'fit-only.jsonl'); y=np.array([r['label']=='reprocess' for r in rows],int)
    inputs={r['key']:r for r in json.loads((out/'capture-data.json').read_text())['cases']}
    if not json.loads((out/'capture/parity.json').read_text())['passed']: raise ValueError('parity failed')
    calibration=Path('sweep/out/accuracy-full/baseline-s0-calibration.json')
    baseline=json.loads(Path('sweep/out/codegraph-baseline/plan.json').read_text())
    if digest(calibration)!=baseline['sha256'][str(calibration.resolve())]: raise ValueError('pack changed')
    temperature=json.loads(calibration.read_text())['temperatures']['choice']
    model_id='sha256:'+digest(baseline['model'])
    gaps={}
    for name,path in [('original',BASE/'capture-v1/captures.jsonl'),('structured',out/'capture/captures.jsonl')]:
        cache={r['original_key']:r for r in lines(path) if r['original_key'] in inputs}
        assert set(cache)==set(inputs)
        for row in rows:
            state=row['state'] if name=='original' else inputs[row['key']]['state']
            c=cache[row['key']]
            assert c['input_sha256']==identity([state,row['question']])
            assert c['model_id']==model_id
        gaps[name]=np.array([(cache[r['key']]['logits'][cache[r['key']]['keys'].index('reprocess')]-
                            cache[r['key']]['logits'][cache[r['key']]['keys'].index('discard')])/temperature for r in rows])
    p={'original_pack':expit(gaps['original']),'structured_pack':expit(gaps['structured'])}
    p.update({k:np.empty(len(rows)) for k in ['class_prior','rules_with_prior','original_affine_oof','structured_affine_oof']})
    decisions=[rule(extract(r['state']['observations'])) for r in rows]
    for keys in plan['folds']:
        held=np.array([i for i,r in enumerate(rows) if r['key'] in keys]); train=np.setdiff1d(np.arange(len(rows)),held)
        prior=(y[train].sum()+.5)/(len(train)+1)
        p['class_prior'][held]=prior
        for i in held:p['rules_with_prior'][i]=prior if decisions[i] is None else (.9 if decisions[i]=='reprocess' else .1)
        for name,x in gaps.items():
            a,b=fit_affine(x[train],y[train],plan['affine_penalty'])
            p[name+'_affine_oof'][held]=expit(a*x[held]+b)
    summaries={}
    for name,prob in p.items():
        result=metrics(prob,rows)
        pos=prob[y==1,None];neg=prob[y==0][None,:]
        result['auc']=float(((pos>neg)+.5*(pos==neg)).mean())
        result['balanced_accuracy']=float(np.mean([v['correct']/v['n'] for v in result['per_label'].values()]))
        summaries[name]=result
    selected=[i for i,d in enumerate(decisions) if d is not None]
    report={'scope':plan['scope'],'n':len(rows),'summaries':summaries,
            'rule_coverage':{'n':len(selected),'correct':sum(decisions[i]==rows[i]['label'] for i in selected)},
            'field_coverage':{k:sum(extract(r['state']['observations'])[k]['value']!='unknown' for r in rows) for k in extract([])},
            'limitations':plan['limitations'],'capture':json.loads((out/'capture/status.json').read_text())}
    write_lines(out/'predictions.jsonl',[{'key':r['key'],'label':r['label'],'rule':decisions[i],
                  'p_reprocess':{k:float(v[i]) for k,v in p.items()}} for i,r in enumerate(rows)])
    json_file(out/'results.json',report)
    json_file(out/'manifest.json',{'sha256':{str(p):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','score']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':setup(args.out);print('Prepared 25 fitting groups; no old holdout included.')
    else:print(json.dumps(score(args.out),indent=2))
