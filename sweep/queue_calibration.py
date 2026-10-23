"""Small, source-documented queue-action pilot; no deployment or risk certificate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from sweep.codegraph_judgment import identity
from sweep.queue_evidence import digest


def fit_affine(x, y, penalty):
    x=np.asarray(x,float); y=np.asarray(y,float)
    def objective(theta):
        a,b=theta; z=a*x+b; prior=(a-1)**2+b*b
        loss=np.mean(np.logaddexp(0,z)-y*z)+penalty/len(y)*prior
        residual=expit(z)-y
        gradient=np.array([np.mean(residual*x),np.mean(residual)])
        gradient+=2*penalty/len(y)*np.array([a-1,b])
        return loss,gradient
    fit=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(0.,10.),(-10.,10.)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('calibration optimizer failed')
    return fit.x


def loss(p,y):
    p=np.clip(np.asarray(p),1e-12,1-1e-12)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def select(x,y,seed):
    """Use fitting labels only; correct log loss, not mean probability."""
    x=np.asarray(x);y=np.asarray(y)
    folds=np.array_split(np.random.default_rng(seed).permutation(len(y)),5)
    scores={'pack':float(loss(expit(x),y).mean())}
    for penalty in [.1,1.,10.,100.]:
        predicted=np.empty(len(y))
        for held in folds:
            train=np.setdiff1d(np.arange(len(y)),held)
            a,b=fit_affine(x[train],y[train],penalty)
            predicted[held]=expit(a*x[held]+b)
        scores[str(penalty)]=float(loss(predicted,y).mean())
    chosen=min(scores,key=scores.get)
    return chosen,scores


def metrics(p,rows):
    p=np.asarray(p);y=np.array([r['label']=='reprocess' for r in rows],int)
    hit=(p>=.5)==y
    return {'n':len(rows),'correct':int(hit.sum()),'accuracy':float(hit.mean()),
        'brier':float((2*(p-y)**2).mean()),'log_loss':float(loss(p,y).mean()),
        'per_label':{label:{'n':int((y==value).sum()),
            'correct':int(hit[y==value].sum())} for label,value in [('discard',0),('reprocess',1)]},
        'strata':{level:{'n':sum(r['evidence_level']==level for r in rows),
            'correct':int(hit[[r['evidence_level']==level for r in rows]].sum())}
            for level in sorted({r['evidence_level'] for r in rows})}}


def verify_files(manifest,root):
    for name,expected in manifest.get('outputs',{}).items():
        if digest(root/name)!=expected:raise ValueError(f'frozen output changed: {name}')
    for name,expected in manifest.get('source_files',{}).items():
        if digest(name)!=expected:raise ValueError('pinned evidence changed')


def run(cohort,capture,out,calibration):
    if out.exists():raise ValueError('use a fresh comparison directory')
    plan=json.loads((cohort/'study-plan.json').read_text())
    verify_files(json.loads((cohort/'manifest.json').read_text()),cohort)
    if not json.loads((capture/'parity.json').read_text())['passed']:
        raise ValueError('serving parity did not pass')
    baseline_path=Path(__file__).resolve().parents[1]/'sweep/out/codegraph-baseline/plan.json'
    baseline_plan=json.loads(baseline_path.read_text())
    if (calibration.resolve()!=Path(baseline_plan['calibration']).resolve()
            or digest(calibration)!=baseline_plan['sha256'][str(calibration.resolve())]):
        raise ValueError('frozen pack calibration changed')
    cases=[json.loads(l) for l in (cohort/'primary.jsonl').read_text().splitlines()]
    cached={r['original_key']:r for r in [json.loads(l) for l in (capture/'captures.jsonl').read_text().splitlines()]}
    if set(cached)!={c['key'] for c in cases}:raise ValueError('capture/case set mismatch')
    if len({r['model_id'] for r in cached.values()})!=1:raise ValueError('mixed models')
    temperature=json.loads(calibration.read_text()).get('temperatures',{}).get('choice',1.)
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('invalid pack temperature')
    for c in cases:
        r=cached[c['key']]
        if r['input_sha256']!=identity([c['state'],c['question']]):raise ValueError('case input changed')
        if c['gold_status']!='source_documented_action':raise ValueError('unknown label provenance')
        if set(r['keys'])!={'reprocess','discard'} or not np.isfinite(r['logits']).all():
            raise ValueError('invalid option logits')
        c['gap']=(r['logits'][r['keys'].index('reprocess')]-r['logits'][r['keys'].index('discard')])/temperature
    role_by_family={}
    for c in cases:
        if c['family_id'] in role_by_family:raise ValueError('primary cohort repeats a family')
        role_by_family[c['family_id']]=c['role']
    fit=[c for c in cases if c['role']=='fit'];test=[c for c in cases if c['role']=='test']
    if len(fit)!=25 or len(test)!=16:raise ValueError('frozen role sizes changed')
    x_test=np.array([r['gap'] for r in test]);base=expit(x_test)
    y_test=np.array([r['label']=='reprocess' for r in test],int)
    results=[]
    for seed in plan['draw_seeds']:
        ordered=sorted(fit,key=lambda r:identity(['queue-fit-order-v1',seed,r['key']]))
        for budget in plan['budgets']:
            subset=ordered[:budget]
            if budget:
                x=np.array([r['gap'] for r in subset]);y=np.array([r['label']=='reprocess' for r in subset],int)
                chosen,cv=select(x,y,seed)
                theta=np.array([1.,0.]) if chosen=='pack' else fit_affine(x,y,float(chosen))
                predicted=expit(theta[0]*x_test+theta[1])
                prior=np.full(len(test),(y.sum()+.5)/(len(y)+1))
            else:
                chosen,cv,theta,predicted,prior='pack',{},np.array([1.,0.]),base,None
            old=(base>=.5)==y_test;new=(predicted>=.5)==y_test
            results.append({'seed':seed,'budget':budget,'chosen':chosen,'cv_log_loss':cv,
                'parameters':theta.tolist(),'fit_keys':[r['key'] for r in subset],
                'fit_labels':dict(Counter(r['label'] for r in subset)),
                'metrics':metrics(predicted,test),'prior_only':None if prior is None else metrics(prior,test),
                'corrections':int((new & ~old).sum()),'regressions':int((old & ~new).sum()),
                'p_reprocess':predicted.tolist()})
    rng=np.random.default_rng(2026102399);indices=rng.integers(0,len(test),size=(10000,len(test)))
    summary={}
    for budget in plan['budgets']:
        runs=[r for r in results if r['budget']==budget]
        q=np.array([r['p_reprocess'] for r in runs])
        delta=(((q>=.5)==y_test).astype(float)-((base>=.5)==y_test)).mean(0)
        interval=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
        summary[str(budget)]={'mean_accuracy':float(np.mean([r['metrics']['accuracy'] for r in runs])),
            'mean_brier':float(np.mean([r['metrics']['brier'] for r in runs])),
            'mean_log_loss':float(np.mean([r['metrics']['log_loss'] for r in runs])),
            'kept':dict(Counter(r['chosen'] for r in runs)),
            'corrections':[r['corrections'] for r in runs],'regressions':[r['regressions'] for r in runs],
            'paired_accuracy_delta95':interval,'meaningful_accuracy_gain_established':interval[0]>0}
    report={'state':'complete','label_provenance':'source_documented_action','independently_human_adjudicated':False,
        'fit_groups':25,'test_groups':16,'baseline':metrics(base,test),'summary':summary,'runs':results,
        'test_cases':[{'key':r['key'],'story_id':r['story_id'],'family_id':r['family_id'],
                       'label':r['label'],'evidence_level':r['evidence_level']} for r in test],
        'training_backbone':False,'deployed':False,'certificate':False,
        'interpretation':'Small selected historical action-prediction pilot. Sparse-alert results do not measure diagnosis. Source documentation is not independent adjudication or proof of the best operational action. Repeated draws share one held-out set.'}
    out.mkdir(parents=True)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    paths=[cohort/'manifest.json',cohort/'study-plan.json',capture/'captures.jsonl',capture/'parity.json',capture/'plan.json',baseline_path,calibration,Path(__file__)]
    (out/'manifest.json').write_text(json.dumps({'inputs':{str(p):digest(p) for p in paths}},indent=2)+'\n')
    (out/'frozen-runner.py').write_bytes(Path(__file__).read_bytes())
    return {k:report[k] for k in ['baseline','summary']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cohort','capture','out','calibration']:p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(args.cohort,args.capture,args.out,args.calibration),indent=2))
