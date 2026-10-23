"""Source-audited Codegraph contract probes through the frozen 2B SDK runtime."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from lars.train.accuracy import load_capture
from sweep.accuracy_ladder import digest
from sweep.experiment import atomic_json
from sweep.local_calibration import read_jsonl, write_jsonl

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'sweep/out/codegraph-calibration'
FULL=ROOT/'sweep/out/accuracy-full'


def audit_tier(row):
    source=row['source'].rsplit('/',1)[-1]
    if source=='story_http_retry':
        return 'quarantined','Wiki says three retries/non-transient fail-fast; code has three total attempts and catches/retries HttpRequestException from permanent errors.'
    if source=='internal_refund' and row['state']['transaction_type']=='Withdrawal':
        return 'quarantined','README excludes withdrawals, but WithdrawalTransaction.Refund and public service dispatch implement refunds.'
    if source=='domain_return':
        return 'quarantined','Constructed facts omit runtime guards/context, including SellerX business division and matching returned-domain record.'
    if source in ('billing_retry','invoice_closure','internal_refund'):
        return 'implementation_corroborated','Expected rule agrees with inspected pinned implementation; billing retry and invoice closure also have matching test-source assertions.'
    return 'documentation_only','Expected answer follows the supplied pinned policy; no executable decision rule independently established.'


def source_content(path):
    value=json.loads(path.read_text())
    if isinstance(value.get('content'),list):
        value=json.loads(value['content'][0]['text'])
    if 'result' in value:
        return '\n'.join(line['text'] for line in value['result']['lines'])
    return value['content']


def pressure():
    return int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True).strip())


def guarded(command, log_path, timeout=600, check_pressure=pressure):
    """Controls ONLY the direct child created here; never process groups or other sessions."""
    if check_pressure()!=1:
        raise RuntimeError('memory pressure is not normal; no worker started')
    with Path(log_path).open('w') as log:
        child=subprocess.Popen(list(map(str,command)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        started=time.monotonic()
        try:
            while child.poll() is None:
                if check_pressure()!=1:
                    raise RuntimeError('memory guard: stopping only this capture child')
                if time.monotonic()-started>timeout:
                    raise RuntimeError('capture time budget exceeded')
                time.sleep(.25)
            if child.returncode:
                raise RuntimeError(f'capture exited{child.returncode}; see {log_path}')
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait()


def probabilities(row):
    z=np.asarray(row['logits'],dtype=float)
    if z.shape!=(len(row['keys']),) or not np.isfinite(z).all():
        raise ValueError('invalid capture logits')
    p=np.exp(z-z.max());return p/p.sum()


def parity(actual,expected,tolerance=.001):
    if len(actual)!=len(expected):raise ValueError('control population differs')
    maximum=0.;flips=0
    for row,ref in zip(actual,expected):
        if row['keys']!=ref['keys'] or row['kind']!=ref['kind']:
            raise ValueError('control option semantics differ')
        p=probabilities(row);q=np.asarray(ref['p'])
        maximum=max(maximum,float(abs(p-q).max()));flips+=int(p.argmax()!=q.argmax())
    return {'passed':maximum<=tolerance and flips==0,'max_probability_delta':maximum,'label_flips':flips,'rows':len(actual)}


def prepare(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise ValueError('preserve previous plan; use a fresh directory')
    manifest=json.loads((CORPUS/'cases-manifest.json').read_text())
    for name,sha in manifest['outputs'].items():
        if digest(CORPUS/name)!=sha:raise ValueError('draft corpus changed')
    audit_files=list((out/'audit-sources').glob('*.json'))
    needed={'billing_config':'BillingMethodNotFoundException','billing_tests':'TotalAttempts = 16',
        'billing_consumer':'catch (DuplicateLineItemException','billing_invoice':'CheckZeroBalance()',
        'billing_method':'LineItems.Sum(li => li.Amount) != 0','billing_close_tests':'ZeroingCredit_OnInvoice_ClosesAndNotifiesOnce',
        'refund_status':'FabalTransactionStatus.Completed','withdrawal_refund':'Refund(decimal',
        'http_client':'attempt <= MaxRetries','domain_closed':'SellerX'}
    for name,phrase in needed.items():
        if phrase not in source_content(out/f'audit-sources/{name}.json'):
            raise ValueError(f'missing source corroboration: {name}')
    rows=read_jsonl(CORPUS/'cases.wire.jsonl')
    audited=[]
    for row in rows:
        tier,reason=audit_tier(row)
        audited.append({'key':row['record_key'],'source':row['source'],'family_id':row['family_id'],
            'tier':tier,'reason':reason,'gold_status':row['gold_status'],'source_reference':row['source_reference']})
    write_jsonl(out/'audit.jsonl',audited);write_jsonl(out/'cases.wire.jsonl',rows)
    controls=read_jsonl(FULL/'data/typed-regression.wire.jsonl')[:8]
    expected=load_capture(FULL/'baseline-s0-typed-regression.jsonl')
    write_jsonl(out/'controls.wire.jsonl',controls)
    write_jsonl(out/'controls.expected.jsonl',[expected[row['record_key']] for row in controls])
    previous=json.loads((FULL/'baseline-s0-typed-regression.done.json').read_text())['identity']
    runtime=ROOT/'target/release/lars-dev';model=FULL/'baseline-s0-Q4_K_M.gguf';template=ROOT/'sweep/templates/Qwen3.5-2B.json'
    for path in [runtime,model,template]:
        if digest(path)!=previous['inputs'][str(path)]:raise ValueError('original SDK/model/template identity changed')
    cal=FULL/'baseline-s0-calibration.json'
    pinned=[out/'cases.wire.jsonl',out/'audit.jsonl',out/'controls.wire.jsonl',out/'controls.expected.jsonl',
        cal,runtime,model,template,Path(__file__).resolve(),*audit_files]
    plan={'format':1,'rows':len(rows),'tier_counts':dict(Counter(r['tier'] for r in audited)),
        'runtime':str(runtime),'model':str(model),'template':str(template),'calibration':str(cal),
        'sha256':{str(p.resolve()):digest(p) for p in pinned},'guard':'normal memory pressure required; own direct child only;600s timeout',
        'capture':'Original SDK defaults including flash attention off; try hidden extraction only if8-control serving parity passes, else logits-only fallback must pass.',
        'interpretation':'Synthetic policy-following diagnostic. Source-reviewed is not independently human/company-reviewed gold; C# test sources inspected, not executed. Quarantined cases never contribute to implementation-corroborated score. No calibration fit, training, certification or deployment claims.'}
    atomic_json(out/'plan.json',plan);atomic_json(out/'status.json',{'state':'prepared','rows':len(rows),'tier_counts':plan['tier_counts']})
    print(json.dumps(plan['tier_counts'],indent=2))


def summarize(rows):
    counts=Counter(r['gold_key'] for r in rows)
    return {'n':len(rows),'correct':sum(r['correct'] for r in rows),'accuracy':float(np.mean([r['correct'] for r in rows])),
        'families':len({r['family_id'] for r in rows}),
        'per_source_majority_reference':sum(max(Counter(r['gold_key'] for r in rows if r['source']==s).values()) for s in {r['source'] for r in rows})/len(rows)}


def capture(args):
    if not args.allow_gpu:raise ValueError('--allow-gpu required')
    out=args.out.resolve();m=json.loads((out/'plan.json').read_text())
    if any(digest(Path(path))!=sha for path,sha in m['sha256'].items()):raise ValueError('frozen input/code changed')
    import fcntl
    with (out/'capture.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'results.json').exists():raise ValueError('completed result exists')
        started=time.time();atomic_json(out/'status.json',{'state':'control_parity','started':started})
        def invoke(name,data,hidden):
            command=[m['runtime'],'capture','--model',m['model'],'--template',m['template'],'--data',str(data),'--out',str(out/f'{name}.raw.jsonl')]
            if hidden:command+=['--hidden',str(out/f'{name}.f32')]
            guarded(command,out/f'{name}.log')
            return read_jsonl(out/f'{name}.raw.jsonl')
        try:
            actual=invoke('controls-hidden',out/'controls.wire.jsonl',True)
            control=parity(actual,read_jsonl(out/'controls.expected.jsonl'));hidden=control['passed']
            atomic_json(out/'controls-hidden.parity.json',control)
            if not hidden:
                control=parity(invoke('controls-logits',out/'controls.wire.jsonl',False),read_jsonl(out/'controls.expected.jsonl'))
                atomic_json(out/'controls-logits.parity.json',control)
                if not control['passed']:raise ValueError('SDK baseline control parity failed')
            atomic_json(out/'status.json',{'state':'capturing_cases','started':started,'hidden':hidden})
            raw=invoke('cases',out/'cases.wire.jsonl',hidden)
            cases=read_jsonl(out/'cases.wire.jsonl');audit=read_jsonl(out/'audit.jsonl')
            if len(raw)!=len(cases):raise ValueError('capture omitted decisions')
            scored=[];cal=json.loads(Path(m['calibration']).read_text())
            for row,case,review in zip(raw,cases,audit):
                if row['label']!=case['label'] or row['kind']!=case['question']['type'] or set(row['keys'])!=set(case['question']['criteria']):
                    raise ValueError('SDK option/label alignment differs')
                p=probabilities(row);gold=row['keys'].index(case['label']);z=np.asarray(row['logits'])
                temperature=cal.get('temperatures',{}).get(row['kind'],1.)
                cp=np.exp((z-z.max())/temperature);cp/=cp.sum()
                scored.append(review|{'kind':row['kind'],'keys':row['keys'],'gold_key':case['label'],
                    'gold_index':gold,'predicted_key':row['keys'][int(p.argmax())],'correct':bool(p.argmax()==gold),
                    'logits':row['logits'],'raw_p':p.tolist(),'pack_p':cp.tolist()})
            dim=None
            if hidden:
                h=np.fromfile(out/'cases.f32',dtype='<f4')
                if h.size%len(cases) or not h.size or not np.isfinite(h).all():raise ValueError('hidden output is not row-aligned/finite')
                dim=h.size//len(cases)
            write_jsonl(out/'scored.jsonl',scored)
            by_source={s:summarize([r for r in scored if r['source']==s]) for s in sorted({r['source'] for r in scored})}
            by_tier={s:summarize([r for r in scored if r['tier']==s]) for s in sorted({r['tier'] for r in scored})}
            result={'all_policy_following':summarize(scored),'per_source':by_source,'per_audit_tier':by_tier,
                'macro_source_accuracy':float(np.mean([r['accuracy'] for r in by_source.values()])),
                'serving_parity':control,'hidden_dimension':dim,'elapsed_seconds':time.time()-started,
                'interpretation':m['interpretation']}
            atomic_json(out/'results.json',result);atomic_json(out/'status.json',{'state':'complete','finished':time.time(),'rows':len(scored)})
            print(json.dumps(result,indent=2))
        except BaseException as exc:
            atomic_json(out/'status.json',{'state':'failed','error':str(exc),'finished':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['prepare','capture'])
    p.add_argument('--out',type=Path,default=Path('sweep/out/codegraph-baseline'));p.add_argument('--allow-gpu',action='store_true')
    a=p.parse_args();globals()[a.command](a)
