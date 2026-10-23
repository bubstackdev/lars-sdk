"""Cache parity-checked 2B outputs for reviewed cases without fitting draft labels."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

from sweep.codegraph_baseline import guarded, parity, probabilities
from sweep.codegraph_judgment import file_hash, identity

ROOT = Path(__file__).resolve().parents[1]


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_lines(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def json_file(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def prepare(data_path, out):
    if out.exists():
        raise ValueError('use a fresh capture directory')
    previous = ROOT/'sweep/out/codegraph-baseline'
    baseline = json.loads((previous/'plan.json').read_text())
    for key in ('runtime', 'model', 'template'):
        if file_hash(baseline[key]) != baseline['sha256'][baseline[key]]:
            raise ValueError('baseline runtime/model/template changed')
    data = json.loads(data_path.read_text())
    out.mkdir(parents=True)
    wire = [{'record_key':c['key'],'state':c['state'],'question':c['question'],
             'label':c['draft_label'],'gold_status':'source_interpreted_unreviewed'} for c in data['cases']]
    write_lines(out/'cases.wire.jsonl', wire)
    for name in ('controls.wire.jsonl','controls.expected.jsonl'):
        if file_hash(previous/name) != baseline['sha256'][str(previous/name)]:
            raise ValueError('baseline controls changed')
        shutil.copy2(previous/name, out/name)
    paths = [data_path,out/'cases.wire.jsonl',out/'controls.wire.jsonl',out/'controls.expected.jsonl',
             Path(__file__),ROOT/'sweep/codegraph_baseline.py']+[Path(baseline[k]) for k in ('runtime','model','template')]
    plan = {k:baseline[k] for k in ('runtime','model','template')}
    plan.update({'data_path':str(data_path.resolve()),'cases':len(wire),
                 'sha256':{str(p.resolve()):file_hash(p) for p in paths},
                 'model_id': 'sha256:'+file_hash(baseline['model']),
                 'hidden':False,'fitting':False,'certification':False,
                 'draft_label_use':'Native capture requires a label for bookkeeping/id only; it is excluded from model input. No draft labels are fitted, certified or promoted.',
                 'guard':'normal memory pressure, own direct child only, 600s timeout'})
    json_file(out/'plan.json',plan)
    shutil.copy2(__file__,out/'frozen-driver.py')
    return plan


def capture(out):
    plan = json.loads((out/'plan.json').read_text())
    if any(file_hash(p) != sha for p,sha in plan['sha256'].items()):
        raise ValueError('frozen capture inputs changed')
    if (out/'captures.jsonl').exists():
        raise ValueError('completed capture exists')
    with (out/'capture.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started = time.monotonic()
        def invoke(name, source):
            guarded([plan['runtime'],'capture','--model',plan['model'],'--template',plan['template'],
                     '--data',str(source),'--out',str(out/f'{name}.raw.jsonl')],out/f'{name}.log')
            return lines(out/f'{name}.raw.jsonl')
        try:
            json_file(out/'status.json',{'state':'control_parity'})
            control = parity(invoke('controls',out/'controls.wire.jsonl'),lines(out/'controls.expected.jsonl'))
            json_file(out/'parity.json',control)
            if not control['passed']:
                raise ValueError('serving parity failed')
            json_file(out/'status.json',{'state':'capturing_cases'})
            raw = invoke('cases',out/'cases.wire.jsonl')
            wire = lines(out/'cases.wire.jsonl')
            if len(raw) != len(wire):
                raise ValueError('missing captures')
            saved = []
            for row,case in zip(raw,wire):
                if (row['id'] != identity([case['state'],case['question'],case['label']])[:16]
                        or row['label'] != case['label'] or row['kind'] != case['question']['type']
                        or set(row['keys']) != set(case['question']['criteria'])):
                    raise ValueError('native row/options identity mismatch')
                probabilities(row)  # Validate finite logits and option alignment.
                saved.append({'original_key':case['record_key'],'state':case['state'],
                              'question':case['question'],'input_sha256':identity([case['state'],case['question']]),
                              'kind':row['kind'],'keys':row['keys'],'logits':row['logits'],
                              'model_id':plan['model_id']})
            write_lines(out/'captures.jsonl',saved)
            result = {'state':'complete','rows':len(saved),'serving_parity':control,
                      'elapsed_seconds':time.monotonic()-started,'fit_or_training':False,
                      'hidden':False,'human_labels_used':False}
            json_file(out/'status.json',result)
            return result
        except BaseException as error:
            json_file(out/'status.json',{'state':'failed','error':str(error)})
            raise


def join_reviewed(export, captures):
    cached = {r['original_key']:r for r in captures}
    joined, recapture = [], []
    for row in export['reviewed_cases']:
        if row.get('gold_status') != 'reviewed':
            raise ValueError('only human-reviewed export rows can enter the learning curve')
        cache = cached.get(row['original_key'])
        if cache is None or cache['input_sha256'] != identity([row['state'],row['question']]):
            recapture.append(row['original_key']);continue
        if set(cache['keys']) != set(row['question']['criteria']) or row['label'] not in cache['keys']:
            raise ValueError('reviewed option semantics changed')
        hard = cache['keys'].index(row['label'])
        joined.append({'key':row['key'],'state':row['state'],'question':row['question'],
                       'family_id':row['family_id'],'role':row['role'],
                       'gold_status':'reviewed','model_id':cache['model_id'],
                       'predicate_id':'codegraph/judgment/'+identity(row['question'])[:16],
                       'workflow':'codegraph/judgment','kind':cache['kind'],'keys':cache['keys'],
                       'logits':cache['logits'],'hard_index':hard,
                       'target':[float(i==hard) for i in range(len(cache['keys']))]})
    return joined, recapture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare');prep.add_argument('--data',type=Path,required=True);prep.add_argument('--out',type=Path,required=True)
    run = sub.add_parser('capture');run.add_argument('--out',type=Path,required=True);run.add_argument('--allow-gpu',action='store_true')
    join = sub.add_parser('join');join.add_argument('--reviews',type=Path,required=True);join.add_argument('--capture',type=Path,required=True);join.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.data,args.out);result={'prepared':True,'cases':result['cases']}
    elif args.command == 'capture':
        if not args.allow_gpu:raise ValueError('--allow-gpu required')
        result = capture(args.out)
    else:
        if args.out.exists():raise ValueError('preserve previous joined labels; choose a fresh path')
        joined,recapture=join_reviewed(json.loads(args.reviews.read_text()),lines(args.capture))
        write_lines(args.out,joined)
        result={'joined':len(joined),'needs_recapture':recapture,'fitting':False}
        json_file(args.out.with_suffix('.status.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
