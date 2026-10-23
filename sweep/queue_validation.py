"""Admission gate for fresh queue validation; never scores an undersized cohort."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from sweep.codegraph_judgment import identity
from sweep.queue_evidence import QUESTION, anchor, digest, events, make_case
from sweep.resolution_replay import timestamp


def gate(rows, target):
    counts=Counter(r['label'] for r in rows)
    return {'passed':len({r['family_id'] for r in rows})>=target['minimum_groups']
            and all(counts[label]>=target['minimum_per_label'] for label in QUESTION['criteria'])
            and sum(r['evidence_level']=='diagnostic' for r in rows)>=target['minimum_diagnostic_groups'],
            'groups':len({r['family_id'] for r in rows}),'per_label':dict(counts),
            'diagnostic':sum(r['evidence_level']=='diagnostic' for r in rows),'required':target}


def returned_case(root, spec):
    """Accept an explicit past return to a consumer queue, not generic 'cleared'.

    Extends admission wording only; frozen model inputs/features stay unchanged.
    """
    quote=spec['outcome_quote']
    if not re.fullmatch(r'(?:After the deploy, I pushed the error queue messages back to their parent queue and all messages were cleared\.|Pushed all messages back to their normal queues and they cleared\.)',quote):
        raise ValueError('not an explicit completed return to consumer queues')
    source=events(root,spec['story_id']);outcome=anchor(source,quote)
    cutoff=spec.get('action_not_before',outcome['recorded_at'])
    if timestamp(cutoff)>timestamp(outcome['recorded_at']):raise ValueError('invalid cutoff')
    evidence=[anchor(source,q,before=cutoff) for q in spec['observations']]
    if not evidence or any(re.search(r'\b(?:purged|shoveled|shovelled|replayed|requeued|redrived)\b|(?:password|pwd|client_secret)\s*=|<\w*:pw>',e['quote'],re.I) for e in evidence):
        raise ValueError('empty or unsafe evidence')
    state={'observations':[e['quote'] for e in evidence]};h=identity([state,QUESTION])
    return {'key':identity([spec['story_id'],h])[:24],'input_sha256':h,
        'story_id':spec['story_id'],'state':state,'question':QUESTION,'label':'reprocess',
        'gold_status':'source_documented_action','independently_human_adjudicated':False,
        'input_evidence':evidence,'outcome_evidence':outcome,'action_cutoff':cutoff,
        'notes':spec['notes'],'evidence_level':spec['evidence_level'],
        'source_sha256':digest(root/'stories'/f"{spec['story_id']}.json"),
        'history_sha256':digest(root/'history'/f"{spec['story_id']}.json")}


def dependence(sources):
    parent={};owners={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def join(a,b):parent[find(a)]=find(b)
    for sid,s in sources.items():
        find(sid)
        for link in s.get('story_links',[]):join(link['subject_id'],link['object_id'])
        if s.get('parent_story_id'):join(sid,s['parent_story_id'])
        text=(s.get('description') or '')+'\n'+'\n'.join(c.get('text') or '' for c in s.get('comments',[]))
        refs=set(re.findall(r'https://turncommerce\.pagerduty\.com/incidents/([A-Za-z0-9]+)',text))
        refs|=set(re.findall(r'https://gitlab\.tcdevops\.com/[^\s)<>]+/-/(?:merge_requests/\d+|commit/[0-9a-fA-F]{8,40})',text))
        for ref in refs:
            ref=ref.lower()
            if ref in owners:join(sid,owners[ref])
            else:owners[ref]=sid
    return {sid:find(sid) for sid in parent}


def build(root, selection, prior_root):
    if (root/'report.json').exists():raise ValueError('preserve existing audit')
    plan=json.loads((root/'study-plan.json').read_text())
    for path,sha in plan['frozen_hashes'].items():
        if digest(path)!=sha:raise ValueError('frozen implementation changed')
    previous=set(plan['excluded_previously_seen_story_ids'])
    fresh={int(p.stem):json.loads(p.read_text()) for p in (root/'stories').glob('*.json')}
    old={int(p.stem):json.loads(p.read_text()) for p in (prior_root/'stories').glob('*.json')}
    families=dependence(old|fresh)
    reviewed=json.loads(selection.read_text());specs={s['story_id']:s for s in reviewed['cases']}
    admitted=[];audit=[]
    for sid,s in sorted(fresh.items()):
        overlap=sorted(k for k in previous if families.get(k,k)==families[sid])
        entry={'story_id':sid,'name':s['name'],'url':s['app_url'],'source_sha256':digest(root/'stories'/f'{sid}.json')}
        if sid in previous or overlap:
            entry.update(status='excluded',reason='dependence_on_previously_seen_story',prior_members=overlap)
        elif sid in specs:
            spec=specs[sid]
            case=returned_case(root,spec) if spec.get('explicit_return') else make_case(root,spec)
            case.update(family_id='fresh-queue:'+str(families[sid]),role='validation_candidate')
            admitted.append(case);entry.update(status='admitted',case_key=case['key'])
        else:
            entry.update(status='excluded',reason=reviewed['exclusions'].get(str(sid),'no_verified_completed_queue_disposition_in_screened_record'))
        audit.append(entry)
    # One representative per family, deterministic and independent of model outputs.
    reps={}
    for c in sorted(admitted,key=lambda c:identity(['fresh-representative-v1',c['key']])):reps.setdefault(c['family_id'],c)
    rows=list(reps.values());admission=gate(rows,plan['target'])
    def write(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
    (root/'cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in rows))
    write('admission-audit.json',audit)
    report={'status':'ready_for_frozen_scoring' if admission['passed'] else 'insufficient_fresh_evidence',
        'fresh_records_screened':len(fresh),'history_records':len(list((root/'history').glob('*.json'))),
        'prior_records_excluded':len(previous),'gate':admission,
        'exclusion_counts':dict(Counter(a.get('reason') for a in audit if a['status']=='excluded')),
        'gpu_used':False,'model_predictions_computed':False,'training_started':False,
        'limitations':['Search coverage is bounded to saved queries; not an exhaustive Codegraph crawl.',
                      'Records lacking a verified final disposition remain excluded, not guessed.',
                      'Chronology uses recorded evidence timestamps; independent execution clocks unavailable.',
                      'Some earlier repairs are documented before cleanup; this is a decision checkpoint, not necessarily initial intake.',
                      'Fresh cases are quarantined from fitting; never reuse them for tuning and call them validation.'],
        'next':'Collect additional independently timed queue outcomes using prospective-case-template.json. No scoring until frozen minimum passes.'}
    write('report.json',report)
    write('prospective-case-template.json',{
        'incident_id':None,'source_url':None,'related_stories_incidents_fixes':[],
        'checkpoint':{'recorded_at':None,'evidence_url':None,'observations':[],
                      'instruction':'Save and pin this checkpoint before the queue action; missing facts remain unknown.'},
        'outcome':{'action_started_at':None,'recorded_at':None,'evidence_url':None,'quote':None,'documented_mechanism':None,
                   'allowed_mechanisms':['reprocess','discard'],
                   'instruction':'Fill only after completion; record explicit queue mechanism separately from success. Never edit checkpoint using later findings.'},
        'inclusion':'All eligible consecutive incidents, both mechanisms; no model-dependent admission.'})
    (root/'frozen-admission-builder.py').write_bytes(Path(__file__).read_bytes())
    paths=[p for p in root.rglob('*') if p.is_file() and p.name!='manifest.json']+list((prior_root/'stories').glob('*.json'))
    write('manifest.json',{'sha256':{str(p):digest(p) for p in paths}})
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--prior-root',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(build(args.root,args.selection,args.prior_root),indent=2))
