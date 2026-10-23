"""Rebuild source-backed replay fixtures from dated observations and later outcomes.

Current descriptions, curator summaries and invented triage labels are never model
inputs. These small replay fixtures are not an adaptation or certification dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from sweep.codegraph_judgment import file_hash, identity, verify


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:
        raise ValueError('timestamps must carry a timezone')
    return result


def comment(source, cid):
    return next(c for c in source['comments'] if c['id']==cid)


def source_anchor(sources, sid, cid, quote):
    source = sources[sid]; row = comment(source,cid)
    if not quote or quote not in row['text']:
        raise ValueError(f'{sid}/{cid}: quote is missing from the source')
    start = row['text'].index(quote)
    return {'story_id':sid,'comment_id':cid,'quote':quote,'start':start,'end':start+len(quote),
            'created_at':row['created_at'],'updated_at':row['updated_at'],
            'url':row.get('url',source['url'])}


def history_anchor(histories, sources, sid, event_id, quote, creation_only=True):
    """Pin event bytes; inputs permit creation, outcomes also permit later edits."""
    event = next(e for e in histories[sid]['history'] if e['id'] == event_id)
    actions = [a for a in event['actions'] if a.get('action') == 'create'
               and a.get('id') == sid and not a.get('entity_type')]
    kind='original_description'
    if len(actions) == 1:
        text = actions[0].get('description', '')
    elif not creation_only:
        revisions=[a['changes']['description']['new'] for a in event['actions']
                   if a.get('action')=='update' and a.get('id')==sid
                   and not a.get('entity_type')
                   and 'description' in a.get('changes',{})]
        if len(revisions)!=1:
            raise ValueError('history outcome must reference a description revision')
        text=revisions[0]
        kind='description_revision'
    else:
        raise ValueError('history input must reference the story creation event')
    if not quote or quote not in text:
        raise ValueError('history quote is missing from the creation event or revision')
    start = text.index(quote)
    return {'story_id': sid, 'history_event_id': event_id, 'source_kind': kind,
            'quote': quote, 'start': start, 'end': start + len(quote),
            'created_at': event['changed_at'], 'updated_at': event['changed_at'],
            'url': sources[sid]['url']}


def make_case(selection, sources, prior, histories=None):
    sid = selection['story_id']
    cutoff_ref = selection['cutoff']
    if 'history_event_id' in cutoff_ref:
        cutoff_event=next(e for e in (histories or {})[cutoff_ref['story_id']]['history']
                          if e['id']==cutoff_ref['history_event_id'])
        cutoff={'created_at':cutoff_event['changed_at']}
    else:
        cutoff = comment(sources[cutoff_ref['story_id']],cutoff_ref['comment_id'])
    cutoff_time = timestamp(cutoff['created_at'])
    input_anchors=[]
    for anchor in selection['inputs']:
        source_id=anchor.get('story_id',sid)
        if 'history_event_id' in anchor:
            evidence=history_anchor(histories or {},sources,source_id,anchor['history_event_id'],anchor['quote'])
        else:
            evidence=source_anchor(sources,source_id,anchor['comment_id'],anchor['quote'])
        if timestamp(evidence['updated_at']) >= cutoff_time:
            raise ValueError('input was edited at or after the recorded finding/correction cutoff')
        if timestamp(evidence['created_at']) > timestamp(evidence['updated_at']):
            raise ValueError('invalid input event chronology')
        input_anchors.append(evidence)
    target=selection['outcome']
    if 'history_event_id' in target:
        outcome_anchors=[history_anchor(histories or {},sources,sid,target['history_event_id'],quote,
                                        creation_only=False) for quote in target['quotes']]
    else:
        outcome_anchors=[source_anchor(sources,sid,target['comment_id'],quote) for quote in target['quotes']]
    if any(timestamp(a['created_at'])<cutoff_time for a in outcome_anchors):
        raise ValueError('outcome predates the finding/correction cutoff')
    if not input_anchors or not outcome_anchors:
        raise ValueError('both observation and outcome evidence are required')
    state={'observations':[a['quote'] for a in input_anchors]}
    # A static instruction and verbatim earlier quotes are the entire model input.
    model_input={'state':state,'instructions':'Identify the documented historical outcome associated with these observations.'}
    expected='\n'.join(a['quote'] for a in outcome_anchors)
    if any(a['quote'] in '\n'.join(state['observations']) for a in outcome_anchors):
        raise ValueError('outcome leaked into model input')
    return {'key':identity(model_input)[:24],'story_id':sid,'title':selection['description'],
            'scope':selection['scope'],'family_id':prior['family_id'],'role':prior['role'],
            'model_input':model_input,'expected_outcome':expected,
            'label_status':'documented_outcome','independently_human_adjudicated':False,
            'input_evidence':input_anchors,'outcome_evidence':outcome_anchors,
            'cutoff':{**cutoff_ref,'created_at':cutoff['created_at']},
            'notes':selection['notes'],
            'limitations':'Chronology follows pinned source metadata and explicit selected cutoffs. Original descriptions, when used, come only from recorded story creation events. Association with a later outcome does not establish that it was uniquely inferable from the earlier observations.'}


EXCLUSIONS = {
    'unresolved_or_unconfirmed_outcome': {139522,146458,140893,137027,131037,128089,130810,137049,
        38517,35511,108031,56557,120166,76270,112797,143327,54007,82624,139444,131930},
    'same_ticket_different_incident': {94171},
    'resolution_or_policy_ambiguous': {138775,139454,129112,147386,147952},
    'earlier_facts_already_contain_the_finding': {132209,136164,135748,142615},
    'insufficient_pre_finding_observations': {132795,140008,144627},
    'related_duplicate_retained_elsewhere': {140516},
}
REASONS = {
    'unresolved_or_unconfirmed_outcome':'The saved record does not establish a sufficiently specific final diagnosis/resolution, or records competing explanations or an unverified mitigation.',
    'same_ticket_different_incident':'The later out-of-memory finding refers to another outage. It cannot label the earlier outage observations merely because both are in the same ticket.',
    'resolution_or_policy_ambiguous':'Later notes revise the explanation or leave policy/causal conditions unclear. Excluded instead of assigning the user a relabeling task.',
    'earlier_facts_already_contain_the_finding':'The available dated notes already contain diagnostic interpretation, or the recorded fix is too vague. They do not establish a clean earlier observation point.',
    'insufficient_pre_finding_observations':'Earlier comments lack the observed failure/context required to make a useful replay; the relevant current description cannot be verified as pre-diagnosis.',
    'related_duplicate_retained_elsewhere':'A related case with timestamped pre-finding observations is retained; the linked incident stays in one dependence group.',
    'no_verified_pre_finding_version':'The current description may contain later research. No version history or suitable earlier comment observations establish a clean input, even when a fix is documented.',
}


def build(previous, selection_path, out):
    if out.exists():
        raise ValueError('use a fresh directory; preserve prior artifacts and reviews')
    verify(previous)
    manifest=json.loads((previous/'manifest.json').read_text())
    sources={int(sid):json.loads(Path(ref['path']).read_text()) for sid,ref in manifest['source_snapshots'].items()}
    previous_rows={r['key']:r for r in read_lines(previous/'cases.draft.jsonl')}
    previous_evidence=read_lines(previous/'evidence.jsonl')
    prior_by_story={r['story_id']:previous_rows[r['key']] for r in previous_evidence}
    selection=json.loads(selection_path.read_text())
    histories={}
    history_hashes={}
    for sid,ref in selection.get('history_snapshots',{}).items():
        path=Path(ref['path'])
        if file_hash(path)!=ref['sha256']:
            raise ValueError('history snapshot hash mismatch')
        histories[int(sid)]=json.loads(path.read_text())
        history_hashes[sid]=ref
    cases=[make_case(s,sources,prior_by_story[s['story_id']],histories) for s in selection['cases']]
    if len({c['story_id'] for c in cases})!=len(cases):
        raise ValueError('duplicate incident')
    family_roles={}
    for case in cases:
        family=case['family_id']
        if family in family_roles and family_roles[family]!=case['role']:
            raise ValueError('family crosses roles')
        family_roles[family]=case['role']
    kept={c['story_id'] for c in cases};audit=[]
    for ref in previous_evidence:
        sid=ref['story_id']
        if sid in kept:
            audit.append({'story_id':sid,'status':'replay_fixture','reason':'Verbatim dated observations precede a separately recorded finding/resolution; expected output comes directly from the later record.'})
        else:
            override=selection.get('exclusion_overrides',{}).get(str(sid))
            if override:
                audit.append({'story_id':sid,'status':'excluded',**override})
                continue
            code=next((code for code,ids in EXCLUSIONS.items() if sid in ids),'no_verified_pre_finding_version')
            audit.append({'story_id':sid,'status':'excluded','reason_code':code,'reason':REASONS[code]})
    out.mkdir(parents=True)
    def jsonl(name, rows):
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    jsonl('cases.jsonl',cases)
    jsonl('inputs.jsonl',[{'key':c['key'],**c['model_input']} for c in cases])
    jsonl('expected.jsonl',[{'key':c['key'],'expected_outcome':c['expected_outcome'],
                            'label_status':c['label_status'],'source':c['outcome_evidence']} for c in cases])
    jsonl('audit.jsonl',audit)
    report={'state':'replay_fixtures_only','date':'2026-10-23','previous_cases':len(prior_by_story),
            'retained_cases':len(cases),'excluded_cases':len(audit)-len(cases),
            'human_review_required':0,'families':len(family_roles),
            'scopes':dict(Counter(c['scope'] for c in cases)),
            'roles':dict(Counter(c['role'] for c in cases)),
            'exclusion_counts':dict(Counter(r['reason_code'] for r in audit if r['status']=='excluded')),
            'retired_metrics':['probe-v4','probe-v4-final'],
            'retirement_reason':'Curator-created triage categories and diagnostic hindsight in previous inputs. Prior proxy accuracy is not evidence of local adaptation.',
            'adaptation_ready':False,'certification_ready':False,'new_model_scoring':False,
            'next_step':selection.get('next_step','Collect original ticket versions or independently timestamped observations and explicit outcomes for more independent incidents. These fixtures validate the replay format; they cannot support a learning curve.'),
            'source_access':selection.get('source_access','Saved source snapshots only.'),
            'limitations':['Observations are exact earlier comment excerpts or pinned original story creation-event excerpts.',
                           'Sparse symptom reports may be compatible with more than one cause; retrospective association does not prove causal identifiability.',
                           'One outcome per incident supplies no independent fitting/evaluation replication for that outcome.',
                           'Neither a closed ticket nor an agent confidence score creates an authoritative outcome label.'],
            'cases':cases,'audit':audit}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'frozen-driver.py').write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps({'previous_manifest_sha256':file_hash(previous/'manifest.json'),
        'selection_sha256':file_hash(selection_path),
        'source_hashes':{sid:r['sha256'] for sid,r in manifest['source_snapshots'].items()},
        'history_snapshots':history_hashes,
        'outputs':{p.name:file_hash(p) for p in out.iterdir() if p.is_file()}},indent=2)+'\n')
    return {k:v for k,v in report.items() if k not in {'cases','audit'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();print(json.dumps(build(args.previous,args.selection,args.out),indent=2))


if __name__=='__main__':main()
