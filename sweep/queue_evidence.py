"""Freeze queue-disposition cases from evidence recorded before the action.

This is an offline prediction study of documented actions, not an operator or a
claim that an action was uniquely inferable or appropriate in a new incident.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from sweep.resolution_replay import timestamp

QUESTION = {
    'type': 'choice',
    'instructions': 'Using only these recorded pre-action observations, predict the final queue-handling mechanism documented for this incident. Redelivery may require a code, data, configuration, or dependency repair first and may complete as an expected no-op. Purging may follow an unsuccessful test redelivery. This is an offline historical prediction, never permission to operate a queue.',
    'criteria': {
        'reprocess': 'Final cleanup used consumer redelivery or reprocessing, after any necessary repair.',
        'discard': 'Final cleanup removed the failed messages rather than delivering them to a consumer again.',
    },
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def events(root, sid):
    source = json.loads((root/'stories'/f'{sid}.json').read_text())
    history = json.loads((root/'history'/f'{sid}.json').read_text())
    result = []
    for event in history['history']:
        for action in event['actions']:
            if action.get('id') != sid or action.get('entity_type'):
                continue
            if action.get('action') == 'create':
                text = action.get('description', '')
            elif 'description' in action.get('changes', {}):
                text = action['changes']['description']['new']
            else:
                continue
            result.append({'kind': 'history', 'event_id': event['id'],
                'recorded_at': event['changed_at'], 'text': text, 'url': source['app_url']})
            if action.get('action')=='create' and action.get('name'):
                result.append({'kind':'creation_title','event_id':event['id'],
                    'recorded_at':event['changed_at'],'text':action['name'],'url':source['app_url']})
    for comment in source.get('comments', []):
        if comment.get('deleted') or not comment.get('text'):
            continue
        result.append({'kind': 'comment', 'event_id': comment['id'],
            'recorded_at': comment['updated_at'], 'created_at': comment['created_at'],
            'text': comment['text'], 'url': comment.get('app_url', source['app_url'])})
    return sorted(result, key=lambda e: timestamp(e['recorded_at']))


def anchor(rows, needle, before=None):
    eligible = [r for r in rows if needle and needle in r['text']
                and (before is None or timestamp(r['recorded_at']) < timestamp(before))]
    if not eligible:
        raise ValueError(f'no earlier exact source for {needle[:90]!r}')
    row = eligible[0]
    start = row['text'].index(needle)
    return {k: v for k, v in row.items() if k != 'text'} | {
        'quote': needle, 'start': start, 'end': start+len(needle)}


def make_case(root, spec):
    sid = spec['story_id']
    rows = events(root, sid)
    outcome = anchor(rows, spec['outcome_quote'])
    cutoff = spec.get('action_not_before', outcome['recorded_at'])
    if timestamp(cutoff) > timestamp(outcome['recorded_at']):
        raise ValueError('action lower bound is later than its completion report')
    observations = [anchor(rows, quote, before=cutoff) for quote in spec['observations']]
    if not observations:
        raise ValueError('observations required')
    for row in observations:
        text = row['quote']
        if re.search(r'\b(?:purged|shoveled|shovelled|replayed|requeued|redrived)\b', text, re.I):
            raise ValueError('completed queue action leaked into observations')
        if re.search(r'(?:password|pwd|client_secret)\s*=|<\w*:pw>', text, re.I):
            raise ValueError('credential-shaped input')
    if spec['label'] not in QUESTION['criteria']:
        raise ValueError('unknown disposition')
    action_patterns={
        'reprocess':r'\b(?:shoveled|shovelled|reshovelled|requeued|reprocessed|replayed|redrove|redrived|republished)\b|shovel.*successfully|pushed.*queue.*processed',
        'discard':r'\b(?:purged|dropped|deleted)\b',
    }
    if not re.search(action_patterns[spec['label']],outcome['quote'],re.I|re.S):
        raise ValueError('outcome does not explicitly record the selected queue action')
    if re.search(r'\b(?:will|should|could|can|must)\s+(?:be\s+)?(?:replayed|reprocessed|shoveled|shovelled|purged|dropped)\b',outcome['quote'],re.I):
        raise ValueError('a proposed action is not a completed outcome')
    state = {'observations': [r['quote'] for r in observations]}
    identity = json.dumps([state, QUESTION], sort_keys=True).encode()
    input_hash=hashlib.sha256(identity).hexdigest()
    case_key=hashlib.sha256(f'{sid}:{input_hash}'.encode()).hexdigest()[:24]
    return {'key': case_key, 'input_sha256':input_hash, 'story_id': sid,
        'state': state, 'question': QUESTION, 'label': spec['label'],
        'gold_status': 'source_documented_action', 'independently_human_adjudicated': False,
        'input_evidence': observations, 'outcome_evidence': outcome,
        'action_cutoff': cutoff, 'notes': spec['notes'],
        'evidence_level':spec.get('evidence_level','diagnostic'),
        'source_sha256': digest(root/'stories'/f'{sid}.json'),
        'history_sha256': digest(root/'history'/f'{sid}.json')}


def groups(root, cases):
    """Join linked stories, shared incident IDs, and shared fixes.

    Identical observations in genuinely different incidents are repeated outcomes,
    not paraphrases of one incident. Track them separately as a signal limitation.
    """
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def join(a, b):
        parent[find(a)] = find(b)
    owners = {}
    # Follow formal links through all saved intermediate stories too.
    for path in sorted((root/'stories').glob('*.json')):
        source=json.loads(path.read_text())
        for link in source.get('story_links',[]):
            join(link['subject_id'],link['object_id'])
        if source.get('parent_story_id'):
            join(source['id'],source['parent_story_id'])
    for case in cases:
        sid = case['story_id']; find(sid)
        source = json.loads((root/'stories'/f'{sid}.json').read_text())
        for link in source.get('story_links', []):
            join(sid, link['subject_id']); join(sid, link['object_id'])
        if source.get('parent_story_id'):
            join(sid, source['parent_story_id'])
        # Include prior case snapshots in links through shared external stubs.
        text = source.get('description', '') or ''
        text += '\n'+'\n'.join(c.get('text') or '' for c in source.get('comments', []))
        refs = set(re.findall(r'https://turncommerce\.pagerduty\.com/incidents/([A-Za-z0-9]+)', text))
        refs |= set(re.findall(r'https://gitlab\.tcdevops\.com/[^\s)<>]+/-/merge_requests/\d+', text))
        refs |= set(re.findall(r'https://gitlab\.tcdevops\.com/[^\s)<>]+/-/commit/[0-9a-fA-F]{8,40}',text))
        refs={ref.lower() for ref in refs}
        for ref in refs:
            if ref in owners: join(sid, owners[ref])
            else: owners[ref] = sid
    # Same-day December 24 cleanup alerts may share an upstream incident.
    if 133736 in parent and 133737 in parent:
        join(133736,133737)
    members = {}
    for sid in parent:
        members.setdefault(find(sid), []).append(sid)
    for case in cases:
        case['family_id'] = 'queue:'+str(min(members[find(case['story_id'])]))
        case['role'] = 'unassigned'
    return cases


def build(root, selection, out):
    if out.exists():
        raise ValueError('use a fresh output directory')
    specs = json.loads(selection.read_text())['cases']
    cases = groups(root, [make_case(root, spec) for spec in specs])
    if len({c['story_id'] for c in cases}) != len(cases):
        raise ValueError('one selected episode per story')
    out.mkdir(parents=True)
    (out/'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in cases))
    (out/'question.json').write_text(json.dumps(QUESTION, indent=2)+'\n')
    (out/'frozen-builder.py').write_bytes(Path(__file__).read_bytes())
    refs = sorted((root/'stories').glob('*.json'))
    refs += [root/'history'/f"{c['story_id']}.json" for c in cases]
    (out/'manifest.json').write_text(json.dumps({'selection_sha256': digest(selection),
        'source_files': {str(p): digest(p) for p in refs},
        'outputs': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}, indent=2)+'\n')
    return cases
