"""Freeze a local, reviewable historical support-triage corpus.

No inference, generated paraphrases, training, external writes, or gold promotion.
Source snapshots and manual curation live in gitignored local outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


OPTIONS = {
    'application_engineering': 'Investigate or change application logic for a reproducible defect or missing supported behavior.',
    'operational_state_recovery': 'Repair or reconcile identified stuck, partial, or inconsistent operational state; this does not rule out later prevention work.',
    'customer_or_integration_configuration': 'Correct a demonstrated customer configuration or service integration setting; do not assume customer fault.',
    'infrastructure_or_upstream': 'Investigate or restore demonstrated host, network, dependency, or authoritative upstream failure.',
    'expected_policy_behavior': 'Explain or use the intended workflow or restriction for the primary symptom; a secondary UX issue may still exist.',
    'insufficient_evidence': 'Collect discriminating evidence when the available observations do not support a specific track.',
}
QUESTION = {
    'type': 'choice',
    'instructions': (
        'Choose the primary investigation track best supported by this support report and its '
        'observations at the reconstructed decision point. Address the primary symptom, not every '
        'possible future improvement. A workaround or successful retry does not establish a root '
        'cause. Prefer insufficient_evidence when competing explanations remain unresolved. '
        'This is an offline recommendation, not permission to execute any remediation.'
    ),
    'criteria': OPTIONS,
}
ROLES = ('fit', 'fit', 'certificate', 'test')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    return re.sub(r'\s+', ' ', value).strip().casefold()


def family_role(family):
    # Fixed without inspecting labels, class balance, or model predictions.
    return ROLES[int(identity(['judgment-family-split-v1', family]), 16) % len(ROLES)]


def model_input(case):
    # Explicit allowlist: resolution, source titles/URLs, labels and roles stay out.
    return {'state': {'report': case['summary'], 'observations': case['observations']},
            'question': QUESTION}


def grouped_families(seed, sources):
    """Conservatively join formal story links, including shared external stubs.

    A 'relates to' edge can over-group independent work. That costs support, but
    prevents relying on an unexamined related ticket as independent evaluation.
    """
    parents = {}
    def root(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = root(parents[key])
        return parents[key]
    def join(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)
    for row in seed['rows']:
        join(f'story:{row["story_id"]}', f'family:{row["family_id"]}')
    for sid, source in sources.items():
        if source.get('parent_story_id'):
            join(f'story:{sid}', f'story:{source["parent_story_id"]}')
        for link in source.get('story_links', []):
            join(f'story:{link["subject_id"]}', f'story:{link["object_id"]}')
    return {r['story_id']: root(f'story:{r["story_id"]}') for r in seed['rows']}


def validate(seed, sources):
    if seed.get('status') != 'source_interpreted_unreviewed':
        raise ValueError('this builder cannot promote labels to reviewed gold')
    if not seed.get('rows'):
        raise ValueError('empty corpus')
    seen_ids, seen_inputs = set(), {}
    for row in seed['rows']:
        sid = row['story_id']
        if sid in seen_ids:
            raise ValueError('duplicate source story: do not inflate support with variants')
        seen_ids.add(sid)
        source = sources[sid]
        if source['id'] != sid or not source.get('updated_at'):
            raise ValueError('source identity or retrieval revision missing')
        if row['draft_label'] not in OPTIONS or not row.get('family_id'):
            raise ValueError('invalid label/family')
        if not row.get('summary') or not row.get('observations') or not row.get('review_notes'):
            raise ValueError('missing reconstructed facts or review notes')
        payload = model_input(row)
        rendered = json.dumps(payload, ensure_ascii=False)
        if re.search(r'https?://|[\w.+-]+@[\w.-]+\.[a-z]{2,}|<(?:\w+:)?pw>|sc-\d+', rendered, re.I):
            raise ValueError('source identifiers, contact details, or credential markup in model input')
        key = identity(normalize(json.dumps(payload['state'], sort_keys=True)))
        if key in seen_inputs:
            raise ValueError('duplicate normalized input, even across source stories')
        seen_inputs[key] = sid
        if not row.get('evidence'):
            raise ValueError('source evidence required')
        comments = {c['id']: c for c in source.get('comments', [])}
        for anchor in row['evidence']:
            cid = anchor.get('comment_id')
            text = source['description'] if cid is None else comments[cid]['text']
            if not anchor['quote'] or anchor['quote'] not in text:
                raise ValueError(f'{sid}: evidence quote missing from selected source field')


def readiness(rows):
    counts = Counter(r['role'] for r in rows)
    families = {role: len({r['family_id'] for r in rows if r['role'] == role})
                for role in ('fit', 'certificate', 'test')}
    return {
        'rows_by_role': dict(counts), 'families_by_role': families,
        'draft_classes_by_role': {role: dict(Counter(r['draft_label'] for r in rows if r['role'] == role))
                                  for role in ('fit', 'certificate', 'test')},
        'missing_draft_classes_by_role': {role: sorted(set(OPTIONS)-{r['draft_label'] for r in rows if r['role'] == role})
                                          for role in ('fit', 'certificate', 'test')},
        'independently_reviewed_labels': 0,
        'calibration_ready': False,
        'blocking_reasons': ['All labels remain source-interpreted drafts.',
                             'Family clusters and reconstructed decision points need independent review.',
                             'Expand independently sampled cases to cover all classes in every role; do not move existing families to improve scores.',
                             'Small certification population cannot establish a 5% accepted-error bound.'],
        'minimum_all_correct_independent_certification_cases_at_5pct_existing_wilson_rule': 73,
        'budgets': {str(n): {
            'candidate_fit_family_count_sufficient': families['fit'] >= n,
            'reviewed_fit_labels_sufficient': False,
        } for n in (10, 25, 50, 100, 200)},
        'interpretation': 'Families are conservative proposed dependence groups, not proven independent incidents. Row counts do not increase independent support.',
    }


def build(source_dir, curation_path, out):
    if out.exists():
        raise ValueError('choose a fresh output directory; preserve frozen inputs')
    seed = json.loads(curation_path.read_text())
    source_paths = {int(p.stem.split('-')[1]): p for p in source_dir.glob('story-*.json')}
    sources = {sid: json.loads(p.read_text()) for sid, p in source_paths.items()}
    validate(seed, sources)
    groups = grouped_families(seed, sources)
    rows, evidence = [], []
    sources_manifest = {str(sid): {'path': str(path.resolve()), 'sha256': file_hash(path),
                                  'updated_at': sources[sid]['updated_at']}
                        for sid, path in source_paths.items()}
    for case in seed['rows']:
        sid = case['story_id']; source = sources[sid]
        payload = model_input(case)
        key = identity(payload)[:24]
        family = groups[sid]
        role = family_role(family)
        rows.append({'key': key, 'family_id': family, 'role': role,
                     'gold_status': seed['status'], 'draft_label': case['draft_label'], **payload})
        evidence.append({'key': key, 'story_id': sid, 'proposed_family': case['family_id'], 'source_url': source['url'],
                         'source_updated_at': source['updated_at'], 'evidence': case['evidence'],
                         'review_notes': case['review_notes'], 'source_sha256': file_hash(source_paths[sid]),
                         'decision_point': 'Curated reconstruction from current ticket snapshot, not a verified immutable pre-resolution event.',
                         'input_basis': 'Description and investigative observations paraphrased by the curator; not necessarily the original intake.',
                         'label_origin': 'Agent interpretation of ticket evidence; not independent human adjudication.'})
        comments = {c['id']: c for c in source.get('comments', [])}
        anchor_text = '\n'.join(source['description'] if a.get('comment_id') is None else comments[a['comment_id']]['text']
                                for a in case['evidence'])
        evidence[-1]['automated_or_assisted_evidence_detected'] = bool(
            re.search(r'claude|AI[- ]assisted|AI Triage|basecamp-triage|Codex', anchor_text, re.I))
        evidence[-1]['independent_human_confirmation'] = False
    out.mkdir(parents=True)
    def jsonl(name, data):
        (out/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data))
    jsonl('cases.draft.jsonl', rows)
    # Inference input has no labels, provenance, family, role, or outcomes.
    jsonl('inputs.jsonl', [{'key': r['key'], 'state': r['state'], 'question': r['question']} for r in rows])
    jsonl('evidence.jsonl', evidence)
    with (out/'review-blind.csv').open('w', newline='') as handle:
        fields = ['key', 'report', 'observations', 'reviewed_label', 'reviewer', 'decision_point_valid',
                  'family_review', 'notes']
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({'key': row['key'], 'report': row['state']['report'],
                             'observations': json.dumps(row['state']['observations'])})
    report = readiness(rows)
    report.update({'cases': len(rows), 'families': len({r['family_id'] for r in rows}),
                   'draft_label_counts': dict(Counter(r['draft_label'] for r in rows)),
                   'excluded_story_ids': sorted(set(sources)-{c['story_id'] for c in seed['rows']}),
                   'exclusion_reason': 'Not curated into this seed: unclear or revised evidence, duplicated incident variants, automated diagnosis without independent corroboration, or insufficient relevant detail. Closed status alone was never treated as a label.'})
    (out/'readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'frozen-driver.py').write_bytes(Path(__file__).read_bytes())
    manifest = {'schema_version': 1, 'question_sha256': identity(QUESTION),
                'curation_path': str(curation_path.resolve()), 'curation_sha256': file_hash(curation_path),
                'source_snapshots': sources_manifest, 'driver_sha256': file_hash(__file__),
                'split_rule': 'sha256([judgment-family-split-v1,family]) modulo 4: fit,fit,certificate,test',
                'review_required_before_calibration': True,
                'no_prospective_or_representative_sampling_claim': True,
                'outputs': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return report


def verify(out):
    manifest = json.loads((out/'manifest.json').read_text())
    if file_hash(out/'frozen-driver.py') != manifest['driver_sha256']:
        raise ValueError('frozen driver changed')
    for name, expected in manifest['outputs'].items():
        if file_hash(out/name) != expected:
            raise ValueError(f'frozen artifact changed: {name}')
    if file_hash(manifest['curation_path']) != manifest['curation_sha256']:
        raise ValueError('curation changed')
    for source in manifest['source_snapshots'].values():
        if file_hash(source['path']) != source['sha256']:
            raise ValueError('source snapshot changed')
    return {'verified': True, 'outputs': len(manifest['outputs']),
            'sources': len(manifest['source_snapshots'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('build')
    make.add_argument('--sources', type=Path, required=True)
    make.add_argument('--curation', type=Path, required=True)
    make.add_argument('--out', type=Path, required=True)
    check = sub.add_parser('verify'); check.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.sources, args.curation, args.out) if args.command == 'build' else verify(args.out)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
