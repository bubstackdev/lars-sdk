"""Prepare an audited review queue without changing a frozen corpus or its gold status."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re

from sweep.codegraph_judgment import file_hash, verify


def local_excerpt(text, quote):
    at = text.index(quote)
    excerpt = text[max(0, at-450):min(len(text), at+len(quote)+600)]
    excerpt = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[email omitted]', excerpt)
    excerpt = re.sub(r'(?i)(support code\s*:?\s*)\d+', r'\1[omitted]', excerpt)
    excerpt = re.sub(r'(<(?:\w+:)?pw>)[\s\S]*?(</(?:\w+:)?pw>)', r'\1[omitted]\2', excerpt)
    return excerpt


def prepare(bundle, assessments_path, out, corroboration=None):
    if out.exists():
        raise ValueError('use a fresh review directory')
    verify(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    assessments = json.loads(assessments_path.read_text())
    rows = [json.loads(l) for l in (bundle/'cases.draft.jsonl').read_text().splitlines()]
    evidence = {r['key']: r for r in map(json.loads, (bundle/'evidence.jsonl').read_text().splitlines())}
    cases = []
    for row in rows:
        ref = evidence[row['key']]; sid = ref['story_id']
        assessment = assessments[str(sid)]
        confidence = assessment.get('confidence')
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1 or not assessment.get('reason')
                or not isinstance(assessment.get('unresolved'), list)):
            raise ValueError('confidence must be a finite judgment estimate with rationale and unresolved items')
        source = json.loads(Path(manifest['source_snapshots'][str(sid)]['path']).read_text())
        comments = {c['id']: c for c in source.get('comments', [])}
        quotes = []
        for anchor in ref['evidence']:
            field = source if anchor['comment_id'] is None else comments[anchor['comment_id']]
            text = field['description'] if anchor['comment_id'] is None else field['text']
            quotes.append({'title': 'Ticket evidence' if anchor['comment_id'] is None else 'Investigation / follow-up note',
                           'quote': local_excerpt(text, anchor['quote']),
                           'url': field.get('url', source['url']),
                           'revision': field.get('updated_at', source['updated_at'])})
        text = (row['state']['report']+' '+ref['proposed_family']).casefold()
        domain = next((label for pattern, label in [('dns|nameserver|cname|zone-cut','DNS'),
            ('whois|privacy|contact','WHOIS'), ('cart|checkout|wallet','Checkout'),
            ('transfer|registry','Transfers'),('refund|payment|salesorder|fabal|auction','Payments'),
            ('auth|scope|token|openid','Identity')] if re.search(pattern, text)), 'Infrastructure')
        cases.append(row | {'story_id': sid, 'domain': domain,
                            'confidence': confidence, 'assessment': assessment,
                            'needs_human_review': confidence < .8,
                            'review_status': 'pending', 'evidence': quotes,
                            'automated_evidence_detected': ref['automated_or_assisted_evidence_detected']})
    attached = []
    if corroboration:
        for sid, name in [(139454,'privacy.json'), (144043,'wallet.json')]:
            path = corroboration/name
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for case in cases:
                if case['story_id'] == sid:
                    case['evidence'].append({'title': 'Pinned implementation excerpt (inspected, not executed)',
                        'quote': data['content'], 'revision': data['ref'],
                        'url': f'https://gitlab.tcdevops.com/{data["project"]}/-/blob/{data["ref"]}/{data["filePath"]}#L{data["contentStartLine"]}'})
            attached.append(path)
        for sid, sha in [(140008,'2f3248717a6205de21573c6b5f81132d3869fdd7'),
                         (145388,'7a6c718d'), (145388,'091b9a6b')]:
            path = corroboration/f'commit-{sha}.json'
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            diffs = [d for d in data['diffs'] if d['newPath'].endswith('.cs')]
            for case in cases:
                if case['story_id'] == sid:
                    for diff in diffs:
                        case['evidence'].append({'title': 'Pinned change: '+diff['newPath'],
                            'quote': diff['diff'], 'revision': sha,
                            'url': f'https://gitlab.tcdevops.com/{data["project"]}/-/commit/{sha}'})
            attached.append(path)
    metadata = {'corpus_bundle': str(bundle.resolve()), 'manifest_sha256': file_hash(bundle/'manifest.json'),
                'assessments_sha256': file_hash(assessments_path),
                'confidence_definition': 'Uncalibrated agent judgment that the proposed track is supported by the reconstructed facts. Not model output probability, independent adjudication, or a deployment guarantee.',
                'threshold': .8, 'human_review_required_below_threshold': True,
                'all_labels_still_unreviewed': True, 'cases': len(cases),
                'below_threshold': sum(c['confidence'] < .8 for c in cases),
                'high_confidence': sum(c['confidence'] >= .8 for c in cases),
                'groups': len({c['family_id'] for c in cases}),
                'additional_corroboration': {str(p.resolve()): file_hash(p) for p in attached}}
    out.mkdir(parents=True)
    (out/'queue.json').write_text(json.dumps({'metadata': metadata, 'cases': cases}, indent=2)+'\n')
    uncertain = sorted((c for c in cases if c['confidence'] < .8), key=lambda c: c['confidence'])
    (out/'needs-review.json').write_text(json.dumps(uncertain, indent=2)+'\n')
    with (out/'needs-review.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['key','story_id','confidence','report','proposed_label','reason','unresolved'])
        writer.writeheader()
        for c in uncertain:
            writer.writerow({'key':c['key'],'story_id':c['story_id'],'confidence':c['confidence'],
                             'report':c['state']['report'],'proposed_label':c['draft_label'],
                             'reason':c['assessment']['reason'],'unresolved':' | '.join(c['assessment']['unresolved'])})
    (out/'manifest.json').write_text(json.dumps(metadata | {
        'outputs':{p.name:file_hash(p) for p in out.iterdir() if p.is_file()},
        'class_counts':dict(Counter(c['draft_label'] for c in cases))},indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--assessments', type=Path, required=True)
    parser.add_argument('--corroboration', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle,args.assessments,args.out,args.corroboration),indent=2))


if __name__ == '__main__':
    main()
