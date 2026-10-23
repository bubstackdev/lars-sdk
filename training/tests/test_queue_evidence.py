import json
from pathlib import Path
import tempfile
import unittest

from sweep.queue_evidence import make_case, groups


class QueueEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'stories').mkdir();(self.root/'history').mkdir()
        self.source={'id':1,'app_url':'https://example.invalid/1','description':'Later hindsight.',
            'comments':[{'id':20,'created_at':'2026-01-03T00:00:00Z','updated_at':'2026-01-03T00:00:00Z',
                         'text':'I requeued the message and it processed successfully.'}]}
        self.history={'history':[{'id':'create','changed_at':'2026-01-01T00:00:00Z',
            'actions':[{'id':1,'action':'create','name':'Error queue alert','description':'Missing type mapping.'}]},
            {'id':'edit','changed_at':'2026-01-02T00:00:00Z','actions':[{'id':1,'action':'update',
                'changes':{'description':{'old':'Missing type mapping.','new':'A dependency cannot be resolved.'}}}]}]}
        self.spec={'story_id':1,'label':'reprocess','observations':['Missing type mapping.'],
                   'outcome_quote':self.source['comments'][0]['text'],'notes':'Test'}
        self.save()

    def tearDown(self):self.temp.cleanup()

    def save(self):
        (self.root/'stories/1.json').write_text(json.dumps(self.source))
        (self.root/'history/1.json').write_text(json.dumps(self.history))

    def test_current_description_and_outcome_are_not_features(self):
        row=make_case(self.root,self.spec)
        self.assertEqual(row['state'],{'observations':['Missing type mapping.']})
        self.assertNotIn('processed',str(row['state']))
        self.assertFalse(row['independently_human_adjudicated'])

    def test_earlier_historical_revision_is_allowed(self):
        self.spec['observations']=['A dependency cannot be resolved.']
        row=make_case(self.root,self.spec)
        self.assertEqual(row['input_evidence'][0]['event_id'],'edit')

    def test_same_note_findings_are_rejected(self):
        self.source['comments'][0]['text']='A dependency is absent. I requeued it successfully.'
        self.spec['outcome_quote']='I requeued it successfully.'
        self.spec['observations']=['A dependency is absent.'];self.save()
        with self.assertRaisesRegex(ValueError,'no earlier exact source'):
            make_case(self.root,self.spec)

    def test_late_comment_edit_cannot_supply_earlier_features(self):
        self.source['comments'].append({'id':10,'created_at':'2026-01-01T00:00:00Z',
            'updated_at':'2026-01-04T00:00:00Z','text':'Originally observed, later edited.'})
        self.spec['observations']=['Originally observed, later edited.'];self.save()
        with self.assertRaisesRegex(ValueError,'no earlier exact source'):
            make_case(self.root,self.spec)

    def test_backdated_action_bound_is_enforced(self):
        self.spec['observations']=['A dependency cannot be resolved.']
        self.spec['action_not_before']='2026-01-01T12:00:00Z'
        with self.assertRaisesRegex(ValueError,'no earlier exact source'):
            make_case(self.root,self.spec)

    def test_label_requires_explicit_completed_action(self):
        self.spec['label']='discard'
        with self.assertRaisesRegex(ValueError,'explicitly record'):
            make_case(self.root,self.spec)
        self.spec['label']='reprocess';self.spec['outcome_quote']='It can be replayed.'
        self.source['comments'][0]['text']=self.spec['outcome_quote'];self.save()
        with self.assertRaisesRegex(ValueError,'proposed action'):
            make_case(self.root,self.spec)

    def test_independent_repeated_observations_have_distinct_keys(self):
        a=make_case(self.root,self.spec)
        source=dict(self.source,id=2,comments=self.source['comments'])
        history=json.loads(json.dumps(self.history))
        for e in history['history']:
            for action in e['actions']:action['id']=2
        (self.root/'stories/2.json').write_text(json.dumps(source))
        (self.root/'history/2.json').write_text(json.dumps(history))
        b=make_case(self.root,dict(self.spec,story_id=2))
        self.assertNotEqual(a['key'],b['key'])
        self.assertEqual(a['input_sha256'],b['input_sha256'])
        rows=groups(self.root,[a,b])
        self.assertNotEqual(rows[0]['family_id'],rows[1]['family_id'])


if __name__=='__main__':unittest.main()
