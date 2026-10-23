import copy
import unittest

from sweep.resolution_replay import make_case


class ReplayEvidenceTests(unittest.TestCase):
    def fixture(self):
        sources={1:{'url':'https://example.invalid/1','comments':[
            {'id':10,'created_at':'2026-01-01T01:00:00Z','updated_at':'2026-01-01T01:00:00Z','text':'The request returns an error.'},
            {'id':20,'created_at':'2026-01-02T01:00:00Z','updated_at':'2026-01-02T01:00:00Z','text':'A missing reference was found.'},
            {'id':30,'created_at':'2026-01-03T01:00:00Z','updated_at':'2026-01-03T01:00:00Z','text':'The missing reference was restored. The request now works.'}],
            'description':'Later root cause, private account and resolution must never become features.'}}
        selection={'story_id':1,'description':'Metadata title','inputs':[{'comment_id':10,'quote':'The request returns an error.'}],
                   'cutoff':{'story_id':1,'comment_id':20},'outcome':{'comment_id':30,'quotes':['The missing reference was restored.']},
                   'scope':'pre_diagnosis','notes':'Metadata only'}
        prior={'family_id':'one','role':'test'}
        return selection,sources,prior

    def test_features_use_only_exact_earlier_quotes(self):
        selection,sources,prior=self.fixture()
        result=make_case(selection,sources,prior)
        self.assertEqual(result['model_input']['state']['observations'],['The request returns an error.'])
        self.assertEqual(result['expected_outcome'],'The missing reference was restored.')
        self.assertEqual(result['label_status'],'documented_outcome')
        self.assertNotIn('Metadata title',str(result['model_input']))
        self.assertNotIn('Later root cause',str(result['model_input']))
        self.assertEqual(result['role'],'test')

    def test_late_edit_of_early_comment_is_rejected(self):
        selection,sources,prior=self.fixture()
        sources[1]['comments'][0]['updated_at']='2026-01-03T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'edited at or after'):
            make_case(selection,sources,prior)

    def test_wrong_source_and_outcome_in_input_are_rejected(self):
        selection,sources,prior=self.fixture()
        selection['inputs'][0]['quote']='Invented facts'
        with self.assertRaisesRegex(ValueError,'missing from the source'):
            make_case(selection,sources,prior)
        selection,sources,prior=self.fixture()
        sources[1]['comments'][0]['text']='The missing reference was restored.'
        selection['inputs'][0]['quote']='The missing reference was restored.'
        with self.assertRaisesRegex(ValueError,'outcome leaked'):
            make_case(selection,sources,prior)

    def test_future_outcomes_cannot_change_features(self):
        selection,sources,prior=self.fixture()
        first=make_case(selection,sources,prior)
        other=copy.deepcopy(sources);other[1]['comments'][2]['text']='The request was rerouted instead.'
        selection['outcome']['quotes']=['The request was rerouted instead.']
        second=make_case(selection,other,prior)
        self.assertEqual(first['model_input'],second['model_input'])
        self.assertEqual(first['key'],second['key'])
        self.assertNotEqual(first['expected_outcome'],second['expected_outcome'])

    def test_earlier_related_ticket_cutoff_is_enforced(self):
        selection,sources,prior=self.fixture()
        sources[2]={'url':'https://example.invalid/2','comments':[{'id':5,'created_at':'2026-01-01T00:00:00Z',
                    'updated_at':'2026-01-01T00:00:00Z','text':'Related-ticket diagnosis already known.'}]}
        selection['cutoff']={'story_id':2,'comment_id':5}
        with self.assertRaisesRegex(ValueError,'edited at or after'):
            make_case(selection,sources,prior)

    def test_creation_history_survives_later_description_edits(self):
        selection,sources,prior=self.fixture()
        histories={1:{'history':[{'id':'create-1','changed_at':'2026-01-01T00:00:00Z',
            'actions':[{'id':1,'action':'create','description':'Initial request fails.'}]}]}}
        selection['inputs']=[{'history_event_id':'create-1','quote':'Initial request fails.'}]
        first=make_case(selection,sources,prior,histories)
        sources[1]['description']='A later edit includes the entire solution.'
        second=make_case(selection,sources,prior,histories)
        self.assertEqual(first['model_input'],second['model_input'])
        self.assertEqual(first['input_evidence'][0]['source_kind'],'original_description')
        histories[1]['history'][0]['changed_at']='2026-01-04T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'edited at or after'):
            make_case(selection,sources,prior,histories)

    def test_history_rejects_updates_and_unanchored_quotes(self):
        selection,sources,prior=self.fixture()
        histories={1:{'history':[{'id':'event','changed_at':'2026-01-01T00:00:00Z',
            'actions':[{'id':1,'action':'update','description':'Initial request fails.'}]}]}}
        selection['inputs']=[{'history_event_id':'event','quote':'Initial request fails.'}]
        with self.assertRaisesRegex(ValueError,'creation event'):
            make_case(selection,sources,prior,histories)
        histories[1]['history'][0]['actions'][0]['action']='create'
        selection['inputs'][0]['quote']='Invented symptom'
        with self.assertRaisesRegex(ValueError,'missing from the creation event'):
            make_case(selection,sources,prior,histories)

    def test_later_description_revision_can_document_an_outcome(self):
        selection,sources,prior=self.fixture()
        histories={1:{'history':[{'id':'repair','changed_at':'2026-01-03T00:00:00Z',
            'actions':[{'id':1,'action':'update','changes':{'description':{
                'old':'Earlier report.','new':'The repair succeeded.'}}}]}]}}
        selection['outcome']={'history_event_id':'repair','quotes':['The repair succeeded.']}
        result=make_case(selection,sources,prior,histories)
        self.assertEqual(result['expected_outcome'],'The repair succeeded.')
        self.assertEqual(result['outcome_evidence'][0]['source_kind'],'description_revision')
        self.assertNotIn('repair succeeded',str(result['model_input']))
        histories[1]['history'][0]['changed_at']='2026-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'outcome predates'):
            make_case(selection,sources,prior,histories)

    def test_history_cutoff_blocks_later_inputs(self):
        selection,sources,prior=self.fixture()
        histories={1:{'history':[{'id':'finding','changed_at':'2025-12-31T00:00:00Z','actions':[]}]}}
        selection['cutoff']={'story_id':1,'history_event_id':'finding'}
        with self.assertRaisesRegex(ValueError,'edited at or after'):
            make_case(selection,sources,prior,histories)


if __name__=='__main__':unittest.main()
