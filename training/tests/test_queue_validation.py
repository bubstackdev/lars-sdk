import json
from pathlib import Path
import tempfile
import unittest

from sweep.queue_validation import dependence, gate, returned_case


class ValidationGateTests(unittest.TestCase):
    def test_majority_only_cannot_pass(self):
        target={'minimum_groups':20,'minimum_per_label':5,'minimum_diagnostic_groups':10}
        rows=[{'family_id':str(i),'label':'reprocess','evidence_level':'diagnostic'} for i in range(30)]
        self.assertFalse(gate(rows,target)['passed'])
        for row in rows[:5]:row['label']='discard'
        self.assertTrue(gate(rows,target)['passed'])
        for row in rows:row['family_id']='one-incident'
        self.assertFalse(gate(rows,target)['passed'])

    def test_dependence_chases_intermediate_links_and_shared_fix(self):
        sources={1:{'story_links':[{'subject_id':1,'object_id':9}]},
                 9:{'parent_story_id':2},2:{'description':'https://gitlab.tcdevops.com/team/repo/-/merge_requests/12'},
                 3:{'description':'https://gitlab.tcdevops.com/team/repo/-/merge_requests/12/'},4:{}}
        families=dependence(sources)
        self.assertEqual(families[1],families[3])
        self.assertNotEqual(families[1],families[4])

    def test_generic_cleared_or_proposal_is_not_explicit_return(self):
        for quote in ['Queue cleared.','We should push messages back to their normal queues.']:
            with self.assertRaisesRegex(ValueError,'explicit completed'):
                returned_case(None,{'outcome_quote':quote})

    def test_explicit_return_still_requires_earlier_evidence(self):
        quote='Pushed all messages back to their normal queues and they cleared.'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'stories').mkdir();(root/'history').mkdir()
            source={'app_url':'https://example.invalid/1','comments':[{
                'id':2,'created_at':'2026-01-02T00:00:00Z','updated_at':'2026-01-02T00:00:00Z',
                'text':'Late diagnosis. '+quote}]}
            history={'history':[{'id':'create','changed_at':'2026-01-01T00:00:00Z',
                'actions':[{'id':1,'action':'create','description':'Original fault.'}]}]}
            (root/'stories/1.json').write_text(json.dumps(source));(root/'history/1.json').write_text(json.dumps(history))
            spec={'story_id':1,'outcome_quote':quote,'observations':['Original fault.'],
                  'notes':'Test','evidence_level':'diagnostic'}
            self.assertEqual(returned_case(root,spec)['state']['observations'],['Original fault.'])
            spec['observations']=['Late diagnosis.']
            with self.assertRaisesRegex(ValueError,'no earlier exact source'):returned_case(root,spec)


if __name__=='__main__':unittest.main()
