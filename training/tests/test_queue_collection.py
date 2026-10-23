from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from sweep.queue_collection import Collection


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=Collection(Path(self.temp.name)/'cases.sqlite3')
        self.start=datetime(2026,10,23,10,tzinfo=timezone.utc)

    def tearDown(self):self.temp.cleanup()

    def checkpoint(self,incident='incident-1'):
        with patch('sweep.queue_collection.now',return_value=self.start.isoformat()):
            return self.store.save('checkpoint',{'incident_id':incident,'source_url':'https://example.invalid/incident',
                'observations':['Original exception.'],'related_refs':[],'captured_by':'Operator'})

    def assessment(self,reviewer,rating='either',confidence=.7):
        return self.store.save('assessment',{'incident_id':'incident-1','reviewer':reviewer,'rating':rating,
            'confidence':confidence,'checkpoint_only':True,'rationale':'The checkpoint does not establish replay safety.'})

    def outcome(self,started=None):
        with patch('sweep.queue_collection.now',return_value=(self.start+timedelta(minutes=10)).isoformat()):
            return self.store.save('outcome',{'incident_id':'incident-1','source_url':'https://example.invalid/outcome',
                'mechanism':'discard','action_started_at':started or (self.start+timedelta(minutes=5)).isoformat(),
                'quote':'I purged the failed messages.','recorded_by':'Operator','success':'confirmed'})

    def test_entries_are_immutable_and_survive_restart(self):
        self.checkpoint()
        with self.assertRaisesRegex(ValueError,'already frozen'):self.checkpoint()
        with self.store.connect() as con:
            with self.assertRaises(sqlite3.IntegrityError):con.execute("UPDATE events SET payload='{}'")
            with self.assertRaises(sqlite3.IntegrityError):con.execute('DELETE FROM events')
        self.assertEqual(Collection(self.store.path).summary()['cases'][0]['observations'],['Original exception.'])

    def test_outcome_cannot_precede_checkpoint_or_be_future(self):
        self.checkpoint()
        for delta in [-1,0,20]:
            with self.assertRaisesRegex(ValueError,'Action must start'):
                self.outcome((self.start+timedelta(minutes=delta)).isoformat())
        self.outcome()
        with self.assertRaisesRegex(ValueError,'already frozen'):self.outcome()

    def test_blind_summary_never_contains_outcomes_or_assessments(self):
        self.checkpoint();self.assessment('Alice');self.outcome()
        serialized=json.dumps(self.store.summary())
        self.assertNotIn('purged',serialized);self.assertNotIn('discard',serialized)
        self.assertNotIn('either',serialized);self.assertNotIn('Alice',serialized)
        self.assertEqual(self.store.summary()['cases'][0]['assessment_count'],1)

    def test_reviewer_identity_case_cannot_create_second_vote(self):
        self.checkpoint();self.assessment('Alice')
        with self.assertRaisesRegex(ValueError,'already frozen'):self.assessment(' ALICE ')
        self.assessment('Bob');self.outcome()
        report=self.store.export()
        self.assertEqual(report['counts']['two_reviewers'],1)
        self.assertEqual(report['review_needed'],['incident-1'])
        self.assertFalse(report['validation_ready'])
        self.assertFalse(report['cases'][0]['two_reviewer_agreement_on_single_action'])

    def test_two_decisive_votes_are_not_automatic_gold(self):
        self.checkpoint();self.assessment('Alice','discard_only',.9);self.assessment('Bob','discard_only',.9);self.outcome()
        report=self.store.export()
        self.assertTrue(report['cases'][0]['agrees_with_recorded_action'])
        self.assertFalse(report['target_review']['ready_for_discussion'])
        self.assertFalse(report['validation_ready'])
        self.assertEqual(report['cases'][0]['state'],{'observations':['Original exception.']})

    def test_reject_credential_url_and_unblinded_review(self):
        with self.assertRaisesRegex(ValueError,'credentials'):
            self.store.save('checkpoint',{'incident_id':'x','source_url':'https://name:secret@example.invalid',
                'observations':['fault'],'captured_by':'Me'})
        self.checkpoint()
        with self.assertRaisesRegex(ValueError,'checkpoint-only'):
            self.store.save('assessment',{'incident_id':'incident-1','reviewer':'Alice','rating':'either',
                'confidence':.8,'rationale':'Both','checkpoint_only':False})


if __name__=='__main__':unittest.main()
