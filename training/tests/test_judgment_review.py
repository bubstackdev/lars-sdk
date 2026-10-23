import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sweep.codegraph_judgment import QUESTION
from sweep.judgment_review import Conflict, ReviewStore, make_server


def fixture():
    return {'metadata': {'test_fixture': True}, 'cases': [{
        'key': 'case-one', 'story_id': 100, 'domain': 'Checkout',
        'draft_label': 'application_engineering', 'confidence': .70,
        'state': {'report': 'A request fails.', 'observations': ['A handler drops a required field.']},
        'question': QUESTION, 'family_id': 'family:one', 'role': 'fit',
        'assessment': {'reason': 'The cause may need clarification.', 'unresolved': ['Check the source.']},
        'evidence': [{'title': 'Source', 'quote': '<script>window.BAD=true</script>',
                      'url': 'https://app.shortcut.com/turncommerce/story/100'}],
    }]}


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name)/'reviews.sqlite3'
        self.data = fixture(); self.store = ReviewStore(self.data, self.db)

    def payload(self, **changes):
        return {'revision': 0, 'status': 'approved', 'reviewer': 'Test reviewer',
                'label': 'application_engineering', 'facts_checked': True,
                'notes': '', **changes}

    def test_durable_corrections_and_conflicting_tabs(self):
        review = self.store.save('case-one', self.payload(label='insufficient_evidence', notes='The source has no diagnosis.'))
        self.assertEqual(review['status'], 'corrected')
        self.assertEqual(self.store.cases['case-one']['draft_label'], 'application_engineering')
        reopened = ReviewStore(self.data, self.db)
        self.assertEqual(reopened.snapshot()['cases'][0]['review']['label'], 'insufficient_evidence')
        with self.assertRaises(Conflict):
            reopened.save('case-one', self.payload())
        self.assertEqual(len(reopened.history('case-one')), 1)

    def test_human_gate_exclusion_and_reopen_audit(self):
        self.assertEqual(self.store.export()['reviewed_cases'], [])
        with self.assertRaisesRegex(ValueError, 'facts'):
            self.store.save('case-one', self.payload(facts_checked=False))
        with self.assertRaisesRegex(ValueError, 'explain'):
            self.store.save('case-one', self.payload(status='excluded'))
        self.store.save('case-one', self.payload())
        self.assertEqual(self.store.export()['reviewed_cases'][0]['gold_status'], 'reviewed')
        self.store.save('case-one', self.payload(revision=1, status='pending'))
        self.assertEqual(self.store.export()['reviewed_cases'], [])
        self.assertEqual([r['revision'] for r in self.store.history('case-one')], [1, 2])

    def test_fact_changes_are_corrections_and_need_a_reason(self):
        state = {'report':'A request fails.', 'observations':['The handler was not inspected.']}
        with self.assertRaisesRegex(ValueError, 'explain'):
            self.store.save('case-one', self.payload(state=state))
        saved = self.store.save('case-one', self.payload(state=state,notes='Corrected an unsupported observation.'))
        self.assertEqual(saved['status'], 'corrected')
        self.assertEqual(self.store.export()['reviewed_cases'][0]['state'], state)

    def test_different_cohort_cannot_reuse_reviews(self):
        other = copy.deepcopy(self.data); other['cases'][0]['confidence'] = .9
        with self.assertRaisesRegex(ValueError, 'different case snapshot'):
            ReviewStore(other, self.db)

    def test_family_concerns_and_duplicate_edited_states_block_export(self):
        self.store.save('case-one',self.payload(family_concern='May share an incident with evaluation.'))
        self.assertEqual(self.store.export()['reviewed_cases'], [])
        other = copy.deepcopy(self.data); second = copy.deepcopy(other['cases'][0])
        second.update(key='case-two',family_id='family:two',role='test')
        second['state'] = {'report':'A different request fails.', 'observations':['No trace.']}
        other['cases'].append(second)
        store = ReviewStore(other,Path(self.temp.name)/'other.sqlite3')
        store.save('case-one',self.payload())
        store.save('case-two',self.payload(state=other['cases'][0]['state'],notes='Corrected the report.'))
        exported = store.export()
        self.assertEqual(exported['reviewed_cases'], [])
        self.assertEqual(len(exported['duplicate_cases_held_back']),2)


class ReviewHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.store = ReviewStore(fixture(),Path(self.temp.name)/'reviews.sqlite3')
        self.server = make_server(self.store,port=0)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.stop)
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def stop(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def request(self,path,**kw):
        try:
            response = urlopen(Request(self.base+path,**kw))
        except HTTPError as error:
            response = error
        with response:
            return response.status,response.headers,response.read()

    def test_http_requires_local_token_and_validates_payload(self):
        status,headers,body = self.request('/api/cases')
        self.assertEqual(status,200)
        token = json.loads(body)['csrf_token']
        payload = {'key':'case-one','revision':0,'status':'approved','label':'application_engineering',
                   'facts_checked':True,'reviewer':'Test reviewer'}
        data = json.dumps(payload).encode()
        self.assertEqual(self.request('/api/review',data=data,headers={'Content-Type':'application/json'})[0],403)
        base_headers = {'Content-Type':'application/json','X-Review-Token':token}
        self.assertEqual(self.request('/api/review',data=data,headers=base_headers|{'Origin':'https://foreign.invalid'})[0],403)
        self.assertEqual(self.request('/api/review',data=data,headers=base_headers)[0],200)
        self.assertEqual(self.request('/api/review',data=data,headers=base_headers)[0],409)
        self.assertEqual(len(self.store.history('case-one')),1)

    def test_static_routes_and_headers(self):
        for path in ['/','/app.js','/style.css']:
            status,headers,body=self.request(path)
            self.assertEqual(status,200)
            self.assertIn("frame-ancestors 'none'",headers['Content-Security-Policy'])
        self.assertEqual(self.request('/../sources/private.json')[0],404)
        self.assertEqual(self.request('/api/cases',headers={'Host':'foreign.invalid'})[0],403)

    def test_retired_review_archive_is_read_only(self):
        self.stop()
        self.server=make_server(self.store,port=0,benchmark={'retained_cases':1,'human_review_required':0})
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        status,_,body=self.request('/api/cases');snapshot=json.loads(body)
        self.assertTrue(snapshot['metadata']['archived'])
        self.assertEqual(self.request('/api/benchmark')[0],200)
        self.assertIn(b'Historical replay audit',self.request('/')[2])
        self.assertEqual(self.request('/review')[0],200)
        payload={'key':'case-one','revision':0,'status':'approved','reviewer':'Test reviewer',
                 'label':'application_engineering','facts_checked':True}
        status,_,body=self.request('/api/review',data=json.dumps(payload).encode(),headers={
            'Content-Type':'application/json','X-Review-Token':snapshot['csrf_token']})
        self.assertEqual(status,410)
        self.assertEqual(self.store.history('case-one'),[])


if __name__=='__main__':
    unittest.main()
