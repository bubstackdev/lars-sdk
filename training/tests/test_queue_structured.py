import unittest

from sweep.queue_structured import extract, rule


class StructuredEvidenceTests(unittest.TestCase):
    def test_alert_does_not_invent_diagnosis(self):
        fields=extract(['PD: DomainInventoryApi - Error Queues with Ready Messages'])
        self.assertTrue(all(v=={'value':'unknown','evidence':[]} for v in fields.values()))
        self.assertIsNone(rule(fields))

    def test_terminal_domain_is_not_completed_work(self):
        quote='Fraud return marked the domain Deleted; EPP 2303 ObjectDoesNotExist.'
        fields=extract([quote])
        self.assertEqual(fields['domain_status']['value'],['Deleted'])
        self.assertEqual(fields['error_codes']['evidence'],[quote])
        self.assertEqual(fields['work_already_completed']['value'],'unknown')
        self.assertIsNone(rule(fields))

    def test_completed_work_and_repaired_data_differ(self):
        self.assertEqual(rule(extract(['These domains were manually restored by Richard'])),'discard')
        self.assertEqual(rule(extract(['After de-duping the table and fixing things up, messages remain'])),'reprocess')

    def test_exception_does_not_imply_replay_safe(self):
        fields=extract(['System.NullReferenceException'])
        self.assertEqual(fields['exception_types']['value'],['System.NullReferenceException'])
        self.assertEqual(fields['known_replay_behavior']['value'],'unknown')
        self.assertIsNone(rule(fields))


if __name__=='__main__':unittest.main()
