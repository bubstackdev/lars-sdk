import copy
import unittest

from sweep.codegraph_judgment import QUESTION, identity
from sweep.judgment_capture import join_reviewed


class ReviewedCaptureTests(unittest.TestCase):
    def fixture(self):
        state={'report':'A report','observations':['A supported observation.']}
        keys=list(reversed(list(QUESTION['criteria'])))
        row={'key':'new','original_key':'old','state':state,'question':QUESTION,
             'label':'application_engineering','family_id':'one','role':'fit','gold_status':'reviewed'}
        capture={'original_key':'old','state':state,'question':QUESTION,
                 'input_sha256':identity([state,QUESTION]),'kind':'choice',
                 'keys':keys,'logits':[1.]*len(keys),'model_id':'frozen-model'}
        return row,capture

    def test_label_corrections_align_to_native_key_order(self):
        row,cache=self.fixture()
        joined,pending=join_reviewed({'reviewed_cases':[row]},[cache])
        self.assertEqual(pending,[])
        self.assertEqual(joined[0]['keys'][joined[0]['hard_index']],row['label'])
        self.assertEqual(joined[0]['target'][joined[0]['hard_index']],1.)
        self.assertEqual(joined[0]['model_id'],'frozen-model')

    def test_changed_facts_need_recapture_and_drafts_are_rejected(self):
        row,cache=self.fixture();changed=copy.deepcopy(row)
        changed['state']['report']='Corrected report'
        joined,pending=join_reviewed({'reviewed_cases':[changed]},[cache])
        self.assertEqual(joined,[]);self.assertEqual(pending,['old'])
        row['gold_status']='source_interpreted_unreviewed'
        with self.assertRaisesRegex(ValueError,'human-reviewed'):
            join_reviewed({'reviewed_cases':[row]},[cache])


if __name__=='__main__':unittest.main()
