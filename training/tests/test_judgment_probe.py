import copy
import json
from pathlib import Path
import tempfile
import unittest

from sweep.judgment_probe import prepare_rows, probe, reject_retired
from sweep.codegraph_judgment import QUESTION, identity


class AgentProbeIsolationTests(unittest.TestCase):
    def test_retired_question_cannot_be_rerun_through_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            marker=Path(temp)/'retirement.json'
            marker.write_text(json.dumps({'retired_question_sha256':[identity(QUESTION)]}))
            with self.assertRaisesRegex(ValueError,'retired'):
                reject_retired({'cases':[{'question':QUESTION}]},marker)
            reject_retired({'cases':[{'question':{'instructions':'A different task'}}]},marker)

    def test_threshold_and_family_selection_do_not_promote_gold(self):
        cases=[];captures=[]
        for i,confidence in enumerate([.799,.8,.9]):
            state={'report':str(i),'observations':['Observed']}
            cases.append({'key':str(i),'family_id':'one','role':'fit','confidence':confidence,
                          'state':state,'question':QUESTION,'draft_label':'insufficient_evidence'})
            captures.append({'original_key':str(i),'input_sha256':identity([state,QUESTION]),
                             'keys':list(QUESTION['criteria']),'logits':[1.]*6,'kind':'choice','model_id':'one-model'})
        rows=prepare_rows({'cases':cases},captures)
        self.assertEqual(len(rows),1)
        self.assertNotEqual(rows[0]['key'],'0')
        self.assertEqual(rows[0]['gold_status'],'agent_assessed')
        # Explicit human corrections bring an uncertain case into a future run;
        # deferred/excluded high-confidence cases must stay out.
        cases[0]['family_id']='human-family'
        cases[0]['review']={'status':'corrected','state':cases[0]['state'],
                            'label':'application_engineering','family_concern':''}
        cases[1]['review']={'status':'excluded'}
        cases[2]['review']={'status':'deferred'}
        corrected=prepare_rows({'cases':cases},captures)
        self.assertEqual(len(corrected),1)
        self.assertEqual(corrected[0]['gold_status'],'reviewed')
        self.assertEqual(corrected[0]['keys'][corrected[0]['hard_index']],'application_engineering')

    def test_evaluation_and_certificate_labels_cannot_choose_method(self):
        rows=[]
        for i in range(14):
            role='fit' if i<10 else ('certificate' if i<12 else 'test')
            hard=i%2
            rows.append({'key':str(i),'family_id':str(i),'role':role,'kind':'choice','keys':['a','b'],
                         'logits':[float(i%3),float((i+1)%3)],'hard_index':hard,
                         'target':[float(hard==0),float(hard==1)]})
        calibration={'temperatures':{'choice':1.2}}
        first=probe(rows,calibration,budgets=(0,10),draws=1)
        altered=copy.deepcopy(rows)
        for row in altered:
            if row['role']!='fit':
                row['hard_index']=1-row['hard_index'];row['target']=list(reversed(row['target']))
        second=probe(altered,calibration,budgets=(0,10),draws=1)
        for a,b in zip(first['runs'],second['runs']):
            self.assertEqual(a['fit_keys'],b['fit_keys']);self.assertEqual(a['kept'],b['kept'])
            self.assertEqual(a['cv'],b['cv']);self.assertEqual(a['test_logq'],b['test_logq'])


if __name__=='__main__':unittest.main()
