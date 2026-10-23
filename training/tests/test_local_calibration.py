import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from sweep.local_calibration import arrays, certificate, metrics, prepare, run, select, validate
from sweep.codegraph_calibration_cases import billing_action, http_action, payout_action


def fixture_rows():
    rows=[]
    for role,n in [('fit',30),('certificate',75),('test',20)]:
        for i in range(n):
            x=(i+.5)/n*4-2
            rows.append({'key':f'{role}-{i}','state':{'measurement':x,'event':f'{role}-{i}'},
                'family_id':f'{role}-{i}','role':role,'predicate_id':'fixture/q','workflow':'fixture',
                'model_id':'fixture','gold_status':'benchmark','kind':'noul','keys':['yes','no'],
                'question':{'type':'noul','instructions':'Measurement positive?'},
                'logits':[x+.8,0.],'hard_index':int(x<=0),'target':[float(x>0),float(x<=0)]})
    return rows


class LocalCalibrationTests(unittest.TestCase):
    def test_state_family_and_gold_guards(self):
        rows=fixture_rows();validate(rows)
        changed=copy.deepcopy(rows);changed[-1]['state']=changed[0]['state']
        with self.assertRaisesRegex(ValueError,'overlaps roles'):validate(changed)
        changed=copy.deepcopy(rows);changed[-1]['state']=changed[0]['state'] | {'event':' FIT-0 '}
        with self.assertRaisesRegex(ValueError,'overlaps roles'):validate(changed)
        changed=copy.deepcopy(rows);changed[-1]['family_id']=changed[0]['family_id']
        with self.assertRaisesRegex(ValueError,'overlaps roles'):validate(changed)
        changed=copy.deepcopy(rows);changed[0]['gold_status']='rule_derived_unreviewed'
        with self.assertRaisesRegex(ValueError,'unreviewed'):validate(changed)
        changed=copy.deepcopy(rows);changed[0]['model_id']='other'
        with self.assertRaisesRegex(ValueError,'mix model'):validate(changed)
        changed=copy.deepcopy(rows);changed[-1]['question']['instructions']='different'
        with self.assertRaisesRegex(ValueError,'changes question'):validate(changed)

    def test_certificate_matches_minimum_and_fails_closed(self):
        self.assertIsNone(certificate(np.tile([.999,.001],(72,1)),np.zeros(72,int)))
        c=certificate(np.tile([.999,.001],(73,1)),np.zeros(73,int))
        self.assertEqual(c['accepted'],73);self.assertLessEqual(c['error_upper95'],.05)
        p=np.vstack([np.tile([.999,.001],(73,1)),np.tile([.55,.45],(500,1))])
        y=np.zeros(573,int);y[:5]=1
        self.assertIsNone(certificate(p,y))  # Cannot skip the failed first group for a later passing group.
        value=metrics(np.array([[.9,.1]]),[{'hard_index':1,'target':[.9,.1]}],None)
        self.assertEqual(value['accuracy'],0);self.assertEqual(value['coverage'],0)
        self.assertIsNone(value['accepted_error']);self.assertEqual(value['accepted_error_upper95'],1)

    def test_test_labels_cannot_select_model_and_budgets_are_nested(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cal=root/'cal.json';cal.write_text('{}')
            outcomes=[]
            for variant in [0,1]:
                rows=fixture_rows()
                if variant:
                    for row in rows:
                        if row['role']=='test':row['hard_index']=1-row['hard_index']
                data=root/f'input{variant}.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                args=SimpleNamespace(out=root/f'out{variant}',input=data,calibration=cal,draws=1)
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare(args)
                    captured=args.out/'captures.jsonl';original_bytes=captured.read_bytes()
                    captured.write_bytes(original_bytes+b'\n')
                    with self.assertRaisesRegex(ValueError,'frozen'):run(args)
                    captured.write_bytes(original_bytes)
                    run(args)
                result=json.loads((args.out/'results.json').read_text());outcomes.append(result)
                complete=[r for r in result['runs'] if r['state']=='complete']
                self.assertEqual([r['budget'] for r in complete],[0,10,25])
                self.assertTrue(set(complete[1]['fit_keys'])<=set(complete[2]['fit_keys']))
                self.assertEqual(result['summary']['50']['state'],'unsupported')
            a,b=outcomes
            self.assertEqual([(r.get('kept'),r.get('certificate')) for r in a['runs']],
                             [(r.get('kept'),r.get('certificate')) for r in b['runs']])
            self.assertNotEqual(a['summary']['25']['macro_accuracy'],b['summary']['25']['macro_accuracy'])

    def test_source_contract_boundaries_and_precedence(self):
        self.assertEqual(billing_action('BillingMethodNotFoundException',1,3),'retry_in_process')
        self.assertEqual(billing_action('BillingMethodNotFoundException',1,4),'delayed_redelivery')
        self.assertEqual(billing_action('BillingMethodNotFoundException',4,4),'fault')
        self.assertEqual(billing_action('DuplicateLineItemException',4,4),'skip_duplicate')
        self.assertEqual(billing_action('deadlock',1,1),'fault')
        self.assertEqual(http_action(429,0),'retry_after_2s')
        self.assertEqual(http_action(503,2),'retry_after_8s')
        self.assertEqual(http_action(503,3),'fail')
        self.assertEqual(http_action(401,0),'fail');self.assertEqual(http_action(600,0),'fail')
        self.assertEqual(payout_action('Paid',True),'needs_attention')
        self.assertEqual(payout_action('Paid',False),'completed')

    def test_hidden_head_is_available_only_at_supported_budget(self):
        rows=[]
        for i in range(50):
            x=(i+.5)/50*4-2
            rows.append({'kind':'choice','keys':['positive','negative'],'logits':[0.,0.],
                         'hard_index':int(x<0),'hidden':[x,x*x,1.]})
        chosen, losses=select(arrays(rows),{},np.random.default_rng(17))
        self.assertTrue(chosen.startswith('head:'),chosen)
        self.assertLess(losses[chosen],losses['pack'])
        _, small_losses=select(arrays(rows[:25]),{},np.random.default_rng(17))
        self.assertFalse(any(name.startswith('head:') for name in small_losses))
