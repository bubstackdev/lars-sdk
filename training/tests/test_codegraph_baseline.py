from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sweep.codegraph_baseline import audit_tier, guarded, parity


class CodegraphBaselineTests(unittest.TestCase):
    def test_conflicts_cannot_enter_implementation_score(self):
        for source,state,tier in [('billing_retry',{},'implementation_corroborated'),
            ('invoice_closure',{},'implementation_corroborated'),
            ('internal_refund',{'transaction_type':'User Purchase'},'implementation_corroborated'),
            ('internal_refund',{'transaction_type':'Withdrawal'},'quarantined'),
            ('story_http_retry',{},'quarantined'),('domain_return',{},'quarantined'),
            ('payout_result',{},'documentation_only'),('escalation_contact',{},'documentation_only')]:
            self.assertEqual(audit_tier({'source':'codegraph/'+source,'state':state})[0],tier)

    def test_control_parity_rejects_option_changes_and_small_label_flips(self):
        row={'kind':'choice','keys':['a','b'],'logits':[0.,0.]}
        self.assertTrue(parity([row],[row|{'p':[.5,.5]}])['passed'])
        self.assertFalse(parity([row],[row|{'p':[.49999,.50001]}])['passed'])
        with self.assertRaisesRegex(ValueError,'option semantics'):
            parity([row],[row|{'keys':['b','a'],'p':[.5,.5]}])

    def test_memory_guard_controls_only_its_own_child(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'log'
            with patch('sweep.codegraph_baseline.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError,'no worker started'):
                    guarded(['not-launched'],log,check_pressure=lambda:2)
                spawn.assert_not_called()
            sibling=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
            try:
                calls=iter([1,2])
                with self.assertRaisesRegex(RuntimeError,'only this capture child'):
                    guarded([sys.executable,'-c','import time;time.sleep(10)'],log,check_pressure=lambda:next(calls))
                self.assertIsNone(sibling.poll())
            finally:
                sibling.terminate();sibling.wait(timeout=5)
