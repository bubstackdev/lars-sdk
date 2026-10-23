import unittest

import numpy as np
from scipy.special import expit

from sweep.queue_calibration import fit_affine,loss,select,metrics


class QueueCalibrationTests(unittest.TestCase):
    def test_fit_can_correct_a_probability_bias(self):
        x=np.zeros(20);y=np.array([1]*16+[0]*4)
        a,b=fit_affine(x,y,.1)
        self.assertLess(loss(expit(a*x+b),y).mean(),loss(expit(x),y).mean())
        self.assertTrue(0<=a<=10)

    def test_strong_regularization_keeps_the_pack_prior(self):
        x=np.array([-1.,1.,2.,-2.]);y=np.array([1.,0.,0.,1.])
        a,b=fit_affine(x,y,100000.)
        self.assertAlmostEqual(a,1.,places=3)
        self.assertAlmostEqual(b,0.,places=3)

    def test_selection_is_deterministic_and_uses_log_loss(self):
        x=np.linspace(-2,2,15);y=np.array([0]*7+[1]*8)
        first=select(x,y,42);second=select(x,y,42)
        self.assertEqual(first,second)
        self.assertAlmostEqual(first[1]['pack'],loss(expit(x),y).mean())
        self.assertEqual(set(first[1]),{'pack','0.1','1.0','10.0','100.0'})

    def test_strata_and_label_support_are_reported(self):
        rows=[{'label':'reprocess','evidence_level':'diagnostic'},
              {'label':'discard','evidence_level':'alert_or_symptom_only'}]
        result=metrics(np.array([.9,.8]),rows)
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['strata']['diagnostic'],{'n':1,'correct':1})
        self.assertEqual(result['per_label']['discard'],{'n':1,'correct':0})


if __name__=='__main__':unittest.main()
