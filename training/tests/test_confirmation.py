import json
from pathlib import Path
import sys
import tempfile
import unittest

from lars.train.accuracy import metrics
from lars.train.data import Record
from sweep.confirmation import (canonical_sdk, certificates, paired_interval, promotion, save_dataset)
from sweep.experiment import Runner


class ConfirmationTests(unittest.TestCase):
    def test_hard_labels_are_separate_from_soft_brier_targets(self):
        row={"kind":"choice","p":[.2,.8],"target":[.8,.2],"hard_index":1}
        result=metrics([row])
        self.assertEqual(result["accuracy"],1.)
        self.assertAlmostEqual(result["brier"],.72)

    def test_sdk_capture_retains_hard_label_and_rejects_reordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            r=Record("row","source","score","a\u2028b","Rate this",["low","medium","high"],[.1,.8,.1])
            save_dataset(root,"eval",[(r,0)])
            raw=root/"raw.jsonl"
            row={"kind":"score","keys":["0","1","2"],"label":"0","logits":[3.,1.,0.]}
            raw.write_text(json.dumps(row)+"\n")
            canonical_sdk(raw,root,"eval",root/"out.jsonl")
            captured=json.loads((root/"out.jsonl").read_text())
            self.assertEqual(captured["hard_index"],0)
            self.assertEqual(captured["target"],[.1,.8,.1])
            raw.write_text(json.dumps(row|{"keys":["1","0","2"]})+"\n")
            with self.assertRaises(ValueError):
                canonical_sdk(raw,root,"eval",root/"bad.jsonl")

    def test_certificates_require_enough_independent_labels(self):
        def rows(n):
            return {str(i):{"source":"s","kind":"choice","keys":["a","b"],"p":[.999,.001],
                "hard_index":0,"target":[1.,0.]} for i in range(n)}
        self.assertEqual(certificates(rows(72),rows(100))["accepted"],0)
        self.assertEqual(certificates(rows(100),rows(100))["accepted"],100)
        self.assertEqual(certificates({},rows(100))["accepted"],0)

    def test_paired_interval_clusters_repeated_states_and_seeds(self):
        def row(p):
            return {"source":"s","kind":"choice","state_key":"shared","target":[1.,0.],"hard_index":0,"p":p}
        a={"a":row([.1,.9]),"b":row([.1,.9])}
        b={"a":row([.9,.1]),"b":row([.9,.1])}
        result=paired_interval([a]*3,[b]*3,draws=20)
        self.assertEqual(result["ci95"],[1.,1.])
        self.assertIsNone(paired_interval([a],[b],score_only=True,draws=20))

    def test_promotion_rejects_brier_regression_and_uncertified_coverage(self):
        primary={"baseline":{"accuracy":.70,"brier":.30,"score_accuracy":.45},
                 "distilled":{"accuracy":.72,"brier":.29,"score_accuracy":.49},
                 "accuracy_deltas":[.02,.02,.02],"score_accuracy_deltas":[.04,.04,.04]}
        intervals={"accuracy":{"ci95":[.01,.03]},"score_accuracy":{"ci95":[.02,.06]}}
        coverage={"baseline":{"macro_coverage":.15},"distilled":{"macro_coverage":.15,"accepted":500,
                   "accepted_error":.02,"worst_seed_error_upper95":.04}}
        self.assertTrue(promotion({"confirmation":primary},intervals,coverage)["eligible"])
        ood={"baseline":{"accuracy":.7,"brier":.3},"distilled":{"accuracy":.7,"brier":.33}}
        self.assertFalse(promotion({"confirmation":primary,"ood":ood},intervals,coverage)["eligible"])
        coverage["distilled"]["accepted"]=0
        self.assertFalse(promotion({"confirmation":primary},intervals,coverage)["eligible"])

    def test_resume_checks_outputs_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";target=root/"output"
            source.write_text("first")
            runner=Runner(root/"run")
            cmd=[sys.executable,"-c","from pathlib import Path; import sys; Path(sys.argv[2]).write_text(Path(sys.argv[1]).read_text())",str(source),str(target)]
            runner.stage("copy",command=cmd,inputs=[source],outputs=[target])
            target.write_text("corrupt")
            runner.stage("copy",command=cmd,inputs=[source],outputs=[target])
            self.assertEqual(target.read_text(),"first")
            source.write_text("second")
            runner.stage("copy",command=cmd,inputs=[source],outputs=[target])
            self.assertEqual(target.read_text(),"second")


if __name__=="__main__":
    unittest.main()
