import copy
import json
from pathlib import Path
import tempfile
import unittest

from sweep.codegraph_judgment import build, family_role, grouped_families, model_input, validate, verify


class JudgmentCorpusTests(unittest.TestCase):
    def fixture(self):
        case = {'story_id': 1, 'family_id': 'incident-a', 'draft_label': 'insufficient_evidence',
                'summary': 'A request intermittently fails.', 'observations': ['No trace is available.'],
                'evidence': [{'comment_id': 8, 'quote': 'Need a trace'}],
                'review_notes': 'OUTCOME_SENTINEL should never reach model input.'}
        source = {'id': 1, 'url': 'https://example.invalid/story/1', 'updated_at': '2026-10-22',
                  'description': 'A report', 'comments': [{'id': 8, 'text': 'Need a trace to diagnose.'}]}
        return {'status': 'source_interpreted_unreviewed', 'rows': [case]}, {1: source}

    def test_outcomes_and_identifiers_cannot_enter_rendering(self):
        seed, sources = self.fixture(); row = seed['rows'][0]
        before = model_input(row)
        row.update({'review_notes': 'ANOTHER_OUTCOME', 'draft_label': 'application_engineering',
                    'story_id': 2, 'family_id': 'other', 'resolution': 'LEAK'})
        self.assertEqual(before, model_input(row))
        bad = copy.deepcopy(seed); bad['rows'][0]['story_id'] = 1
        bad['rows'][0]['summary'] = 'Contact customer@example.com'
        with self.assertRaisesRegex(ValueError, 'contact details'):
            validate(bad, sources)

    def test_duplicates_and_false_gold_are_rejected(self):
        seed, sources = self.fixture()
        validate(seed, sources)
        bad = copy.deepcopy(seed); bad['status'] = 'reviewed'
        with self.assertRaisesRegex(ValueError, 'cannot promote'):
            validate(bad, sources)
        bad = copy.deepcopy(seed); bad['rows'].append(copy.deepcopy(bad['rows'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate source'):
            validate(bad, sources)
        bad = copy.deepcopy(seed); bad['rows'][0]['evidence'][0]['quote'] = 'Not in source'
        with self.assertRaisesRegex(ValueError, 'quote missing'):
            validate(bad, sources)

    def test_formal_links_through_shared_stubs_cannot_cross_roles(self):
        seed, sources = self.fixture()
        second = copy.deepcopy(seed['rows'][0]); second.update(story_id=2, family_id='incident-b')
        seed['rows'].append(second); sources[2] = copy.deepcopy(sources[1])
        for sid in sources:
            sources[sid]['story_links'] = [{'subject_id': sid, 'object_id': 999}]
        groups = grouped_families(seed, sources)
        self.assertEqual(groups[1], groups[2])
        self.assertEqual(family_role(groups[1]), family_role(groups[2]))

    def test_frozen_bundle_keeps_review_and_model_inputs_separate(self):
        seed, sources = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); src = root/'sources'; src.mkdir()
            (src/'story-1.json').write_text(json.dumps(sources[1]))
            curation = root/'curation.json'; curation.write_text(json.dumps(seed))
            out = root/'bundle'
            report = build(src, curation, out)
            self.assertFalse(report['calibration_ready'])
            self.assertEqual(report['independently_reviewed_labels'], 0)
            self.assertTrue(verify(out)['verified'])
            inputs = (out/'inputs.jsonl').read_text()
            self.assertNotIn('OUTCOME_SENTINEL', inputs)
            self.assertNotIn('draft_label', inputs)
            blind = (out/'review-blind.csv').read_text()
            self.assertNotIn('OUTCOME_SENTINEL', blind)
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                build(src, curation, out)
            (out/'inputs.jsonl').write_text(inputs+'\n')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                verify(out)


if __name__ == '__main__':
    unittest.main()
