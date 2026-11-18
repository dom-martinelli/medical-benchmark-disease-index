import json
import unittest

import numpy as np

from stages.code import format_code, icd_chapter
from stages.collect import _redistribution_allowed, parse_medbullets, parse_mmlu
from stages.label import _normalize_response
from stages.models import parse_json_object
from stages.report import expected_answer_text, wrong_option_choice_map


class PipelineTests(unittest.TestCase):
    def test_mmlu_parser(self):
        row = parse_mmlu({"question": "Q", "choices": ["one", "two", "three", "four"], "answer": 2}, "test")
        self.assertEqual(row["answer_key"], "C")
        self.assertTrue(row["has_answer_key"])

    def test_mmlu_parquet_array_parser(self):
        row = parse_mmlu({"question": "Q", "choices": np.array(["one", "two", "three", "four"]), "answer": 1}, "test")
        self.assertEqual(json.loads(row["options"])["B"], "two")

    def test_medbullets_options_are_not_duplicated(self):
        row = parse_medbullets({"idx": "1", "question": "Q", "options": {"A": "x"}, "answer": "A", "explanation": "why"}, "op5_test")
        self.assertEqual(json.loads(row["options"]), {"A": "x"})
        self.assertEqual(row["explanation"], "why")

    def test_model_json_fences(self):
        self.assertEqual(parse_json_object('```json\n{"conditions": []}\n```')["conditions"], [])

    def test_quote_is_capped_and_role_validated(self):
        obj = {"conditions": [{"condition_name": "asthma", "role": "tested", "supporting_quote": "one two three four five six seven eight nine ten eleven twelve thirteen", "confidence": "high"}]}
        row = _normalize_response(obj)[0]
        self.assertEqual(len(row["supporting_quote"].split()), 12)
        self.assertEqual(row["role"], "tested")

    def test_no_condition_sentinel(self):
        row = _normalize_response({"conditions": [], "no_condition_category": "ethics or law"})[0]
        self.assertEqual(row["condition_name"], "")
        self.assertEqual(row["no_condition_category"], "ethics or law")

    def test_codes_and_chapters(self):
        self.assertEqual(format_code("E119"), "E11.9")
        self.assertTrue(icd_chapter("C509").startswith("II."))
        self.assertTrue(icd_chapter("D649").startswith("III."))
        self.assertTrue(icd_chapter("H919").startswith("VIII."))

    def test_conservative_license_rule(self):
        self.assertTrue(_redistribution_allowed("mit"))
        self.assertTrue(_redistribution_allowed("apache-2.0"))
        self.assertFalse(_redistribution_allowed("unknown"))
        self.assertFalse(_redistribution_allowed("mit", "HealthBench"))
        self.assertTrue(_redistribution_allowed("unknown", "MedQA"))
        self.assertFalse(_redistribution_allowed("unknown", "MedBullets"))

    def test_expected_answer_text(self):
        self.assertEqual(expected_answer_text('{"A":"alpha","B":"beta"}', "B"), "beta")
        self.assertEqual(expected_answer_text('{"A":"yes","B":"no"}', "yes"), "yes")
        self.assertEqual(expected_answer_text("{}", "free text"), "free text")

    def test_wrong_option_choice_map(self):
        options = '{"A":"Meningioma","B":"Astrocytoma","C":"Vascular malformation"}'
        labels = [{"condition_name":"Astrocytoma", "supporting_quote":"Meningioma, Astrocytoma"}]
        self.assertEqual(wrong_option_choice_map(options, "A", labels), {"B": ["Astrocytoma"]})
        self.assertEqual(wrong_option_choice_map(options, "B", labels), {})


if __name__ == "__main__":
    unittest.main()
