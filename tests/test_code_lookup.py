"""Regression cases for the Stage 3 ICD-10-CM lookup.

The first lookup used partial fuzzy matching and took the first description that
contained the condition name, which sent "sepsis" to Salmonella sepsis and
"anaphylaxis" to a personal-history code. These cases pin the fix. They need the
pinned reference files in data/reference and are skipped without them.

Lookup version 4 cases come from an audit of 8,605 real labels (2026-09-14):
common diseases and abbreviations that got no code, cancers coded as carcinoma in
situ, qualified codes (postprocedural, newborn, maternal, traumatic) chosen for
plain names, and wrong-organ near misses. None of these tests call a model."""

import unittest

from stages.code import REFERENCE_DIR

HAVE_REFERENCE = all(
    (REFERENCE_DIR / name).exists()
    for name in ("icd10cm-April-1-2026.zip", "DXCCSR-v2026-1.zip", "icd10cm-April-1-2026-tables-index.zip")
)

EXPECTED = {
    "sepsis": "A41.9",
    "pneumonia": "J18.9",
    "hypertension": "I10",
    "myocardial infarction": "I21.9",
    "heart attack": "I21.9",
    "anaphylaxis": "T78.2",
    "hemophilia A": "D66",
    "tuberculosis": "A15.9",
    "Alzheimer's disease": "G30",
    "common cold": "J00",
    "asthma": "J45",
    "celiac disease": "K90.0",
}

# Condition name -> accepted code prefixes. A prefix stands for the code family
# where ICD-10-CM allows several correct codes.
EXPECTED_FAMILY = {
    # Common diseases that were unmatched.
    "colorectal cancer": ("C18", "C19", "C20"),
    "Graves' disease": ("E05.0",),
    "Graves’ disease": ("E05.0",),
    "Graves disease": ("E05.0",),
    "cervical cancer": ("C53",),
    "Kawasaki disease": ("M30.3",),
    "osteosarcoma": ("C40", "C41"),
    "nephrolithiasis": ("N20.0",),
    "Ewing's sarcoma": ("C40", "C41"),
    "neuroblastoma": ("C74.9",),
    "non-small cell lung carcinoma": ("C34.9",),
    # Abbreviations.
    "COPD": ("J44.9",),
    "SLE": ("M32.9",),
    "SIADH": ("E22.2",),
    "DIC": ("D65",),
    "TB": ("A15", "A16", "A17", "A18", "A19"),
    "CKD": ("N18.9",),
    "PCOS": ("E28.2",),
    "GVHD": ("D89.81",),
    # Lookup version 5: correct codes that version 4 threw away because they carry a
    # newborn, maternal, complication, or traumatic qualifier the disease inherently has.
    "congenital CMV infection": ("P35.1",),
    "meconium aspiration syndrome": ("P24.0",),
    "malignant hyperthermia": ("T88.3",),
    "hemolytic transfusion reaction": ("T80.9",),
    "rotator cuff tear": ("M75.1",),
    "uterine leiomyoma": ("D25",),
    "Menier's disease": ("H81.0",),
    "idiopathic intracranial hypertension": ("G93.2",),
    "cephalhematoma": ("P12.0",),
    "Ahogryposis multiplex cngenita": ("Q74.3",),
    # E08 ("due to underlying condition") replaced by the plain type 2 code.
    "diabetic ketoacidosis": ("E11.1", "E10.1"),
    "diabetic nephropathy": ("E11.2", "E10.2"),
    "diabetic neuropathy": ("E11.4", "E10.4"),
    "DLBCL": ("C83.3",),
    "DVT": ("I82.4",),
    # Cancers that were coded as carcinoma in situ.
    "rectal cancer": ("C20",),
    "oral cancer": ("C06", "C03", "C04", "C05"),
    "laryngeal carcinoma": ("C32.9",),
    "cancer larynx": ("C32.9",),
    "carcinoma larynx": ("C32.9",),
    "Endometrial carcinoma": ("C54.1",),
    "colorectal carcinoma": ("C18", "C19", "C20"),
    "invasive bladder cancer": ("C67",),
    "localized prostate cancer": ("C61",),
    "carcinoma breast": ("C50",),
    # Qualified codes chosen for plain names.
    "septicemic shock": ("R65.21", "A41.9"),
    "chronic hypertension": ("I10",),
    "arterial hypertension": ("I10",),
    "disseminated intravascular coagulation": ("D65",),
    "diabetic retinopathy": ("E11.3", "E10.3"),
    "subarachnoid hemorrhage": ("I60",),
    "seizures": ("R56.9",),
    "uveitis": ("H20",),
    # Wrong-organ and near-miss matches.
    "scaphoid fracture": ("S62.0",),
    "choriocarcinoma": ("C58",),
    "gastric cancer": ("C16.9",),
    "squamous cell carcinoma of tongue": ("C02",),
    "squamous cell carcinoma of vulva": ("C51",),
    "T-cell leukemia": ("C91.5", "C91.6"),
    "enamel fluorosis": ("K00.3",),
    "Varicella zoster": ("B02",),
    "zoster": ("B02",),
    "Supracondylar fracture of humerus": ("S42.4",),
}

# Codes that were wrong before and must not come back, where the right answer is
# debatable or there may be none.
NOT_EXPECTED = {
    "premature ovarian failure": "E89",
    "chronic hypertension": "I97",
    "septicemic shock": "T81",
    "cesarean delivery": "P03",
    "neonatal death": "P01",
    "arterial hypertension": "O16",
    "disseminated intravascular coagulation": "P60",
    "diabetic retinopathy": "E08",
    "subarachnoid hemorrhage": "S06",
    "seizures": "R56.1",
    "uveitis": "H44.13",
    "squamous cell carcinoma of tongue": "C44",
    "gastric cancer": "K31",
    "choriocarcinoma": "C69",
    "posterior dislocation of knee joint": "S53",
}


@unittest.skipUnless(HAVE_REFERENCE, "pinned ICD-10-CM, CCSR, and index files are not downloaded")
class CodeLookupRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stages.code import CodeLookup, load_reference

        reference, _ = load_reference()
        cls.lookup = CodeLookup(reference)

    def test_common_conditions_get_the_general_code(self):
        for condition, code in EXPECTED.items():
            with self.subTest(condition=condition):
                self.assertEqual(self.lookup.match(condition)["code"], code)

    def test_history_codes_only_when_asked_for(self):
        self.assertFalse(self.lookup.match("anaphylaxis")["code"].startswith("Z"))

    def test_every_matched_code_has_a_ccsr_category(self):
        for condition in EXPECTED:
            with self.subTest(condition=condition):
                self.assertNotEqual(self.lookup.match(condition)["ccsr_category"], "unmatched")

    def test_audit_cases_get_the_right_code_family(self):
        for condition, prefixes in EXPECTED_FAMILY.items():
            with self.subTest(condition=condition):
                code = self.lookup.match(condition)["code"]
                self.assertTrue(code.startswith(prefixes), f"{condition!r} -> {code}, expected {prefixes}")

    def test_audit_wrong_codes_do_not_come_back(self):
        for condition, prefix in NOT_EXPECTED.items():
            with self.subTest(condition=condition):
                self.assertFalse(self.lookup.match(condition)["code"].startswith(prefix))

    def test_cancer_names_never_get_in_situ_codes(self):
        for condition in ("rectal cancer", "oral cancer", "laryngeal carcinoma", "Endometrial carcinoma", "carcinoma breast"):
            with self.subTest(condition=condition):
                self.assertTrue(self.lookup.match(condition)["code"].startswith("C"))

    def test_in_situ_only_when_asked_for(self):
        self.assertTrue(self.lookup.match("carcinoma in situ of larynx")["code"].startswith("D02"))

    def test_rerank_candidates_are_real_codes(self):
        codes = set(self.lookup.codes)
        shown = self.lookup.candidates("gastric cancer")
        self.assertTrue(0 < len(shown) <= self.lookup.RERANK_LIMIT)
        self.assertTrue(all(self.lookup.codes[i] in codes for i in shown))

    def test_rerank_cannot_return_an_invented_code(self):
        import tempfile
        from pathlib import Path

        class FakeClient:
            model_name, settings = "fake", {}

            def __init__(self, answer):
                self.answer = answer

            def generate_json(self, prompt):
                return self.answer

        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "rerank.jsonl"
            base = self.lookup.match("gastric cancer")
            out_of_range = self.lookup.rerank("gastric cancer", FakeClient({"choice": 99}), base, checkpoint)
            self.assertEqual(out_of_range["code"], base["code"])
            invented = self.lookup.rerank("gastric cancer", FakeClient({"choice": "C99.9"}), base, checkpoint)
            self.assertEqual(invented["code"], base["code"])
            declined = self.lookup.rerank("gastric cancer", FakeClient({"choice": "none"}), base, checkpoint)
            self.assertEqual(declined["code"], "unmatched")
            self.assertTrue(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
