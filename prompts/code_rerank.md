# ICD-10-CM candidate rerank prompt, version 1.0.0

You are checking which official ICD-10-CM code best matches a condition name. The candidates below were taken from the official ICD-10-CM files. You may only choose one of them, or say none fits. Never write a code yourself.

Rules:
- Choose the candidate that means the same condition as the name, at the same level of detail or the closest more general code.
- Do not choose a candidate that adds something the name does not say, such as "in situ", "postprocedural", "newborn", "maternal" or "pregnancy", "traumatic", a different body site, or a different organism.
- If the name is not a disease or health condition (for example a cell type, a drug, a lab value, or a procedure), or no candidate means the same thing, answer "none".

Return one JSON object and nothing else:

```json
{"choice": 0, "reason": "short reason"}
```

Use the candidate number for `choice`, or `"none"`.

CONDITION NAME:
{{CONDITION}}

CANDIDATES:
{{CANDIDATES}}
