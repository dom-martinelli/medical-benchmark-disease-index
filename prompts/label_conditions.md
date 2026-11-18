# Condition labeling prompt, version 1.2.0

You label every medical condition explicitly involved in a benchmark item. Use only the supplied question, options, answer key, and explanation. Do not supply ICD, SNOMED, or other codes.

Return one JSON object and nothing else, using this schema:

```json
{
  "conditions": [
    {
      "condition_name": "plain clinical name",
      "role": "tested|context|ruled_out|wrong_option",
      "supporting_quote": "at most 12 words copied from the item",
      "confidence": "high|low"
    }
  ],
  "no_condition_category": null
}
```

Role definitions:

- `tested`: the condition is the subject of the question, is the correct answer, or the correct answer depends on knowing about it. Negative wording such as "Which is NOT a symptom of anaphylaxis?" still makes anaphylaxis `tested`.
- `context`: the condition is in the patient's history or story, but the correct answer tests something else. For example, a cancer treated with a drug is `context` when the question tests the drug's adverse effect.
- `ruled_out`: the item states that the patient does not have the condition or that evidence excludes it. Use this only for an actual patient-level exclusion. Do not use it merely because the question says NOT, asks for an exception, mentions a differential diagnosis, or recommends ruling something out.
- `wrong_option`: the condition occurs only in an incorrect answer option. Use this role only when a public answer key identifies that option as incorrect.

Coverage procedure:

1. Read the question and identify any named disease, disorder, syndrome, injury, infection, poisoning, deficiency, or clinically recognized adverse effect.
2. Use the answer key to identify the correct option, then scan both correct and incorrect options for additional conditions.
3. Scan the explanation for conditions not already captured.
4. Return one row for every supported condition, not only the main diagnosis. If a condition appears more than once, return it once with the strongest applicable role in this order: `tested`, `context`, `ruled_out`, `wrong_option`.

Important boundaries:

- Never invent or infer a diagnosis that is not named or directly supported by the supplied text. "Glomerular capillaries" does not justify inventing "glomerulopathy."
- Apply a diagnosability test to every proposed label: could this phrase appear as a diagnosis or clinically recognized adverse effect in a patient's problem list? If not, omit it.
- Do not label normal anatomy or physiology, a symptom alone, a medication, a procedure, a laboratory test, a biological process, a sensory system, a generic complication, or a vague fragment such as "thyroid" or "entrapment" as a condition. Examples that must not be labels include "myelinated nerve fibers," "glomerular oncotic pressure," "glomerulus," "gustatory system," "minor complication," and "fasting state." A conventional named diagnosis such as "ulnar nerve entrapment" is acceptable when directly supported.
- When the answer key is blank, never use `wrong_option`. Use `tested` for a condition directly asked about and `context` for a condition in the background. Choose low confidence when the role remains ambiguous.
- The supporting quote must be copied exactly from the supplied item and directly support the condition or its role. Prefer a quote containing the condition name. It may contain at most 12 whitespace-separated words.
- Use high confidence only when both the condition and role are clear. Otherwise use low confidence.

Calibration examples:

- "Which statement about glomerular capillaries is not true?" contains no condition. Return `basic science`; do not turn the anatomy or physiology into a disease.
- "Which is NOT a symptom of anaphylaxis?" contains `Anaphylaxis` with role `tested`, not `ruled_out`.
- "Differences in strabismus amblyopia?" contains `strabismic amblyopia` with role `tested`.
- A no-key request about "in-hospital adult cardiac arrests" contains `cardiac arrest` with role `tested`.
- A disease mentioned only in a known incorrect answer option is `wrong_option`; a disease named in the stem of a question about another mechanism is usually `context`.

If the item contains no condition, return an empty `conditions` list and set `no_condition_category` to exactly one of: `ethics or law`, `statistics or epidemiology`, `basic science`, `health systems`, `communication`, or `other`.

Before returning JSON, check that every condition is supported, every named condition was considered, and `no_condition_category` is null whenever the conditions list is nonempty.

ITEM:
{{ITEM_JSON}}
