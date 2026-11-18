# Label review prompt, version 1.0.1

Review each proposed condition label against the supplied benchmark item. Do not add new conditions and do not produce medical codes.

Return one JSON object and nothing else:

```json
{
  "reviews": [
    {
      "label_index": 0,
      "supported": true,
      "reason": "short reason grounded in the item"
    }
  ]
}
```

Mark a label supported only when both the condition and its assigned role are supported by the item text.

The input supplies an explicit `label_index` for every proposed label. Return exactly one review for every supplied label, using each supplied index exactly once. If the labels list is empty, return `{"reviews": []}`.

ITEM AND LABELS:
{{ITEM_AND_LABELS_JSON}}
