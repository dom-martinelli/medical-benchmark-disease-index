# What it would cost to label every item in every medical AI benchmark

*Prepared 14 Sep 2026. OpenAI Batch API prices (50% off list; results within 24 hours). Prices from https://developers.openai.com/api/docs/pricing.*

## The short answer

Labeling the **evaluation items of all clinical benchmarks (~100–140k items) costs roughly $40–100** on a mid-size model. It only reaches hundreds or thousands of dollars if you include training splits or two very large synthetic datasets that are not clinical questions.

## By scope

43 datasets: 42 screened medical QA benchmarks plus HealthBench. Three have no public data (PediatricsQA, SDBench, PUMA) and count as zero.

| Scope | Items | gpt-5-nano | gpt-5-mini | gpt-4.1-mini | gpt-5.4-mini |
|---|---:|---:|---:|---:|---:|
| A. Everything ever released, including training splits | 3.05M | $180 | $890 | $1,220 | $2,480 |
| B. Test and validation sets only | 1.09M | $55 | $270 | $365 | $745 |
| C. B without DDXPlus and MedConceptsQA | 138k | $7 | $37 | $50 | $100 |
| D. C without datasets that re-copy other benchmarks | 103k | $6 | $28 | $38 | $77 |

**Recommended: scope C or D on gpt-4.1-mini, about $40–50.**

## Why two datasets dominate the cost

- **MedConceptsQA** (820k test items) is automatically generated vocabulary questions: "what does medical code X mean?" Labeling them with diseases is circular.
- **DDXPlus** (135k test items) is computer-generated patients covering only 49 diseases.

Together they are 88% of scope B.

Scope D also drops ClinCheckBench, MedLFQA, PedMedQA, ReMedQA, and MedThink-Bench, whose items are copied from other benchmarks already counted.

## How the estimate was made

- **Short items** (multiple choice, yes/no, short questions): about 1,230 input and 65 output tokens each, **measured** on 9,000 items already labeled.
- **Long items** (clinical cases, conversations, long-form answers): about 2,200 input and 110 output tokens each, **estimated**.
- Item counts come from each dataset's paper, Hugging Face card, or direct download.

## Could be off because

- Some case-based datasets (MedXpertQA, RareArena, MedR-Bench) may be longer than assumed, which would raise the cost.
- GPT-5 models spend hidden "reasoning" tokens unless reasoning is set to minimal.
- About 1,100 tokens of every call are identical instructions, so prompt caching could lower the cost. Whether caching stacks with the Batch discount is not verified.
- Adding a second model for reliability checks adds roughly 5–10%.

## Current status of the pilot

The six-benchmark version (14,088 items) is being labeled on a local model at no cost and should finish this week.
