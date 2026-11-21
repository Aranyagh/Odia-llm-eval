# How well does a small open LLM handle Odia?

An evaluation of **Gemma 4 E4B** (`google/gemma-4-E4B-it`, 4-bit) on Odia, Hindi and English translation, using 100 sentences from Flores-IN (IndicGenBench).

**Short answer:** Gemma reads Odia much better than it writes it. English→Odia scores chrF++ 23.8 against 52.1 for Odia→English, and 15% of its Odia outputs degenerate into repetition loops. Its tokenizer needs 4.88 tokens per Odia word, against 1.39 for Hindi.

## Results (100 test sentences, greedy decoding, max 512 new tokens)

| Task | chrF++ | 95% bootstrap range | Outputs hitting the 512-token limit |
|---|---|---|---|
| English → Odia | 23.8 | [from score_task] | 15 / 100 |
| English → Hindi | 52.8 | [from score_task] | 0 / 100 |
| Odia → English | 52.1 | [from score_task] | 0 / 100 |
| Hindi → English | 62.4 | [from score_task] | 0 / 100 |

- Odia→English and Hindi→English use the same English references, so they are directly comparable: Hindi→English is higher by **10.3 chrF++** (95% paired bootstrap range 6.9 to 13.5).
- Scores into different target languages (English→Odia vs English→Hindi) are not directly comparable, because chrF++ depends on the language and script.

## Tokenizer fertility (Gemma tokenizer, 200 aligned sentences)

| | Tokens per word | Tokens per character |
|---|---|---|
| Odia | 4.88 | 0.708 |
| Hindi | 1.39 | 0.273 |
| English | 1.24 | 0.205 |

Odia text is split into nearly one token per character. This is a measurement of the tokenizer, and this study does not show that it causes the quality gap.

## English → Odia error profile

| Category | n | chrF++ |
|---|---|---|
| Clean Odia (Odia letters ≥ 90%) | 71 | 30.1 |
| Odia with stray letters from other scripts | 14 | 27.1 |
| Repetition loop (hit the limit) | 15 | 8.0 |
| Wrong main script | 0 | n/a |

Categories are exclusive, with loops taking priority, and chrF++ per category is descriptive because the categories are defined from the outputs. All 15 limit-hitting outputs have a zlib compression ratio of 0.02 to 0.11, while all other outputs have 0.42 or more, so they are repetitive text. Excluding the loops raises English→Odia from 23.8 to 29.7, still far below Odia→English. [Add human ratings here when available.]

## Prompt selection (dev set, 20 sentences)

Chosen before testing, with the rule "pick the prompt with fewer wrong-script outputs":

| Prompt | chrF++ | Wrong script | Odia share ≥ 0.9 | Hit limit |
|---|---|---|---|---|
| A: plain | 21.1 | 5 / 20 | 9 / 20 | 1 / 20 |
| B: "use the Odia script only" | 20.7 | 1 / 20 | 13 / 20 | 4 / 20 |

Prompt B was frozen (`frozen_prompts.json`). The Hindi prompt copies its structure with the Devanagari script. The into-English prompts have no script instruction.

## Data and method

- Flores-IN from `google/IndicGenBench_flores_in` (test split). The Odia and Hindi files hold the same 1,012 English sentences in a different order, so sentences were matched by their English source. A random sample of 200 (seed 0) was drawn, and the first 100 were used for the test runs. The data files are not redistributed here; download them from Hugging Face.
- Metric: chrF++ (sacreBLEU, `word_order=2`); bootstrap over sentences (1,000 resamples).
- Script analysis uses Unicode script names of each output's letters.

## Reproduce

Open `odia_eval.ipynb` in Colab with a T4 GPU, install `transformers`, `accelerate`, `bitsandbytes>=0.46.1`, `sacrebleu`, `huggingface_hub`, and run the cells in order. Versions used: transformers 5.23.0, accelerate 1.16.0, bitsandbytes 0.50.2, sacrebleu 2.6.0.

## Limitations

- One model (4-bit quantized); no comparison with other models, so no conclusion about tokenizers across models.
- 100 sentences, one benchmark. FLORES is public and may overlap with training data.
- chrF++ is an automatic metric; it was not yet validated against human ratings for Odia [update when done].
- Source sentences in each language are separate human translations, so style differences may affect the gap.
- Greedy decoding and a 512-token limit; loops may behave differently with other decoding settings.

## Files

`gemma_*.jsonl` (all model outputs), `summary_gemma.csv`, `frozen_prompts.json`, `prompt_dev_results.json`, `fertility.json`, notebook.
