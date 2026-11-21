# How well does a small open LLM handle Odia?

An evaluation of **Gemma 4 E4B** (`google/gemma-4-E4B-it`, 4-bit) on Odia, Hindi and English translation, using 100 sentences from Flores-IN (IndicGenBench).

**Short answer:** Gemma 4 E4B writes Odia poorly. English→Odia is its weakest task (chrF++ 23.8, against 52.8 for English→Hindi, though scores into different languages are only roughly comparable), and 15% of its Odia outputs degenerate into repetition loops. On the fair comparison, reading, Hindi→English beats Odia→English by 10.3 chrF++ (95% paired bootstrap range 6.9 to 13.5). Its tokenizer needs 4.88 tokens per Odia word, against 1.39 for Hindi.

## Results (100 test sentences, greedy decoding, max 512 new tokens)

| Task | chrF++ | 95% bootstrap range | Outputs hitting the 512-token limit |
|---|---|---|---|
| English → Odia | 23.8 | [21.0, 26.6] | 15 / 100 |
| English → Hindi | 52.8 | [49.6, 55.9] | 0 / 100 |
| Odia → English | 52.1 | [49.7, 54.7] | 0 / 100 |
| Hindi → English | 62.4 | [59.3, 65.4] | 0 / 100 |

- Odia→English and Hindi→English use the same English references, so they are directly comparable: Hindi→English is higher by **10.3 chrF++** (95% paired bootstrap range 6.9 to 13.5).
- Scores into different target languages (English→Odia vs English→Hindi) are not directly comparable, because chrF++ depends on the language and script.
- FLORES is public, so Hindi→English in particular may be inflated by training-data overlap.

## Tokenizer fertility (Gemma tokenizer, 200 aligned sentences)

| | Tokens per word | Tokens per character |
|---|---|---|
| Odia | 4.88 | 0.708 |
| Hindi | 1.39 | 0.273 |
| English | 1.24 | 0.205 |

Odia text is split into nearly one token per character. This measures the tokenizer. This study does not show that it causes the quality gap.

## English → Odia error profile

| Category | n | chrF++ |
|---|---|---|
| Clean Odia (Odia letters ≥ 90%) | 71 | 30.1 |
| Odia with stray letters from other scripts | 14 | 27.1 |
| Repetition loop (hit the limit) | 15 | 8.0 |
| Wrong main script | 0 | n/a |

Categories are exclusive, with loops taking priority. The chrF++ per category is descriptive, because the categories are defined from the outputs themselves. All 15 limit-hitting outputs have a zlib compression ratio of 0.02 to 0.11, while every other output has 0.42 or more, so they are repetitive text. Excluding the loops raises English→Odia from 23.8 to 29.7, which is still far below Odia→English.

**Human ratings:** pending (30 outputs sent to an Odia reader).

## Prompt selection (dev set, 20 sentences)

Chosen before testing, with the rule "pick the prompt with fewer wrong-script outputs":

| Prompt | chrF++ | Wrong script | Odia share ≥ 0.9 | Hit limit |
|---|---|---|---|---|
| A: plain | 21.1 | 5 / 20 | 9 / 20 | 1 / 20 |
| B: "use the Odia script only" | 20.7 | 1 / 20 | 13 / 20 | 4 / 20 |

Prompt B was frozen (`frozen_prompts.json`). The Hindi prompt copies its structure with the Devanagari script. The into-English prompts have no script instruction.

## Data and method

- Flores-IN from `google/IndicGenBench_flores_in` on Hugging Face (test split). The Odia and Hindi files contain the same 1,012 English sentences in a different order, so sentences were matched by their English source. A random sample of 200 (`random.seed(0)`) was drawn, and the first 100 were used for the test runs.
- **This repository does not contain the benchmark data.** The output files hold only the model's generations, with sentence ids. Download the data from Hugging Face and recreate the sample with the same seed to reproduce the references.
- Metric: chrF++ (sacreBLEU, `word_order=2`); bootstrap over sentences (1,000 resamples, seed 0).
- Script analysis uses the Unicode script names of each output's letters.

## Reproduce

Open `odia_eval.ipynb` in Google Colab with a T4 GPU, install `transformers`, `accelerate`, `bitsandbytes>=0.46.1`, `sacrebleu` and `huggingface_hub`, and run the cells in order. Versions used: transformers 5.23.0, accelerate 1.16.0, bitsandbytes 0.50.2, sacrebleu 2.6.0.

## Files

| Path | What it is |
|---|---|
| `Outputs/gemma_*_outputs.jsonl` | Model generations for each task (ids, outputs, token counts) |
| `Results/summary_gemma.csv` | chrF++ and limit-hit counts per task |
| `Results/fertility.json` | Tokenizer fertility numbers |
| `Results/prompt_dev_results.json` | Prompt comparison on the dev set |
| `frozen_prompts.json` | The exact prompts used |
| `odia_eval.ipynb` | All the code |

## Limitations

- One model (4-bit quantized), with no comparison to other models, so no conclusion about tokenizers across models.
- 100 sentences from one benchmark. FLORES is public and may overlap with training data.
- chrF++ is an automatic metric and has not yet been validated against human ratings for Odia.
- The source sentences in each language are separate human translations, so style differences may affect the gap.
- Greedy decoding with a 512-token limit; loops may behave differently with other decoding settings.
