# Arabic → Tamil MT with multilingual pretrained models

This repo runs the experimental roadmap from the review paper *Bridging a Distant Language Pair: A Review of
Arabic-to-Tamil Machine Translation Using Multilingual Pretrained Models*. It tests two hypotheses on
FLORES+ devtest (1012 sentences). chrF++ is the primary metric.

| | Hypothesis | Measured as | Threshold |
|---|---|---|---|
| **H1** | NLLB-200-distilled-600M is worse at arb→tam than at eng→tam (same sentences, same Tamil references) | chrF++(`direct.nllb.eng`) − chrF++(`direct.nllb.arb`) | ≥ 3.0 |
| **H2** | Pivoting arb→eng (NLLB) →tam (IndicTrans2) recovers part of that gap | chrF++(`pivot.B1`) − chrF++(`direct.nllb.arb`) | ≥ 2.0 |

**Verdicts.** Every Δ comes with a 95% CI from a paired bootstrap (1000 resamples).
- **PASS:** the CI's lower bound is at or above the threshold.
- **INCONCLUSIVE:** the point estimate meets the threshold but the CI's lower bound is below it.
- **FAIL:** the point estimate is below the threshold.

## Steps → gaps

| Command | What it runs | Gap / hypothesis | Systems written |
|---|---|---|---|
| `a2t step1` | Direct NLLB-200 and M2M100 into `tam_Taml` from (a) Arabic as written, (b) Arabic with U+064B–U+0652, U+0670 and U+0640 removed, (c) English as a control | **G1** (direct quality of multilingual models for arb→tam), **G2** (orthography: short vowels), **G4** (distance: same content from English vs Arabic), **H1** | `direct.{nllb,m2m100}.{arb,arb_nodiac,eng}` |
| `a2t step2` | Pivot routes, with IndicTrans2 writing the Tamil. **B1**: arb→eng (NLLB) → tam (IT2 en-indic 200M). **B2**: arb→mal (NLLB) → tam (IT2 indic-indic 320M). Two control rows: an all-NLLB B1, and IT2 translating the human English (B1's upper bound). Picks the Step-3 route on **dev**. | **G3** (pivoting), **H2** | `pivot.B1`, `pivot.B2`, `control.B1_allnllb`, `control.it2_humaneng` |
| `a2t evaluate` | chrF++ (`word_order=2`), spBLEU (`tokenize="flores200"`), CIs, paired bootstrap, H1/H2 verdicts | – | `outputs/results.{csv,md}`, `outputs/comparisons.csv` |
| `a2t step3 [--pilot]` | Synthetic arb↔tam corpus. Arabic Wikipedia goes through the selected route; Tamil Wikipedia goes through its reverse. Filtering is described below. | **G5** (no parallel arb–tam data) | `outputs/synth/corpus.*.jsonl` |
| `a2t step4 [--no-tune]` | LoRA (peft) on frozen NLLB-200, trained on the synthetic corpus in both directions. Grid search and checkpoint choice use FLORES **dev** only; devtest is decoded once at the end. | **G6** (direct adapted model) | `lora.arb`, `lora.tam2arb`, `direct.nllb.tam2arb` |
| `a2t step5` | Prompted-LLM baseline stub, same input/output contract, disabled by default | **G7** | `llm.arb` |
| `a2t all` | step1 + step2 + evaluate. It never starts Steps 3–4. | | |

Run `a2t evaluate` again after Step 4. It then also reports LoRA vs direct NLLB, LoRA vs its pivot teacher,
and the reverse direction.

### Paired comparisons in the report
- **H1:** `direct.nllb.eng` vs `direct.nllb.arb`
- **H2:** `pivot.B1` vs `direct.nllb.arb`
- B1 vs B2
- B1 vs all-NLLB B1: the gain from IndicTrans2 writing the Tamil
- all-NLLB B1 vs direct NLLB: the gain from pivoting alone, with the same model on both hops
- IT2 on human English vs B1. How much does the arb→eng hop cost?
- `arb_nodiac` vs `arb`
- M2M100 vs NLLB
- LoRA vs direct NLLB
- LoRA vs the selected pivot route
- LoRA tam→arb vs NLLB tam→arb

## Setup (Windows, Linux, macOS)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (Git Bash); on Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA; use the plain index for CPU/MPS
pip install -e ".[dev]"           # exact tested versions: pip install -r requirements.lock
```

On Windows, IndicTransToolkit has no prebuilt wheel. pip compiles it from source, which needs the MSVC C++ Build
Tools (Visual Studio Build Tools 2022, "Desktop development with C++").

**Gated repositories.** Accept the terms on each of these pages. All of them approve you automatically.
- [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- [ai4bharat/indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
- [ai4bharat/indictrans2-indic-indic-dist-320M](https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M)
- [ai4bharat/indictrans2-indic-en-dist-200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M) (Step 3 only)

Then log in and check access:

```bash
hf auth login
a2t check
a2t download
```

`hf auth login` takes a read token. `a2t check` reports access for every repo. `a2t download` fetches FLORES+
and all models and prints the commit SHAs that are pinned in `configs/default.yaml`.

## Reproducibility
- **One config file.** [configs/default.yaml](configs/default.yaml) sets models and pinned revisions, language
  codes, beam size (4), batch size, dtype, device (`auto` picks cuda, then mps, then cpu), the seed, and the
  settings for every step.
- **Seeds.** Python, NumPy and torch are seeded and cuDNN runs deterministically. Bootstrap indices are seeded,
  and every system scored against the same references shares them.
- **Environment log.** Each run appends library versions, GPU, git commit and the full config to
  `outputs/env.jsonl`. Translation speed goes to `outputs/throughput.jsonl`.
- **Translation cache.** Translations are cached in `outputs/translations/<model>/*.jsonl` and appended after
  every batch, so an interrupted run resumes where it stopped. The cache key hashes:
  - model ID and revision
  - LoRA adapter weights, if any
  - language pair
  - generation settings
  - the exact input sentences

  Changing any of these starts a new cache file instead of mixing outputs.
- **Pausing Step 4.** LoRA training saves a resumable checkpoint every `step4.train.save_steps` steps (500,
  about 10 min on an RTX 4060; only the newest is kept) and at every epoch end (always kept). Stop the job at
  any time and run `a2t step4` again: it resumes weights, optimizer, schedule and data position, and skips
  finished tuning runs. Only epoch-end checkpoints are scored on dev for selection.
- **FLORES+ is not FLORES-200.** Some translations have been corrected since FLORES-200, so scores will not
  exactly match published FLORES-200 numbers. The dataset revision is pinned (`flores.revision`). Text is NFC
  normalised on load.
- **Diacritic stripping on FLORES.** FLORES Arabic is mostly written without short vowels, so stripping changes
  little. `outputs/diacritic_stats.devtest.json` records how many characters and sentences actually changed;
  read the G2 row with that in mind.
- **Licences.** NLLB-200 is CC-BY-NC-4.0, M2M100 and IndicTrans2 are MIT, FLORES+ is CC-BY-SA-4.0, and
  Wikipedia is CC-BY-SA-3.0.

## Step 3 filtering

Monolingual source: `wikimedia/wikipedia` snapshot 20231101. Arabic uses the first parquet shard; Tamil uses
the whole subset. Filters, in order:

1. Split into sentences and NFC normalise.
2. Keep sentences of 3–200 whitespace tokens.
3. Require at least 80% of letters in the expected script.
4. Dedupe after normalisation.
5. Keep at most 5 sentences per article.
6. Remove any sentence that matches FLORES dev or devtest after normalisation, or shares a word 8-gram with one.
   This uses both Arabic and Tamil FLORES.
7. Shuffle with the seed and take N.
8. Translate in 5k chunks. `--pilot` gives exactly the first chunk of the full run, so its work is reused.
9. On the synthetic side, drop outputs that are empty, out of length range, in the wrong script, or that
   overlap FLORES.
10. Drop pairs whose token-length ratio is above 3, and drop duplicate pairs.

`outputs/synth/stats.*.json` gives counts for every filter.

## Layout
```
configs/default.yaml
src/a2t/  data.py  translate.py  pivot.py  synth.py  train_lora.py  evaluate.py  llm_stub.py  cli.py  utils.py
tests/    test_diacritics.py  test_flores_loader.py  test_metrics.py     (pytest -m "not online")
outputs/  translations/  systems/  synth/  lora/  results.csv  results.md  comparisons.csv  env.jsonl
```
