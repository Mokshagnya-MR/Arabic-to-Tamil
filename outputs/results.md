# Results — FLORES+ devtest (rev 1147e2dc)

## Hypotheses

| | metric | a | b | a | b | Δ (a−b) | 95% CI of Δ | threshold | verdict |
|---|---|---|---|---|---|---|---|---|---|
| H1 | chrf++ | `direct.nllb.eng` | `direct.nllb.arb` | 50.76 | 43.38 | +7.39 | [+6.68, +8.09] | ≥ 3.0 | **PASS** |
| H1 | bleu | `direct.nllb.eng` | `direct.nllb.arb` | 32.33 | 22.93 | +9.40 | [+8.04, +10.50] | — | (secondary) |
| H2 | chrf++ | `pivot.B1` | `direct.nllb.arb` | 48.67 | 43.38 | +5.29 | [+4.71, +5.89] | ≥ 2.0 | **PASS** |
| H2 | bleu | `pivot.B1` | `direct.nllb.arb` | 29.47 | 22.93 | +6.54 | [+5.42, +7.73] | — | (secondary) |

Verdict rule: PASS if the CI lower bound ≥ threshold; INCONCLUSIVE if the point estimate ≥ threshold but the CI lower bound is below it; FAIL if the point estimate < threshold.

## Systems

| system | route | source | target | chrF++ [95% CI] | BLEU [95% CI] |
|---|---|---|---|---|---|
| `direct.nllb.tam2arb` | tam_Taml->arb_Arab | tam | arb_Arab | 37.92 [37.18, 38.66] | 21.46 [20.50, 22.27] |
| `lora.tam2arb` | tam_Taml->arb_Arab (LoRA) | tam | arb_Arab | 38.97 [38.23, 39.70] | 22.35 [21.51, 23.18] |
| `control.B1_allnllb` | arb_Arab -> eng_Latn (nllb) -> tam_Taml (nllb) | arb | tam_Taml | 44.80 [43.99, 45.52] | 25.06 [24.08, 25.98] |
| `control.it2_humaneng` | eng_Latn -> tam_Taml (it2_en_indic) | eng | tam_Taml | 56.73 [55.79, 57.61] | 39.36 [38.11, 40.57] |
| `direct.m2m100.arb` | arb_Arab->tam_Taml | arb | tam_Taml | 11.53 [11.02, 11.99] | 1.23 [0.99, 1.49] |
| `direct.m2m100.arb_nodiac` | arb_Arab->tam_Taml | arb_nodiac | tam_Taml | 11.91 [11.37, 12.42] | 1.38 [1.11, 1.65] |
| `direct.m2m100.eng` | eng_Latn->tam_Taml | eng | tam_Taml | 16.98 [16.11, 17.86] | 4.04 [3.58, 4.52] |
| `direct.nllb.arb` | arb_Arab->tam_Taml | arb | tam_Taml | 43.38 [42.57, 44.13] | 22.93 [21.68, 24.03] |
| `direct.nllb.arb_nodiac` | arb_Arab->tam_Taml | arb_nodiac | tam_Taml | 43.49 [42.72, 44.21] | 23.32 [22.12, 24.39] |
| `direct.nllb.eng` | eng_Latn->tam_Taml | eng | tam_Taml | 50.76 [49.87, 51.62] | 32.33 [30.92, 33.31] |
| `lora.arb` | arb_Arab->tam_Taml (LoRA) | arb | tam_Taml | 45.74 [45.01, 46.40] | 25.31 [24.31, 26.18] |
| `pivot.B1` | arb_Arab -> eng_Latn (nllb) -> tam_Taml (it2_en_indic) | arb | tam_Taml | 48.67 [47.88, 49.39] | 29.47 [28.44, 30.42] |
| `pivot.B2` | arb_Arab -> mal_Mlym (nllb) -> tam_Taml (it2_indic_indic) | arb | tam_Taml | 41.47 [40.70, 42.17] | 22.16 [21.04, 22.93] |

## Paired comparisons

| a | b | metric | a | b | Δ | 95% CI | p |
|---|---|---|---|---|---|---|---|
| `pivot.B1` | `pivot.B2` | chrf++ | 48.67 | 41.47 | +7.19 | [+6.63, +7.73] | 0.000 |
| `pivot.B1` | `pivot.B2` | bleu | 29.47 | 22.16 | +7.31 | [+6.64, +8.23] | 0.000 |
| `pivot.B1` | `control.B1_allnllb` | chrf++ | 48.67 | 44.80 | +3.86 | [+3.39, +4.35] | 0.000 |
| `pivot.B1` | `control.B1_allnllb` | bleu | 29.47 | 25.06 | +4.41 | [+3.85, +5.00] | 0.000 |
| `control.B1_allnllb` | `direct.nllb.arb` | chrf++ | 44.80 | 43.38 | +1.42 | [+0.87, +2.01] | 0.000 |
| `control.B1_allnllb` | `direct.nllb.arb` | bleu | 25.06 | 22.93 | +2.13 | [+1.10, +3.28] | 0.000 |
| `control.it2_humaneng` | `pivot.B1` | chrf++ | 56.73 | 48.67 | +8.06 | [+7.51, +8.60] | 0.000 |
| `control.it2_humaneng` | `pivot.B1` | bleu | 39.36 | 29.47 | +9.89 | [+9.18, +10.66] | 0.000 |
| `direct.nllb.arb_nodiac` | `direct.nllb.arb` | chrf++ | 43.49 | 43.38 | +0.11 | [-0.21, +0.43] | 0.474 |
| `direct.nllb.arb_nodiac` | `direct.nllb.arb` | bleu | 23.32 | 22.93 | +0.39 | [-0.46, +1.30] | 0.420 |
| `direct.m2m100.arb` | `direct.nllb.arb` | chrf++ | 11.53 | 43.38 | -31.85 | [-32.72, -30.96] | 0.000 |
| `direct.m2m100.arb` | `direct.nllb.arb` | bleu | 1.23 | 22.93 | -21.69 | [-22.81, -20.39] | 0.000 |
| `lora.arb` | `direct.nllb.arb` | chrf++ | 45.74 | 43.38 | +2.36 | [+1.88, +2.86] | 0.000 |
| `lora.arb` | `direct.nllb.arb` | bleu | 25.31 | 22.93 | +2.38 | [+1.40, +3.36] | 0.000 |
| `lora.arb` | `pivot.B1` | chrf++ | 45.74 | 48.67 | -2.93 | [-3.39, -2.45] | 0.000 |
| `lora.arb` | `pivot.B1` | bleu | 25.31 | 29.47 | -4.16 | [-4.82, -3.54] | 0.000 |
| `lora.tam2arb` | `direct.nllb.tam2arb` | chrf++ | 38.97 | 37.92 | +1.05 | [+0.60, +1.48] | 0.000 |
| `lora.tam2arb` | `direct.nllb.tam2arb` | bleu | 22.35 | 21.46 | +0.89 | [+0.45, +1.68] | 0.000 |

## Diacritic stripping (source b)

566/1012 Arabic sentences changed (55.93%), 1326 characters removed (1.1401% of characters).

Step-3 route: **B1** (chrF++ on dev: B1 49.15, B2 42.22).

## sacrebleu signatures

- chrf++: `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`
- bleu: `nrefs:1|case:mixed|eff:no|tok:flores200|smooth:exp|version:2.6.0`

Bootstrap: 1000 paired resamples, seed 1234, percentile intervals.
