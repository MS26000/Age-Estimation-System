### A. All approaches, identical photos

Test photos used: **120**

| Approach | Bracket accuracy (95% CI) | Within 1 bracket | Macro-F1 | MAE years (95% CI) | MAE n |
|---|---|---|---|---|---|
| cv_probe | 53.3% (44.2%-62.5%) | 85.8% | 0.534 | 6.35 (5.06-7.74) | 108 |
| cv_zeroshot | 42.5% (34.2%-50.8%) | 87.5% | 0.417 | 7.11 (5.76-8.56) | 108 |
| llm_gemini | 64.2% (55.8%-72.5%) | 98.3% | 0.640 | 4.35 (3.30-5.52) | 108 |

**Confidence quality** (is a confidence of 0.8 right about 80% of the time?)

| Approach | Mean confidence | Accuracy | ECE (lower is better) | Acc. top-half confidence | Acc. bottom-half |
|---|---|---|---|---|---|
| cv_probe | 0.64 | 0.53 | 0.156 | 63.3% | 43.3% |
| cv_zeroshot | 0.50 | 0.42 | 0.104 | 58.3% | 26.7% |
| llm_gemini | 0.86 | 0.64 | 0.223 | 64.7% | 0.0% |

**Per-bracket recall**

| Approach | Under 18 | 18-24 | 25-34 | 35-44 | 45-54 | 55+ |
|---|---|---|---|---|---|---|
| cv_probe | 75.0% | 45.0% | 30.0% | 50.0% | 50.0% | 70.0% |
| cv_zeroshot | 55.0% | 15.0% | 75.0% | 40.0% | 15.0% | 55.0% |
| llm_gemini | 55.0% | 45.0% | 70.0% | 90.0% | 45.0% | 80.0% |

**MAE by true bracket (years)**

| Approach | Under 18 | 18-24 | 25-34 | 35-44 | 45-54 | 55+ |
|---|---|---|---|---|---|---|
| cv_probe | 2.63 | 5.84 | 6.10 | 5.04 | 6.17 | 12.28 |
| cv_zeroshot | 4.96 | 3.87 | 3.93 | 6.26 | 7.51 | 16.20 |
| llm_gemini | 2.70 | 3.23 | 3.80 | 3.29 | 5.37 | 7.84 |

**Accuracy by type of photo date** (camera timestamp = the date has a time of day)

| Approach | Camera timestamp | n | Date typed by uploader | n |
|---|---|---|---|---|
| cv_probe | 58.7% | 46 | 50.0% | 74 |
| cv_zeroshot | 45.7% | 46 | 40.5% | 74 |
| llm_gemini | 65.2% | 46 | 63.5% | 74 |

**Paired comparison** (photos where only one approach was right; exact binomial test)

| A | B | only A right | only B right | p-value |
|---|---|---|---|---|
| cv_probe | cv_zeroshot | 29 | 16 | 0.072 |
| cv_probe | llm_gemini | 16 | 29 | 0.072 |
| cv_zeroshot | llm_gemini | 8 | 34 | 0.000 |

### B. CV approaches on the full test split

Test photos used: **270**

| Approach | Bracket accuracy (95% CI) | Within 1 bracket | Macro-F1 | MAE years (95% CI) | MAE n |
|---|---|---|---|---|---|
| cv_probe | 54.1% (47.8%-60.4%) | 85.9% | 0.540 | 6.22 (5.45-7.03) | 247 |
| cv_zeroshot | 42.6% (36.7%-48.1%) | 86.3% | 0.422 | 7.43 (6.60-8.33) | 247 |

**Confidence quality** (is a confidence of 0.8 right about 80% of the time?)

| Approach | Mean confidence | Accuracy | ECE (lower is better) | Acc. top-half confidence | Acc. bottom-half |
|---|---|---|---|---|---|
| cv_probe | 0.62 | 0.54 | 0.075 | 67.4% | 40.7% |
| cv_zeroshot | 0.49 | 0.43 | 0.114 | 54.1% | 31.1% |

**Per-bracket recall**

| Approach | Under 18 | 18-24 | 25-34 | 35-44 | 45-54 | 55+ |
|---|---|---|---|---|---|---|
| cv_probe | 71.1% | 46.7% | 35.6% | 42.2% | 51.1% | 77.8% |
| cv_zeroshot | 51.1% | 15.6% | 75.6% | 35.6% | 20.0% | 57.8% |

**MAE by true bracket (years)**

| Approach | Under 18 | 18-24 | 25-34 | 35-44 | 45-54 | 55+ |
|---|---|---|---|---|---|---|
| cv_probe | 3.76 | 5.66 | 6.33 | 7.46 | 5.70 | 8.61 |
| cv_zeroshot | 5.47 | 3.94 | 4.33 | 8.01 | 9.47 | 13.85 |

**Accuracy by type of photo date** (camera timestamp = the date has a time of day)

| Approach | Camera timestamp | n | Date typed by uploader | n |
|---|---|---|---|---|
| cv_probe | 55.7% | 122 | 52.7% | 148 |
| cv_zeroshot | 45.1% | 122 | 40.5% | 148 |
