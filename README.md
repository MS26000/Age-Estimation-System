---
title: Age at the time of the photo
emoji: 📸
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Age at the time of the photo (Echooo AI take-home)

Give the API a Wikipedia page of a person. It finds a usable photo on the page, and returns how old the
person was when the photo was taken (age bracket, plus an estimated exact age).

- **Live API:** `<<FILL: https://<user>-<space>.hf.space>>`  (docs: `/docs`)
- **Candidate:** Muhammad Shahzad, candidate number 0167

```bash
curl -X POST https://<<FILL>>.hf.space/predict -H "Content-Type: application/json" \
     -d '{"url": "https://en.wikipedia.org/wiki/Grace_VanderWaal"}'
```
```json
{"person": "...", "status": "ok", "image_used": "...", "photo_date": "2018-07-22",
 "predicted_age_bracket": "Under 18", "predicted_age": 15, "confidence": 0.71, "approach": "cv"}
```
If the page has no usable photo the API answers `"status": "no_usable_photo"` with the reason, and never guesses.
Optional request field `"approach": "cv" | "llm"` (the `llm` option exists only if an API key is configured on the server).

## 1. Approach in one paragraph
`<<FILL after step 6: 4-5 sentences. Which approach won, by how much, and the most likely reasons.>>`

## 2. Data (collected only from Wikipedia / Wikidata / Wikimedia Commons)
No ready-made or labelled face-age dataset was used for training or testing.

| Step | Count | Note |
|---|---|---|
| Candidate people (Wikidata: human, English Wikipedia article, image, day-precision birth date; 8 occupations x 7 birth decades) | 5,182 | raw query rows 12,279 -> 11,182 after de-duplication -> sampled to about 5,000, balanced by birth decade |
| With a usable labelled photo (Wikipedia lead image, on Commons, photo date known, certain bracket) | 4,613 | dropped: 159 unparseable dates, 146 no lead image, 138 too small, 81 no photo date, 20 ambiguous bracket, 13 bad file type, 8 not on Commons, 4 implausible/impossible dates |
| After face filter (one clear main face) and de-duplication | `<<FILL from step 3>>` | `<<FILL funnel: duplicate_image, no_face, multiple_faces, face_too_small, ...>>` |
| Train / validation / test | `<<FILL>>` | test set is balanced across brackets; split by person (one photo per person) |

**How the true age bracket was computed.** True age = completed years between the person's birth date (Wikidata,
day precision) and the photo date (Commons `DateTimeOriginal`: camera EXIF or the date typed by the uploader).
Dates with day, month or year precision are accepted. For month/year dates we compute the youngest and oldest
possible age and keep the photo only if both fall in the same bracket. Vague dates (circa, before, decades,
ranges) and dates that the text says are upload, scan or publication dates are dropped. Exact age (for MAE) is
used only for day-precision dates.

## 3. Models
| | Model | Notes |
|---|---|---|
| Face detection | YuNet (OpenCV Zoo) | one clear main face required: score >= 0.8, at least 2x as wide as any other face, >= 60 px |
| CV zero-shot | `nateraw/vit-age-classifier` (ViT-base) | **pretrained on the FairFace dataset by its author**; allowed as a pretrained model, but note it for transparency. Its 9 age bins are mapped to our 6 brackets by interval overlap |
| CV + linear probe | same ViT, frozen, plus logistic regression (bracket) and ridge regression (age) trained on our train split | not full fine-tuning; weights stored as numpy in `models/cv_probe.npz` |
| LLM | `<<FILL: Qwen2.5-VL-3B-Instruct (local) and/or Gemini model name>>` | prompted for JSON (age, bracket, confidence); sees only the face crop: no name, no dates, no page text |

## 4. Results (test set, same photos for every approach)
`<<PASTE results/results.md>>`

Confusion matrices: `results/confusion_*.png`. Largest errors: `results/worst_errors_*.csv`.

## 5. Why one approach works better
`<<FILL: use the numbers: per-bracket recall, MAE by bracket, calibration (ECE), paired test.>>`

## 6. What didn't work / shortcuts / limitations (honest list)
- Image download: fetching all ~4,400 thumbnails was far too slow because Wikimedia rate-limits thumbnails at non-standard widths (I had asked for 800 px; standard widths are 20/40/60/120/250/330/500/960/1280/1920/3840) and my first downloader retried silently. Final version (stage 3 of the notebook) downloads a bracket-balanced subset (up to 260 photos per bracket, `<<FILL: n>>` in total), uses original files or 960 px thumbnails and resizes locally. Because of this, the train split is balanced by bracket, not natural.
- Rate limits: the first Wikipedia/Commons run lost 1,951 candidates to HTTP 429; fixed with longer back-off and slower requests.
- Wikidata timeouts: one query cell (actors born in the 1970s) failed repeatedly and was split into 5-year queries.
- Date parsing: the first parser rejected 328 clean dates (formats like "Taken on 6 April 2006"); a second pass recovered 169.
  A later review showed the first parser also accepted dates whose text said "upload"/"scan"; those are now rejected. `<<FILL: how many rows this removed>>`
- Label noise: Commons dates are not guaranteed to be capture dates. Manual check of 30 random photos: `<<FILL: x/30 wrong>>`.
  Accuracy is reported separately for camera-timestamp dates and uploader-typed dates.
- Sampling bias: people are notable (actors, athletes, politicians, ...) with freely licensed photos; sampling inside each
  occupation/decade cell is not random (random ordering times out on Wikidata). We did not measure demographic balance.
- "Under 18" photos are rare, so that bracket has the smallest sample. `<<FILL: numbers>>`
- The pretrained CV model was trained on FairFace (different image style: no stage lighting, mostly frontal portraits).
- The LLM free tier: `<<FILL: what happened with quotas / model availability>>`
- API: photos with several similar-sized faces are refused rather than guessed. If the lead image is unusable the API tries
  other images on the page whose file name contains the person's name (`image_source` field says which was used).
- Free hosting sleeps after inactivity; the first request after a sleep can take about a minute.

## 7. Reproduce
Everything (data collection, labelling, face filter, both approaches, evaluation, API files) is in the notebook
`echooo_age_estimation.ipynb`. Run it top to bottom on a CPU machine; finished stages are skipped when their output files exist.
API locally: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && uvicorn app:app --port 7860`.
Tests (no network or GPU needed): `pytest -q tests/test_offline.py`.

## 8. Secrets
No API keys are stored in the repo. The optional LLM approach reads `GEMINI_API_KEY` (and `GEMINI_MODEL`) from the server
environment (Hugging Face Space secrets).
