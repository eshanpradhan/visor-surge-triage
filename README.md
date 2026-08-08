# VISOR — chest X-ray severity prediction (COVID-19-NY-SBU)

Predicting critical-care escalation from a single admission chest radiograph plus
clinical data at presentation.

**Label.** `severe = is_icu OR was_ventilated` — an escalation/urgency outcome.
Death is deliberately excluded: 69 of 183 deceased patients were never admitted
to the ICU and never ventilated, consistent with comfort-care goals-of-care
decisions rather than missed escalation. `deceased` is carried as a secondary
reporting metric only.

**Cohort.** 1384 labelled patients, 1365 with imaging, one earliest chest film
each. 20.0% severe. Patient-level 70/15/15 split, stratified, no patient in more
than one split.

---

## Pipeline, in execution order

Each script is runnable standalone and writes its output to disk. Later stages
read those files rather than recomputing.

Per-patient derived files (labels, cohort, manifest) are gitignored per data-use
terms and regenerate by running the pipeline in the order listed above. The same
applies to the saved embedding and prediction arrays (`*.npz`), which carry
patient identifiers and outcome labels alongside the features. Committed outputs
are aggregate only: column-level specifications, fitted summary statistics, and
fold- or model-level metrics.

### Data preparation

| # | Script | What it does |
|---|--------|--------------|
| 1 | `build_labels.py` | Derives `severe` from `is_icu`/`was_ventilated`; carries `deceased` as a secondary metric. → `visor_labels.csv` |
| 2 | `build_cohort.py` | Joins labels to the CXR archive via `metadata.csv`; asserts metadata patient IDs match on-disk directories. → `visor_cohort.csv` |
| 3 | `build_manifest.py` | Selects one earliest raw (non-enhanced) film per patient. Removes the image-count confound: severe patients carry ~12× more films (mean 37.3 vs 3.2). → `visor_manifest.csv` |
| 4 | `build_split.py` | Stratified 70/15/15 patient-level split, `random_state=42`; asserts no patient crosses splits. → `visor_manifest_split.csv` |
| 5 | `compute_norm_stats.py` | Pixel mean/std from **train images only**. → `norm_stats.json` |

### Feature governance

| # | Script | What it does |
|---|--------|--------------|
| 6 | `build_feature_spec.py` | Classifies all 131 clinical columns as SAFE / LEAK / LEAK_RISK / DROP with a written reason each; asserts full coverage. → `feature_spec.csv` |
| 7 | `features.py` | Builds the feature frame under that contract. Raises `LeakageError` on any excluded **or unclassified** column. `drop_redundant=True` removes 50 threshold flags duplicating a numeric column. 104 SAFE → **54 features** |
| 8 | `impute.py` | Train-only medians (numeric) and modes (categorical). Asserts no missingness-indicator column is ever created. → `impute_stats.json` |
| 9 | `clinical_encoding.py` | `ClinicalEncoder`: log1p for heavy-tailed labs, ordinal/one-hot encoding, train-fit standardization. **Imports no torch** — see "OpenMP" below. 54 → **76 encoded dims** |
| 10 | `dataset.py` | PyTorch `Dataset`/`DataLoader`. Aspect-preserving pad → 224×224, L→3ch, train-only augmentation (±7° rotation, 0.10 brightness/contrast jitter) |

### Models

| # | Script | What it does |
|---|--------|--------------|
| 11 | `baseline.py` | Clinical-only logistic regression + LightGBM, 5-fold CV with preprocessing refit per fold |
| 12 | `image_baseline.py` | Frozen ImageNet ResNet-50 embeddings + logistic/MLP heads; also A/B tests dataset vs ImageNet normalization |
| 13 | `finetune.py` | Unfreezes `layer4` only. All BatchNorm held in eval mode. Early stopping on an inner slice, never the fold holdout. `--multiseed` runs seeds 42/7/2024 |
| 14 | `fusion_model.py` | Image branch (2048→64) ⊕ clinical MLP (76→64→32) → joint head. Modality dropout p=0.2. Supports `drop_columns` and `use_image=False` |
| 15 | `run_ablations.py` | Calendar-free fusion vs clinical-only MLP, matched architecture |
| 16 | `multiseed_ablations.py` | The above at 3 seeds, paired by seed. → `multiseed_ablations.csv` |

### Final evaluation — test set, single pass

| # | Script | What it does |
|---|--------|--------------|
| 17 | `final_test_clinical.py` | Stage 1: LightGBM models. Asserts torch is **not** imported. → `test_preds_clinical.npz` |
| 18 | `final_test_torch.py` | Stage 2: image-only and both fusion variants. Asserts lightgbm is **not** imported. → `test_preds_torch.npz` |
| 19 | `final_test_report.py` | Stage 3: merges predictions, computes the final table and calibration. → `final_test_results.csv`, `calibration_test.png` |
| 20 | `calibrate.py` | Post-hoc Platt/isotonic scaling, fitted on val predictions from a train-only model. → `calibration_fusion_before_after.png` |
| 21 | `save_final_weights.py` | Trains the final models once and writes parameters to `models/` so inference runs without retraining or redownloading the image archive. → `models/` |

### Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard: pick a patient, see the radiograph, the calibrated fusion risk and
tier, a Grad-CAM overlay, clinical attributions for the fusion model, and the
clinical-only model's score for comparison. Inference runs locally and no patient
data leaves the machine.

One network call, first boot only: the checkpoint holds just the trained tensors,
so torchvision fetches the frozen ImageNet ResNet-50 (~98 MB) if the torch cache
is cold. The app surfaces this as its own progress state rather than letting a
fresh deployment sit on a generic spinner for a minute. Cached thereafter.

**Two modes, selected automatically.** With the gitignored cohort files present
the app scores five real held-out test patients. Without them — a fresh clone, or
a deployment — it falls back to **demo mode**: three synthetic cases from
`demo_data.py` paired with public NIH ChestX-ray14 sample images in
`demo_assets/`, behind a disclaimer banner. Force it locally with
`VISOR_DEMO_MODE=1`.

Demo-mode scores are not clinically meaningful. The model was trained on SBU
radiographs; an NIH image with invented labs is out of distribution on both
inputs. Demo mode shows that the pipeline runs, nothing more — all reported
performance comes from the held-out SBU test split.

Demo mode needs no training data: the encoder is restored from
`models/encoder_state.json` and unspecified variables are filled from
`models/impute_stats_final.json`.

`clinical_service.py` exists because LightGBM cannot be called from a process
that has imported torch (see the environment note below); the app sends it an
encoded feature matrix and gets scores back, which keeps it working in both modes.

### Saved models (`models/`, committed)

Fitted parameters and aggregate preprocessing statistics only — no patient rows.
See `models/MODEL_CARD.json` for provenance of each file.

| file | what |
|---|---|
| `fusion_no_calendar.pt` | 68 trained tensors, **fp16** (30.3 MB), cast to fp32 on load. ResNet-50 below `layer4` is frozen at ImageNet init and **not** saved — torchvision fetches it |
| `calibrator_isotonic.json` | Isotonic calibrator; applies **only** to the checkpoint above |
| `clinical_lightgbm_{naive,calendar_ablated}.txt` | LightGBM boosters |
| `encoder_state.json` | Fitted `ClinicalEncoder`: vocabularies, log1p selection, standardization |
| `impute_stats_final.json` | Train medians/modes, calendar column dropped |

**Half precision.** The checkpoint is stored fp16 to halve its size for cloning
and deployment; the forward pass still runs in fp32. Verified: all three demo
cases produce identical probabilities and identical tiers, and val/test AUPRC and
Brier are unchanged to 3 dp. One metric moves — val AUROC 0.796 → 0.795 — through
a single tie flip, with no patient's predicted probability shifting by more than
1e-4. `verify_fp16.py` records that comparison.

**These are fit on the train split (n=955), not train+val (n=1160).** That is
deliberate: the isotonic calibrator must be fitted on predictions the model made
out-of-sample, and val is the only split available for that. Saving the
train+val model would mean shipping a calibrator fitted against a different
model's outputs. The trade is slightly lower discrimination than the test table
in exchange for trustworthy probabilities.

---

## Results

Cross-validated on train+val (n=1160), patient-disjoint folds, preprocessing
refit inside every fold.

| model | CV AUROC | test AUROC | test AUPRC |
|---|---|---|---|
| clinical LightGBM, naive | 0.843 | 0.835 | 0.565 |
| clinical LightGBM, calendar-ablated | 0.804 | 0.813 | 0.578 |
| image-only, `layer4` fine-tuned (3 seeds) | 0.756 ± 0.003 | 0.802 | 0.522 |
| fusion, with calendar | 0.826 ± 0.011 | 0.874 | 0.677 |
| **fusion, no calendar (3 seeds)** | **0.793 ± 0.009** | **0.834** | **0.658** |
| clinical MLP, no calendar (3 seeds) | 0.767 ± 0.001 | — | — |

**Primary result:** fusion beats a matched clinical-only model by **+0.026 ± 0.007
AUROC** and **+0.026 ± 0.004 AUPRC**, positive in 3/3 seeds (paired t-test over 3
seeds, p=0.025 — directional consistency carries this claim, not the p-value).

Four of five models scored above their CV estimate on test, so this particular
205-patient split appears mildly easy. Quote the CV figure; test confirms the
ranking rather than sharpening it (±0.06–0.08 CI at 41 positives).

**Secondary:** fusion roughly halves within-seed fold variance versus the
clinical MLP (0.023 vs 0.045) — the image branch stabilizes folds where clinical
features alone do poorly.

---

## Confounds found and handled

Each was tested rather than assumed; two hypotheses were checked and rejected.

| finding | evidence | handling |
|---|---|---|
| **Informative missingness** | Among ~208 patients missing a basic metabolic panel, **zero** are severe (Fisher OR 0.00, p≈2.5e-22) — the panel was ordered on everyone sick enough to escalate | No missingness indicators anywhere; enforced by assertion in `impute.py` |
| **Image-count confound** | Severe patients average 37.3 films vs 3.2 | One earliest film per patient |
| **Calendar wave** | Severe prevalence 41.6% (month 1) vs 11.9% (month 2), χ²=146.5, p<1e-30. Worth 4–5 AUROC points | Reported both with and without; primary result is calendar-free |
| **Enhanced duplicates** | 6700 contrast-processed copies of the same acquisitions | Raw arm only |
| **Post-imputation constants** | 6 columns >75% missing pre-imputation | Dropped, with the `blood_pH` threshold flags following their source |
| ~~Study type (PORT vs VIEWONLY)~~ | **Rejected**: 22.6% vs 18.4%, χ² p=0.176 | No action needed |
| ~~Image geometry~~ | **Rejected**: height vs severe, Mann-Whitney p=0.360 | No action needed |

**Calendar independence of the image branch** was verified directly: a linear
probe recovering admission month from embeddings reads 0.624 (frozen) and 0.636
(fine-tuned) — fine-tuning improved severity AUROC by +0.063 without increasing
the calendar signal, so it learned pathology rather than acquisition drift.

---

## Known limitations

1. **Calibration — addressed.** Raw outputs are badly overconfident: fusion's top
   test bin predicted 0.832 against an observed 0.561. `class_weight='balanced'` /
   `pos_weight` optimize ranking at the cost of probability scale. Isotonic
   regression fitted on val predictions (from a train-only model, so val is
   genuinely held out) cuts test Brier from **0.214 to 0.111**, with the worst bin
   gap falling from 0.403 to 0.041. Report calibrated probabilities; the raw
   sigmoid outputs are ranking scores, not probabilities.
2. **Mode imputation.** ~20% of comorbidity flags were null and filled with the
   train mode rather than an "Unknown" level, because an "Unknown" level is a
   missingness indicator and missingness leaks severity here. May understate true
   comorbidity prevalence.
3. **Small test set.** 205 patients, 41 positives. ±0.06–0.08 on AUROC.
4. **`antibiotics_use_v` / `nsaid_use_v`** dropped conservatively — the `_v`
   suffix suggests baseline use but the data dictionary was not available to
   confirm. Recoverable if it says pre-admission.
5. **Troponin** log1p is near-identity on its 0.01–1.8 range, so it remains at
   16.9σ. Matters only for the linear baseline; tree models are scale-invariant.
6. **Single-architecture comparison.** LightGBM (0.804) beats the fusion model's
   MLP clinical branch (0.767) on identical features, so fusion-vs-LightGBM mixes
   modality and architecture effects. The matched MLP comparison is the clean one.

---

## Environment note: LightGBM and PyTorch cannot share a process

Both bundle their own OpenMP runtime. On macOS the combination fails two ways:
with torch imported first a multithreaded `lgb.train` **segfaults** (exit 139, no
traceback); with LightGBM imported first it survives one call then **deadlocks**
at 0% CPU. Hence `clinical_encoding.py` is torch-free, `baseline.py` imports from
it, and the final evaluation runs as three separate processes, each asserting the
other library is absent.

Run background jobs with `python3 -u` — buffered stdout is discarded on segfault,
which makes the crash look like silence.
