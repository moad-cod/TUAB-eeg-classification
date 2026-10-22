# TUAB EEG Feature Extraction Protocol — Implemented Version 1.0

**Research:** Reliability of Spatial Explanations in Deep Learning-Based EEG Abnormality Detection  
**Primary dataset:** TUAB v3.0.1  
**Implementation:** `notebooks/03-TUAB-EEG-Feature-Extraction.ipynb`  
**Preprocessing source:** `notebooks/02-tuab-eeg-preprocessing-and-signal-harmonizati.ipynb`

## 1. Purpose and scope

Notebook 03 creates an interpretable recording-level feature table for
normal-versus-abnormal EEG classification. The features serve two purposes:

1. Establish a traditional machine-learning baseline.
2. Provide channel- and region-level EEG characteristics that can contextualize
   later CNN spatial explanations.

Notebook 03 performs feature extraction, aggregation, integrity checks, and
export. It does not fit scalers, imputers, feature selectors, or classification
models. Those operations belong in Notebook 04 and must be learned from the
development-training partition only.

This document describes the feature extractor that is currently implemented.
Potential improvements and sensitivity experiments are identified separately
and must not be confused with the Version 1.0 output.

## 2. Validated input dataset

Notebook 03 consumes the completed output of Notebook 02.

| Property | Implemented value |
|---|---:|
| Recordings | 2,993 |
| Total epochs | 512,137 |
| Channels | 19 |
| Samples per epoch | 1,600 |
| Epoch duration | 8 seconds |
| Sampling frequency | 200 Hz |
| Signal dtype | `float32` |
| Signal units | Microvolts |
| Stored signal | Original TUAB reference representation |
| `NO_SCREENING_FLAGS` epochs | 457,577 |
| `SUSPECT` epochs | 54,560 |

Each recording is stored in a compressed NPZ file with:

```text
original_reference: (n_epochs, 19, 1600), float32, µV
sfreq:              scalar, 200
channels:           (19,), fixed channel order
```

The fixed channel order is:

```text
Fp1, Fp2, F3, F4, F7, F8, Fz,
C3, C4, Cz,
T3, T4, T5, T6,
P3, P4, Pz,
O1, O2
```

The metadata integration uses:

```text
tuab_preprocessing/metadata/tuab_patient_grouped_splits.csv
tuab_preprocessing/full_tuab/processing_manifest.csv
```

The tables are joined one-to-one using the original recording filepath.

## 3. Patient-independent partitions

Notebook 03 preserves the partitions created in Notebook 02:

| Split | Normal | Abnormal | Total recordings |
|---|---:|---:|---:|
| Train | 1,097 | 1,076 | 2,173 |
| Validation | 274 | 270 | 544 |
| Evaluation | 150 | 126 | 276 |

The notebook verifies that patient sets are disjoint between train,
validation, and evaluation. No feature operation changes these assignments.

Labels belong to complete recordings, not individual epochs. Notebook 03
therefore produces exactly one feature row per recording.

## 4. Input validation

Before full extraction, Notebook 03 validates a six-recording pilot containing
three normal and three abnormal training recordings.

For every pilot NPZ file it verifies:

- Required arrays are present.
- The signal has three dimensions.
- Every epoch has shape `(19, 1600)`.
- Signal dtype is `float32`.
- Sampling frequency is 200 Hz.
- Channel names and ordering match the expected montage.
- No NaN or infinite signal samples are present.
- The NPZ epoch count matches the preprocessing manifest.

The pilot validation completed successfully.

## 5. Reference transformation

The NPZ files contain the original TUAB signal representation. For the primary
feature table, Notebook 03 recalculates a 19-channel common average reference
for every epoch and time sample:

\[
x^{CAR}_{e,c,t}
=x_{e,c,t}-\frac{1}{19}\sum_{j=1}^{19}x_{e,j,t}
\]

This is a comparison between the original TUAB reference configuration and a
newly calculated 19-channel CAR, not between referenced and unreferenced EEG.

The implementation uses `float32` input and output. Pilot checks verify:

- Output shape and dtype are unchanged.
- All CAR values are finite.
- The maximum across-channel mean is below `1e-4 µV`.
- Pairwise channel differences are preserved within `1e-3 µV`.

The exported Version 1.0 feature table contains only dynamically recomputed CAR
features. Original-reference features are reserved for a future sensitivity
analysis.

## 6. Welch power spectral density

PSD is calculated independently for every epoch and channel using
`scipy.signal.welch`.

| Parameter | Implemented value |
|---|---:|
| Sampling frequency | 200 Hz |
| Window | Hann |
| Segment length | 400 samples / 2 seconds |
| Overlap | 200 samples / 50% |
| Detrending | Constant |
| Scaling | Density |
| Frequency-bin spacing | 0.5 Hz |
| Output unit | µV²/Hz |

The pilot PSD has shape `(n_epochs, 19, 201)` and covers 0–100 Hz. Feature
calculations retain the 0.5–40 Hz range.

The 0.5 Hz bin lies at the preprocessing high-pass boundary, and the 30–40 Hz
band lies near the low-pass transition. These effects must be considered when
interpreting delta and low-gamma features. Excluding low gamma is a planned
sensitivity analysis, not part of the Version 1.0 extraction.

## 7. Spectral band-power features

PSD is integrated using trapezoidal integration. The implementation
interpolates the PSD at exact band boundaries before integration.

| Band | Range |
|---|---:|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Low gamma | 30–40 Hz |

For every band, channel, and epoch, Notebook 03 calculates:

- Linear absolute band power in µV².
- Relative band power as band power divided by total 0.5–40 Hz power.

This creates 10 band-power feature families:

```text
delta_absolute_power       delta_relative_power
theta_absolute_power       theta_relative_power
alpha_absolute_power       alpha_relative_power
beta_absolute_power        beta_relative_power
low_gamma_absolute_power   low_gamma_relative_power
```

Absolute power is stored on the linear scale in Version 1.0. A log-power
representation may be derived in Notebook 04 within the leakage-safe training
pipeline or evaluated through a separately versioned extraction.

Relative-power division currently uses a lower bound of `1e-12 µV²` for total
power. Pilot and final recording-level outputs contained no non-finite values.

The five band intervals partition 0.5–40 Hz, and the pilot verifies that the
sum of integrated band powers matches total integrated power within `1e-5`
relative error.

## 8. Spectral descriptors and ratios

Notebook 03 calculates seven additional spectral feature families.

### 8.1 Spectral descriptors

- Normalized spectral entropy over 0.5–40 Hz.
- Spectral edge frequency containing 90% of cumulative power.
- Spectral edge frequency containing 95% of cumulative power.

Entropy is normalized by the logarithm of the number of retained frequency
bins. Spectral edge frequencies use trapezoidal cumulative integration and
linear interpolation within the selected frequency interval.

### 8.2 Power ratios

- Delta/alpha.
- Theta/alpha.
- Theta/beta.
- Slow/fast: `(delta + theta) / (alpha + beta)`.

Ratios are calculated from linear absolute band powers. A denominator must be
greater than `1e-12 µV²`; otherwise the epoch-channel ratio is represented as
missing. The implementation never substitutes zero for an undefined ratio.

These ratios are included in the Version 1.0 combined table. Because fixed
floors and power ratios can produce unstable extremes at very low denominator
power, Notebook 04 must compare models with and without ratio columns.

## 9. Temporal features

Notebook 03 calculates seven temporal feature families per epoch and channel:

1. Root mean square amplitude.
2. Interquartile range.
3. Median absolute deviation.
4. Mean absolute successive difference, named `line_length` in the table.
5. Peak-to-peak amplitude.
6. Hjorth mobility.
7. Hjorth complexity.

All amplitude features operate on microvolt signals. No per-epoch
standardization is performed.

For Hjorth features, variance values at or below `1e-12` are treated as
degenerate. Undefined mobility or complexity values are stored as missing at
epoch level. The final recording aggregates contained no missing values.

The implemented `line_length` is the mean absolute successive difference:

\[
LL_{mean}=\frac{1}{N-1}\sum_{t=1}^{N-1}|x_{t+1}-x_t|
\]

This definition must be used consistently in subsequent notebooks.

## 10. Implemented feature dimensionality

Notebook 03 contains 24 epoch-level feature families:

| Family | Count |
|---|---:|
| Absolute and relative band powers | 10 |
| Spectral entropy and edge frequencies | 3 |
| Power ratios | 4 |
| Temporal and Hjorth features | 7 |
| **Total** | **24** |

The current extraction intentionally retains all implemented families in one
combined table. Notebook 04 must define model experiments by selecting columns
from this frozen table rather than changing evaluation data after inspecting
results.

## 11. Recording-level channel aggregation

Every epoch-level channel feature is aggregated across all epochs belonging to
a recording using:

- Median.
- Interquartile range.

This produces:

\[
24\ feature\ families \times 19\ channels \times 2\ summaries
=912\ channel\ features
\]

Median and IQR reduce sensitivity to isolated extreme epochs while retaining
all epochs.

## 12. Regional aggregation

Notebook 03 uses five non-overlapping channel regions:

| Region | Channels |
|---|---|
| Frontal | Fp1, Fp2, F3, F4, F7, F8, Fz |
| Central | C3, C4, Cz |
| Temporal | T3, T4, T5, T6 |
| Parietal | P3, P4, Pz |
| Occipital | O1, O2 |

For each epoch and feature family, the regional value is the median across
channels in that region. Regional epoch values are then aggregated across
epochs using the median and IQR.

This produces:

\[
24\ feature\ families \times 5\ regions \times 2\ summaries
=240\ regional\ features
\]

Channel features are retained alongside regional summaries. The total is:

\[
912+240=1,152\ recording-level\ features
\]

## 13. FE-All definition

The implemented Version 1.0 table is **FE-All**:

- Every complete preprocessed epoch contributes to feature aggregation.
- Both `NO_SCREENING_FLAGS` and `SUSPECT` epochs are retained.
- QC status and artifact counts are not used as predictors.
- One row is produced for every recording.
- Median and IQR provide robust aggregation.

Although QC files are available from Notebook 02, Version 1.0 does not load
them during feature extraction. Therefore, the current table supports the
all-epoch experiment but not a clean-only reconstruction from its aggregated
values.

## 14. Planned FE-QC sensitivity table

FE-QC is a planned sensitivity analysis and is not part of the current
Notebook 03 output.

FE-QC must:

- Use the same reference transformation and feature formulas as FE-All.
- Load the matching epoch-level QC file for every recording.
- Retain only epochs with `qc_status == "NO_SCREENING_FLAGS"`.
- Exclude QC flags and suspect rates from the predictor columns.
- Use the same median and IQR aggregation.
- Report recordings with insufficient unflagged epochs by split and label.
- Apply a predefined minimum-clean-epoch rule before model evaluation.
- Compare FE-All and FE-QC on the common eligible recording set.

An optional `FE-All + QC rates` experiment may be used only as a confounding
diagnostic and must not replace the primary FE-All result.

## 15. Feature experiments for Notebook 04

Notebook 03 exports one combined 1,152-feature table. Notebook 04 should define
these column subsets before model fitting:

| Experiment | Included feature families | Research question |
|---|---|---|
| E1 — Spectral | Band powers, entropy, edge frequencies; ratios evaluated as an ablation | How much information comes from spectral characteristics? |
| E2 — Spectral + Temporal | E1 plus seven temporal/Hjorth families | Do temporal features add useful information? |
| E3 — Regional | E2 plus regional columns | Does regional aggregation improve performance or stability? |

Use identical patient-independent splits and evaluation procedures for all
experiments. The original-reference and FE-QC sensitivity analyses should
follow the primary CAR FE-All baseline.

## 16. Pilot validation and runtime benchmark

The six-recording pilot verifies:

- 1,152 unique feature columns.
- Three normal and three abnormal recordings.
- No missing or infinite recording-level pilot values.
- Correct channel and regional dimensionality.

The benchmark observed:

| Measurement | Result |
|---|---:|
| Mean time per recording | 1.07 seconds |
| Estimated full runtime | 0.89 hours |
| Maximum traced Python memory | 327.96 MB |

The completed extraction processed all 2,993 recordings in 41.57 minutes with
zero recorded exceptions.

Python memory tracing does not necessarily include every native NumPy/SciPy
allocation, so it should be interpreted as an approximate lower-bound measure.

## 17. Resumable extraction

Notebook 03 processes one recording at a time and checkpoints every 50
successful recordings.

Checkpoint:

```text
tuab_feature_extraction/checkpoints/extraction_progress.csv
```

Error log location:

```text
tuab_feature_extraction/logs/extraction_errors.csv
```

The checkpoint is written to a temporary file and atomically replaced. On
resume, the notebook verifies filepath uniqueness and the expected column set,
then processes only paths not already present.

The completed Version 1.0 run contains:

- 2,993 completed recordings.
- Zero extraction exceptions.
- No error-log file because no errors occurred.

The checkpoint does not currently hash source NPZ files or verify every saved
row before skipping it. Stronger source and row verification is recommended
for a future extraction version.

## 18. Feature schema

Notebook 03 freezes the sorted feature-column order after pilot validation and
writes:

```text
tuab_feature_extraction/metadata/feature_schema.json
```

The schema records:

- Schema version.
- CAR reference name.
- Sampling frequency and epoch duration.
- Feature count and ordered feature names.
- Channel order and regional definitions.
- Frequency bands.
- Welch segment and overlap settings.
- Channel and regional aggregation procedures.

Feature names use:

```text
car__<channel>__<feature>__<aggregation>
car__region_<region>__<feature>__<aggregation>
```

Examples:

```text
car__T5__theta_absolute_power__median
car__O1__spectral_entropy__iqr
car__region_temporal__hjorth_complexity__median
```

## 19. Final export and audit

Notebook 03 exports:

```text
tuab_feature_extraction/features/tuab_recording_features.parquet
```

The Parquet file uses PyArrow with Zstandard compression. It is read back and
compared exactly with the in-memory DataFrame.

Final artifact properties:

| Property | Result |
|---|---:|
| Rows | 2,993 |
| Metadata columns | 4 |
| Feature columns | 1,152 |
| Total columns | 1,156 |
| Duplicate recording paths | 0 |
| Infinite values | 0 |
| Final missing values | 0 |
| Parquet size | 28.68 MB |

The four exported metadata columns are:

```text
filepath
patient_id
label
split
```

These columns are provenance and grouping variables, not model predictors.

During extraction, some epoch-level regional operations emitted
`All-NaN slice` warnings. Final recording-level aggregates contained no NaN,
but Notebook 04 should not interpret that as proof that no intermediate value
was undefined. A future feature version should explicitly count and export
pre-aggregation missingness.

## 20. Leakage controls for Notebook 04

Notebook 04 must fit the following using development-training patients only:

- Missing-value imputation, if required.
- Log or other feature transforms not frozen in Notebook 03.
- Feature scaling.
- Variance filtering.
- Correlation filtering.
- Univariate or model-based feature selection.
- PCA or other dimensionality reduction.
- Model hyperparameters.
- Classification thresholds.

The validation split may guide predefined model selection. The official TUAB
evaluation split must remain untouched until the complete feature and model
pipeline is frozen.

Models must never receive filepath or patient ID as predictors. Recording
duration or epoch count should not be added as primary predictors because they
may encode acquisition behavior rather than pathology.

## 21. Known limitations and required reporting

Version 1.0 must be reported with these limitations:

- It exports one combined feature table rather than separate E1–E3 tables.
- Absolute powers are linear rather than log-transformed.
- Ratios use a fixed `1e-12 µV²` denominator criterion.
- Dominant frequency, alpha peak frequency, standard deviation, and maximum
  absolute amplitude are not included.
- All epoch-level QC categories are aggregated together.
- Intermediate missing-value counts are not preserved.
- The checkpoint does not validate source hashes or schema version before
  skipping completed rows.
- The final table does not include recording ID, epoch count, or protocol
  version as columns; these remain available through the preprocessing
  manifest and feature schema.

These limitations do not invalidate the exported table, but they define the
scope of claims that can be made from it and the ablations required in Notebook
04.

## 22. Reproducibility requirements

Any model using the Version 1.0 table must record:

- Feature schema version.
- Exact selected feature columns.
- Whether ratios were included.
- Whether regional columns were included.
- Any log transformation performed in Notebook 04.
- Training-only imputation and scaling parameters.
- Patient-independent split membership.
- Random seeds and software versions.
- FE-All or FE-QC designation.

Changes to feature formulas, numerical floors, reference handling, aggregation,
or QC inclusion require a new feature-schema version and a separately named
output artifact.
