# EEG Preprocessing Protocol — Version 2.0

**Research:** Reliability of Spatial Explanations in Deep Learning-Based EEG Abnormality Detection: A Cross-Dataset Study

**Primary dataset:** TUAB v3.0.1  
**External validation dataset:** NMT Scalp EEG

## 1. Objective

Establish a reproducible, pathology-preserving EEG preprocessing pipeline for normal-versus-abnormal classification and spatial explainability.

The primary pipeline must:

- Preserve clinically relevant abnormal EEG activity.
- Avoid unnecessary modification of spatial EEG patterns.
- Maintain consistent model inputs across TUAB and NMT.
- Prevent patient-level and preprocessing-related data leakage.
- Document all quality-control decisions.

All learned preprocessing parameters and data-dependent thresholds must be estimated using TUAB training data only. The finalized protocol must be frozen before evaluating the held-out TUAB evaluation partition or NMT.

## 2. Preprocessing configuration

| Component | Final proposed configuration |
|---|---|
| EEG channels | Standard 10–20 montage, 19 scalp channels |
| Channel order | Fp1, Fp2, F3, F4, F7, F8, Fz, C3, C4, Cz, T3, T4, T5, T6, P3, P4, Pz, O1, O2 |
| Channel naming | Standardized channel-name mapping |
| Initial QC | Detect missing, flat, clipped, or severely corrupted channels |
| Referencing | Common Average Reference (CAR), subject to bad-channel handling rules |
| Bandpass filtering | 0.5–40 Hz, zero-phase FIR |
| Notch filtering | Disabled by default |
| Resampling | 200 Hz with appropriate anti-alias filtering |
| Epoching | Non-overlapping 8-second windows |
| Artifact detection | Conservative technical-artifact detection |
| Channel interpolation | Disabled in the primary pipeline |
| Epoch rejection | Only clearly unusable epochs under validated, predefined rules |
| Normalization | Physical-unit conversion to µV; no data-dependent scaling in the primary baseline |
| Sensitivity experiment | Training-derived channel-wise robust scaling |
| CNN input | `(19, 1600)` per epoch |
| Prediction aggregation | Epoch predictions aggregated to recording level |

**Status:** This is the proposed protocol. Artifact thresholds, CAR handling, and rejection criteria require pilot validation before the protocol is frozen.

## 3. Preprocessing sequence

### Step 1 — Dataset partitioning

- Preserve the official TUAB training and evaluation partitions.
- Create a patient-grouped training/validation split within the official training partition.
- Keep the official TUAB evaluation partition untouched during development.
- Reserve NMT exclusively for external validation.
- Fit all learned parameters using development-training recordings only; use validation data for model and protocol selection.

### Step 2 — Channel selection and standardization

Select the same 19 scalp EEG channels from every recording.

Standardize channel names and preserve the fixed ordering.

Exclude auxiliary channels such as ECG, EMG, and non-scalp reference channels from the model input.

Retain auxiliary-channel availability as metadata for potential artifact investigations.

**Missing-channel policy:** Flag recordings with missing required channels. Do not silently replace missing electrodes with zeros or synthetic signals.

### Step 3 — Initial channel-quality assessment

Before calculating CAR, inspect each channel for technical failures:

- Sustained flatlines or near-zero variation.
- Signal clipping or saturation.
- Abrupt electrode discontinuities.
- Severe isolated-channel corruption.

Generate channel-quality flags without automatically deleting unusual EEG activity.

### Step 4 — Common Average Reference

Apply CAR using the available reliable scalp electrodes:

\[
V_i^{CAR}(t)=V_i(t)-\frac{1}{|G|}\sum_{j\in G}V_j(t)
\]

Where \(G\) is the set of channels accepted for reference calculation.

**Important constraints:**

- Do not automatically interpolate bad channels.
- A recording with a bad required channel cannot automatically become a valid 19-channel model input.
- Define a consistent policy for such recordings during pilot validation.
- Record channels excluded from the CAR calculation.
- Audit whether varying CAR membership influences regional explanations.

### Step 5 — Filtering

Apply:

- High-pass cutoff: 0.5 Hz.
- Low-pass cutoff: 40 Hz.
- Zero-phase FIR filtering for offline analysis.
- No routine 50/60 Hz notch filtering.

Inspect power spectral density before and after filtering.

Apply filtering to continuous recordings before epoch extraction, with appropriate edge handling.

### Step 6 — Resampling

Resample all recordings to 200 Hz using appropriate anti-alias filtering.

This harmonizes TUAB with the 200 Hz NMT recordings.

Document original and target sampling frequencies.

### Step 7 — Epoching

Divide recordings into fixed, non-overlapping 8-second windows.

At 200 Hz:

\[
N=8\times200=1600
\]

Each epoch has the shape:

`(19 channels, 1600 samples)`

Discard incomplete final windows and record their duration.

Preserve recording IDs, patient IDs, epoch start times, and labels.

## 4. Conservative artifact-quality control

### 4.1 Objective

Identify technically unreliable EEG without systematically removing clinically relevant abnormalities.

Artifact detection must distinguish between **flagging suspicious signals** and **rejecting unusable signals**.

### 4.2 Detection methods

| Artifact category | Detection approach | Default action |
|---|---|---|
| Sustained flatline | Near-zero peak-to-peak amplitude or variance within consecutive short subwindows | Flag; reject if confirmed unusable |
| Clipping/saturation | Repeated saturation plateaus or evidence of acquisition limits | Flag; reject severe cases |
| Abrupt electrode discontinuity | Large sudden steps with supporting temporal or channel-specific evidence | Flag for inspection |
| Extreme amplitude | Peak-to-peak amplitude exceeding a training-derived threshold | Flag, retain by default |
| Unusual spectral activity | Abnormal PSD patterns or excessive narrowband interference | Flag, retain by default |
| Suspected ocular or muscle artifact | Temporal, spatial, and spectral evidence | Flag, retain by default |

**No universal amplitude threshold will be treated as proof of artifact.**

Numerical thresholds must be documented, justified using literature where applicable, and evaluated using TUAB training recordings.

### 4.3 Quality-control categories

| Status | Definition | Primary pipeline action |
|---|---|---|
| CLEAN | No detected technical problems | Retain |
| SUSPECT | Unusual amplitude, morphology, or uncertain artifact | Retain with QC flags |
| UNUSABLE | Strong evidence of technical failure under validated rules | Reject and document |

High-amplitude activity alone must not cause rejection.

### 4.4 Channel handling

The primary pipeline will not automatically interpolate channels.

If a required channel is unusable, the recording or affected epochs must follow a predefined exclusion policy, rather than silently introducing artificial channel values.

Autoreject and channel interpolation may be evaluated later as alternative preprocessing methods.

### 4.5 QC validation

Before finalizing artifact thresholds:

1. Inspect representative normal and abnormal training recordings.
2. Evaluate flagged and rejected epochs.
3. Check whether technical failures are correctly identified.
4. Compare QC distributions and rejection rates between classes.
5. Investigate whether potentially pathological patterns are disproportionately flagged.
6. Freeze the selected QC rules before held-out evaluation.

Expert-confirmed abnormality preservation should be measured if appropriate annotations and qualified review are available. Recording-level normal/abnormal labels alone cannot establish whether an individual epoch contains pathology.

## 5. Normalization strategy

### 5.1 Primary experiment

Use EEG amplitudes expressed in microvolts:

\[
x_{\mu V}=x_V\times10^6
\]

Do not apply data-dependent robust scaling or per-epoch standardization in the primary baseline.

This preserves amplitude information that may be relevant to abnormality classification.

### 5.2 Sensitivity experiment — Robust scaling

Evaluate channel-wise robust scaling separately:

\[
x'_{c,t}=
\frac{x_{c,t}-\operatorname{median}_{train,c}}
{\operatorname{IQR}_{train,c}}
\]

Estimate channel-specific medians and interquartile ranges from development-training data only.

Freeze the parameters and apply them unchanged to validation, held-out TUAB evaluation, and NMT.

Compare classification performance and spatial attribution stability against the unscaled baseline.

## 6. Spatial explainability safeguards

Because the study investigates regional importance:

- Preserve the same electrode ordering in every epoch.
- Avoid automatic channel interpolation in the primary pipeline.
- Document excluded channels and CAR reference membership.
- Keep preprocessing identical across CNN architectures.
- Track which epochs contribute to recording-level predictions.
- Evaluate whether preprocessing choices change regional attributions.
- Do not interpret channel importance as direct localization of a pathological brain source.

## 7. Quality-control audit and reproducibility

Store the following metadata:

| Metadata | Purpose |
|---|---|
| Patient ID | Patient-independent evaluation |
| Recording ID | Traceability |
| Dataset and split | TUAB/NMT provenance |
| Original sampling frequency | Acquisition documentation |
| Channel availability | Montage consistency |
| Bad-channel flags | Technical quality assessment |
| CAR reference membership | Referencing reproducibility |
| Epoch start/end times | Temporal traceability |
| QC metrics and flags | Artifact audit |
| Rejection status and reason | Transparent exclusions |
| Final epoch count | Retained-data reporting |
| Preprocessing version | Reproducibility |

Report recording-level and epoch-level retention rates separately for normal and abnormal groups.

## 8. Planned sensitivity experiments

After establishing the primary baseline, investigate:

| Experiment | Comparison |
|---|---|
| Artifact handling | Conservative QC vs. Autoreject-based rejection |
| Channel repair | No interpolation vs. conservative local interpolation |
| Normalization | No scaling vs. robust scaling |
| Referencing | CAR vs. alternative reference where scientifically comparable |

Select any alternative configurations using TUAB development data only.

The official TUAB evaluation partition and NMT must not be used to choose preprocessing parameters.

## 9. Final implementation sequence

1. Confirm patient-grouped data partitioning.
2. Standardize channel names and ordering.
3. Conduct initial channel-quality assessment.
4. Validate CAR and bad-channel handling.
5. Apply filtering and resampling.
6. Generate 8-second epochs.
7. Apply conservative epoch-level QC.
8. Export clean and suspect epochs with complete QC metadata.
9. Train initial ML/CNN baselines using the primary normalization strategy.
10. Conduct preprocessing sensitivity experiments.
11. Freeze the complete pipeline for final TUAB and NMT evaluation.

**Core principle:** Preserve potentially pathological EEG activity unless there is strong, documented evidence of technical corruption. Quality control should improve reliability without silently changing the clinical information or spatial patterns under investigation.