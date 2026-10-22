# TUAB preprocessing figures

These figures document preprocessing experiments already embedded in
`notebooks/02-tuab-eeg-preprocessing-and-signal-harmonizati.ipynb`. They were
copied without regeneration or rescaling. Unless noted otherwise, they are
pilot analyses and must not be interpreted as full-dataset performance or
quality estimates.

## Initial signal quality control

### `tuab_initial_channel_qc.png`

- **Demonstrates:** Distributions of channel-segment peak-to-peak amplitude,
  RMS amplitude, and minimum one-second peak-to-peak amplitude for normal and
  abnormal recordings.
- **Notebook section:** Channel Standardization and Initial Quality Control,
  initial QC pilot visualization.
- **Scope:** Pilot experiment.
- **Limitations:** The sampled recordings and short signal segments do not
  characterize every TUAB recording. The distributions support threshold
  inspection; they do not automatically reject channels or epochs.

## Filtering and resampling

### `tuab_filter_resample_psd.png`

- **Demonstrates:** Power spectral density before filtering, after the
  0.5–40 Hz FIR filter, and after resampling to 200 Hz for representative
  normal and abnormal recordings.
- **Notebook section:** Bandpass filtering and resampling.
- **Scope:** Pilot experiment.
- **Limitations:** The comparison uses representative recordings and does not
  establish dataset-wide spectral behavior.

## Reference comparison

### `tuab_car_waveform_comparison.png`

- **Demonstrates:** Ten-second waveform excerpts under the original reference
  and common average reference (CAR) for selected channels from representative
  normal and abnormal recordings.
- **Notebook section:** Original reference vs. CAR; waveform visualization.
- **Scope:** Pilot experiment.
- **Limitations:** Only a small channel subset and two recordings are shown.
  The production dataset stores only the original-reference representation;
  CAR can be calculated during model training.

### `tuab_car_psd_comparison.png`

- **Demonstrates:** Power spectral density under the original reference and
  CAR for representative normal and abnormal recordings.
- **Notebook section:** Original reference vs. CAR; PSD comparison.
- **Scope:** Pilot experiment.
- **Limitations:** The plot is a qualitative comparison from representative
  recordings, not a full-dataset statistical analysis.

## Figures intentionally not published

Two existing flatline-inspection images are excluded because their filenames
and titles contain recording identifiers. No existing standalone figures were
found for channel standardization, epoch visualization, or full-preprocessing
validation. Missing figures were not regenerated and notebook cells were not
executed.
