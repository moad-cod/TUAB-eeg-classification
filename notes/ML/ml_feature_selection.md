# E1 — EEG Feature Selection: Experimental Report

Generated: 2026-10-24 11:46 UTC

## 1. Research objective

Evaluate whether feature selection can reduce handcrafted EEG
feature dimensionality while maintaining normal-versus-abnormal
EEG classification performance.

## 2. Dataset and experimental design

- Dataset: TUH Abnormal EEG Corpus (TUAB).
- Input representation: recording-level handcrafted EEG features.
- Total candidate features: 1152.
- Eligible training recordings: 2165.
- Training patients: 1662.
- Outcome: normal versus abnormal EEG recording.
- Primary metric: recording-level ROC-AUC.
- Outer validation: five-fold stratified patient-grouped CV.
- Inner validation: five-fold stratified patient-grouped CV.
- Random seed: 42.
- Preprocessing, feature selection, and hyperparameter tuning
  performed within the training portions of the CV procedure.
- Held-out evaluation recordings were not used for model selection.

## 3. Compared methods

1. Baseline: all features + RBF SVM.
2. SelectKBest: ANOVA selection + RBF SVM.
3. L1: L1-regularized logistic regression selection + RBF SVM.
4. RFE: recursive elimination with L2 logistic regression
   + RBF SVM.

## 4. Nested-CV performance

| Method | ROC-AUC (mean ± SD) | Mean features | Reduction |
|---|---:|---:|---:|
| Baseline | 0.9132 ± 0.0105 | 1152.0 | 0.0% |
| SelectKBest | 0.9186 ± 0.0172 | 500.0 | 56.6% |
| L1 | 0.9116 ± 0.0110 | 326.8 | 71.6% |
| RFE | 0.9094 ± 0.0123 | 440.0 | 61.8% |

SelectKBest produced the highest observed mean outer-fold ROC-AUC.
However, its advantage over the full-feature baseline is small,
and the existing paired uncertainty interval includes zero.

## 5. Feature-selection stability

Chance-adjusted overlap results:

| L1 Logistic Regression | 0.4130 |
| RFE Logistic Regression | 0.3554 |
| SelectKBest | 0.9028 |

SelectKBest exhibited the highest selection consistency across
the five outer training folds.

The outer training subsets overlap, so fold-pair stability
measurements are not independent replications.

## 6. EEG feature interpretation

### Most frequently selected spatial groups

| Spatial group | Selection rate |
|---|---:|
| T3 | 57.50% |
| T4 | 53.33% |
| C4 | 51.67% |
| region_temporal | 48.75% |
| P4 | 48.75% |
| P3 | 48.33% |
| region_central | 47.50% |
| region_frontal | 46.67% |
| T6 | 45.83% |
| region_parietal | 45.42% |

### Feature families

| Family | Selection rate |
|---|---:|
| Spectral ratios | 61.77% |
| Other spectral | 47.22% |
| Time-domain | 39.46% |
| Spectral power | 37.67% |

Alpha relative power was selected in all five folds for
all 48 corresponding features.

These results describe feature-selection frequency, not causal
feature importance or anatomical localization of abnormalities.

## 7. Limitations

- SelectKBest selected k=500, the largest tested value, in all
  five folds; larger k values were not evaluated.
- SelectKBest's SVM C also reached the tested upper boundary.
- L1 selected feature counts varied substantially across folds.
- RFE uses logistic regression as its elimination estimator,
  rather than eliminating features based directly on the SVM.
- The study uses recording-level labels and metrics.
- Some patients have recordings with both labels; assigning
  one patient-level label would be inappropriate.
- Paired bootstrap comparisons are exploratory and were not
  adjusted for multiple comparisons.
- Pooled out-of-fold SVM decision scores may differ in scale
  between folds; mean outer-fold AUC remains the primary result.
- Feature-selection frequency does not establish clinical
  regional attribution.
- External dataset generalization has not yet been evaluated.

## 8. Conclusion

SelectKBest is the leading feature-selection candidate under
the evaluated search space, with mean ROC-AUC
0.9186
and a feature reduction of
56.6%.

The full-feature RBF SVM remains an important reference baseline.
No definitive superiority claim is made.

## 9. Next experiment

E2 — Deep learning baselines on harmonized EEG recordings.

Planned work:
- Verify the preprocessed EEG signal representation.
- Establish identical patient-grouped evaluation partitions.
- Train established EEG CNN architectures.
- Compare CNNs with E1 ML baselines under comparable conditions.
- Reserve regional attribution and cross-dataset validation
  for subsequent experiments.
