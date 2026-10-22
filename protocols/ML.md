# Proposed ML Protocol — TUAB Normal vs Abnormal EEG

Objective: Evaluate whether spectral, temporal, and regional EEG features can distinguish normal from abnormal recordings, and determine whether QC screening improves classification performance and model interpretability.

I recommend a controlled, patient-independent, recording-level ML experiment using FE-All and FE-QC.

## 1. Experimental design

The primary comparison uses the same 2,979 eligible recordings. A secondary experiment evaluates FE-All on the complete 2,993-recording cohort.

## 2. Data splitting and leakage prevention

We will preserve the patient-grouped splits established during preprocessing.

| Split | FE-All recordings | Purpose |
|---|---:|---|
| Train | 2,173 | Model fitting and grouped cross-validation |
| Validation | 544 | Model selection and threshold calibration |
| Evaluation | 276 | Final untouched testing |

For FE-QC, the eligible counts are 2,165 train, 538 validation, and 276 evaluation.

Critical rule: No patient may appear in more than one split. We will use `StratifiedGroupKFold` for five-fold cross-validation within training, with `patient_id` as the grouping variable.

## 3. Models and preprocessing

| Model | Scaling | Proposed hyperparameters |
|---|---|---|
| Dummy Classifier | No | Prior baseline |
| Logistic Regression | StandardScaler | `C`, L2 regularization |
| SVM (RBF) | StandardScaler | `C`, `gamma` |
| Random Forest | No | `n_estimators`, `max_depth`, `min_samples_leaf` |
| XGBoost | No | `max_depth`, `learning_rate`, `n_estimators`, `subsample` |

For all models:

- Feature imputation, if required, is fitted on training data only.
- Feature selection is fitted inside CV folds, never on the full dataset before splitting.
- Hyperparameter search budgets and CV folds remain identical between FE-All and FE-QC.
- Use random seed `42` wherever supported.
- Do not include patient ID, filepath, split, or QC coverage as predictive features.

We should begin with all 1,152 features using regularized models, then investigate training-only feature selection as a separate experiment.

## 4. Evaluation metrics

Primary metric: ROC-AUC.

| Metric | Purpose |
|---|---|
| ROC-AUC | Overall ranking/discrimination |
| Balanced accuracy | Classification performance across both classes |
| Sensitivity | Correct detection of abnormal recordings |
| Specificity | Correct detection of normal recordings |
| F1-score | Balance of precision and recall |
| 95% confidence intervals | Quantify uncertainty |

We will evaluate at the recording level and additionally examine patient-level performance by aggregating multiple recording predictions per patient.

Classification thresholds should be selected using validation data, not the evaluation split. Confidence intervals should account for patient clustering.

## 5. QC sensitivity and robustness analysis

Our FE audit found that abnormal recordings have lower mean epoch acceptance, particularly in training (87.18% versus 92.52%).

Therefore, our analysis must address potential selection bias.

We will compare:

1. FE-All vs FE-QC on identical eligible recordings.
2. FE-All on the complete cohort.
3. Performance by QC-coverage subgroup, especially recordings retaining 25% or fewer epochs.
4. Feature importance and regional attribution stability between the two conditions.

A performance increase after QC screening is not sufficient evidence that clinical information was preserved.

## 6. Interpretability

For classical ML, we'll analyze permutation importance on held-out validation data and examine stability across patient-grouped folds.

We will aggregate feature importance by:

- Frequency band: delta, theta, alpha, beta, low gamma.
- Feature family: spectral, temporal, regional.
- Brain region: frontal, central, temporal, parietal, occipital.

Because many EEG features are correlated, individual permutation importance may be misleading. We should complement it with grouped-feature analyses and regional ablation experiments.

## 7. Experiment sequence

| Experiment | Deliverable |
|---|---|
| ML-01 | Data integrity, paired cohorts, split verification |
| ML-02 | Dummy + Logistic Regression baselines |
| ML-03 | SVM, Random Forest, XGBoost |
| ML-04 | Hyperparameter tuning and grouped CV |
| ML-05 | FE-All vs FE-QC paired comparison |
| ML-06 | Feature importance and regional ablation |
| ML-07 | Final locked evaluation and uncertainty analysis |

## Final recommendation

Use ROC-AUC as the primary metric, five-fold patient-grouped CV for training, the existing validation split for model selection, and the evaluation split only after the complete protocol is finalized.

For scientific rigor, the FE-All vs FE-QC comparison should be a predefined sensitivity analysis, not an assumption that QC must improve performance.

Our first notebook step will be `ML-01`: load both Parquet datasets, validate feature schemas, confirm patient independence, and establish the paired cohorts before fitting any models.
