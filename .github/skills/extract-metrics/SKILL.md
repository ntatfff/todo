---
name: extract-metrics
description: "Use when extracting evaluation metrics from all Jupyter notebooks in a provided folder and saving them as CSV."
---

# Extract Metrics From Notebooks

Use this skill when a folder contains one or more notebooks whose latest output includes sklearn-style classification metrics, and the goal is to turn that data into a clean, reusable CSV dataset.

## Workflow

1. Accept the provided folder path.
2. Recursively find all .ipynb files in that folder.
3. For each notebook, inspect the latest cell that produces the evaluation metrics.
4. Extract the values from the standard sklearn-style sections:

```
Sklearn binary metrics:
accuracy: 0.9383
precision: 0.7642
recall: 0.8174
f1: 0.7899
mcc: 0.7544
roc_auc: 0.9670
pr_auc: 0.8772
log_loss: 0.1780
brier_score: 0.0482

Confusion matrix:
 [[667  29]
 [ 21  94]]

Classification report:
               precision    recall  f1-score   support

           0     0.9695    0.9583    0.9639       696
           1     0.7642    0.8174    0.7899       115

    accuracy                         0.9383       811
   macro avg     0.8669    0.8879    0.8769       811
weighted avg     0.9404    0.9383    0.9392       811
```

## Extraction Rules

- From the Sklearn binary metrics section, extract accuracy, precision, recall, f1, mcc, roc_auc, and pr_auc.
- From the Confusion matrix section, extract the full matrix.
- From the Classification report section, extract support_0 and support_1.
- From the filename, extract the radiomics class, the kernel, and the model name.
  - Example: output/kernel3/GradientBoostedTreesModel.1.firstorder.Tensorflow.ipynb -> class=firstorder, kernel=kernel3, model=GradientBoostedTreesModel
- Create a CSV with this column order:
  class, kernel, model, accuracy, precision, recall, f1, mcc, roc_auc, pr_auc, support_0, support_1, confusion_matrix
- Each row should correspond to one notebook's metrics.
- Save the CSV in the same directory as the notebook, with the name metrics.csv.
- If a metrics.csv already exists, replace it.
- If the notebook path contains nested folders, keep the CSV next to its corresponding notebook, not in a separate aggregate directory.

## Validation

After extraction:

- Confirm that the number of rows extracted matches the number of notebooks processed.
- Spot-check at least one row against the notebook output before finishing.
- If a notebook does not contain a recognizable metrics block, report it clearly and skip it without silently dropping data.

## When To Ask A Question

Ask for clarification if any of these are unclear:

- Which notebook cell is used to extract the metrics from
- Whether the operation should process a single notebook or a whole folder

## Example Prompts

- Extract the metrics data from all notebooks in this folder into CSV.
- Compare the notebook outputs with the generated CSV and fix any mismatches.