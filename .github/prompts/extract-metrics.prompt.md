---
description: "Extract evaluation metrics from all Jupyter notebooks in a provided folder and save the results as CSV."
name: "Extract Notebook Metrics"
argument-hint: "Path to folder containing .ipynb files"
agent: "agent"
---

Use the workflow in the extract-metrics skill for processing a folder of notebooks and producing a metrics.csv for each matching notebook.

Process all .ipynb files under the provided folder path, extract the relevant evaluation metrics for each notebook, and save the results in the same directory as each notebook.

Follow the skill’s rules for:
- identifying the latest metrics-producing cell
- extracting the expected metric fields,  confusion matrix and support values
- parsing class, kernel, and model from the filename
- validating row counts and spot-checking the output
- skipping notebooks that do not contain recognizable metrics without silently dropping them

Example invocations:
- /extract-notebook-metrics /workspaces/todo
- /extract-notebook-metrics /workspaces/todo/dataset
