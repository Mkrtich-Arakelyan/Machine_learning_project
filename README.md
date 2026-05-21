[README.md](https://github.com/user-attachments/files/28111230/README.md)
# Bankruptcy Prediction Using Machine Learning

This project compares eight classification algorithms for bankruptcy prediction using the Polish Companies Bankruptcy dataset from the UCI Machine Learning Repository. The goal is to predict whether a company will go bankrupt based on financial ratio data.

## Project Overview

The dataset contains five separate forecasting horizons:

* Year 1: predicts bankruptcy 5 years ahead
* Year 2: predicts bankruptcy 4 years ahead
* Year 3: predicts bankruptcy 3 years ahead
* Year 4: predicts bankruptcy 2 years ahead
* Year 5: predicts bankruptcy 1 year ahead

Each dataset contains 64 financial ratio features and a binary target variable:

* `0` = non-bankrupt company
* `1` = bankrupt company

The dataset is highly imbalanced, so accuracy alone is not enough for model evaluation. The project therefore focuses on precision, recall, F1-score, ROC-AUC, confusion matrices, and threshold analysis.

## Models Used

The project compares the following eight classification algorithms:

* k-Nearest Neighbors
* Naive Bayes
* Logistic Regression
* Linear Discriminant Analysis
* Quadratic Discriminant Analysis
* Decision Tree
* Support Vector Machine
* Random Forest

## Methodology

The workflow uses scikit-learn pipelines for preprocessing, model training, and evaluation.

Main steps:

* Load yearly ARFF datasets
* Rename financial ratio columns into informative names
* Apply median imputation for missing values
* Clip extreme values for selected models
* Apply standard scaling for scale-sensitive models
* Use stratified 80/20 train-test split
* Tune models with GridSearchCV using F1-score
* Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
* Perform threshold analysis for Random Forest

## Main Findings

Random Forest was the strongest overall model. It achieved the best F1-score in Years 2-5 and the highest ROC-AUC across all five forecasting horizons. Decision Tree achieved the best F1-score in Year 1.

Threshold analysis showed that the default classification threshold of 0.5 was not always optimal. For Random Forest, lowering the threshold to 0.4 improved F1-score in Years 1, 4, and 5.

Overall, the project shows that bankruptcy prediction should be treated as a risk-ranking and decision-support problem, not simply as a high-accuracy classification task.

Project Structure
.
├── Dataset.py
├── model\_configs.py
├── evaluations.py
├── experiments.py
├── visualization.py
├── main.py
├── results/
├── figures/
├── report/
└── README.md
```

## File Descriptions

### `Dataset.py`

Loads the yearly ARFF datasets, converts the target variable, renames columns, and returns feature matrix `X` and target vector `y`.

### `model\_configs.py`

Defines model pipelines and hyperparameter grids for GridSearchCV.

### `evaluations.py`

Calculates evaluation metrics, including accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix values, and threshold-analysis results.

### `experiments.py`

Runs the full experiment for each forecasting year, including train-test split, model tuning, evaluation, and result saving.

### `visualization.py`

Generates model comparison charts, ROC curves, confusion matrices, threshold plots, and across-year visualizations.

### `main.py`

Runs the full workflow for all five forecasting years.

## How to Run

Install the required packages:

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

Run the full experiment:

```bash
python main.py
```

The script generates result CSV files in the `results/` folder and figures in the `figures/` folder.

## Report

The full written report is included in the `Bankruptcy prediction report.pdfS/` folder. It contains the methodology, model comparison tables, ROC curves, confusion matrices, threshold analysis, and final conclusions.

## Dataset Source

UCI Machine Learning Repository: Polish Companies Bankruptcy Data  
https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

