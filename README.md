
# Machine Failure Prediction Using Machine Learning

**Predictive Maintenance | Binary Classification | Python | scikit-learn**

## Overview

This project explores machine learning approaches for identifying potentially faulty factory equipment based on operating conditions.

Using a dataset of **20,000 machine records**, five supervised classification algorithms were compared, and a Random Forest model was optimised to improve failure detection while keeping false alarms low.

### Key Results

The optimised Random Forest achieved:

- **Accuracy:** 99.50%
- **Precision:** 99.15%
- **Recall:** 86.03%
- **F1-score:** 92.13%

The project demonstrates data preprocessing, exploratory data analysis, feature engineering, model comparison, hyperparameter optimisation and classification evaluation.

> **Project Scope:** This is an offline machine-status classification experiment, not a deployed real-time predictive maintenance system. It predicts whether equipment is working or faulty rather than forecasting exactly when a future failure will occur.

---

## Problem Statement

Unexpected machine failures can disrupt industrial operations, increase maintenance costs and cause production downtime.

Machine learning can support predictive maintenance by identifying potentially faulty equipment from operating measurements.

The main objective of this project is to develop a classification model that:

1. Identifies faulty machines accurately.
2. Minimises missed failures (false negatives).
3. Reduces unnecessary maintenance alerts (false positives).
4. Provides insights into important machine operating conditions.

Because faulty machines represent a small proportion of the dataset, accuracy alone is insufficient to evaluate model performance.

Special attention is given to **precision, recall and F1-score**.

---

## Dataset

The project uses `factory_data.csv`, containing 20,000 machine records.

### Dataset Summary

| Property | Value |
|---|---|
| Total records | 20,000 |
| Original columns | 9 |
| Target variable | Machine Status |
| Working machines | 19,322 (96.61%) |
| Faulty machines | 678 (3.39%) |
| Problem type | Binary Classification |

### Input Features

The available predictors include:

- Machine quality
- Ambient temperature
- Process temperature
- Rotation speed
- Torque
- Tool wear

Unique ID and Product ID are excluded from the model inputs.

### Missing Values

| Feature | Missing Records |
|---|---:|
| Quality | 991 |
| Process T (C) | 400 |
| Rotation Speed (rpm) | 1,188 |

The dataset is not included in this repository. An authorised copy must be placed in the `data/` directory before running the notebook.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

Exploratory data analysis was conducted to understand the dataset and identify factors associated with machine failures.

The analysis included:

- Examining feature distributions.
- Identifying missing values and duplicates.
- Investigating outliers.
- Visualising relationships between operating measurements.
- Comparing healthy and faulty machine conditions.
- Analysing correlations between numerical features.
- Investigating class imbalance.

The analysis revealed that faulty machines account for only **3.39%** of the dataset, making evaluation metrics beyond accuracy particularly important.

### 2. Data Cleaning and Preprocessing

Data preprocessing was performed to prepare the dataset for model training.

Steps included:

- Checked for duplicate records; none were identified.
- Removed `Unique ID` and `Product ID` from the predictors.
- Encoded machine quality categories (`L`, `M`, `H`) numerically.
- Handled missing numerical values using median imputation.
- Applied feature standardisation where appropriate.
- Divided the dataset into training and testing subsets.

**Train-test split:**

- Training: 80%
- Testing: 20%
- Stratified sampling
- Random state: 42

Preprocessing transformations were fitted using the training data to reduce data leakage.

### 3. Feature Engineering

Two additional features were created to capture relationships between machine operating conditions.

**Power**

```python
Power = Torque (Nm) * Rotation Speed (rpm)
```

This feature combines torque and rotational speed.

**Temperature Difference**

```python
Temp_Diff = Process T (C) - Ambient T (C)
```

This feature measures the difference between process and surrounding temperatures.

These engineered features were included in model development and feature-importance analysis.

---

## Machine Learning Models

Five supervised machine learning algorithms were trained and evaluated.

### Models Compared

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbours (KNN)
5. Support Vector Machine (SVM)

The models were evaluated using accuracy, precision, recall and F1-score.

### Baseline Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.78% | 14.51% | 83.09% | 24.70% |
| Decision Tree | 99.33% | 90.98% | 88.97% | 89.96% |
| Random Forest | 99.38% | 100.00% | 81.62% | 89.88% |
| KNN | 98.05% | 85.37% | 51.47% | 64.22% |
| SVM | 93.50% | 33.51% | 92.65% | 49.22% |

**Key Observation:**

The baseline Random Forest achieved 100% precision, meaning it produced no false-positive predictions on the test set.

However, its recall of 81.62% indicated that it still failed to detect some faulty machines.

This motivated further hyperparameter optimisation.

---

## Random Forest Hyperparameter Optimisation

Random Forest was further optimised using `GridSearchCV`.

### Hyperparameters Explored

| Parameter | Values |
|---|---|
| n_estimators | 100, 200 |
| max_depth | 5, 10, 20 |
| min_samples_split | 5, 10 |
| min_samples_leaf | 2, 4 |

The search used:

- Five-fold cross-validation
- F1-score as the optimisation metric

### Before and After Optimisation

| Metric | Baseline Random Forest | Tuned Random Forest |
|---|---:|---:|
| Accuracy | 99.38% | **99.50%** |
| Precision | 100.00% | **99.15%** |
| Recall | 81.62% | **86.03%** |
| F1-score | 89.88% | **92.13%** |
| False Positives | 0 | 1 |
| False Negatives | 25 | **19** |

### Key Findings

The optimised Random Forest:

- Improved recall by approximately **4.41 percentage points**.
- Improved F1-score from 89.88% to 92.13%.
- Reduced missed faulty machines from 25 to 19.
- Maintained high precision at 99.15%.

Although one additional false-positive prediction occurred, the tuned model identified more faulty machines.

This demonstrates the importance of balancing precision and recall for machine failure detection.

---

## Feature Importance Analysis

Random Forest feature importance was examined to identify predictors associated with machine failure classification.

### Top Five Features

| Feature | Importance |
|---|---:|
| Power | 0.215 |
| Torque (Nm) | 0.198 |
| Rotation Speed (rpm) | 0.195 |
| Tool Wear (min) | 0.142 |
| Temp_Diff | 0.138 |

Power, torque and rotation speed were among the most influential features in the trained Random Forest model.

These scores describe model-specific predictive importance and should not be interpreted as causal relationships.

---

## Technologies Used

**Programming Language**
- Python

**Data Analysis**
- Pandas
- NumPy

**Data Visualisation**
- Matplotlib

**Machine Learning**
- scikit-learn
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM

**Model Optimisation**
- GridSearchCV
- Optuna

**Development Environment**
- Jupyter Notebook

---

## Repository Structure

```text
machine-failure-prediction-ml/
│
├── README.md
├── machine_failure_prediction.ipynb
├── requirements.txt
├── .gitignore
│
├── data/
│   └── .gitkeep
│
└── docs/
    └── Machine_Failure_Presentation.pdf
```

The presentation is optional and should only be included if permitted for public sharing.

---

## Installation and Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/machine-failure-prediction-ml.git

cd machine-failure-prediction-ml
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
optuna
notebook
```

### Step 3: Prepare the Dataset

Place the authorised `factory_data.csv` file inside the `data/` folder.

Update the notebook's data-loading path to:

```python
df = pd.read_csv("data/factory_data.csv")
```

### Step 4: Run the Notebook

```bash
jupyter notebook machine_failure_prediction.ipynb
```

Execute the notebook cells sequentially to reproduce the preprocessing, training and evaluation workflow.

Results may vary slightly depending on package versions and stochastic tuning experiments.

---

## Limitations

### 1. Class Imbalance

Only 3.39% of the machines are labelled faulty.

High accuracy does not necessarily indicate strong failure detection performance.

Precision, recall, F1-score and confusion matrices are therefore essential evaluation metrics.

### 2. Dataset Generalisation

The models were evaluated on a single held-out test split.

Additional testing on new operating conditions would be necessary before production use.

### 3. Hyperparameter Selection

The exploratory Optuna Decision Tree experiment in the original notebook used the test set during tuning.

Future experiments should use cross-validation or a separate validation set to avoid biased performance estimates.

The reported optimised Random Forest results are based on the GridSearchCV experiment.

### 4. Production Readiness

The project focuses on offline model development and evaluation.

Deployment, live machine monitoring and real-time alerting were not implemented.

---

## Future Improvements

Potential extensions include:

- Applying additional class-imbalance handling techniques.
- Investigating precision-recall curves and classification thresholds.
- Evaluating models across multiple random seeds.
- Integrating preprocessing and classification into scikit-learn Pipelines.
- Testing generalisation on new machine operating records.
- Developing a monitoring dashboard for machine failure predictions.

---

## Conclusion

This project explored five machine learning algorithms for machine failure classification and demonstrated how feature engineering and hyperparameter optimisation can improve predictive performance.

The optimised Random Forest achieved:

- **99.50% Accuracy**
- **99.15% Precision**
- **86.03% Recall**
- **92.13% F1-score**

The model correctly identified **117 out of 136 faulty machines** in the test set while producing only one false-positive alert.

The project highlights the importance of selecting evaluation metrics according to the problem, particularly when false negatives and false positives have different operational consequences.
