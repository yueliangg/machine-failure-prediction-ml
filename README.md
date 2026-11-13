# Machine Failure Prediction (Classification)

Predicting whether a factory machine is **Working** or **Faulty** from its sensor readings, so that maintenance can be scheduled *before* a breakdown happens.

> **Final model:** GridSearch-tuned Random Forest. It catches **118 of 136 faulty machines** with only **2 false alarms** out of 3,864 working machines (Precision 0.983, Recall 0.868, F1 0.922).

---

## Problem Overview

| | |
|---|---|
| **Task** | Binary classification |
| **Dataset** | `factory_data.csv`: 20,000 machine records, 9 columns |
| **Target** | `Machine Status` (0 = Working, 1 = Faulty) |
| **Main challenge** | Heavy class imbalance: only **3.4%** of machines are faulty |

Because 96.6% of machines are working, a model that always predicts "Working" scores **96.6% accuracy** while catching **zero** failures. Models are therefore judged on **precision, recall and F1** for the Faulty class, not on accuracy.

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Optuna` · `Matplotlib` · `Seaborn` · `Jupyter`

---

## Workflow

1. **Exploratory Data Analysis:** class imbalance, feature distributions, working vs faulty comparisons, failure rate by quality and a correlation heatmap.
2. **Data Cleaning:** checked for duplicates and dropped the ID columns.
3. **Preprocessing (leakage-safe):**
   - 80/20 **stratified** train–test split.
   - Missing values (`Quality`, `Process T`, `Rotation Speed`) imputed with the **training-set median after the split**.
   - `Quality` ordinally encoded (L = 0, M = 1, H = 2).
   - `StandardScaler` for the distance- and linear-based models.
4. **Feature Engineering:**
   - `Power` = Torque × Rotation Speed (mechanical load)
   - `Temp_Diff` = Process T − Ambient T (heat produced by the machine)
5. **Modelling:** Dummy baseline, Logistic Regression, Decision Tree, Random Forest, KNN and SVM, with `class_weight="balanced"` where supported.
6. **Hyperparameter Tuning:**
   - Random Forest tuned with `GridSearchCV` (5-fold stratified CV, F1 scoring).
   - Decision Tree tuned with `Optuna` (5-fold CV) as a challenger model.
7. **Interpretation:** feature importance from the tuned Random Forest.

---

## Results

Test set: 4,000 machines, of which 136 are faulty.

| Model | Precision | Recall | F1 | False Alarms | Missed Failures |
|---|---|---|---|---|---|
| Dummy (always Working) | 0.000 | 0.000 | 0.000 | 0 | 136 |
| Logistic Regression | 0.164 | 0.816 | 0.273 | 567 | 25 |
| SVM | 0.314 | 0.919 | 0.468 | 273 | 11 |
| KNN | 0.821 | 0.507 | 0.627 | 15 | 67 |
| Decision Tree (Base) | 0.896 | 0.882 | 0.889 | 14 | 16 |
| Random Forest (Base) | 1.000 | 0.809 | 0.894 | 0 | 26 |
| Decision Tree (Optuna) | 0.945 | 0.882 | 0.913 | 7 | 16 |
| **Random Forest (GridSearch Tuned)** | **0.983** | **0.868** | **0.922** | **2** | **18** |

**Why the tuned Random Forest was chosen:**
- It has the highest F1 (0.922) and the fewest false alarms of any tuned model.
- Tuning caught **8 more failures** than the base Random Forest, at the cost of only **2 false alarms**.
- It is more reliable than the tuned Decision Tree, with a cross-validated F1 of **0.892** compared with 0.833.

---

## Key Insights

- **Faulty machines run under heavier load:**
  - median torque **53.7 vs 39.9 Nm**
  - median speed **1,365 vs 1,507 rpm**
  - median tool wear **165 vs 107 min**
- **Failures happen at both extremes.** Some faulty machines have very low torque or run above 2,500 rpm. These non-linear patterns are why tree-based models performed best.
- **Top predictors:** Tool Wear, Torque, Rotation Speed and the engineered **Power** feature, which together make up about 80% of the model's importance.
- **Low-quality machines fail most often** (3.91%, compared with 2.13% for high quality), but Quality adds little once the mechanical features are known.

---

## ▶️ How to Run

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install pandas numpy matplotlib seaborn scikit-learn optuna
jupyter notebook machine-status_prediction.ipynb
```

In the notebook, set the dataset path in the loading cell:

```python
df = pd.read_csv("data/factory_data.csv")
```

Then select **Kernel → Restart & Run All**. This takes about 2 minutes, because of GridSearchCV and Optuna. Results are reproducible (`random_state=42`, seeded Optuna sampler).

---

## Future Improvements

- **Lower the decision threshold** to catch more failures, if missed breakdowns cost more than inspections.
- Try **SMOTE** oversampling as an alternative to class weighting.
- Study the low-torque / high-speed failures separately, as they may be a different failure type.

---

## Author

**Thawdar Lin Wai**
Diploma student, Singapore Polytechnic
