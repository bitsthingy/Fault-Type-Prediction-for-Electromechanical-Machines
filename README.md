# Fault Type Prediction in Rotating Machinery

Classifying the **fault type of a machine** from its vibration measurements using classical machine learning.

## Main Idea

Vibration signatures change depending on the mechanical fault (imbalance, misalignment, bearing wear, looseness, etc.). This project turns raw vibration readings into machine-level features and trains models to predict the fault class, then interprets the important features in mechanical terms.

## Dataset

- [Mechanical Analysis](https://www.kaggle.com/datasets/heitornunes/mechanical-analysis) (Kaggle)
- 9,254 raw measurements, aggregated into **209 machine instances**
- 6 fault classes (imbalanced: class 2 = 69 samples, class 3 = 14 samples)

## Approach

| Step | Details |
|------|---------|
| Aggregation | Pivoted per-sensor readings (`dir` = measurement direction/type) into one row per machine (`mis_*`, `misr_*`) |
| Preprocessing | Median imputation, standard scaling, variance threshold |
| Split | 80/20 stratified train/test |
| Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| Tuning | `GridSearchCV` with 5-fold CV |
| Evaluation | Accuracy, confusion matrices, classification report, feature importance |

## Results

| Model | Test Accuracy |
|-------|---------------|
| **Logistic Regression** | **0.738** |
| Random Forest | 0.690 |
| Gradient Boosting | 0.690 |
| XGBoost | 0.690 |
| Decision Tree | 0.667 |

- Classes 2 and 6 are predicted well (F1 of 0.90 and 1.00).
- Rare classes (3, 4, 5) are the weak spot, which is expected with only 14–18 samples each.
- Simple linear models beat tree ensembles, likely because the dataset is very small.

## Run It

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn kaggle
```

Open `Fault_Type_Prediction.ipynb` (built for Google Colab). It needs a `kaggle.json` API key to download the dataset.

## Tech Stack

Python · scikit-learn · XGBoost · pandas · matplotlib · seaborn
