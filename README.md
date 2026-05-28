# Diabetes Prediction Challenge

This repository contains our work for Kaggle's **Diabetes Prediction Challenge**. The task is a binary classification problem where the goal is to predict whether a patient is diagnosed with diabetes using structured clinical and lifestyle data.

## Project Overview

The dataset contains **1,000,000 patient records**, split into **700,000 training samples** and **300,000 test samples**. It includes numerical health indicators such as age, BMI, physical activity, blood pressure, cholesterol levels, triglycerides, and diet score, as well as categorical variables such as gender, ethnicity, education level, income level, smoking status, and employment status.

The main objective was to maximize ROC-AUC while building a model that generalizes well beyond leaderboard-specific artifacts.

## Team

Team name: **diabetes**

Team members:
- To Sum Yi
- HEO Sunghak
- Mok Pan Yu

## Methodology

### 1. Exploratory Data Analysis

We first examined the target distribution, feature distributions, Pearson and Spearman correlations, and mutual information scores. The analysis showed that most individual features had only weak to moderate correlation with the target. Mutual information helped identify potential nonlinear relationships, especially in features such as physical activity and alcohol consumption.

Dimensionality reduction methods such as PCA, LDA, QDA, UMAP, and ISOMAP were also explored. These projections showed substantial class overlap, suggesting that the problem is not easily solved using simple linear separation.

### 2. Feature Engineering

Several feature engineering strategies were tested:

- Interaction features such as BMI × physical activity, waist-to-hip ratio × physical activity, LDL cholesterol × physical activity, and BMI × age
- Log transformations for skewed features such as physical activity and triglycerides
- Binning for age, physical activity, and alcohol consumption
- Polynomial feature expansion
- mRMR-based feature selection
- KAN-based neural symbolic features

Overall, manual and automated feature engineering produced limited improvement. In several cases, engineered features slightly reduced performance, indicating that the original features were already informative enough and that tree-based models could capture most nonlinear relationships internally.

### 3. Model Development

We trained and compared several models:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost
- CatBoost
- TabM
- AutoGluon benchmark model

Logistic Regression and Random Forest were used as baseline models. Gradient boosting models significantly outperformed the baselines, with LightGBM, XGBoost, and CatBoost becoming the main models for later experiments.

### 4. Hyperparameter Tuning

RandomizedSearchCV with 5-fold cross-validation was used to tune the main gradient boosting models. The search focused on model complexity, learning rate, regularization, and sampling parameters.

Best tuned single-model leaderboard scores:

| Model | Private Score |
|---|---:|
| LightGBM | 0.69373 |
| CatBoost | 0.69345 |
| XGBoost | 0.69265 |

### 5. Ensemble Methods

We used blending as the main ensemble strategy. Blending was performed by taking weighted averages of predicted probabilities from LightGBM, XGBoost, and CatBoost.

Best no-ID blending results:

| Model Blend | Private Score |
|---|---:|
| LGB + XGB + CAT 60:25:15 | 0.69413 |
| LGB + XGB + CAT 70:15:15 | 0.69412 |
| LGB + XGB + CAT 65:25:10 | 0.69408 |
| LGB + XGB 70:30 | 0.69394 |
| LGB + XGB 60:40 | 0.69392 |

The best final model was a **triple blend of LightGBM, XGBoost, and CatBoost with weights 60:25:15**, achieving a private score of **0.69413**.

## ID Feature Analysis

Kaggle discussions and high-ranking solutions suggested that the `id` column contained hidden ordering effects and distribution shifts. We tested this hypothesis and confirmed that including ID improved leaderboard scores significantly.

However, ID does not represent meaningful clinical information and may exploit dataset-specific artifacts. Therefore, the final model excluded ID to prioritize robustness, interpretability, and generalization.

## Final Result

Our final selected model was a no-ID blended ensemble of:

- LightGBM
- XGBoost
- CatBoost

Final private score:

```text
0.69413
```

This model was selected because it provided the best balance between leaderboard performance and principled modeling practice.

## Key Takeaways

- Manual feature engineering provided limited improvement.
- Tree-based models were more effective than linear and deep learning baselines for this tabular dataset.
- Hyperparameter tuning improved individual model performance.
- Blending multiple gradient boosting models improved stability and performance.
- ID-based features improved leaderboard scores but were excluded from the final model due to leakage and generalization concerns.

## Repository Structure

A recommended repository structure is shown below:

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── baseline_logistic.ipynb
│   ├── baseline_RF.ipynb
│   ├── lightgbm.ipynb
│   ├── lightgbm_polynomial.ipynb
│   └── lightgbm_with_id.ipynb
├── submissions/
│   ├── logistic_baseline.csv
│   ├── lightgbm_baseline.csv
│   ├── blend_noID_lgb_xgb_cat_60_25_15.csv
│   └── ...
├── reports/
│   ├── Report.pdf
│   └── Individual Report - Sunghak Heo.pdf
└── README.md
```

## Dependencies

Main Python libraries used:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
lightgbm
xgboost
catboost
autogluon
```

Install common dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost catboost
```

## How to Run

1. Download the competition data from Kaggle.
2. Place `train.csv`, `test.csv`, and `sample_submission.csv` inside the `data/` folder.
3. Run the EDA notebooks to reproduce the analysis.
4. Train individual models such as LightGBM, XGBoost, and CatBoost.
5. Generate prediction files for each model.
6. Blend model predictions using weighted averaging.
7. Save the final submission CSV.

## Notes

The dataset used in this project is from Kaggle's Playground Series. Competition data should be downloaded directly from Kaggle according to the competition rules.
