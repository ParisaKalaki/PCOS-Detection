# 🧠 PCOS Detection Using Machine Learning

A supervised machine learning pipeline designed to predict the likelihood of **Polycystic Ovary Syndrome (PCOS)** based on clinical and lifestyle data. This project aims to support early detection and provide interpretable insights for medical professionals and researchers.

---

## 📊 Overview

- Cleaned and preprocessed the dataset (handled outliers, skewness, and missing values).
- Performed **feature selection** using:
  - Point-biserial correlation
  - Variance Inflation Factor (VIF)
- Trained and compared multiple models:
  - Logistic Regression
  - Decision Tree
  - Random Forest (best performance)
- Applied **SMOTE** to balance classes and reduce false negatives.
- Evaluated models using key metrics: recall, F1-score, and AUC-ROC.

---

## 🧪 Technologies Used

- **Programming Language:** Python
- **Libraries:** `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn (SMOTE)`
- **Environment:** Jupyter Notebook

---

## 📈 Exploratory Data Analysis (EDA)

### Correlation

![Correlation](plots/Correlation.png)

### imbalanced target

![imbalanced-target](plots/imbalanced-target.png)

---

## 🔍 Feature Selection

### Point-Biserial Correlation Heatmap

![point-biserial-correlation](plots/point-biserial-correlation.png)

---

## 🤖 Model Performance

| Model              | Recall   | F1-Score | AUC-ROC  |
| ------------------ | -------- | -------- | -------- |
| LogisticRegression | 0.78     | 0.76     | 0.83     |
| Decision Tree      | 0.81     | 0.77     | 0.85     |
| Random Forest      | **0.86** | **0.82** | **0.89** |

> Random Forest performed best in identifying positive PCOS cases with the highest recall and overall balanced metrics.

---

## 🔍 Feature Importance (Random Forest)

![Feature Importance](plots/feature_importance.png)

---

## Confusion Matrix (Random Forest)

![RF-confusion-matrix](plots/RF-confusion-matrix.png)

---

## ROC (Random Forest)

![ROC](plots/ROC.png)

---

## Precision Recall (Random Forest)

![RF-precision-recall](RF-precision-recall.png)

---

## Learning Curve (Random Forest)

![RF-learning-curve](plots/RF-learning-curve.png)

---

## 📁 Project Structure

```
pcos-ml-detection/
│
├── data/
│ └── PCOS_data.csv
├── notebooks/
│ └── pcos_analysis.ipynb
├── plots/
│ ├── correlation.png
│ ├── feature-importance.png
│ └── ...
└── README.md
```

---

## 📌 Key Takeaways

- **Feature Selection** improved accuracy and interpretability by reducing noise.
- **Preprocessing** steps like outlier handling, skew correction, and SMOTE balanced the dataset.
- **Scaling** was essential for linear models to perform effectively.
- **Visualizations** (box plots, bar charts, pie charts) helped reveal data patterns clearly.
- **Hyperparameter Tuning** boosted performance, but required care to avoid overfitting.
- **Collaboration** was enhanced through clear notebook comments and task distribution.
- **Recall-Focused Evaluation** minimized false negatives—critical for medical predictions.

---

## 📝 Dataset

The dataset was sourced from Kaggle: [PCOS Dataset](<[https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)>)

---
