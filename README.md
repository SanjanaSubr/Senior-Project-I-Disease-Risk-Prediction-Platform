# LifeFactors – Machine Learning-Based Disease Risk Prediction Web Application

LifeFactors is a machine learning-powered health risk prediction system developed as our Senior Project (Thesis I) at Assumption University.

The system predicts the probability of:
- Diabetes
- Heart Disease (Myocardial Infarction / Coronary Heart Disease)

The goal was to build an end-to-end ML pipeline — from dataset preprocessing and model selection to deployment within a functional web application.

---

## Project Objective

Traditional disease diagnosis often requires invasive and resource-intensive medical procedures.  
LifeFactors explores whether supervised machine learning models can:

- Identify high-risk individuals earlier
- Provide non-invasive probability-based screening
- Support preventative healthcare decisions using behavioral and biometric data

This system is intended as a predictive aid — not a diagnostic tool.

---

## Datasets Used

### Diabetes (BRFSS 2015)
- 254,680 training samples
- 21 health-related features
- Balanced unseen dataset: 70,692 samples
- Sampled 50,000 rows for computational efficiency

Features included BMI, blood pressure, cholesterol, age category, physical activity, mental health, smoking, income, education, etc.

---

### Heart Disease (Two Approaches)

1. Clinical bloodwork dataset (15 medical features)
2. BRFSS 2020 lifestyle-based dataset (40,000 balanced samples)

This allowed comparison between:
- Clinical diagnostic features
- Behavioral risk factor prediction

---

## Machine Learning Pipeline

### Feature Engineering
- Information Value (IV) & Weight of Evidence (WOE) analysis
- Quantile binning for continuous variables
- Feature selection based on predictive strength
- StandardScaler (z-score normalization)

### Algorithms Compared

- Random Forest
- Logistic Regression
- Support Vector Machine
- Gradient Boosting
- Gaussian Naïve Bayes
- K-Nearest Neighbors

### Hyperparameter Tuning
- GridSearchCV for cross-validation
- Tuned `n_estimators`, `max_depth`, `max_features`
- SVM regularization parameter (C), kernel selection
- Logistic regression solver (saga) for large datasets

Best-performing models were serialized using `joblib` and deployed in the web application backend.

---

## Results

### Diabetes Prediction (BRFSS 2015)

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **75.64%** |
| SVM | 75.33% |
| Gradient Boosting | 74.85% |
| Random Forest | 74.48% |
| Naïve Bayes | 71.69% |
| KNN | 69.88% |

Top performer: **Logistic Regression (75.64%)**

---

### Heart Disease – Clinical Bloodwork Dataset

| Model | Accuracy |
|-------|----------|
| Support Vector Machine | **88.52%** |
| Gradient Boosting | 86.89% |
| Random Forest | 85.24% |
| Naïve Bayes | 81.97% |
| Logistic Regression | 78.69% |
| KNN | 70.49% |

Top performer: **Support Vector Machine (88.52%)**

---

### Heart Disease – BRFSS 2020 (Lifestyle Dataset)

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **78.64%** |
| Random Forest | 78.55% |
| SVM | 78.40% |
| Gradient Boosting | 78.20% |
| Naïve Bayes | 72.20% |
| KNN | 69.58% |

Top performer: **Logistic Regression (78.64%)**

---

## System Architecture

1. User inputs health indicators via web interface
2. Flask backend processes and scales inputs
3. Serialized ML model loads dynamically
4. Model returns probability-based prediction
5. Risk result is displayed to user

---

## Technical Stack

- Python
- Scikit-learn
- Pandas / NumPy
- Flask
- HTML / CSS
- Joblib (model persistence)

---

## Academic Presentation

This project was formally presented at the Senior Project Exhibition under the supervision of:

Dr. Anilkumar Kothalil Gopalakrishnan  
Department of Computer Science, Assumption University

### Senior Project Exhibition Poster
<p align="center">
  <img src="Exhibit Poster.png" width="800">
</p>

---

## What This Project Demonstrates

- Applied supervised machine learning in healthcare
- Feature importance analysis using IV/WOE
- Hyperparameter optimization with cross-validation
- Comparative model evaluation
- End-to-end ML deployment in a web application
- Research-driven development in a collaborative team

---

LifeFactors reflects the integration of machine learning and healthcare analytics, demonstrating how predictive modeling can support early risk identification in chronic diseases.
