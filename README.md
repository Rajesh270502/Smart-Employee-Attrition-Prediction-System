# Smart Employee Attrition Prediction System

## Project Overview

This project focuses on predicting employee attrition using machine learning techniques. The system analyzes employee-related information such as job role, work environment, income, overtime, promotions, and satisfaction levels to identify employees who are likely to leave the organization.

The project also includes:

* Data preprocessing
* Feature engineering
* Class balancing using SMOTE
* Multiple machine learning models
* Model evaluation
* Explainable AI techniques
* Risk score generation
* Employee retention recommendation system
* Model deployment preparation

---

# Dataset Information

The dataset contains employee-related records used for attrition prediction.

Initial dataset information:

* Total Rows: 1470
* Total Columns: 35

The dataset includes both numerical and categorical features.

Important columns include:

* Age
* Attrition
* BusinessTravel
* Department
* DistanceFromHome
* Education
* EnvironmentSatisfaction
* Gender
* JobRole
* JobSatisfaction
* MaritalStatus
* MonthlyIncome
* OverTime
* TotalWorkingYears
* WorkLifeBalance
* YearsAtCompany
* YearsSinceLastPromotion
* YearsWithCurrManager

Dataset inspection was performed using:

```python
df.info()
df.shape
df.isnull().sum()
```

Output Summary:

* Dataset contained 1470 employee records.
* Dataset contained 35 columns.
* No missing values were present.
* Numerical and categorical features were identified successfully.

---

# Step 1: Importing Required Libraries

The following libraries were imported:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Machine learning libraries:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
```

Additional libraries:

```python
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import joblib
```

---

# Step 2: Data Loading

The dataset was loaded using:

```python
df = pd.read_csv("Employee Attrition.csv")
```

The dataset structure and data types were verified using:

```python
df.info()
```

---

# Step 3: Checking Missing Values

Missing values were checked using:

```python
df.isnull().sum()
```

Output:

* No missing values were found in the dataset.
* All columns contained complete records.

---

# Step 4: Removing Duplicate Records

Duplicate records were removed using:

```python
df = df.drop_duplicates()
```

Output:

```python
Dataset Shape After Removing Duplicates: (1470, 35)
```

Observation:

* No duplicate rows were found.
* Dataset shape remained unchanged.

---

# Step 5: Dropping Unnecessary Columns

The following columns were removed:

```python
drop_cols = [
    'EmployeeCount',
    'StandardHours',
    'Over18',
    'EmployeeNumber'
]
```

Reason:

These columns do not contribute significantly to attrition prediction.

---

# Step 6: Feature Engineering

Several new features were created to improve model performance.

## 1. OverTimeRisk

```python
df['OverTimeRisk'] = (
    (df['OverTime'] == 'Yes').astype(int) *
    df['DistanceFromHome']
)
```

Purpose:

Measures burnout risk based on overtime and commuting distance.

---

## 2. SatisfactionIndex

```python
df['SatisfactionIndex'] = (
    df['EnvironmentSatisfaction'] +
    df['JobSatisfaction']
)
```

Purpose:

Combines employee satisfaction metrics into one feature.

---

## 3. StockOptionWeight

```python
df['StockOptionWeight'] = (
    df['StockOptionLevel'] * df['MonthlyIncome']
)
```

Purpose:

Measures the impact of stock benefits on retention.

---

## 4. ExperienceGap

```python
df['ExperienceGap'] = (
    df['TotalWorkingYears'] -
    df['YearsAtCompany']
)
```

Purpose:

Measures difference between total experience and company experience.

---

## 5. CareerStability

```python
df['CareerStability'] = (
    df['YearsInCurrentRole'] /
    (df['YearsAtCompany'] + 1)
)
```

Purpose:

Measures role stability within the company.

---

## 6. CareerGrowthRisk

```python
df['CareerGrowthRisk'] = (
    df['YearsSinceLastPromotion'] /
    (df['JobLevel'] + 1)
)
```

Purpose:

Identifies employees with slower career growth.

---

## 7. UnderPromotionRisk

```python
df['UnderPromotionRisk'] = (
    df['TotalWorkingYears'] /
    (df['JobLevel'] + 1)
)
```

Purpose:

Measures risk of under-promotion.

---

## 8. BurnoutRisk

```python
df['BurnoutRisk'] = (
    df['JobLevel'] *
    df['OverTime'] *
    (df['DistanceFromHome'] + 1)
)
```

Purpose:

Captures burnout probability using overtime, job level, and commuting distance.

---

# Step 7: Encoding Categorical Features

Categorical columns were converted into numerical format using Label Encoding.

```python
from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
```

Purpose:

Machine learning algorithms require numerical input.

---

# Step 8: Feature and Target Separation

Features and target variable were separated.

```python
X = df.drop('Attrition', axis=1)
y = df['Attrition']
```

Target Variable:

* Attrition

Classes:

* 0 = Employee Stays
* 1 = Employee Leaves

---

# Step 9: Handling Class Imbalance Using SMOTE

SMOTE was applied to balance the dataset.

```python
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

Output:

```python
After SMOTE:
Attrition
1    1233
0    1233
```

Observation:

* Dataset became perfectly balanced.
* Both classes contained 1233 records.

---

# Step 10: Feature Scaling

Standardization was performed using StandardScaler.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)
```

Purpose:

Ensures all features are on a similar scale.

---

# Step 11: Train-Test Split

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_resampled,
    test_size=0.2,
    random_state=42
)
```

Output:

```python
Training Shape: (1972, 38)
Testing Shape: (494, 38)
```

---

# Step 12: Experience Category Feature

A new categorical feature was created.

```python
df['ExperienceCategory'] = pd.cut(
    df['YearsAtCompany'],
    bins=[0, 3, 7, 15, 40],
    labels=['Junior', 'Mid', 'Senior', 'Expert']
)
```

Purpose:

Groups employees based on company experience.

Categories:

* Junior
* Mid
* Senior
* Expert

---

# Step 13: Data Visualization

Attrition distribution was visualized using Seaborn.

```python
sns.countplot(x=y_resampled)
```

Observation:

* The balanced dataset showed equal representation of attrition classes.

---

# Step 14: Machine Learning Models Used

The following machine learning models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. XGBoost

---

# Step 15: Model Evaluation

Performance metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

Model comparison results:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 86.43%   | 88.64%    | 83.19% | 85.84%   | 93.68%  |
| Decision Tree       | 87.24%   | 85.49%    | 89.34% | 87.37%   | 87.27%  |
| Random Forest       | 92.51%   | 91.90%    | 93.03% | 92.46%   | 97.79%  |
| SVM                 | 91.49%   | 92.80%    | 89.75% | 91.25%   | 96.57%  |
| XGBoost             | 94.13%   | 94.24%    | 93.85% | 94.05%   | 97.87%  |

Observation:

* XGBoost achieved the best overall performance.
* Random Forest also produced strong results.
* Logistic Regression produced the lowest accuracy among tested models.

---

# Step 16: Best Model Selection

XGBoost was selected as the final model.

```python
best_model = XGBClassifier(eval_metric='logloss')
```

Reason:

XGBoost achieved the highest:

* Accuracy
* Precision
* F1 Score
* ROC-AUC

Final Performance:

```python
Accuracy : 94.13 %
Precision: 94.24 %
Recall   : 93.85 %
F1 Score : 94.05 %
ROC-AUC  : 97.87 %
```

---

# Step 17: Confusion Matrix

A confusion matrix was generated to evaluate prediction quality.

Purpose:

* Measures correct and incorrect predictions.
* Helps identify false positives and false negatives.

---

# Step 18: Explainable AI

## Feature Importance

Feature importance analysis was performed using XGBoost.

```python
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
})
```

Purpose:

Identifies the most influential features affecting attrition.

---

## SHAP Explainability

SHAP values were used for model interpretability.

```python
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

Purpose:

Explains how each feature contributes to employee attrition prediction.

---

# Step 19: Risk Score Generation

Risk scores were generated using prediction probabilities.

```python
y_prob = best_model.predict_proba(X_test)[:, 1]
```

Risk categories:

* Low Risk
* Medium Risk
* High Risk

Sample Output:

```python
Risk Score Risk Category
0.020694   Low Risk
0.999250   High Risk
```

Risk Distribution:

```python
Low Risk       246
High Risk      228
Medium Risk     20
```

Observation:

* Most employees were classified into Low Risk or High Risk groups.
* Medium Risk employees were fewer.

---

# Step 20: Employee Attrition Prediction Function

A prediction function was created for new employee data.

```python
def predict_employee(employee_data: dict):
```

Output Example:

```python
Attrition Prediction : YES - Will Leave
Risk Score           : 0.9912
Risk Category        : High Risk
```

Purpose:

Allows prediction for individual employee records.

---

# Step 21: HR Recommendation System

A recommendation engine was created to provide retention strategies.

```python
def generate_recommendations(employee_data: dict, risk_score: float, risk_label: str):
```

Sample Recommendations:

* Reduce overtime hours.
* Offer flexible work options.
* Conduct feedback sessions.
* Improve work-life balance.
* Provide promotion opportunities.
* Introduce wellness programs.
* Improve workplace culture.
* Create personalized retention plans.

Purpose:

Provides actionable HR strategies for employee retention.

---

# Step 22: Model Saving and Deployment Preparation

The trained model and preprocessing objects were saved.

```python
joblib.dump(best_model, 'attrition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')
```

Output:

```python
Model, Scaler, Encoders, and Columns saved successfully!
```

Purpose:

Prepares the project for deployment using Streamlit.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SHAP
* SMOTE
* Joblib
* Streamlit

---

# Final Conclusion

The Smart Employee Attrition Prediction System successfully predicts employee attrition using machine learning techniques.

Key achievements:

* Performed complete data preprocessing.
* Engineered meaningful HR-related features.
* Balanced dataset using SMOTE.
* Evaluated multiple machine learning algorithms.
* Achieved 94.13% accuracy using XGBoost.
* Implemented Explainable AI using SHAP.
* Generated employee risk scores.
* Developed HR retention recommendation system.
* Prepared the project for deployment.

The project can help organizations proactively identify employees at risk of leaving and improve employee retention strategies using data-driven insights.
