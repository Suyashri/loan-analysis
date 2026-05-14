# =========================================================
# LOAN APPROVAL PREDICTION PROJECT
# =========================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# LOAD DATASET
# -----------------------------
# Replace with your dataset path
import pandas as pd
df = pd.read_csv("loan_data.csv")

# -----------------------------
# BASIC INFORMATION
# -----------------------------

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# =========================================================
# DATA CLEANING
# =========================================================

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# =========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

# -----------------------------
# LOAN APPROVAL COUNT
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Distribution")
plt.show()

# -----------------------------
# GENDER VS LOAN STATUS
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Loan_Status', data=df)
plt.title("Gender vs Loan Status")
plt.show()

# -----------------------------
# EDUCATION VS LOAN STATUS
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.title("Education vs Loan Status")
plt.show()

# -----------------------------
# APPLICANT INCOME DISTRIBUTION
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['ApplicantIncome'], bins=30, kde=True)
plt.title("Applicant Income Distribution")
plt.show()

# -----------------------------
# CREDIT HISTORY VS LOAN STATUS
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Credit History vs Loan Status")
plt.show()

# =========================================================
# DATA PREPROCESSING
# =========================================================

# -----------------------------
# LABEL ENCODING
# -----------------------------
label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# MODEL BUILDING
# =========================================================

# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================================
# DECISION TREE
# =========================================================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))

# =========================================================
# RANDOM FOREST
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10,6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")
plt.show()

# =========================================================
# FINAL INSIGHTS
# =========================================================

print("\n===== PROJECT INSIGHTS =====")

print("""
1. Credit history plays a major role in loan approval.
2. Applicant income influences approval chances.
3. Education and marital status may impact approvals.
4. Random Forest generally performs better for prediction.
5. Data preprocessing and feature engineering improve accuracy.
""")