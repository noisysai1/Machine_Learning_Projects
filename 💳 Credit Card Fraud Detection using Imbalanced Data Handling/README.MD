📌 Overview

This project builds a machine learning pipeline to detect fraudulent credit card transactions. 
Since fraud cases are extremely rare, the dataset is highly imbalanced, which makes standard modeling approaches ineffective.

This project focuses on solving that challenge using data resampling techniques, proper evaluation metrics and model comparison.

🎯 Problem Statement

Traditional ML models fail on imbalanced datasets because they prioritize majority classes.
Here:

~99.8% transactions → Legitimate
~0.2% transactions → Fraud

Goal:
👉 Build a model that correctly identifies fraud without being biased toward majority data.


🧠 Step-by-Step Implementation

🔹 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

🔹 Step 2: Load Dataset
df = pd.read_csv("creditcard.csv")
df.head()

🔹 Step 3: Data Exploration (EDA)
Dataset Shape
df.shape
Class Distribution
df['Class'].value_counts()


👉 Observation:

Fraud cases are extremely low → Imbalanced dataset
Visualization
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")


🔹 Step 4: Data Preprocessing
Checked for null values
df.isnull().sum()
Feature scaling (if needed for models like Logistic Regression)

                 
🔹 Step 5: Split Data
from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

                 
🔹 Step 6: Handle Imbalanced Data
✅ Method 1: Under-sampling
Reduce majority class samples
✅ Method 2: Over-sampling (SMOTE)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

                 
👉 Result:

Balanced dataset for better learning

🔹 Step 7: Model Training
Logistic Regression
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_resampled, y_resampled)
Random Forest
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_resampled, y_resampled)


🔹 Step 8: Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
Predictions
y_pred = model_rf.predict(X_test)
Metrics
print(classification_report(y_test, y_pred))

                 
📊 Evaluation Metrics Explained
                 
Precision → How many predicted frauds are actually fraud
Recall → How many actual frauds we detected
F1 Score → Balance between Precision & Recall
ROC-AUC → Model's ability to distinguish classes

📈 Results
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	Good	Moderate	High	Balanced
Random Forest	Best	High	High	Strong

🔍 Key Learnings
Imbalanced data can mislead models
Accuracy is not a reliable metric here
SMOTE significantly improves model performance
Ensemble models (Random Forest) perform better

⚠️ Challenges Faced
Severe class imbalance
Avoiding overfitting after oversampling
Choosing correct evaluation metrics

🚀 Future Enhancements
Implement XGBoost / LightGBM
Perform hyperparameter tuning
Build real-time fraud detection API
Deploy using Flask / FastAPI
Integrate dashboard (Power BI / Streamlit)

📁 Project Structure
├── creditcard.csv
├── Hanling_Imbalanced_Data.ipynb
├── README.md
