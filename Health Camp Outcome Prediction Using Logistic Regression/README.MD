# 🏥 Healthcare Analytics – Health Camp Outcome Prediction

## 📌 Project Overview

This project applies **data preprocessing, feature engineering, data integration, feature scaling and  machine learning** to healthcare camp data in order to build a binary classification model for predicting patient health-camp outcomes.

The dataset contains information about patients registering for multiple health camps, along with anonymized registration variables and attendance/outcome information collected from three different types of health camps.

Because the three health-camp formats record outcomes differently, the project first converts their respective outcome measurements into a common binary variable called `hs_binary`. 
The processed attendance information is then merged with patient registration data and used to train a **Logistic Regression** classification model.

The project demonstrates an end-to-end machine learning workflow involving:

- Data loading and inspection
- Data cleaning
- Target-variable engineering
- Dataset integration
- Missing-value handling
- Date-based feature engineering
- Feature selection
- Feature scaling
- Train/test splitting
- Logistic Regression modeling
- Model evaluation
- Preparation of unseen test data for prediction

---

## 🎯 Project Objective

The main objective is to transform different health-camp outcome measurements into a common binary classification problem and investigate whether information available when a patient registers for a health camp can be used to predict the resulting health-camp outcome.

The model uses the anonymized registration variables:

`Var1`, `Var2`, `Var3`, `Var4`, `Var5`

along with a newly engineered registration-month feature:

`Months`

to predict:

`hs_binary`

---

## 📂 Dataset Description

The project contains multiple datasets representing registrations, patients, health camps, attendance records, test observations and  submission data.

### `Train(1).csv`

Main training registration dataset.

**Shape:** `75,278 rows × 8 columns`

Contains:

- `Patient_ID`
- `Health_Camp_ID`
- `Registration_Date`
- `Var1`
- `Var2`
- `Var3`
- `Var4`
- `Var5`

`Var1` through `Var5` are anonymized discrete numerical variables captured at registration.

There are **334 missing Registration_Date values** in this file.

---

### `First_Health_Camp_Attended.csv`

Contains attendance information for the first health-camp format.

**Shape:** `6,218 rows × 5 columns`

Features include:

- `Patient_ID`
- `Health_Camp_ID`
- `Donation`
- `Health_Score`
- `Unnamed: 4`

For modeling, `Donation` and the empty `Unnamed: 4` column are removed.

The continuous `Health_Score` is converted into the binary target.

```text
Health_Score > Mean Health_Score → 1
Health_Score <= Mean Health_Score → 0
```

The mean Health Score in this dataset is approximately:

```text
0.5177
```

Resulting target distribution:

```text
Class 0: 3,052
Class 1: 3,166
```

---

### `Second_Health_Camp_Attended.csv`

Contains attendance information for the second health-camp format.

**Shape:** `7,819 rows × 3 columns`

Features:

- `Patient_ID`
- `Health_Camp_ID`
- `Health Score`

The Health Score is again converted into a binary target using its dataset mean.

```text
Health Score > Mean Health Score → 1
Health Score <= Mean Health Score → 0
```

Mean Health Score:

```text
0.5542
```

Resulting target distribution:

```text
Class 0: 4,184
Class 1: 3,635
```

---

### `Third_Health_Camp_Attended.csv`

Contains information from the third type of health camp.

**Shape:** `6,515 rows × 4 columns`

Features:

- `Patient_ID`
- `Health_Camp_ID`
- `Number_of_stall_visited`
- `Last_Stall_Visited_Number`

For this camp format, the binary target is created from the number of stalls visited:

```text
Number_of_stall_visited == 0 → 0
Number_of_stall_visited > 0  → 1
```

Target distribution:

```text
Class 0:    18
Class 1: 6,497
```

`Last_Stall_Visited_Number` is removed before modeling.

---

### `Patient_Profile.csv`

Contains demographic and profile information about patients.

**Shape:** `37,633 rows × 11 columns`

Available variables include:

- `Patient_ID`
- `Online_Follower`
- `LinkedIn_Shared`
- `Twitter_Shared`
- `Facebook_Shared`
- `Income`
- `Education_Score`
- `Age`
- `First_Interaction`
- `City_Type`
- `Employer_Category`

> **Note:** Although this dataset is supplied with the project, the current notebook does **not merge or use these patient-profile features in the Logistic Regression model**.

---

### `Health_Camp_Detail.csv`

Contains metadata describing individual health camps.

**Shape:** `65 rows × 6 columns`

Features:

- `Health_Camp_ID`
- `Camp_Start_Date`
- `Camp_End_Date`
- `Category1`
- `Category2`
- `Category3`

> **Note:** The current notebook does **not use these camp-level features during model training**.

---

### `test.csv`

Unseen registration data intended for prediction.

**Shape:** `35,249 rows × 8 columns`

Contains:

- `Patient_ID`
- `Health_Camp_ID`
- `Registration_Date`
- `Var1`
- `Var2`
- `Var3`
- `Var4`
- `Var5`

The supplied `test.csv` and `test(1).csv` contain the same data.

---

### `sample_submmission.csv`

Provides the expected prediction/submission structure.

**Shape:** `35,249 rows × 3 columns`

Columns:

```text
Patient_ID
Health_Camp_ID
Outcome
```

The provided sample initializes `Outcome` to `0.5`.

---

### `Data_Dictionary.xlsx`

Provides descriptions for the supplied datasets and their fields, including:

- Health Camp details
- Training registration data
- Patient profiles
- First health-camp attendance
- Second health-camp attendance
- Third health-camp attendance
- Test registration data

---

# 🔄 Machine Learning Workflow

The overall workflow implemented in the notebook is:

```text
        Health Camp Attendance Data
                   │
       ┌───────────┼───────────┐
       │           │           │
    Camp 1      Camp 2      Camp 3
       │           │           │
 Health Score  Health Score  Stall Visits
       │           │           │
       └────── Binary Target ───┘
                   │
             hs_binary
                   │
                   ▼
       Combine Attendance Records
                   │
                   ▼
       Merge with Registration Data
                   │
                   ▼
          Remove Missing Records
                   │
                   ▼
     Registration Date Processing
                   │
                   ▼
        Extract Registration Month
                   │
                   ▼
             Feature Selection
                   │
                   ▼
             Feature Scaling
                   │
                   ▼
           75% Train / 25% Test
                   │
                   ▼
          Logistic Regression
                   │
                   ▼
            Model Evaluation
                   │
                   ▼
        Prepare Unseen Test Data
```

---

# 🧹 1. Data Cleaning

The three attendance datasets have different structures, so dataset-specific cleaning is performed.

For the first health-camp dataset:

```python
df1.drop(['Donation', 'Unnamed: 4'], axis=1, inplace=True)
```

For the third health-camp dataset:

```python
df3.drop(['Last_Stall_Visited_Number'], axis=1, inplace=True)
```

The original continuous outcome columns are removed after the binary target is created.

---

# 🎯 2. Target Variable Engineering

A common target called:

```text
hs_binary
```

is created so that the different camp formats can be represented as a binary classification task.

For the first and second health camps, patients with health scores above the corresponding dataset mean receive class `1`; otherwise they receive class `0`.

For the third health-camp format, visiting at least one stall corresponds to class `1`, while visiting zero stalls corresponds to class `0`.

This step converts heterogeneous health-camp outcome measurements into a common target representation.

---

# 🔗 3. Dataset Integration

The processed attendance datasets are concatenated.

The resulting records are then joined with the main registration dataset using:

```python
Patient_ID
Health_Camp_ID
```

with an inner merge.

Conceptually:

```text
Attendance/Outcome Data
          +
Registration Information
          │
          ▼
 Patient_ID + Health_Camp_ID
          │
          ▼
 Combined Modeling Dataset
```

---

# 🧽 4. Missing Value Handling

After merging the data, missing values are inspected using:

```python
df_new.isnull().sum()
```

The notebook then removes rows containing missing values:

```python
df_ = df_new.dropna(axis=0)
```

The final modeling dataset produced by the notebook contains:

```text
7,669 observations
```

with target distribution:

```text
Class 0: 4,090
Class 1: 3,579
```

---

# 📅 5. Date Feature Engineering

`Registration_Date` is converted from text into a Pandas datetime object.

```python
pd.to_datetime(
    df_['Registration_Date'],
    format='%d-%b-%y'
)
```

The month is then extracted and stored in a new variable:

```text
Months
```

For example:

```text
13-Feb-05 → 2
13-Jan-05 → 1
10-Feb-05 → 2
```

After extracting the month, the original `Registration_Date` field is removed.

This transforms the registration date into a numerical temporal feature that can be consumed by the model.

---

# 🔍 6. Feature Selection

The final modeling table is restricted to:

```text
Patient_ID
Health_Camp_ID
Var1
Var2
Var3
Var4
Var5
Months
hs_binary
```

However, the IDs are not supplied to the model.

The actual independent variables are:

```text
Var1
Var2
Var3
Var4
Var5
Months
```

Target:

```text
hs_binary
```

Therefore:

```python
X = [Var1, Var2, Var3, Var4, Var5, Months]
y = hs_binary
```

This is consistent with the data dictionary's warning that `Patient_ID` and `Health_Camp_ID` are identifiers rather than sequential modeling variables.

---

# ⚖️ 7. Feature Scaling

The project defines a custom Scikit-learn transformer using:

```python
BaseEstimator
TransformerMixin
StandardScaler
```

The custom transformer standardizes selected numerical features while preserving the original DataFrame column order.

The variables selected for scaling are:

```text
Var1
Var2
Var3
Var4
Var5
```

while:

```text
Months
```

is intentionally excluded from scaling.

Standardization conceptually transforms a variable using:

```text
z = (x - mean) / standard deviation
```

This places numerical features onto comparable scales.

---

# ✂️ 8. Train/Test Split

The processed dataset is divided into training and validation subsets.

```python
train_test_split(
    sc_inp,
    y,
    train_size=0.75,
    random_state=0
)
```

The split is:

```text
75% Training
25% Testing
```

Resulting shapes:

```text
X_train: (5751, 6)
X_test : (1918, 6)
```

Using `random_state=0` makes the split reproducible.

---

# 🤖 9. Logistic Regression

A Logistic Regression classifier is created using Scikit-learn.

```python
lg = LogisticRegression()
lg.fit(X_train, y_train)
```

Logistic Regression is appropriate for this workflow because the engineered target has two possible classes:

```text
0
1
```

The notebook reports a training score of approximately:

```text
53.38%
```

---

# 🧪 10. Unseen Test Data Preparation

The supplied test dataset contains:

```text
35,249 observations
```

Its `Registration_Date` field is processed in the same general way by converting it to datetime and extracting the month.

The final selected test features are:

```text
Var1
Var2
Var3
Var4
Var5
Months
```

The notebook then calls:

```python
lg.predict_proba(test_df)
```

to obtain Logistic Regression probability predictions.

---

# 📊 Notebook-Reported Accuracy

The final notebook output displays:

```text
Accuracy Score: 0.5375391032325338
```

or approximately:

```text
53.75%
```

However, this value requires an important qualification.

Immediately before calculating accuracy, the notebook executes:

```python
pred = np.argmax(y_test, axis=1)
```

This overwrites the predictions previously generated using:

```python
lg.predict_proba(test_df)
```

Therefore, the displayed **53.75% value should not be interpreted as a reliable Logistic Regression validation accuracy**.

A corrected evaluation should instead use:

```python
y_pred = lg.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
```

---

# ⚠️ Important Implementation Notes

Reviewing the complete notebook reveals several areas that should be corrected before treating the project as a production-ready predictive model.

### 1. First health-camp data is accidentally omitted during concatenation

After processing `df1`, the notebook contains:

```python
df1_copy = df2.copy()
```

instead of copying the processed first dataset.

As a result, the first attendance dataset is not incorporated as originally intended, while second-camp information is effectively duplicated in the concatenation process.

The intended statement was likely:

```python
df1_copy = df1.copy()
```

---

### 2. Prediction variable is overwritten

The notebook first generates probabilities:

```python
pred = lg.predict_proba(test_df)
```

but then replaces `pred` with:

```python
pred = np.argmax(y_test, axis=1)
```

Consequently, the final reported accuracy is not based on standard model predictions against `X_test`.

---

### 3. Training and unseen-test preprocessing are inconsistent

Training features are passed through the custom StandardScaler transformer, but the final unseen test features are sent directly to:

```python
lg.predict_proba(test_df)
```

without applying the fitted scaler.

The same preprocessing fitted on the training data should also be applied to test data.

A Scikit-learn `Pipeline` would be a safer implementation.

---

### 4. Data leakage in scaling

The scaler is fitted before the train/test split:

```text
Complete dataset
      ↓
Fit StandardScaler
      ↓
Train/Test split
```

A stronger machine-learning workflow is:

```text
Train/Test split
      ↓
Fit scaler ONLY on X_train
      ↓
Transform X_train
      ↓
Transform X_test using same scaler
```

This prevents information from the validation data from influencing preprocessing.

---

### 5. Available datasets are underutilized

`Patient_Profile.csv` contains potentially useful information such as:

```text
Age
Income
Education_Score
Online_Follower
LinkedIn_Shared
Twitter_Shared
Facebook_Shared
City_Type
Employer_Category
```

Similarly, `Health_Camp_Detail.csv` provides:

```text
Camp_Start_Date
Camp_End_Date
Category1
Category2
Category3
```

These datasets are included with the project but are not used in the current notebook.

Incorporating them could provide substantially richer features.

---

### 6. Missing values are simply discarded

The notebook uses:

```python
dropna()
```

which removes incomplete observations.

For an improved model, numerical and categorical imputation strategies could be investigated rather than automatically discarding observations.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Statistical / Machine Learning

- Statsmodels
- Scikit-learn

### Scikit-learn Components

- `BaseEstimator`
- `TransformerMixin`
- `StandardScaler`
- `train_test_split`
- `LogisticRegression`
- `metrics`

### Development Environment

- Jupyter Notebook

---

# 📁 Suggested Repository Structure

```text
healthcare-analytics/
│
├── healthcare_analytics.ipynb
│
├── README.md
│
├── data/
│   ├── Train.csv
│   ├── test.csv
│   ├── Patient_Profile.csv
│   ├── Health_Camp_Detail.csv
│   ├── First_Health_Camp_Attended.csv
│   ├── Second_Health_Camp_Attended.csv
│   ├── Third_Health_Camp_Attended.csv
│   ├── sample_submission.csv
│   └── Data_Dictionary.xlsx
│
└── requirements.txt
```

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd healthcare-analytics
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn jupyter
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
healthcare_analytics.ipynb
```

and execute the cells sequentially.

> The paths inside the original notebook use a `Train/` directory. Adjust the dataset paths if the repository uses a different folder structure.

---

# 💡 Potential Improvements

The current notebook provides a baseline classification workflow. Future improvements could include:

- Correcting the attendance-dataset concatenation
- Creating a unified and clearly defined target
- Using a Scikit-learn `Pipeline`
- Fitting preprocessing only on training data
- Applying identical transformations to validation and test data
- Incorporating `Patient_Profile.csv`
- Incorporating `Health_Camp_Detail.csv`
- Handling missing data through imputation
- Engineering patient-history features
- Engineering camp-duration features
- Creating registration-to-camp time-gap features
- Encoding categorical variables
- Comparing multiple classification algorithms
- Performing cross-validation
- Hyperparameter tuning
- Evaluating ROC-AUC, precision, recall and F1-score
- Handling class imbalance where necessary
- Performing feature-importance analysis
- Generating a final submission file using predicted probabilities

Potential models for comparison include:

```text
Logistic Regression
Random Forest
Gradient Boosting
XGBoost / LightGBM
```

---

# 📈 Key Project Takeaways

This project demonstrates how multiple healthcare datasets with different outcome definitions can be combined into a machine-learning workflow.

The implementation covers several core data-science concepts:

**Data Cleaning** → removing irrelevant fields and incomplete observations.

**Target Engineering** → converting different health-camp measurements into a common binary outcome.

**Data Integration** → joining attendance information with registration records using patient and camp identifiers.

**Feature Engineering** → extracting registration month from date information.

**Feature Scaling** → standardizing numerical registration variables.

**Machine Learning** → training a Logistic Regression classifier.

**Evaluation** → establishing the framework for evaluating classification performance.

The project also highlights an important real-world data-science lesson: **a machine-learning pipeline must be validated not only by its final metric, but also by carefully checking data joins, preprocessing consistency, target construction and  evaluation logic.**

---
