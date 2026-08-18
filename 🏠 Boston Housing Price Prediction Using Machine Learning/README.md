# 🏠 Boston Housing Price Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting the **median value of houses in Boston** using multiple Machine Learning regression algorithms.

The project uses the **Boston Housing Dataset**, which contains information about housing and neighborhood characteristics across different areas of Boston. 
The objective is to understand the factors that influence housing prices and build regression models capable of predicting the median home value.

The complete Machine Learning workflow includes:

* Dataset exploration
* Data preprocessing
* Missing-value analysis
* Exploratory Data Analysis (EDA)
* Outlier analysis
* Feature distribution analysis
* Min-Max Normalization
* Feature Standardization
* Correlation analysis
* Feature selection
* Train-test splitting
* Regression model development
* Cross-validation
* Model evaluation
* Feature importance analysis
* Model comparison

---

# 🎯 Project Objective

The primary objective of this project is to develop Machine Learning regression models that can estimate Boston housing prices based on various socioeconomic, environmental
and housing-related characteristics.

The project also compares several regression algorithms to understand which model performs best on the dataset.

The target variable is:

**MEDV – Median value of owner-occupied homes in $1000s.**

---

# 📊 Dataset Information

The Boston Housing Dataset contains:

* **506 observations**
* **13 predictor variables**
* **1 target variable**
* **MEDV** as the target variable

The original CSV also contains an index-like `Unnamed: 0` column, which is removed during preprocessing.

After removing this unnecessary column, the working dataset contains **14 columns**.

---

# 📚 Dataset Features

| Feature       | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `CRIM`        | Per capita crime rate by town                                      |
| `ZN`          | Proportion of residential land zoned for lots over 25,000 sq. ft.  |
| `INDUS`       | Proportion of non-retail business acres per town                   |
| `CHAS`        | Charles River dummy variable: 1 if tract bounds river, otherwise 0 |
| `NOX`         | Nitric oxide concentration                                         |
| `RM`          | Average number of rooms per dwelling                               |
| `AGE`         | Proportion of owner-occupied units built before 1940               |
| `DIS`         | Weighted distance to five Boston employment centers                |
| `RAD`         | Index of accessibility to radial highways                          |
| `TAX`         | Full-value property tax rate per $10,000                           |
| `PTRATIO`     | Pupil-teacher ratio by town                                        |
| `B` / `BLACK` | Demographic variable included in the original Boston dataset       |
| `LSTAT`       | Percentage of lower-status population                              |
| `MEDV`        | Median value of owner-occupied homes in $1000s                     |

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

## Machine Learning Algorithms

The following regression algorithms were implemented:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Extra Trees Regressor
5. XGBoost Regressor

---

# 📁 Project Structure

```text
Boston-Housing-Price-Prediction/
│
├── Boston Housing Prediction - Regression.ipynb
├── Boston Dataset.csv
├── README.md
└── requirements.txt
```

### File Description

**Boston Housing Prediction - Regression.ipynb**

Contains the complete Machine Learning workflow, including preprocessing, visualization, model training, cross-validation, and evaluation.

**Boston Dataset.csv**

Contains the housing dataset used to train and evaluate the regression models.

**README.md**

Provides complete project documentation.

---

# 🔄 Project Workflow

```text
Boston Housing Dataset
        ↓
Data Loading
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Missing Value Analysis
        ↓
Exploratory Data Analysis
        ↓
Outlier Detection
        ↓
Distribution Analysis
        ↓
Normalization / Standardization
        ↓
Correlation Analysis
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Cross Validation
        ↓
Model Evaluation
        ↓
Feature Importance
        ↓
Model Comparison
```

---

# 1️⃣ Importing Required Libraries

The project begins by importing the libraries required for data manipulation, visualization, preprocessing, and Machine Learning.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
```

The notebook also uses Scikit-learn and XGBoost for model development.

---

# 2️⃣ Loading the Dataset

The Boston Housing dataset is loaded using Pandas.

```python
df = pd.read_csv("Boston Dataset.csv")
```

The unnecessary index column is removed:

```python
df.drop(columns=['Unnamed: 0'], axis=0, inplace=True)
```

The first few observations are then examined using:

```python
df.head()
```

This provides an initial understanding of the structure and values contained in the dataset.

---

# 3️⃣ Understanding the Dataset

Before building Machine Learning models, the dataset is analyzed using descriptive statistics and data-type information.

```python
df.describe()
```

This provides information such as:

* Count
* Mean
* Standard deviation
* Minimum
* Maximum
* 25th percentile
* Median
* 75th percentile

The dataset structure is also inspected using:

```python
df.info()
```

The processed dataset contains **506 rows and 14 variables**.

---

# 4️⃣ Missing Value Analysis

Missing values can negatively affect Machine Learning algorithms.

Therefore, every feature is checked using:

```python
df.isnull().sum()
```

### Result

The dataset contains **no missing values** across the 14 working columns.

Therefore, no missing-value imputation was required.

---

# 5️⃣ Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the dataset before training the models.

EDA helps identify:

* Outliers
* Feature distributions
* Relationships between variables
* Potential correlations
* Variables that may influence house prices

---

# 6️⃣ Outlier Analysis Using Boxplots

Boxplots were created for all features.

```python
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1

plt.tight_layout()
```

### Purpose

Boxplots help identify:

* Extreme observations
* Median values
* Data spread
* Interquartile ranges
* Potential outliers

Some Boston Housing variables contain highly skewed observations, making this step useful for understanding the dataset.

---

# 7️⃣ Feature Distribution Analysis

Distribution plots were generated for every variable.

```python
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.distplot(value, ax=ax[index])
    index += 1

plt.tight_layout()
```

Distribution analysis helps identify whether variables are:

* Normally distributed
* Positively skewed
* Negatively skewed
* Concentrated around particular values
* Affected by extreme observations

---

# 8️⃣ Min-Max Normalization

Selected features were normalized:

```python
cols = ['crim', 'zn', 'tax', 'black']
```

Min-Max Normalization follows:

```text
Xnormalized = (X - Xmin) / (Xmax - Xmin)
```

The notebook performs the transformation using:

```python
minimum = min(df[col])
maximum = max(df[col])

df[col] = (df[col] - minimum) / (maximum - minimum)
```

### Why Normalize Data?

Different variables can have very different numerical ranges.

Normalization helps place selected features on a comparable scale.

---

# 9️⃣ Standardization

The selected variables were subsequently standardized using Scikit-learn's `StandardScaler`.

```python
from sklearn import preprocessing

scalar = preprocessing.StandardScaler()

scaled_cols = scalar.fit_transform(df[cols])
```

The standardized values were converted back into a Pandas DataFrame and assigned to the original dataset.

Standardization transforms variables approximately according to:

```text
Z = (X - Mean) / Standard Deviation
```

This centers the variables around zero and scales them according to their standard deviation.

---

# 🔟 Correlation Matrix

A correlation matrix was created to understand relationships among the variables.

```python
corr = df.corr()

plt.figure(figsize=(20,10))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)
```

### Why Correlation Analysis?

Correlation analysis helps identify:

* Variables associated with house prices
* Positive relationships
* Negative relationships
* Relationships among predictors
* Potentially useful predictive features

---

# 🔎 Important Feature Relationships

The notebook specifically investigates the relationship between `LSTAT` and house prices.

```python
sns.regplot(
    y=df['medv'],
    x=df['lstat']
)
```

It also investigates the relationship between the average number of rooms and house prices:

```python
sns.regplot(
    y=df['medv'],
    x=df['rm']
)
```

These visualizations make it easier to understand how individual housing characteristics relate to `MEDV`.

---

# 1️⃣1️⃣ Feature and Target Separation

The target variable is:

```python
y = df['medv']
```

The input features are created using:

```python
X = df.drop(columns=['medv', 'rad'], axis=1)
```

Therefore, both `MEDV` and `RAD` are excluded from the model input, with `MEDV` used as the prediction target.

---

# 1️⃣2️⃣ Train-Test Split

The project uses Scikit-learn's:

```python
train_test_split
```

The split is performed inside a reusable training function:

```python
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42
)
```

Since `test_size` is not explicitly specified, Scikit-learn uses its default test proportion.

Setting:

```python
random_state=42
```

makes the split reproducible.

---

# 1️⃣3️⃣ Model Evaluation

The project evaluates models primarily using:

## Mean Squared Error (MSE)

```text
MSE = Average((Actual Value - Predicted Value)²)
```

A **lower MSE** indicates that predictions are closer to actual values.

The notebook calculates MSE using:

```python
mean_squared_error(y_test, pred)
```

---

# 1️⃣4️⃣ Cross-Validation

In addition to the test-set MSE, the project uses **5-fold cross-validation**.

```python
cross_val_score(
    model,
    X,
    y,
    scoring='neg_mean_squared_error',
    cv=5
)
```

The negative MSE scores returned by Scikit-learn are converted into positive values and averaged.

Cross-validation provides another way to evaluate model performance across multiple partitions of the dataset rather than depending entirely on one train-test split.

---

# 🤖 Machine Learning Models

## 1. Linear Regression

Linear Regression establishes a linear relationship between the predictor variables and the target.

```python
from sklearn.linear_model import LinearRegression
```

### Notebook Result

```text
MSE: 23.8710
CV Score: 35.5814
```

The notebook also visualizes the learned model coefficients.

---

## 2. Decision Tree Regressor

Decision Trees divide the feature space into regions to make predictions.

```python
from sklearn.tree import DecisionTreeRegressor
```

### Notebook Result

```text
MSE: 10.5054
CV Score: 41.5433
```

The feature importance values generated by the Decision Tree are also visualized.

---

## 3. Random Forest Regressor

Random Forest combines multiple decision trees to produce more robust predictions.

```python
from sklearn.ensemble import RandomForestRegressor
```

### Notebook Result

```text
MSE: 10.7266
CV Score: 21.9430
```

Random Forest also provides feature importance scores that help explain which variables contribute most to predictions.

---

## 4. Extra Trees Regressor

Extra Trees is another ensemble tree-based regression technique.

```python
from sklearn.ensemble import ExtraTreesRegressor
```

### Notebook Result

```text
MSE: 10.9130
CV Score: 20.2666
```

Feature importance is visualized after training the model.

---

## 5. XGBoost Regressor

XGBoost is a gradient-boosting algorithm designed to build a sequence of trees where later trees attempt to correct errors from earlier ones.

```python
import xgboost as xgb

model = xgb.XGBRegressor()
```

### Notebook Result

```text
MSE: 10.0323
CV Score: 17.9713
```

Among the models evaluated in this notebook, **XGBoost produced the lowest test MSE and the lowest reported cross-validation MSE**.

---

# 📈 Model Performance Comparison

| Model             |    Test MSE | 5-Fold CV MSE |
| ----------------- | ----------: | ------------: |
| Linear Regression |     23.8710 |       35.5814 |
| Decision Tree     |     10.5054 |       41.5433 |
| Random Forest     |     10.7266 |       21.9430 |
| Extra Trees       |     10.9130 |       20.2666 |
| **XGBoost**       | **10.0323** |   **17.9713** |

### Best Performing Model

Based on the metrics recorded in the notebook:

**🏆 XGBoost Regressor achieved the strongest overall performance.**

It produced:

* Lowest test MSE: **10.0323**
* Lowest cross-validation MSE: **17.9713**

The Decision Tree achieved a competitive test MSE, but its substantially higher cross-validation error indicates less consistent performance across folds.

---

# 🧠 Key Project Insights

The analysis demonstrates several important concepts in a regression Machine Learning workflow.

### Housing characteristics matter

Variables such as the average number of rooms (`RM`) and socioeconomic indicators such as `LSTAT` show visible relationships with median house value in the project's regression plots.

### Scaling helps prepare numerical variables

Normalization and standardization were applied to selected variables to transform features with different numerical ranges.

### Ensemble models performed strongly

Tree-based ensemble methods such as Random Forest, Extra Trees, and XGBoost achieved substantially lower test MSE than the Linear Regression model in the recorded notebook results.

### Cross-validation matters

Looking only at the test-set error can give an incomplete picture.

For example, the Decision Tree achieved:

```text
Test MSE = 10.5054
```

but:

```text
CV MSE = 41.5433
```

This difference demonstrates why cross-validation is useful when evaluating model generalization.

---

# 💡 What I Learned From This Project

Through this project, I practiced the complete workflow required for a supervised Machine Learning regression problem.

Key skills demonstrated include:

* Loading and exploring datasets using Pandas
* Understanding statistical summaries
* Checking data quality
* Detecting missing values
* Visualizing outliers
* Analyzing feature distributions
* Performing feature scaling
* Creating correlation heatmaps
* Investigating predictor-target relationships
* Separating features and target variables
* Splitting datasets into training and testing sets
* Training multiple regression algorithms
* Evaluating models using Mean Squared Error
* Applying 5-fold cross-validation
* Comparing regression algorithms
* Analyzing model coefficients
* Analyzing feature importance
* Selecting a model based on evaluation results

---

# 🚀 How to Run the Project

## Step 1 — Clone the Repository

```bash
git clone <your-repository-url>
```

## Step 2 — Navigate to the Project

```bash
cd Boston-Housing-Price-Prediction
```

## Step 3 — Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## Step 4 — Start Jupyter Notebook

```bash
jupyter notebook
```

## Step 5 — Open the Notebook

Open:

```text
Boston Housing Prediction - Regression.ipynb
```

Run the notebook cells sequentially to reproduce the analysis.

---



> **Compatibility note:** The notebook was created with older library APIs. For example, `LinearRegression(normalize=True)` has been removed from newer versions of Scikit-learn
>  and `sns.distplot()` is deprecated in modern Seaborn. These lines may need minor updates when running the notebook with current package versions.

---

# 🔮 Future Improvements

The project can be extended with:

* Hyperparameter tuning using GridSearchCV
* RandomizedSearchCV
* Additional regression metrics such as MAE, RMSE, and R²
* Residual analysis
* Prediction-vs-actual visualization
* More systematic outlier treatment
* Feature engineering
* Regularized Linear Regression such as Ridge and Lasso
* Gradient Boosting Regressor
* Model persistence using Joblib or Pickle
* Deployment through Flask, FastAPI, or Streamlit

---

# ⚠️ Dataset Note

The Boston Housing dataset is a historical Machine Learning dataset. One of its original variables encodes a race-related demographic transformation and is now widely considered problematic.

For this reason, this project should be treated primarily as an **educational regression exercise**, rather than as a basis for real-world housing valuation or decision-making.

---

# 🎯 Conclusion

This project demonstrates an end-to-end Machine Learning regression workflow for predicting median Boston housing values.

Starting with raw data, the project performs:

**Data Exploration → Preprocessing → Visualization → Scaling → Correlation Analysis → Feature Selection → Model Training → Cross-Validation → Model Comparison**

Five regression algorithms were evaluated:

* Linear Regression
* Decision Tree
* Random Forest
* Extra Trees
* XGBoost

Based on the results stored in the notebook, **XGBoost achieved the best overall performance**, with a test MSE of approximately **10.03** and a 5-fold cross-validation MSE of approximately **17.97**.

The project demonstrates how exploratory analysis, preprocessing, model comparison, and validation can be combined to develop and evaluate regression-based Machine Learning solutions.

---

## ⭐ Repository Highlights

`Machine Learning` `Regression` `Python` `Scikit-learn` `XGBoost` `Random Forest` `Decision Tree` `Extra Trees` `EDA` `Data Visualization` `Feature Scaling` `Cross Validation` `Predictive Modeling`

