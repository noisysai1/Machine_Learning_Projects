# 🛍️ Black Friday Sales Prediction Using Machine Learning

## 📌 Overview

This project develops a **machine learning regression solution to predict customer purchase amounts during Black Friday sales**.

The analysis uses historical retail transaction data containing customer demographics, product information, occupation, city category, marital status and product categories. 
The project combines **Exploratory Data Analysis (EDA), data preprocessing, feature encoding, correlation analysis and regression modeling** to understand purchasing behavior and build predictive models.

Several regression algorithms are trained and evaluated to determine how effectively customer and product characteristics can be used to predict purchase amounts.

---

## 🎯 Problem Statement

Retailers generate large amounts of customer transaction data during major shopping events such as Black Friday.

Understanding purchasing behavior can help businesses improve:

* Customer segmentation
* Sales forecasting
* Product recommendations
* Inventory planning
* Marketing strategies
* Promotional campaigns

The primary objective of this project is:

> **Predict the purchase amount of a customer based on demographic and product-related attributes.**

This is therefore a **supervised machine learning regression problem**.

---

## 📊 Dataset

The dataset contains approximately:

* **550,069 transactions**
* **12 features**
* Customer demographic information
* Product information
* Purchase amounts

### Dataset Features

| Feature                      | Description                                                |
| ---------------------------- | ---------------------------------------------------------- |
| `User_ID`                    | Unique customer identifier                                 |
| `Product_ID`                 | Unique product identifier                                  |
| `Gender`                     | Gender of the customer                                     |
| `Age`                        | Customer age group                                         |
| `Occupation`                 | Customer occupation code                                   |
| `City_Category`              | Category of the customer's city                            |
| `Stay_In_Current_City_Years` | Number of years the customer has lived in the current city |
| `Marital_Status`             | Customer marital status                                    |
| `Product_Category_1`         | Primary product category                                   |
| `Product_Category_2`         | Secondary product category                                 |
| `Product_Category_3`         | Tertiary product category                                  |
| `Purchase`                   | Purchase amount — **target variable**                      |

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to better understand customer demographics, product characteristics and purchasing behavior.

The analysis includes:

### Purchase Distribution

The distribution of the target variable `Purchase` was analyzed to understand the overall spending behavior of customers.

### Customer & Product Analysis

Count plots were created for:

* Gender
* Age
* Marital Status
* Occupation
* Product Category 1
* Product Category 2
* Product Category 3
* City Category
* Years in Current City

### Bivariate Analysis

Average purchase amounts were compared across important customer characteristics.

The project analyzes relationships such as:

* **Occupation vs Purchase**
* **Age vs Purchase**
* **Gender vs Purchase**

Pivot tables and bar charts were used to visualize these relationships.

---

## 🧹 Data Preprocessing

Several preprocessing steps were performed before model training.

### Missing Value Handling

Missing values were identified in:

* `Product_Category_2`
* `Product_Category_3`

These missing values were replaced with:

```python
-2.0
```

The columns were subsequently converted to `float32`.

### Gender Encoding

Gender was converted into numerical values:


Female → 0
Male   → 1
```

### Label Encoding

Categorical variables were transformed into numerical representations using `LabelEncoder`.

Encoded features include:

```text
Age
City_Category
Stay_In_Current_City_Years
```

---

## 🔗 Correlation Analysis

A correlation matrix was generated to examine relationships between numerical variables.

A **Seaborn heatmap** was used to visualize correlations and identify variables associated with customer purchase amounts.

---

## ✂️ Feature Selection

The following identifier columns were removed before model training:

```text
User_ID
Product_ID
```

The target variable was:

```text
Purchase
```

The resulting feature matrix contains customer demographic and product-related variables used to predict purchase amounts.

---

## 🤖 Machine Learning Models

Four regression algorithms were evaluated.

### 1. Linear Regression

Linear Regression provides a baseline model for estimating purchase amounts using linear relationships between the input features and target variable.

Model coefficients were also visualized to examine the influence of individual features.

### 2. Decision Tree Regressor

A Decision Tree Regressor was trained to capture nonlinear relationships within customer and product data.

Feature importance values were analyzed to determine which variables contributed most to predictions.

### 3. Random Forest Regressor

Random Forest combines multiple decision trees to improve predictive performance and reduce the tendency of individual trees to overfit.

Feature importance analysis was also performed.

### 4. Extra Trees Regressor

The Extra Trees Regressor uses an ensemble of highly randomized decision trees.

It was evaluated alongside the other regression models and its feature importance values were visualized.

---

## 📏 Model Evaluation

The dataset was divided into:

```text
Training Data : 75%
Testing Data  : 25%
```

A fixed random state of `42` was used for reproducibility.

Models were evaluated using:

### Root Mean Squared Error (RMSE)

RMSE measures the average magnitude of prediction errors.

Lower RMSE values indicate better predictive performance.

### 5-Fold Cross-Validation

Each model was additionally evaluated using **5-fold cross-validation** with negative mean squared error as the scoring metric.

The resulting score was converted to RMSE for easier interpretation.

This provides a more reliable estimate of model performance than relying on a single train-test split.

---

## ⚙️ Machine Learning Workflow

```text
Black Friday Dataset
        │
        ▼
Data Exploration
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Missing Value Treatment
        │
        ▼
Categorical Encoding
        │
        ▼
Correlation Analysis
        │
        ▼
Feature / Target Split
        │
        ▼
Train-Test Split
        │
        ▼
 ┌─────────────────────────┐
 │   Regression Models     │
 ├─────────────────────────┤
 │ Linear Regression       │
 │ Decision Tree           │
 │ Random Forest           │
 │ Extra Trees             │
 └─────────────────────────┘
        │
        ▼
RMSE Evaluation
        │
        ▼
5-Fold Cross-Validation
        │
        ▼
Purchase Prediction
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Environment

* Jupyter Notebook / Google Colab

---

## 📂 Suggested Repository Structure

```text
black-friday-sales-prediction/
│
├── Black_Friday_Sales_Prediction_Analysis_Regression.ipynb
├── train.csv
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/black-friday-sales-prediction.git
```

Navigate to the project directory:

```bash
cd black-friday-sales-prediction
```

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Black_Friday_Sales_Prediction_Analysis_Regression.ipynb
```

and run the notebook cells sequentially.

---

## 📈 Key Project Components

This project demonstrates an end-to-end machine learning workflow involving:

* Data loading and inspection
* Statistical analysis
* Exploratory Data Analysis
* Missing-value treatment
* Categorical feature encoding
* Data visualization
* Correlation analysis
* Feature selection
* Train-test splitting
* Regression modeling
* Cross-validation
* RMSE evaluation
* Feature importance analysis
* Purchase prediction

---

## 💡 Business Applications

A purchase prediction model like this can help retailers estimate expected customer spending and support data-driven decisions.

Potential applications include:

**Personalized Marketing**

Identify customer groups likely to have higher purchase values and target them with appropriate promotions.

**Sales Forecasting**

Estimate purchasing behavior during major retail events.

**Inventory Management**

Use predicted demand patterns to support inventory and product planning.

**Customer Segmentation**

Understand how demographic characteristics relate to purchasing behavior.

**Product Strategy**

Analyze the importance of different product categories in driving customer purchases.

---

## 🔮 Future Improvements

The project can be extended by:

* Applying one-hot encoding to nominal categorical variables
* Performing hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
* Comparing additional models such as XGBoost, LightGBM and Gradient Boosting
* Creating additional customer and product-level features
* Performing more extensive outlier analysis
* Building reusable preprocessing pipelines
* Adding MAE and R² alongside RMSE
* Saving the best model for future inference
* Developing a Streamlit or Flask application for real-time predictions
* Deploying the trained model as a REST API

---

## 📝 Conclusion

This project demonstrates how machine learning can be applied to a large-scale retail dataset to predict customer purchase amounts.

The workflow covers the complete process from **data exploration and preprocessing to regression modeling, cross-validation and feature importance analysis**.

By comparing Linear Regression with tree-based and ensemble regression techniques, the project provides a practical demonstration of predictive modeling for retail sales data.

---
