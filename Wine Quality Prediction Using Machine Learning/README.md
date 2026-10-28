# 🍷 Red Wine Quality Prediction Using Machine Learning


## 📌 Project Overview

This project focuses on predicting the quality of red wine based on its physicochemical properties using Machine Learning classification algorithms.

The objective is to analyze the relationship between various chemical characteristics of red wine and its overall quality, perform Exploratory Data Analysis (EDA) and 
develop predictive models that classify wine into **Good Quality** and **Bad Quality** categories.

The project implements and compares three machine learning algorithms:

- Random Forest Classifier
- Stochastic Gradient Descent (SGD) Classifier
- Support Vector Classifier (SVC)

Additionally, **GridSearchCV** is used for hyperparameter tuning and **10-fold cross-validation** is performed to evaluate model performance.


## 🎯 Project Objectives

- Understand the factors influencing red wine quality.
- Explore and visualize relationships between chemical properties and wine quality.
- Perform data preprocessing and feature scaling.
- Convert wine quality scores into binary classification categories.
- Train and evaluate multiple machine learning models.
- Optimize model performance through hyperparameter tuning.
- Compare model performance using accuracy, precision, recall, F1-score and confusion matrices.


## 📊 Dataset Description

The project uses the **Red Wine Quality Dataset**, commonly associated with the UCI Machine Learning Repository.

The dataset contains **1,599 red wine samples**, with 11 physicochemical input features and one quality score.



### Dataset Features

| Feature | Description |
|---|---|
| Fixed Acidity | Concentration of nonvolatile acids in wine |
| Volatile Acidity | Amount of volatile acids, primarily acetic acid |
| Citric Acid | Citric acid concentration |
| Residual Sugar | Sugar remaining after fermentation |
| Chlorides | Salt concentration |
| Free Sulfur Dioxide | Amount of free SO₂ |
| Total Sulfur Dioxide | Total SO₂ concentration |
| Density | Density of the wine |
| pH | Acidity or alkalinity level |
| Sulphates | Sulphate concentration |
| Alcohol | Alcohol percentage |
| Quality | Wine quality score (target variable) |

The original quality scores range from 3 to 8 in the supplied dataset.

### Target Variable Transformation

The original multiclass quality scores are converted into two categories:

- **Bad Quality (0):** Quality score ≤ 6
- **Good Quality (1):** Quality score ≥ 7

This transformation makes the project a binary classification problem.


## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed using Pandas, Matplotlib and Seaborn to understand the dataset and identify patterns between chemical characteristics and wine quality.


### Key Observations

**1. Volatile Acidity**

Wines with higher quality scores generally exhibit lower volatile acidity.

**2. Citric Acid**

Higher-quality wines tend to have relatively higher citric acid concentrations.

**3. Chlorides**

Lower chloride concentrations are generally associated with higher wine quality.

**4. Sulphates**

Higher sulphate concentrations show a positive association with wine quality.

**5. Alcohol Content**

Alcohol content tends to increase as wine quality scores increase.

**6. Fixed Acidity and Residual Sugar**

These features do not show a strong, consistent relationship with wine quality in the visualizations.

These findings represent patterns observed in the dataset and do not necessarily establish causal relationships.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed before model training.

### 1. Target Variable Encoding

Wine quality scores were transformed into binary categories using `pd.cut()` and `LabelEncoder()`.

### 2. Feature Selection

The dataset was separated into independent and dependent variables.

- **X:** 11 physicochemical features
- **y:** Encoded wine quality category

### 3. Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

A random state of 42 was used for the split.

### 4. Feature Scaling

StandardScaler was applied to standardize numerical features.

**Note:** The original notebook refits the scaler on the test dataset. For correct and consistent preprocessing, the scaler should be fitted only on the training data and then used to transform the test data.

```python
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## 🤖 Machine Learning Models

### 1. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification performance.

**Implementation:**

- Algorithm: RandomForestClassifier
- Number of Estimators: 200
- Evaluation: Classification Report and Confusion Matrix

**Result:** Approximately 88% test accuracy in the saved notebook output.

### 2. Stochastic Gradient Descent (SGD) Classifier

SGD is an optimization-based learning method that iteratively updates model parameters to minimize a loss function.

**Implementation:**

- Algorithm: SGDClassifier
- Penalty: None
- Evaluation: Classification Report and Confusion Matrix

**Result:** Approximately 83% test accuracy in the saved notebook output.

### 3. Support Vector Classifier (SVC)

Support Vector Classification identifies a decision boundary that separates different classes.

**Implementation:**

- Algorithm: SVC
- Initial Model: Default parameters
- Evaluation: Classification Report

**Result:** Approximately 88% test accuracy based on the saved classification report.

## 🔧 Hyperparameter Tuning Using GridSearchCV

GridSearchCV was implemented to identify a better combination of hyperparameters for the Support Vector Classifier.

The parameters explored include:

- `C`: Regularization parameter
- `kernel`: Linear and RBF
- `gamma`: Kernel coefficient

The search used **10-fold cross-validation** with accuracy as the scoring metric.

### Best Parameters Recorded

```python
{
    'C': 1.2,
    'gamma': 0.9,
    'kernel': 'rbf'
}
```

After applying the selected parameters, the tuned SVC achieved approximately **90% test accuracy** in the notebook's saved classification report.

## 📈 Model Performance Comparison

| Model | Approximate Test Accuracy |
|---|---|
| Random Forest Classifier | 88.13% |
| SGD Classifier | 83.13% |
| Support Vector Classifier | 87.5% |
| Tuned Support Vector Classifier | 90% |

*Note: These results are based on saved notebook outputs. The notebook contains some different rounded accuracy statements and the results have not been independently rerun after correcting the preprocessing issue.*

### Cross-Validation

10-fold cross-validation was performed on the Random Forest model using the training dataset.

**Recorded Mean Cross-Validation Accuracy: 91.17%**

This is a cross-validation score and should not be interpreted as independent test-set accuracy.

### Model Evaluation Insights

The tuned SVC achieved the highest reported test accuracy. However, the dataset contains significantly more bad-quality samples than good-quality samples.

Because of this imbalance, accuracy alone may not provide a complete understanding of model performance.

Precision, recall, F1-score and confusion matrices are important for evaluating the models, particularly their ability to identify good-quality wines.

## 🛠️ Technologies and Libraries Used

**Programming Language**
- Python

**Development Environment**
- Jupyter Notebook

**Data Analysis**
- Pandas

**Data Visualization**
- Matplotlib
- Seaborn

**Machine Learning**
- Scikit-learn
- RandomForestClassifier
- SGDClassifier
- Support Vector Classifier

**Model Evaluation and Optimization**
- Train-Test Split
- StandardScaler
- LabelEncoder
- Classification Report
- Confusion Matrix
- GridSearchCV
- Cross-Validation

## 📂 Project Structure

```text
red-wine-quality-prediction/
│
├── prediction-of-quality-of-wine.ipynb
│
├── winequality-red.csv
│
└── README.md
```

**File Descriptions**

- `prediction-of-quality-of-wine.ipynb` — Jupyter Notebook containing data analysis, preprocessing, model training and evaluation.
- `winequality-red.csv` — Dataset containing red wine chemical properties and quality scores.
- `README.md` — Project documentation.

## 💡 Key Learnings

Through this project, the following machine learning concepts were explored:

- Understanding and analyzing structured datasets
- Identifying relationships between numerical features
- Performing Exploratory Data Analysis
- Data preprocessing and label encoding
- Feature scaling and train-test splitting
- Implementing supervised classification algorithms
- Evaluating models using classification metrics
- Hyperparameter optimization using GridSearchCV
- Understanding cross-validation
- Comparing machine learning models and interpreting their limitations

## 🔮 Future Enhancements

Potential improvements include:

- Addressing class imbalance using class weighting or resampling techniques.
- Improving evaluation using balanced accuracy, F1-score and precision-recall metrics.
- Implementing a preprocessing pipeline to prevent data leakage.
- Exploring additional classification algorithms such as Gradient Boosting.
- Performing feature importance analysis.
- Improving reproducibility through fixed random seeds.
- Developing a Streamlit web application for interactive wine quality prediction.
- Deploying the trained model for real-time predictions.

## 🏁 Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting red wine quality using physicochemical properties.

By implementing Random Forest, SGD and Support Vector Classification, the project compares multiple approaches to binary classification.

The tuned Support Vector Classifier achieved approximately **90% test accuracy** in the saved notebook results, while Random Forest achieved a mean cross-validation score of approximately **91.17%**.

The analysis also demonstrates the importance of data preprocessing, feature scaling, hyperparameter tuning and evaluating model performance beyond accuracy, especially when working with imbalanced datasets.

## 📚 Dataset Reference

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.

Dataset: UCI Wine Quality Dataset

https://archive.ics.uci.edu/dataset/186/wine+quality

---

**Project Category:** Data Science | Machine Learning | Predictive Analytics | Classification

**Language:** Python

**Environment:** Jupyter Notebook
