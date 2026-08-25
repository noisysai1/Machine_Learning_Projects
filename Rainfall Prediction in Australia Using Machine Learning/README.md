# 🌧️ Rainfall Prediction in Australia Using Machine Learning

## 📌 Project Overview

Weather forecasting is an important real-world application of data science and machine learning. Accurate rainfall prediction can support agriculture, transportation, water-resource planning, disaster preparedness, and everyday decision-making.

This project develops a **Machine Learning classification system to predict whether it will rain tomorrow in Australia** based on historical daily weather observations.

The target variable is:

**`RainTomorrow`**

where:

* `Yes / 1` → Rain is expected tomorrow
* `No / 0` → Rain is not expected tomorrow

The project goes beyond simply training a model. It implements an end-to-end machine-learning workflow including:

* Data loading and exploration
* Feature identification
* Missing-value analysis
* Missing-value imputation
* Exploratory Data Analysis
* Correlation analysis
* Categorical feature encoding
* Date feature engineering
* Outlier detection and treatment
* Class imbalance analysis
* SMOTE oversampling
* Train-test splitting
* Multiple classification algorithms
* Confusion matrices
* Precision, Recall and F1-score evaluation
* ROC analysis
* Model comparison
* Model serialization using Joblib

---

# 🎯 Project Objective

The primary objective of this project is to build a machine-learning model capable of answering:

> **Will it rain tomorrow based on today's weather conditions?**

Rather than relying on one algorithm, multiple classification algorithms are trained and evaluated to determine which model performs best on the Australian weather dataset.

---

# 📂 Dataset

The project uses:

```text
weatherAUS.csv
```

The original dataset contains:

```text
145,460 rows
23 columns
49 weather locations
```

The observations span approximately:

```text
November 2007 – June 2017
```

The dataset contains daily weather observations from multiple locations across Australia.

---

# 🗂️ Dataset Features

The original dataset contains the following columns:

| Feature       | Description                               |
| ------------- | ----------------------------------------- |
| Date          | Date of weather observation               |
| Location      | Australian weather station/location       |
| MinTemp       | Minimum temperature                       |
| MaxTemp       | Maximum temperature                       |
| Rainfall      | Amount of rainfall                        |
| Evaporation   | Evaporation measurement                   |
| Sunshine      | Number of sunshine hours                  |
| WindGustDir   | Direction of strongest wind gust          |
| WindGustSpeed | Speed of strongest wind gust              |
| WindDir9am    | Wind direction at 9 AM                    |
| WindDir3pm    | Wind direction at 3 PM                    |
| WindSpeed9am  | Wind speed at 9 AM                        |
| WindSpeed3pm  | Wind speed at 3 PM                        |
| Humidity9am   | Humidity at 9 AM                          |
| Humidity3pm   | Humidity at 3 PM                          |
| Pressure9am   | Atmospheric pressure at 9 AM              |
| Pressure3pm   | Atmospheric pressure at 3 PM              |
| Cloud9am      | Cloud cover at 9 AM                       |
| Cloud3pm      | Cloud cover at 3 PM                       |
| Temp9am       | Temperature at 9 AM                       |
| Temp3pm       | Temperature at 3 PM                       |
| RainToday     | Whether rain occurred today               |
| RainTomorrow  | Whether rain occurs tomorrow — **Target** |

---

# 🔍 Project Workflow

The project follows the machine-learning pipeline shown below:

```text
Raw Weather Dataset
        ↓
Data Exploration
        ↓
Feature Classification
        ↓
Missing Value Analysis
        ↓
Missing Value Imputation
        ↓
Exploratory Data Analysis
        ↓
Categorical Encoding
        ↓
Date Feature Engineering
        ↓
Outlier Detection
        ↓
Outlier Treatment
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
SMOTE Class Balancing
        ↓
Train Multiple ML Models
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Model Serialization
```

---

# 1️⃣ Importing Required Libraries

The project begins by importing libraries required for data processing, visualization, statistical analysis and machine learning.

Major libraries include:

```python
NumPy
Pandas
Matplotlib
Seaborn
SciPy
Scikit-learn
Imbalanced-learn
CatBoost
XGBoost
Joblib
```

These libraries are used throughout the project for preprocessing, visualization, modeling and evaluation.

---

# 2️⃣ Loading the Dataset

The Australian weather dataset is loaded using Pandas:

```python
df = pd.read_csv("weatherAUS.csv")
```

Pandas is then configured to display all columns, making the dataset easier to inspect during analysis.

---

# 3️⃣ Feature Classification

The features are separated into different groups based on their data types and characteristics.

The notebook identifies:

```text
Numerical Features
Discrete Features
Continuous Features
Categorical Features
```

This separation is useful because different types of variables require different preprocessing techniques.

For example:

* Continuous variables can be analyzed for distributions and outliers.
* Categorical variables require encoding.
* Missing numerical values can be handled using statistical techniques.

---

# 4️⃣ Missing Value Analysis

Real-world weather datasets frequently contain missing observations.

The percentage of missing data for every feature is calculated using:

```python
df.isnull().sum()*100/len(df)
```

This makes it possible to understand the amount of incomplete data before choosing appropriate imputation strategies.

---

# 5️⃣ Handling Missing Values

Different techniques are applied depending on the feature.

## Random Sample Imputation

A custom function is created to perform random sample imputation:

```python
def randomsampleimputation(df, variable):
    random_sample = df[variable].dropna().sample(
        df[variable].isnull().sum(),
        random_state=0
    )

    random_sample.index = df[df[variable].isnull()].index

    df.loc[df[variable].isnull(), variable] = random_sample
```

Random sample imputation is applied to:

```text
Cloud9am
Cloud3pm
Evaporation
Sunshine
```

This technique fills missing observations using randomly selected existing observations from the same feature.

---

# 6️⃣ Exploratory Data Analysis

Exploratory Data Analysis is performed to better understand the relationships and distributions present in the weather data.

Several visualization techniques are used.

## Correlation Heatmap

A Spearman correlation matrix is initially calculated:

```python
corrmat = df.corr(method="spearman")
```

A heatmap is then generated using Seaborn.

The heatmap helps identify relationships among numerical weather variables.

---

# 7️⃣ Feature Distribution Analysis

Distribution plots are generated for continuous variables.

```python
sns.distplot(df[feature])
```

These plots help investigate:

* Distribution shape
* Skewness
* Spread
* Extreme observations
* Potential preprocessing requirements

---

# 8️⃣ Outlier Detection

Box plots are generated for the continuous variables.

```python
sns.boxplot(data[feature])
```

Box plots provide a visual way to identify observations that lie far outside the typical range of each weather variable.

Because extreme observations can influence some machine-learning algorithms, the project performs additional outlier treatment later in the pipeline.

---

# 9️⃣ Median Imputation

For continuous features that still contain missing values, the median value is used.

```python
df[feature] = df[feature].fillna(
    df[feature].median()
)
```

Median imputation is useful for numerical variables because it is less sensitive to extreme observations than the mean.

---

# 🔟 Encoding Rain Variables

The binary categorical variables are converted into numerical representations.

```python
df["RainToday"] = pd.get_dummies(
    df["RainToday"],
    drop_first=True
)

df["RainTomorrow"] = pd.get_dummies(
    df["RainTomorrow"],
    drop_first=True
)
```

The target is therefore represented numerically as:

```text
0 → No Rain
1 → Rain
```

---

# 1️⃣1️⃣ Encoding Wind Direction

Weather variables such as wind direction contain categorical values including:

```text
N
NE
E
SE
S
SW
W
NW
NNW
WNW
NNE
WSW
SSW
SSE
ENE
ESE
```

The project manually maps these categorical values into numerical representations.

The encoded wind features include:

```text
WindGustDir
WindDir9am
WindDir3pm
```

Missing values remaining after the mapping process are filled using the most frequently occurring value.

---

# 1️⃣2️⃣ Encoding Location

The dataset contains weather observations from **49 Australian locations**.

The project examines the relationship between each location and `RainTomorrow` before creating a numerical mapping for locations.

Examples include:

```text
Portland
Cairns
Walpole
Dartmoor
MountGambier
Sydney
Darwin
Hobart
Brisbane
Melbourne
Canberra
Perth
Adelaide
AliceSprings
Uluru
```

The `Location` variable is subsequently represented numerically for machine-learning algorithms.

---

# 1️⃣3️⃣ Date Feature Engineering

The original `Date` column is converted into a Pandas datetime variable.

```python
df["Date"] = pd.to_datetime(df["Date"])
```

Additional temporal features are then extracted:

```python
df["Date_month"] = df["Date"].dt.month
df["Date_day"] = df["Date"].dt.day
```

This allows the models to use information about the month and day rather than treating the complete date as a string.

The original `Date` variable is later excluded from the machine-learning feature matrix.

---

# 1️⃣4️⃣ Target Class Distribution

The distribution of `RainTomorrow` is visualized using a count plot.

```python
sns.countplot(df["RainTomorrow"])
```

The original dataset clearly contains more **No Rain** observations than **Rain** observations.

This creates a **class imbalance problem**.

A machine-learning model trained directly on an imbalanced dataset may become biased toward predicting the majority class.

The project addresses this problem using **SMOTE**.

---

# 1️⃣5️⃣ Outlier Treatment Using IQR

The project uses the **Interquartile Range (IQR)** approach to determine lower and upper boundaries for several continuous features.

The general calculation is:

```text
IQR = Q3 - Q1

Lower Boundary = Q1 - 1.5 × IQR

Upper Boundary = Q3 + 1.5 × IQR
```

Extreme observations are capped at calculated boundaries.

Outlier treatment is performed for variables including:

```text
MinTemp
MaxTemp
Rainfall
Evaporation
WindGustSpeed
WindSpeed9am
WindSpeed3pm
Humidity9am
Pressure9am
Pressure3pm
Temp9am
Temp3pm
```

This reduces the influence of extreme observations during model training.

---

# 1️⃣6️⃣ Normality Analysis

The project also examines feature distributions using histograms and Q-Q plots.

The SciPy function:

```python
stats.probplot()
```

is used to compare continuous variables against a theoretical normal distribution.

This provides additional understanding of:

* Normality
* Skewness
* Distribution behavior
* Extreme values

---

# 1️⃣7️⃣ Saving Preprocessed Data

After preprocessing, the transformed dataset is saved as:

```text
preprocessed_1.csv
```

using:

```python
df.to_csv("preprocessed_1.csv", index=False)
```

This allows the cleaned dataset to be reused without repeating the complete preprocessing workflow.

---

# 1️⃣8️⃣ Feature and Target Separation

The target variable is:

```text
RainTomorrow
```

The feature matrix and target vector are created as:

```python
X = df.drop(["RainTomorrow", "Date"], axis=1)

Y = df["RainTomorrow"]
```

Therefore:

```text
X → Weather predictor variables
Y → RainTomorrow
```

---

# 1️⃣9️⃣ Train-Test Split

The dataset is divided into training and testing subsets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=0
)
```

The project uses:

```text
80% → Training Data
20% → Testing Data
```

Using `stratify=Y` helps preserve the target class proportions in both subsets.

The final test set contains:

```text
29,092 observations
```

---

# 2️⃣0️⃣ Handling Class Imbalance with SMOTE

One of the most important preprocessing steps in this project is handling the imbalance in `RainTomorrow`.

The training target originally contains:

```text
No Rain (0): 90,866
Rain (1):    25,502
```

This shows a significant imbalance between the two classes.

The project applies:

## SMOTE — Synthetic Minority Over-sampling Technique

```python
sm = SMOTE(random_state=0)

X_train_res, y_train_res = sm.fit_resample(
    X_train,
    y_train
)
```

After SMOTE:

```text
No Rain (0): 90,866
Rain (1):    90,866
```

The training data therefore becomes perfectly balanced.

### Why SMOTE?

Instead of simply duplicating minority-class records, SMOTE generates synthetic examples based on neighboring minority-class observations.

This gives the machine-learning algorithms greater exposure to rainfall observations during training.

---

# 🤖 Machine Learning Models

Seven classification algorithms are trained and compared.

```text
1. CatBoost Classifier
2. Random Forest Classifier
3. Logistic Regression
4. Gaussian Naive Bayes
5. K-Nearest Neighbors
6. XGBoost Classifier
7. Support Vector Machine
```

---

# 2️⃣1️⃣ CatBoost Classifier

The CatBoost model is configured as:

```python
CatBoostClassifier(
    iterations=2000,
    eval_metric="AUC"
)
```

### Results

```text
Accuracy: 86.34%
```

Confusion Matrix:

```text
[[21520  1197]
 [ 2777  3598]]
```

Rain-class performance:

```text
Precision: 0.75
Recall:    0.56
F1-score:  0.64
```

CatBoost achieved the **highest overall accuracy among the models evaluated in the notebook**.

---

# 2️⃣2️⃣ Random Forest Classifier

Random Forest combines predictions from multiple decision trees.

```python
rf = RandomForestClassifier()
```

### Results

```text
Accuracy: 84.35%
```

Confusion Matrix:

```text
[[20633  2084]
 [ 2470  3905]]
```

Rain-class metrics:

```text
Precision: 0.65
Recall:    0.61
F1-score:  0.63
```

Random Forest provides strong overall classification performance while identifying a larger proportion of rainy days than CatBoost.

---

# 2️⃣3️⃣ Logistic Regression

```python
logreg = LogisticRegression()
```

### Results

```text
Accuracy: 77.36%
```

Rain-class metrics:

```text
Precision: 0.49
Recall:    0.76
F1-score:  0.60
```

An interesting observation is that Logistic Regression has lower overall accuracy but considerably higher recall for rainy observations.

This demonstrates why **accuracy alone should not be used to evaluate an imbalanced classification problem**.

---

# 2️⃣4️⃣ Gaussian Naive Bayes

```python
gnb = GaussianNB()
```

### Results

```text
Accuracy: 74.91%
```

Rain-class metrics:

```text
Precision: 0.46
Recall:    0.74
F1-score:  0.56
```

The model detects a relatively large percentage of rainy observations but produces more false-positive rainfall predictions.

---

# 2️⃣5️⃣ K-Nearest Neighbors

The project uses:

```python
KNeighborsClassifier(n_neighbors=3)
```

### Results

```text
Accuracy: 75.31%
```

Rain-class metrics:

```text
Precision: 0.46
Recall:    0.72
F1-score:  0.56
```

---

# 2️⃣6️⃣ XGBoost Classifier

```python
xgb = XGBClassifier()
```

### Results

```text
Accuracy: 85.68%
```

Confusion Matrix:

```text
[[21396  1321]
 [ 2844  3531]]
```

Rain-class metrics:

```text
Precision: 0.73
Recall:    0.55
F1-score:  0.63
```

XGBoost produced the **second-highest accuracy** among the tested algorithms.

---

# 2️⃣7️⃣ Support Vector Machine

```python
svc = SVC()
```

### Results

```text
Accuracy: 77.70%
```

Rain-class metrics:

```text
Precision: 0.49
Recall:    0.75
F1-score:  0.60
```

Like Logistic Regression, SVM achieves relatively high recall for rainy days while sacrificing some overall accuracy.

---

# 📊 Model Comparison

| Model                |   Accuracy | Rain Precision | Rain Recall |  Rain F1 |
| -------------------- | ---------: | -------------: | ----------: | -------: |
| 🥇 CatBoost          | **86.34%** |       **0.75** |        0.56 | **0.64** |
| 🥈 XGBoost           | **85.68%** |           0.73 |        0.55 |     0.63 |
| 🥉 Random Forest     | **84.35%** |           0.65 |        0.61 |     0.63 |
| SVM                  |     77.70% |           0.49 |        0.75 |     0.60 |
| Logistic Regression  |     77.36% |           0.49 |    **0.76** |     0.60 |
| KNN                  |     75.31% |           0.46 |        0.72 |     0.56 |
| Gaussian Naive Bayes |     74.91% |           0.46 |        0.74 |     0.56 |

---

# 🏆 Best Performing Model

Based on **overall test accuracy**, the strongest model in this experiment is:

## 🥇 CatBoost Classifier — 86.34%

The three highest-accuracy models are:

```text
CatBoost       → 86.34%
XGBoost        → 85.68%
Random Forest  → 84.35%
```

However, model selection depends on the business objective.

For example, if identifying as many actual rainy days as possible is more important than minimizing false alarms, **recall for the rain class** becomes particularly important.

Logistic Regression achieved a rain recall of approximately:

```text
76%
```

compared with approximately:

```text
56%
```

for CatBoost.

Therefore, this project demonstrates an important machine-learning concept:

> **The model with the highest accuracy is not automatically the best model for every business objective.**

---

# 📈 Evaluation Metrics

The models are evaluated using multiple metrics rather than relying exclusively on accuracy.

## Accuracy

Measures the percentage of all predictions that were correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

## Precision

Answers:

> When the model predicts rain, how often is that prediction correct?

## Recall

Answers:

> Out of all days where rain actually occurred, how many did the model identify?

## F1-Score

Balances Precision and Recall.

## Confusion Matrix

Shows:

```text
True Negatives
False Positives
False Negatives
True Positives
```

## ROC Analysis

ROC curves are generated for the classifiers in the notebook to analyze classification performance across different decision thresholds.

---

# 💾 Saving Trained Models

The project uses Joblib for model serialization.

The notebook saves:

```python
joblib.dump(svc, "svc.pkl")
joblib.dump(xgb, "xgb.pkl")
```

This produces:

```text
svc.pkl
xgb.pkl
```

Saved models can later be loaded into another Python program or application without retraining them.

Example:

```python
import joblib

model = joblib.load("xgb.pkl")
```

---

# 📁 Suggested GitHub Repository Structure

```text
Rainfall-Prediction-Australia/
│
├── RainPrediction2.ipynb
│
├── weatherAUS.csv
│
├── preprocessed_1.csv
│
├── svc.pkl
├── xgb.pkl
│
├── requirements.txt
│
└── README.md
```

If the dataset or `.pkl` files are too large for your preferred repository setup, they can instead be excluded with `.gitignore` and documented separately.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate into the project:

```bash
cd Rainfall-Prediction-Australia
```

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn catboost xgboost joblib
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
RainPrediction2.ipynb
```

and execute the notebook cells sequentially.

---

# 🛠️ Technologies Used

### Programming

```text
Python
```

### Data Processing

```text
Pandas
NumPy
```

### Visualization

```text
Matplotlib
Seaborn
```

### Statistical Analysis

```text
SciPy
```

### Machine Learning

```text
Scikit-learn
CatBoost
XGBoost
```

### Imbalanced Data Handling

```text
SMOTE
Imbalanced-learn
```

### Model Persistence

```text
Joblib
```

### Development Environment

```text
Jupyter Notebook
```

---

# 💡 Key Project Insights

Several important machine-learning lessons can be demonstrated through this project.

### 1. Real-world datasets require extensive preprocessing

The original weather data contains missing values, categorical features, outliers and an imbalanced target variable.

A significant part of building the model therefore involves preparing the data correctly.

### 2. Rain prediction is an imbalanced classification problem

There are substantially more non-rain observations than rain observations.

SMOTE was used to balance the training classes:

```text
Before SMOTE

0 → 90,866
1 → 25,502

After SMOTE

0 → 90,866
1 → 90,866
```

### 3. Ensemble boosting algorithms performed strongly

CatBoost and XGBoost achieved the two highest overall accuracies in the notebook.

### 4. Accuracy does not tell the entire story

Although CatBoost achieved the highest accuracy, Logistic Regression and SVM achieved higher recall for the rainfall class.

This illustrates the trade-off between:

```text
Accuracy
Precision
Recall
False Positives
False Negatives
```

### 5. Weather prediction depends on multiple interacting variables

The project uses temperature, humidity, pressure, rainfall, wind, clouds, sunshine, evaporation, location and temporal information together rather than relying on a single weather measurement.

---

# 🌍 Real-World Applications

A rainfall prediction system can potentially support:

### 🌾 Agriculture

Farmers can use rainfall forecasts when planning irrigation, planting and harvesting activities.

### 🚗 Transportation

Rainfall predictions can support road-safety and transportation planning.

### 💧 Water Resource Management

Weather predictions can contribute to reservoir and water-resource planning.

### 🏙️ Urban Planning

Rainfall information can support drainage and infrastructure planning.

### 👥 Everyday Decision Making

Weather predictions influence travel, outdoor activities and daily planning.

---

# 🚀 Future Improvements

The current project can be extended in several directions.

Potential improvements include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation
* Feature importance analysis
* More advanced feature engineering
* Threshold optimization
* Precision-Recall curve analysis
* Comparing additional boosting algorithms
* Testing alternative class-balancing strategies
* Building reusable preprocessing pipelines
* Deploying the model through Flask or FastAPI
* Creating an interactive Streamlit weather prediction application
* Adding MLflow experiment tracking
* Containerizing the application using Docker
* Deploying the model to a cloud environment

---

# 📌 Skills Demonstrated

This project demonstrates practical experience with:

```text
✔ Python Programming
✔ Data Cleaning
✔ Data Preprocessing
✔ Exploratory Data Analysis
✔ Missing Value Treatment
✔ Statistical Analysis
✔ Data Visualization
✔ Feature Engineering
✔ Categorical Encoding
✔ Outlier Detection
✔ Outlier Treatment
✔ Imbalanced Data Handling
✔ SMOTE
✔ Classification
✔ Ensemble Learning
✔ Boosting Algorithms
✔ Model Evaluation
✔ Confusion Matrix Analysis
✔ Precision / Recall / F1 Analysis
✔ ROC Analysis
✔ Model Comparison
✔ Model Serialization
```

---

# 🎓 What I Learned From This Project

This project provided hands-on experience in building an end-to-end classification workflow using a large real-world dataset.

A major learning point was that successful machine learning involves much more than selecting an algorithm. Data quality, missing-value handling, categorical encoding, outliers and class imbalance all have a significant impact on the final model.

The project also highlighted the importance of comparing multiple models using several evaluation metrics.

In particular, it demonstrated that:

> **A high-accuracy model may still miss an important portion of the minority class.**

For rainfall prediction, this means evaluating recall, precision and F1-score alongside overall accuracy.

---

# 🏁 Conclusion

This project successfully develops and evaluates multiple machine-learning approaches for predicting whether it will rain the following day using historical Australian weather observations.

The workflow begins with a raw dataset of more than **145,000 weather observations**, performs extensive preprocessing and exploratory analysis, addresses class imbalance using **SMOTE**, and compares seven classification algorithms.

Among the tested models:

```text
🥇 CatBoost       → 86.34%
🥈 XGBoost        → 85.68%
🥉 Random Forest  → 84.35%
```

CatBoost achieved the highest overall test accuracy, while other models demonstrated different precision-recall trade-offs.

Overall, the project provides a practical example of solving a real-world **binary classification problem** from raw data preparation through final model evaluation and persistence.

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a **⭐ Star**.

Feel free to fork the repository, experiment with the models and extend the project with additional machine-learning techniques.

---

**Project:** Rainfall Prediction in Australia Using Machine Learning
**Language:** Python
**Problem Type:** Binary Classification
**Target:** `RainTomorrow`
**Dataset Size:** 145,460 observations
**Best Accuracy:** **86.34% — CatBoost Classifier**
