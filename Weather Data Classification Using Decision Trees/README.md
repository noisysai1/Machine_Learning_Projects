# 🌦️ Weather Data Classification Using Decision Trees

A machine learning classification project that uses historical weather observations to predict whether afternoon relative humidity will be **high or low** based on weather conditions measured earlier in the day.

The project demonstrates an end-to-end supervised machine learning workflow using **Python, Pandas, Scikit-learn and Decision Tree Classification**.

---

## 📌 Project Overview

Weather conditions such as temperature, atmospheric pressure, wind speed, wind direction and rainfall can provide useful information about how atmospheric conditions may change throughout the day.

In this project, historical daily weather observations are analyzed and used to build a **Decision Tree Classifier**.

The goal is to predict whether the **relative humidity at 3:00 PM** will exceed a predefined threshold using measurements collected around **9:00 AM**.

The project includes:

- Dataset exploration
- Data cleaning
- Missing-value handling
- Feature selection
- Target-variable engineering
- Train-test splitting
- Decision Tree model training
- Prediction
- Accuracy-based model evaluation

---

# 🎯 Problem Statement

The primary question explored in this project is:

> **Can morning weather measurements be used to predict whether afternoon relative humidity will be high?**

The original dataset contains a continuous measurement for afternoon humidity:

``` 
relative_humidity_3pm
```

To formulate the task as a machine learning classification problem, this continuous variable is transformed into a binary target.

``` 
3 PM Humidity ≤ 24.99%  →  Class 0
3 PM Humidity > 24.99%  →  Class 1
```

The Decision Tree then learns patterns between morning weather conditions and these two afternoon humidity classes.

---

# 📊 Dataset

The project uses the:

``` 
daily_weather.csv
```

dataset.

The original dataset contains:

``` 
1,095 rows
11 columns
```

Each row represents a daily weather observation.

The dataset contains measurements related to:

- Atmospheric pressure
- Air temperature
- Wind direction
- Wind speed
- Maximum wind conditions
- Rain accumulation
- Rain duration
- Relative humidity

---

## 📋 Dataset Features

| Column | Description |
|---|---|
| `number` | Unique observation identifier |
| `air_pressure_9am` | Average atmospheric pressure around 9 AM |
| `air_temp_9am` | Average air temperature around 9 AM |
| `avg_wind_direction_9am` | Average wind direction around 9 AM |
| `avg_wind_speed_9am` | Average wind speed around 9 AM |
| `max_wind_direction_9am` | Maximum wind/gust direction around 9 AM |
| `max_wind_speed_9am` | Maximum wind/gust speed around 9 AM |
| `rain_accumulation_9am` | Rain accumulated during the previous 24 hours |
| `rain_duration_9am` | Duration of rainfall during the previous 24 hours |
| `relative_humidity_9am` | Relative humidity around 9 AM |
| `relative_humidity_3pm` | Relative humidity around 3 PM |

---

# 🧠 Machine Learning Approach

The project treats afternoon humidity prediction as a **binary classification problem**.

The general workflow is:

``` 
Weather Dataset
       │
       ▼
Data Exploration
       │
       ▼
Data Cleaning
       │
       ▼
Handle Missing Values
       │
       ▼
Create Binary Humidity Label
       │
       ▼
Select Morning Features
       │
       ▼
Train / Test Split
       │
       ▼
Decision Tree Classifier
       │
       ▼
Generate Predictions
       │
       ▼
Evaluate Accuracy
```

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- Scikit-learn

### Development Environment

- Jupyter Notebook

### Machine Learning

- Supervised Learning
- Binary Classification
- Decision Tree Classifier

### Data Science Concepts

- Data preprocessing
- Missing-value handling
- Feature selection
- Target engineering
- Train-test splitting
- Model training
- Prediction
- Model evaluation

---

# 📂 Project Structure

``` 
Weather-Data-Classification/
│
├── Weather Data Classification using Decision Trees.ipynb
│
├── daily_weather.csv
│
└── README.md
```

### File Description

**`Weather Data Classification using Decision Trees.ipynb`**

Contains the complete machine learning workflow, including preprocessing, model training, predictions and evaluation.

**`daily_weather.csv`**

Contains the historical weather observations used for model development.

**`README.md`**

Provides documentation and instructions for understanding and running the project.

---

# 🔍 Data Exploration

The first step is importing the required Python libraries.

```python
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```

The weather dataset is then loaded into a Pandas DataFrame.

```python
data = pd.read_csv('./weather/daily_weather.csv')
```

The dataset can be inspected using:

```python
data.head()
```

This provides an initial understanding of the available variables and their values.

---

# 🧹 Data Preprocessing

Raw datasets often require preprocessing before they can be used for machine learning.

The preprocessing performed in this project includes:

1. Removing the observation identifier
2. Detecting missing values
3. Removing incomplete observations
4. Creating the target variable
5. Selecting model features

---

## Removing the Identifier Column

The dataset contains a column named:

``` 
number
```

This column is simply an observation identifier.

Since it does not represent an actual weather condition and does not provide meaningful predictive information, it is removed.

```python
del data['number']
```

---

# 🔎 Missing Value Analysis

The dataset contains missing measurements.

Missing values are identified using:

```python
data[data.isnull().any(axis=1)]
```

Rows containing missing values are removed before model training.

```python
clean_data = data.copy()
clean_data = clean_data.dropna()
```

### Dataset Before Cleaning

``` 
1,095 observations
```

### Dataset After Cleaning

``` 
1,064 observations
```

### Observations Removed

``` 
31 observations
```

Removing incomplete records ensures that the Decision Tree receives complete feature values during training.

---

# 🎯 Target Variable Engineering

The original afternoon humidity variable is:

``` 
relative_humidity_3pm
```

Because this is a continuous numerical measurement, it needs to be transformed into categories for this binary classification task.

A new variable is created:

``` 
high_humidity_label
```

The threshold used in the notebook is:

``` 
24.99%
```

The target is created using:

```python
clean_data['high_humidity_label'] = (
    clean_data['relative_humidity_3pm'] > 24.99
) * 1
```

This creates two classes:

| Class | Condition | Interpretation |
|---|---|---|
| `0` | Humidity ≤ 24.99% | Lower afternoon humidity |
| `1` | Humidity > 24.99% | Higher afternoon humidity |

---

# ⚖️ Target Distribution

After cleaning the data and creating the target variable, the classes are nearly evenly distributed.

``` 
Class 0: 535 observations
Class 1: 529 observations
```

This is useful because a highly imbalanced dataset can cause a classifier to favor the majority class.

In this project, the two classes have approximately equal representation.

---

# 🔧 Feature Selection

The model uses the following morning measurements:

```python
morning_features = [
    'air_pressure_9am',
    'air_temp_9am',
    'avg_wind_direction_9am',
    'avg_wind_speed_9am',
    'max_wind_direction_9am',
    'max_wind_speed_9am',
    'rain_accumulation_9am',
    'rain_duration_9am'
]
```

These eight variables form the feature matrix.

```python
X = clean_data[morning_features].copy()
```

The target variable is:

```python
y = clean_data[['high_humidity_label']].copy()
```

### Input

``` 
Morning Weather Conditions
```

### Output

``` 
High / Low Afternoon Humidity
```

---

# ✂️ Training and Testing Data

To evaluate the model on observations it has not seen during training, the dataset is divided into training and testing subsets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=324
)
```

The split used in this project is approximately:

``` 
Training Data → 67%
Testing Data  → 33%
```

Resulting in approximately:

| Dataset | Observations |
|---|---:|
| Training | 712 |
| Testing | 352 |
| Total | 1,064 |

The `random_state` parameter makes the split reproducible.

---

# 🌳 Decision Tree Classifier

The machine learning algorithm selected for this project is a **Decision Tree Classifier**.

Decision Trees classify observations by repeatedly dividing data according to feature values.

A simplified conceptual tree could look like:

``` 
                 Morning Weather
                       │
              ┌────────┴────────┐
              │                 │
        Condition A?        Condition B?
              │                 │
         ┌────┴────┐       ┌────┴────┐
         │         │       │         │
      Class 0   Class 1  Class 0   Class 1
```

The classifier is created as:

```python
humidity_classifier = DecisionTreeClassifier(
    max_leaf_nodes=10,
    random_state=0
)
```

The model is then trained:

```python
humidity_classifier.fit(
    X_train,
    y_train
)
```

---

# 🌿 Why Limit the Tree?

The model uses:

```python
max_leaf_nodes=10
```

A Decision Tree can continue creating branches until it closely fits the training data.

Restricting the number of leaf nodes controls the size of the tree and reduces model complexity.

In this project, the tree is restricted to a maximum of **10 terminal leaf nodes**.

---

# 🔮 Making Predictions

Once the classifier has learned patterns from the training dataset, it can make predictions on the test dataset.

```python
predictions = humidity_classifier.predict(X_test)
```

The result contains predicted class labels:

``` 
0
1
1
0
1
...
```

where:

``` 
0 → Lower afternoon humidity
1 → Higher afternoon humidity
```

---

# 📈 Model Evaluation

The predictions are compared with the actual test labels.

The notebook evaluates the model using:

```python
accuracy_score(
    y_true=y_test,
    y_pred=predictions
)
```

---

# 🏆 Model Performance

The Decision Tree achieved approximately:

``` 
Accuracy = 0.8153
```

or:

# **81.53% Test Accuracy**

This means the trained Decision Tree correctly classified approximately **81.5% of the observations in the test dataset**.

---

## 📊 Results Summary

| Metric | Result |
|---|---:|
| Original Observations | 1,095 |
| Clean Observations | 1,064 |
| Removed Observations | 31 |
| Input Features | 8 |
| Training Observations | 712 |
| Testing Observations | 352 |
| Maximum Leaf Nodes | 10 |
| Test Accuracy | **81.53%** |

---

# 🔑 Key Findings

The experiment demonstrates that the selected morning weather measurements contain useful predictive information about afternoon humidity.

Using only eight morning measurements, the Decision Tree achieved approximately **81.53% accuracy** on unseen test observations.

The result also demonstrates how a continuous environmental measurement can be converted into a classification problem by defining a threshold and creating a binary target variable.

---

# 💡 Why Decision Trees?

Decision Trees are useful for classification problems because they:

- Can model nonlinear relationships
- Work naturally with numerical features
- Require relatively little preprocessing
- Divide observations using understandable decision rules
- Can capture interactions between multiple variables
- Are straightforward to train using Scikit-learn

They also provide a useful introduction to supervised machine learning because their decision-making process can be visualized as a tree.

---

# 🚀 Installation

## Prerequisites

Make sure Python is installed.

Recommended:

``` 
Python 3.x
Jupyter Notebook
```

Install the required libraries:

```bash
pip install pandas scikit-learn jupyter
```

---

# ▶️ Running the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Enter the Project Directory

```bash
cd Weather-Data-Classification
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the Notebook

Open:

``` 
Weather Data Classification using Decision Trees.ipynb
```

Run the notebook cells sequentially.

> **Note:** Make sure the CSV path in `pd.read_csv()` matches the location of `daily_weather.csv` on the  machine.

---

# 🧪 Reproducibility

The project uses fixed random states:

```python
random_state=324
```

for the train-test split and:

```python
random_state=0
```

for the Decision Tree.

Using fixed random states helps reproduce the same data split and model behavior when the notebook is executed again under the same environment.

---

# 📚 Machine Learning Concepts Demonstrated

This project provides hands-on practice with:

### Data Preparation

- Reading CSV datasets
- Exploring tabular data
- Identifying missing values
- Removing incomplete records
- Removing irrelevant identifier fields

### Feature Engineering

- Creating a binary target
- Applying a classification threshold
- Selecting predictor variables

### Model Development

- Separating features and targets
- Creating training and testing datasets
- Building a Decision Tree
- Controlling tree complexity

### Model Evaluation

- Generating predictions
- Comparing predictions against actual labels
- Calculating classification accuracy

---

# 🔮 Future Improvements

There are several ways this project could be extended.

### 1. Confusion Matrix

A confusion matrix could show:

```
True Positives
True Negatives
False Positives
False Negatives
```

This would provide more information than accuracy alone.

### 2. Precision, Recall and F1-Score

Additional classification metrics could provide a more complete evaluation of model performance.

### 3. Decision Tree Visualization

The trained Decision Tree could be visualized to better understand which weather variables influence its decisions.

### 4. Feature Importance

Decision Tree feature importance could be examined to determine which morning measurements contribute most to the prediction.

### 5. Hyperparameter Tuning

Parameters such as:

``` 
max_depth
min_samples_split
min_samples_leaf
max_leaf_nodes
criterion
```

could be tuned systematically.

### 6. Cross-Validation

K-fold cross-validation could provide a more robust estimate of model performance than a single train-test split.

### 7. Compare Multiple Algorithms

The Decision Tree could be compared against models such as:

``` 
Logistic Regression
Random Forest
K-Nearest Neighbors
Support Vector Machine
Gradient Boosting
```

### 8. Additional Features

The dataset contains:

``` 
relative_humidity_9am
```

but this variable is not included in the notebook's selected eight model features.

A future experiment could evaluate whether including morning relative humidity improves predictive performance.

---

# 📌 Limitations

The current implementation has several limitations:

- Accuracy is the primary evaluation metric.
- Rows containing missing values are removed instead of imputed.
- The model is evaluated using a single train-test split.
- Hyperparameter optimization is not performed.
- The humidity threshold is predefined in the notebook.
- Only one machine learning algorithm is evaluated.
- The model's results apply to the dataset used in this project and should not automatically be generalized to other locations or weather datasets.

---

# 🎓 Learning Outcomes

After completing this project, you should understand how to:

1. Load structured datasets using Pandas.
2. Inspect weather sensor data.
3. Identify and handle missing values.
4. Select relevant machine learning features.
5. Convert continuous measurements into categorical labels.
6. Split datasets into training and testing sets.
7. Train a Decision Tree Classifier.
8. Generate predictions for unseen observations.
9. Calculate classification accuracy.
10. Interpret basic machine learning results.

---

# 📝 Conclusion

This project demonstrates an end-to-end machine learning classification workflow using historical weather observations.

The raw weather data is cleaned and transformed into a supervised learning problem by creating a binary target representing afternoon humidity.
Eight morning weather measurements are then used to train a **Decision Tree Classifier**.

With a maximum of 10 leaf nodes, the trained model achieves approximately:

# **81.53% accuracy**

on the held-out test dataset.

The project provides a practical example of how **Python, Pandas, Scikit-learn, data preprocessing, feature engineering and Decision Trees** can be combined to solve a real-world classification problem.

---
