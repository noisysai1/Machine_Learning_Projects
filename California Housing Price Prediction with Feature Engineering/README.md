# 🏠 California Housing Price Prediction with Feature Engineering

## 📌 Project Overview

This project demonstrates how **feature engineering can improve a machine learning model** by creating more meaningful features from existing data.

The project uses the **California Housing Dataset**, which contains housing and demographic information collected from the **1990 California Census** at the city-block level.

The main objective is to build a **Linear Regression model using TensorFlow** to predict the:

> **Median House Value (`median_house_value`)**

Instead of relying only on the original dataset features, additional features are engineered to better represent the relationship between housing characteristics and property values.

---

## 🎯 Project Objective

The primary goal of this project is to understand how creating appropriate features can help a machine learning model capture useful relationships within data.

### Learning Objectives

* Explore and understand the California Housing dataset
* Analyze descriptive statistics of housing variables
* Split data into training and evaluation datasets
* Perform feature engineering
* Create TensorFlow feature columns
* Apply bucketization to geographic data
* Build a Linear Regression model using TensorFlow Estimator
* Train and evaluate the model
* Monitor model training using TensorBoard
* Understand how engineered features can improve model representation

---

# 📊 Dataset

The project uses the **California Housing Training Dataset**.

The uploaded dataset contains:

* **17,000 observations**
* **9 original variables**

The data is based on the **1990 California Census** and represents information at the city-block level.

### Dataset Features

| Feature              | Description                            |
| -------------------- | -------------------------------------- |
| `longitude`          | Geographic longitude of the block      |
| `latitude`           | Geographic latitude of the block       |
| `housing_median_age` | Median age of houses within the block  |
| `total_rooms`        | Total number of rooms in the block     |
| `total_bedrooms`     | Total number of bedrooms in the block  |
| `population`         | Total population of the block          |
| `households`         | Number of households in the block      |
| `median_income`      | Median household income                |
| `median_house_value` | Median house value — prediction target |

---

# 🧠 Machine Learning Problem

This is a **supervised machine learning regression problem**.

The model attempts to learn the relationship between housing characteristics and:

 
median_house_value
```

### Target Variable

```python
median_house_value
```

During model training, the target is scaled:

```python
y = df['median_house_value'] / 100000
```

This means the model learns house values on a smaller numerical scale.

---

# 🔄 Project Workflow

The overall workflow followed in this project is:

 
California Housing Dataset
        ↓
Data Loading
        ↓
Data Exploration
        ↓
Training / Evaluation Split
        ↓
Feature Engineering
        ↓
TensorFlow Feature Columns
        ↓
Linear Regression Model
        ↓
Model Training
        ↓
Model Evaluation
        ↓
TensorBoard Monitoring
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* TensorFlow
* Math
* Shutil

### Machine Learning

* TensorFlow Estimator API
* Linear Regression
* Feature Engineering
* Numeric Feature Columns
* Bucketized Feature Columns

### Model Monitoring

* TensorBoard

---

# 📂 Project Structure

 
California-Housing-Feature-Engineering/
│
├── a_features.ipynb
├── california_housing_train.csv
└── README.md
```

### File Description

**`a_features.ipynb`**

Contains the complete machine learning workflow including:

* Dataset loading
* Exploratory analysis
* Train/evaluation splitting
* Feature engineering
* TensorFlow input pipeline
* Feature-column creation
* Linear Regression model
* Training and evaluation
* TensorBoard setup

**`california_housing_train.csv`**

Contains the California housing dataset used for the project.

**`README.md`**

Provides complete documentation and explanation of the project.

---

# 🚀 Step-by-Step Implementation

## 1. Import Required Libraries

The project begins by importing the required Python libraries.

```python
import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
```

These libraries are used for:

* Data manipulation
* Numerical operations
* Machine learning
* Model training
* Model directory management

---

# 2. Load the Dataset

The California Housing dataset is loaded into a Pandas DataFrame.

The notebook originally loads the dataset from an online CSV source.

```python
df = pd.read_csv(
    "https://storage.googleapis.com/ml_universities/california_housing_train.csv",
    sep=","
)
```

Once loaded, the data can be explored and prepared for machine learning.

---

# 3. Explore the Dataset

Before training a machine learning model, it is important to understand the structure and statistical properties of the data.

The first few observations are displayed using:

```python
df.head()
```

Descriptive statistics are generated using:

```python
df.describe()
```

This provides information such as:

* Count
* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

This step helps identify the general distribution and scale of the numerical features.

---

# 4. Split the Dataset

The dataset is divided into:

* **Training dataset**
* **Evaluation dataset**

A random seed is used so that the split is reproducible.

```python
np.random.seed(seed=1)

msk = np.random.rand(len(df)) < 0.8

traindf = df[msk]
evaldf = df[~msk]
```

Approximately:

 
80% → Training Data
20% → Evaluation Data
```

The training data is used to teach the model, while the evaluation data is used to assess model performance on observations not used during training.

---

# 🔧 5. Feature Engineering

Feature engineering is the central concept of this project.

Raw variables do not always provide the most useful representation for a machine learning algorithm.

Therefore, two additional features are generated.

---

## Feature 1: Average Rooms per House

The first engineered feature is:

```python
avg_rooms_per_house
```

It is calculated as:

```python
df['avg_rooms_per_house'] = (
    df['total_rooms'] / df['households']
)
```

Mathematically:

 
Average Rooms per House =
Total Rooms / Number of Households
```

### Why is this useful?

`total_rooms` alone can be misleading.

For example, a block containing 5,000 rooms may appear significant, but if that block also contains a very large number of households, individual homes may not actually be large.

By dividing rooms by households, the model gets a more meaningful representation of the approximate number of rooms associated with each household.

The notebook expects this feature to have a **positive relationship with median house value**.

---

**# 6. Average Persons per Room**

The second engineered feature is:

```python
avg_persons_per_room
```

It is calculated as:

```python
df['avg_persons_per_room'] = (
    df['population'] / df['total_rooms']
)
```

Mathematically:

 
Average Persons per Room =
Population / Total Rooms
```

### Why is this useful?

This feature provides an approximate measure of population density relative to available rooms.

The notebook expects this feature to have a **negative relationship with median house value**.

Together, these engineered variables transform raw totals into more informative ratios.

---

# 🧩 Feature Engineering Function

Both features are generated through a reusable function:

```python
def add_more_features(df):

    df['avg_rooms_per_house'] = (
        df['total_rooms'] / df['households']
    )

    df['avg_persons_per_room'] = (
        df['population'] / df['total_rooms']
    )

    return df
```

This allows the same transformations to be consistently applied when preparing data for TensorFlow.

---

# 7. Create the TensorFlow Input Function

TensorFlow requires an input pipeline for supplying features and labels to the estimator.

The project creates this using:

```python
tf.estimator.inputs.pandas_input_fn()
```

The input function:

* Applies feature engineering
* Separates input features from the target
* Scales the target variable
* Creates batches
* Controls epochs
* Shuffles observations

Example:

```python
def make_input_fn(df, num_epochs):

    return tf.estimator.inputs.pandas_input_fn(

        x=add_more_features(df),

        y=df['median_house_value'] / 100000,

        batch_size=128,

        num_epochs=num_epochs,

        shuffle=True,

        queue_capacity=1000,

        num_threads=1
    )
```

---

# 8. TensorFlow Feature Columns

The project defines the features used by the Linear Regression model through TensorFlow feature columns.

```python
def create_feature_cols():

    return [

        tf.feature_column.numeric_column(
            'housing_median_age'
        ),

        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('latitude'),
            boundaries=np.arange(32.0, 42, 1).tolist()
        ),

        tf.feature_column.numeric_column(
            'avg_rooms_per_house'
        ),

        tf.feature_column.numeric_column(
            'avg_persons_per_room'
        ),

        tf.feature_column.numeric_column(
            'median_income'
        )
    ]
```

The model therefore uses:

 
housing_median_age
latitude
avg_rooms_per_house
avg_persons_per_room
median_income
```

---

# 🌎 9. Bucketizing Latitude

One interesting feature transformation used in this project is **bucketization**.

Instead of treating latitude only as one continuous numeric value, latitude values are grouped into intervals.

```python
tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    boundaries=np.arange(32.0, 42, 1).tolist()
)
```

The boundaries are created approximately between:

32° → 42°

```

in one-degree intervals.

### Why bucketize latitude?

Housing prices can vary significantly by geographic region.

Bucketization allows the model to represent different latitude ranges separately instead of assuming that every one-unit change in latitude has exactly the same linear effect.

---

# 🤖 10. Build the Linear Regression Model

The machine learning model used in this project is:

```python
tf.estimator.LinearRegressor
```

The estimator is initialized with the feature columns created earlier.

```python
estimator = tf.estimator.LinearRegressor(
    model_dir=output_dir,
    feature_columns=create_feature_cols()
)
```

Linear Regression attempts to model the target as a weighted combination of the selected features.

Conceptually:

 
Predicted House Value =
    w₁(Housing Age)
  + w₂(Latitude)
  + w₃(Average Rooms per House)
  + w₄(Average Persons per Room)
  + w₅(Median Income)
  + Bias
```

---

# 11. Configure Model Training

TensorFlow's `TrainSpec` is used to define the training process.

```python
train_spec = tf.estimator.TrainSpec(
    input_fn=make_input_fn(traindf, None),
    max_steps=num_train_steps
)
```

The project trains the model for:

 
2,000 training steps
```

---

# 12. Configure Model Evaluation

An evaluation specification is also created:

```python
eval_spec = tf.estimator.EvalSpec(
    input_fn=make_input_fn(evaldf, 1),
    steps=None,
    start_delay_secs=1,
    throttle_secs=5
)
```

This evaluates the trained model using the held-out evaluation dataset.

---

# 13. Train and Evaluate

Training and evaluation are combined using:

```python
tf.estimator.train_and_evaluate(
    estimator,
    train_spec,
    eval_spec
)
```

This allows TensorFlow to manage both the training and evaluation workflow.

---

# 📈 14. TensorBoard

TensorBoard is used to monitor the model training process.

The model output directory is defined as:

```python
OUTDIR = './trained_model'
```

TensorBoard is then started for this directory.

```python
TensorBoard().start(OUTDIR)
```

TensorBoard can help visualize information generated during model training, including training progress and model metrics.

---

# 🧹 15. Start Training from a Clean Model Directory

Before training, the previous model directory is removed.

```python
shutil.rmtree(
    OUTDIR,
    ignore_errors=True
)
```

The TensorFlow summary cache is also cleared:

```python
tf.summary.FileWriterCache.clear()
```

This ensures that each run begins cleanly rather than mixing new model information with files from previous experiments.

---

# ▶️ 16. Run the Model

Finally, the complete training and evaluation pipeline is executed:

```python
train_and_evaluate(
    OUTDIR,
    2000
)
```

The Linear Regression model is therefore trained for **2,000 steps** using the engineered housing features.

---

# 💡 Key Feature Engineering Insights

One of the most important lessons from this project is that **the representation of data matters**.

Consider the following original variables:

 
total_rooms
households
population
```

Individually, these values describe the scale of a city block.

However, combining them creates more meaningful information:

 
total_rooms / households
        ↓
Average Rooms per House
```

and:

 
population / total_rooms
        ↓
Average Persons per Room
```

These ratios can provide the model with information about housing size and density that raw totals may not express as effectively.

---

# 🔍 Features Used by the Final Model

Although the original dataset contains nine columns, the model intentionally uses a selected and transformed subset:

| Model Feature          | Type               |
| ---------------------- | ------------------ |
| `housing_median_age`   | Numeric            |
| `latitude`             | Bucketized         |
| `avg_rooms_per_house`  | Engineered Numeric |
| `avg_persons_per_room` | Engineered Numeric |
| `median_income`        | Numeric            |

Target:

 
median_house_value
```

This demonstrates that machine learning is not simply about passing every available column into an algorithm.

Selecting and representing features appropriately is an important part of model development.

---

# 📊 Dataset → Model Transformation


Original Dataset
│
├── housing_median_age ────────────────┐
│                                      │
├── latitude ──> Bucketization ────────┤
│                                      │
├── total_rooms ───────┐               │
│                      ├─> Rooms/House ┤
├── households ────────┘               │
│                                      ├──> Linear Regression
├── population ────────┐               │
│                      ├─> Persons/Room│
├── total_rooms ───────┘               │
│                                      │
├── median_income ─────────────────────┘
│
└── median_house_value ─────────> TARGET
```

---

# 📌 Key Concepts Demonstrated

This project demonstrates several important machine learning concepts:

### Data Exploration

Understanding the dataset before training the model.

### Train/Evaluation Split

Separating data used for learning from data used for evaluation.

### Feature Engineering

Creating useful variables from existing data.

### Ratio Features

Transforming raw totals into more meaningful per-household or per-room measurements.

### Feature Bucketization

Converting continuous geographic information into discrete intervals.

### Regression

Predicting a continuous numerical value.

### TensorFlow Input Pipelines

Supplying Pandas data to TensorFlow estimators.

### Model Evaluation

Evaluating model behavior using data separate from the training observations.

### TensorBoard

Monitoring information generated during model training.

---

# 🎯 Why This Project Is Useful

This project provides a practical introduction to one of the most important concepts in machine learning:

> **Better features can help a model represent the underlying problem more effectively.**

Real-world machine learning projects often require more than choosing an algorithm.

A large part of the work involves:


Understanding Data
        +
Selecting Features
        +
Transforming Features
        +
Engineering New Features
        +
Training the Model
        +
Evaluating the Results
```

The California Housing dataset provides a useful example because demographic, geographic and housing characteristics can be transformed into variables with clearer real-world interpretations.

---

# 🚀 Possible Future Improvements

The project can be extended in several ways:

* Add additional engineered housing features
* Explore longitude together with latitude
* Perform correlation analysis
* Visualize geographic housing-price patterns
* Compare raw features against engineered features
* Experiment with different latitude bucket sizes
* Evaluate additional regression algorithms
* Add MAE, MSE and RMSE comparisons
* Perform hyperparameter tuning
* Compare TensorFlow Linear Regression with Scikit-learn models
* Experiment with tree-based regression models
* Build a neural-network regression model
* Add prediction visualizations
* Modernize the TensorFlow implementation for current TensorFlow/Keras APIs

---

# 🧪 Example Additional Features for Future Work

Additional features that could be explored include:

```python
bedrooms_per_room = total_bedrooms / total_rooms
```

```python
population_per_household = population / households
```

```python
rooms_per_person = total_rooms / population
```

These are suggestions for future experiments and are **not part of the current notebook implementation**.

---

# 📚 What I Learned

Through this project, I practiced:

* Loading and exploring real-world housing data
* Preparing data for machine learning
* Creating reproducible training/evaluation splits
* Engineering meaningful features
* Understanding feature relationships
* Creating TensorFlow numeric feature columns
* Bucketizing continuous features
* Building a TensorFlow Linear Regression estimator
* Creating TensorFlow input functions
* Training and evaluating regression models
* Monitoring model training with TensorBoard

The project reinforced the importance of **feature engineering and feature representation in machine learning**.

---

# 🏁 Conclusion

This project builds a **TensorFlow Linear Regression model for California housing-price prediction** while focusing primarily on feature engineering.

The original housing data is transformed to create:

 
avg_rooms_per_house
avg_persons_per_room
```

and latitude is represented through bucketized feature columns.

These transformations demonstrate how domain-aware feature engineering can provide machine learning algorithms with more useful representations than raw variables alone.

Overall, the project provides a strong foundation for understanding:

**Data Exploration → Feature Engineering → Feature Representation → Regression → Model Evaluation**

---

## 👨‍💻 Tools Used

Developed as a hands-on Machine Learning project focused on **Feature Engineering, TensorFlow and Regression Analysis**.
