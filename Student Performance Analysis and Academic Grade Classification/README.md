# Student Performance Analysis and Academic Grade Prediction Using Python

## 1. Project Overview

Student academic performance is an important indicator of educational progress and learning outcomes. Understanding the factors associated with student achievement can help educators, researchers and educational institutions identify academic patterns, evaluate learning outcomes and explore opportunities for student support.

This project focuses on analyzing student examination performance using Python, statistical techniques, exploratory data analysis (EDA), data visualization and machine learning.

The dataset contains information about **1,000 students**, including their mathematics, reading and writing examination scores, along with demographic and educational attributes such as gender, race/ethnicity group, parental education level, lunch category and test preparation course completion.

The project investigates academic performance patterns across these attributes, evaluates relationships between examination scores, calculates statistical measures and develops classification models to categorize students based on academic grades.

The analysis is performed using Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, SciPy and Scikit-learn.

### Key Project Highlights

- Analyzed 1,000 student records across 8 original attributes.
- Performed descriptive and inferential statistical analysis.
- Examined academic performance across different student groups.
- Created visualizations to understand score distributions and relationships.
- Developed additional features such as total marks, percentage, pass/fail status and academic grades.
- Applied categorical encoding and feature scaling for machine learning.
- Implemented Logistic Regression and Random Forest classification models.
- Compared classification model accuracy.
- Identified limitations associated with using score-derived features to predict academic grades.

---

## 2. Problem Statement

Student performance can vary across subjects and demographic or educational categories. Identifying these variations through data analysis can provide a clearer understanding of academic outcomes.

The primary problem addressed in this project is understanding how students perform across mathematics, reading and writing examinations and how their performance differs across available demographic and educational attributes.

The project also explores whether machine learning algorithms can classify academic grade categories using information available in the dataset.

### Research Questions

1. What is the average student performance in mathematics, reading and writing?
2. Which subject has the highest and lowest average score?
3. How does academic performance vary by gender?
4. Are there differences in examination scores across race/ethnicity groups?
5. How is parental education level associated with student performance?
6. How do students with completed test preparation courses compare with those who did not complete them?
7. What differences are observed between standard lunch and free/reduced lunch categories?
8. What proportion of students achieve different passing thresholds?
9. How are mathematics, reading and writing scores related?
10. Can classification algorithms predict grade categories derived from students' academic scores?

---

## 3. Project Objectives

The main objective is to perform an end-to-end analysis of student examination performance and explore academic grade classification using machine learning.

### Specific Objectives

**Data Understanding**

Understand the structure of the dataset, identify numerical and categorical attributes and inspect data quality.

**Statistical Analysis**

Calculate descriptive statistics, analyze score distributions, estimate sample statistics and examine score-related probabilities.

**Exploratory Data Analysis**

Explore relationships between academic scores and demographic or educational categories using visualizations and grouped statistics.

**Feature Engineering**

Create new variables representing academic achievement, including total marks, percentage, subject passing status, overall result and grades.

**Machine Learning**

Prepare the dataset for classification and implement Logistic Regression and Random Forest models.

**Model Evaluation**

Compare model predictions against actual grade categories and evaluate their classification accuracy.

---

## 4. Dataset Description

The dataset used in this project is `StudentsPerformance.csv`.

It contains 1,000 student records and 8 original attributes.

### Dataset Summary

| Property | Description |
|---|---|
| Dataset | StudentsPerformance.csv |
| Total records | 1,000 |
| Original attributes | 8 |
| Numerical attributes | 3 |
| Categorical attributes | 5 |
| Missing values | 0 |
| Academic subjects | Mathematics, Reading, Writing |
| Data format | CSV |

### Dataset Attributes

| Attribute | Data Type | Description |
|---|---|---|
| gender | Categorical | Gender of the student |
| race/ethnicity | Categorical | Student demographic group, categorized from A to E |
| parental level of education | Categorical | Highest reported parental education category |
| lunch | Categorical | Standard or free/reduced lunch category |
| test preparation course | Categorical | Whether the test preparation course was completed |
| math score | Numerical | Mathematics examination score |
| reading score | Numerical | Reading examination score |
| writing score | Numerical | Writing examination score |

### Dataset Characteristics

The dataset includes student examination results and background information that can be used to explore academic performance.

The three numerical attributes represent examination scores, while the five categorical attributes provide contextual information about student groups.

The original dataset does not contain the final academic grade used in the classification stage. Grades are generated during feature engineering using the examination scores and grading rules implemented in the notebook.

---

## 5. Technologies and Libraries Used

### Programming Language

**Python**

Python is used for data manipulation, statistical analysis, visualization, feature engineering and machine learning.

### Development Environment

**Jupyter Notebook**

Jupyter Notebook provides an interactive environment for executing Python code, displaying tables, creating visualizations and documenting analytical findings.

### Python Libraries

| Library | Purpose |
|---|---|
| Pandas | Data loading, cleaning, manipulation and aggregation |
| NumPy | Numerical calculations and statistical operations |
| Matplotlib | Creating charts and visualizations |
| Seaborn | Statistical visualizations and grouped comparisons |
| SciPy | Statistical analysis and probability-related calculations |
| Scikit-learn | Data preprocessing, machine learning and model evaluation |

### Machine Learning Algorithms

- Logistic Regression
- Random Forest Classifier

### Machine Learning Preprocessing

- Label Encoding
- Categorical feature mapping
- Train-test splitting
- MinMax scaling

---

## 6. Project Methodology

The project follows a structured data science workflow.

**Step 1:** Import required Python libraries.

**Step 2:** Load the student performance dataset.

**Step 3:** Inspect the dataset structure and identify missing values.

**Step 4:** Perform descriptive statistical analysis.

**Step 5:** Explore sample statistics, score thresholds and probability-related measures.

**Step 6:** Perform exploratory data analysis using visualizations and grouped comparisons.

**Step 7:** Create additional academic performance features.

**Step 8:** Encode categorical attributes.

**Step 9:** Divide the data into training and testing subsets.

**Step 10:** Apply feature scaling.

**Step 11:** Train Logistic Regression and Random Forest classification models.

**Step 12:** Evaluate and compare model performance.

**Step 13:** Interpret findings and identify methodological limitations.

---

## 7. Data Loading and Initial Exploration

The first stage involves loading the dataset into a Pandas DataFrame.

```python
import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")
```

After loading the data, the dataset is inspected using common Pandas operations.

```python
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
```

### Explanation

`head()` displays the first five records, helping identify the dataset structure and column names.

`shape` returns the number of rows and columns.

`info()` provides information about column types and non-null values.

`describe()` calculates summary statistics for numerical attributes.

`isnull().sum()` checks the number of missing values in each column.

### Findings

The dataset contains 1,000 records and 8 original attributes.

No missing values were identified in the dataset.

The numerical score attributes can therefore be used for the analysis without performing missing-value imputation.

The categorical variables are retained for grouped analysis and subsequently transformed for machine learning.

---

## 8. Descriptive Statistical Analysis

Descriptive statistics summarize the main characteristics of the academic examination scores.

The project calculates measures of central tendency and dispersion for mathematics, reading and writing.

### Statistical Measures

**Mean**

Represents the average score of students in a particular subject.

**Median**

Represents the middle score after arranging observations in ascending order.

**Minimum and Maximum**

Identify the lowest and highest observed examination scores.

**Standard Deviation**

Measures the variation in examination scores around their mean.

**Quartiles**

Describe the distribution of scores across different portions of the dataset.

### Average Examination Scores

| Subject | Average Score |
|---|---:|
| Mathematics | 66.089 |
| Reading | 69.169 |
| Writing | 68.054 |

### Interpretation

Reading has the highest average score, approximately 69.17.

Writing follows with an average score of approximately 68.05.

Mathematics has the lowest average score, approximately 66.09.

These results suggest that students in this dataset performed slightly better in reading and writing than in mathematics.

However, differences in average scores do not establish the reasons for those differences.

### Importance of Descriptive Statistics

Descriptive analysis establishes a foundation for the remaining project.

It helps identify general performance patterns, understand score variation and determine which relationships should be investigated further.

---

## 9. Inferential Statistical Analysis

The notebook explores inferential statistical concepts using student examination scores.

This includes sample-based analysis, confidence intervals and calculations related to score thresholds.

### Sample-Based Analysis

The project examines samples from the examination score distributions and calculates sample means.

Sampling helps demonstrate how statistical estimates can be calculated from a subset of observations.

### Confidence Intervals

Confidence intervals provide a range of plausible values for a population parameter under a specified sampling model.

The notebook explores confidence interval calculations as part of its statistical analysis.

These calculations are educational demonstrations. Their broader interpretation depends on assumptions about the sampling process and population.

### Score Threshold Analysis

The project evaluates the proportion of students whose examination scores exceed particular thresholds.

| Condition | Observed Percentage |
|---|---:|
| Mathematics score above 50 | 85.0% |
| Reading score above 50 | 90.3% |
| Writing score above 50 | 87.6% |
| Scores above 40 in all three subjects | 93.9% |
| Scores above 90 in all three subjects | 2.3% |

### Interpretation

Reading has the highest percentage of students scoring above 50.

Mathematics has the lowest percentage of students scoring above 50 among the three subjects.

A large majority of students score above 40 in all three subjects.

Only a small percentage of students achieve scores above 90 in all three subjects.

These results illustrate the distribution of academic achievement within the observed dataset.

---

## 10. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is one of the major components of this project.

EDA is used to identify patterns, compare groups, examine score distributions and understand relationships between variables.

The notebook uses Pandas, Matplotlib and Seaborn for statistical exploration and visualization.

### 10.1 Gender-Based Performance Analysis

The analysis compares academic performance across male and female student groups.

Mathematics, reading and writing scores are examined to understand differences in score distributions and grouped performance measures.

**Purpose:** Identify whether academic performance patterns differ across gender categories in the dataset.

**Interpretation:** Any observed differences describe the sample and should not be treated as evidence that gender causes academic performance differences.

### 10.2 Race/Ethnicity Group Analysis

Students are grouped into five categories, labeled A through E.

The analysis compares examination results across these groups.

**Purpose:** Understand how mathematics, reading and writing performance varies across the available group labels.

**Interpretation:** These group labels are dataset categories and should not be interpreted as detailed demographic identities beyond the information provided.

### 10.3 Parental Education Analysis

The notebook explores academic performance across different parental education categories.

The dataset includes educational qualifications such as high school, some college, associate's degree, bachelor's degree and master's degree.

**Purpose:** Examine whether student examination outcomes differ across parental education categories.

**Interpretation:** The analysis can identify associations, but it cannot determine whether parental education independently causes differences in examination performance.

### 10.4 Lunch Category Analysis

The dataset includes two lunch categories:

- Standard
- Free/reduced

The notebook compares student examination scores across these categories.

**Purpose:** Explore differences in academic outcomes between the two lunch groups.

**Interpretation:** Lunch category can be examined as a contextual variable, but the dataset does not establish individual household income or the causes of score differences.

### 10.5 Test Preparation Course Analysis

Students are categorized according to whether they completed a test preparation course.

The analysis compares examination performance between students who completed the course and those who did not.

**Purpose:** Investigate the association between test preparation completion and academic scores.

**Interpretation:** The comparison is observational and does not establish the causal effectiveness of the course.

### 10.6 Score Distribution Analysis

Score distributions help visualize the spread of academic performance.

The notebook uses statistical charts to explore how examination scores are distributed across students.

These visualizations help identify score concentrations, variation and differences between subjects.

### 10.7 Subject Relationship Analysis

The project examines relationships between mathematics, reading and writing performance.

This analysis helps investigate whether students who perform well in one subject also tend to perform well in other subjects.

Relationships between variables can provide useful descriptive insights but should not be interpreted as causal relationships.

---

## 11. Data Visualization

Data visualization is used throughout the project to communicate analytical findings.

### Visualization Techniques

**Histograms**

Used to understand the distribution of numerical examination scores.

**Count Plots**

Used to examine the frequency of observations in categorical variables.

**Box Plots**

Used to visualize score distributions, medians, quartiles and potential outliers.

**Bar Charts**

Used to compare aggregated performance measures across categories.

**Grouped Visualizations**

Used to compare academic outcomes across combinations of student attributes.

### Benefits of Visualization

Visualizations make statistical results easier to interpret and help reveal patterns that may not be immediately visible in raw tables.

They also provide a clear way to communicate academic performance differences to technical and nontechnical audiences.

---

## 12. Feature Engineering

Feature engineering transforms the original dataset into a form suitable for additional analysis and academic grade classification.

The notebook creates several variables derived from mathematics, reading and writing scores.

### 12.1 Subject Passing Status

Individual passing indicators are created for the three academic subjects.

The grading logic treats scores below 40 as failures.

These indicators are used to identify whether a student has passed each subject.

### 12.2 Total Marks

A total score is calculated by adding mathematics, reading and writing scores.

```python
df["Total_Marks"] = (
    df["math score"]
    + df["reading score"]
    + df["writing score"]
)
```

The maximum possible total is 300.

### 12.3 Percentage

The total marks are converted into a percentage.

```python
df["Percentage"] = df["Total_Marks"] / 3
```

Because each subject is scored out of 100, dividing the total by three produces the overall average percentage.

### 12.4 Overall Passing Status

The overall passing result is determined from the individual subject passing indicators.

A student who fails at least one subject is classified as having failed overall under the notebook's grading logic.

### 12.5 Academic Grade Generation

The project creates an academic grade variable using percentage-based grading conditions and overall passing status.

The grade labels used in the notebook include:

| Grade | Category |
|---|---|
| O | Highest grade category |
| A | High achievement category |
| B | Good achievement category |
| C | Intermediate achievement category |
| D | Lower passing category |
| E | Failing category |

The exact percentage boundaries are defined by the notebook's grading function.

### Importance of Feature Engineering

Feature engineering provides a structured representation of student academic achievement.

It also creates the target grade labels used in the machine learning classification stage.

Because the grades are calculated from examination scores, this relationship must be considered when interpreting model performance.

---

## 13. Machine Learning Implementation

The machine learning stage explores the classification of students into academic grade categories.

Two supervised classification algorithms are implemented:

1. Logistic Regression
2. Random Forest Classifier

### 13.1 Data Preprocessing

Machine learning algorithms require numerical input features.

The notebook transforms categorical attributes using label encoding and manual category mappings.

These transformations convert category labels into numerical representations.

### 13.2 Train-Test Split

The dataset is divided into training and testing subsets.

| Subset | Percentage |
|---|---:|
| Training | 75% |
| Testing | 25% |

The training subset is used to fit the classification models.

The testing subset is used to evaluate predictions on observations not included in model training.

### 13.3 Feature Scaling

MinMax scaling is applied during preprocessing.

MinMax scaling transforms numerical feature values into a specified range, commonly between 0 and 1.

This can be useful for algorithms whose behavior depends on feature magnitude.

For a more robust implementation, the scaler should be fitted only on the training data and then applied to the testing data.

---

## 14. Logistic Regression Model

Logistic Regression is a supervised machine learning algorithm used for classification.

Despite its name, Logistic Regression is commonly used to predict categorical outcomes rather than continuous numerical values.

### Model Objective

The objective is to classify students into academic grade categories based on the selected input features.

### Implementation Process

1. Prepare the numerical feature matrix.
2. Separate input features and target grades.
3. Split the dataset into training and testing subsets.
4. Apply preprocessing.
5. Train the Logistic Regression classifier.
6. Generate predictions on the testing dataset.
7. Compare predicted grades with actual grade labels.

### Model Result

**Recorded Logistic Regression accuracy: 76.8%**

### Interpretation

The model correctly classified approximately 76.8% of the testing observations in the recorded notebook run.

This result provides a baseline for comparing a linear classification approach against a more flexible ensemble-based algorithm.

---

## 15. Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees.

Each tree learns patterns from the training data and their predictions are combined to produce the final classification.

### Model Objective

The objective is to classify students into academic grade categories using the selected features.

### Implementation Process

1. Prepare the training and testing datasets.
2. Initialize the Random Forest classifier.
3. Train the model using the training data.
4. Predict grade categories for the testing data.
5. Calculate classification accuracy.
6. Compare the results with Logistic Regression.

### Model Result

**Recorded Random Forest accuracy: 100%**

### Interpretation

The recorded Random Forest model classified all testing observations correctly.

However, this result requires careful interpretation.

The academic grade labels were generated from examination scores and related engineered variables.

When those same score-derived variables are used as model inputs, the model can learn the rules used to construct the target.

This creates a form of **target leakage**.

Therefore, the 100% accuracy should not be interpreted as evidence that Random Forest can perfectly predict future academic performance.

Instead, it demonstrates that the model can reproduce the score-based grading relationships represented in the dataset.

---

## 16. Model Performance Comparison

| Model | Recorded Accuracy |
|---|---:|
| Logistic Regression | 76.8% |
| Random Forest Classifier | 100% |

### Comparative Analysis

Random Forest achieved a higher recorded classification accuracy than Logistic Regression.

This difference is consistent with the ability of tree-based algorithms to represent complex decision boundaries and threshold-based rules.

However, because the target grades are derived from score-related input features, the comparison does not represent a leakage-free assessment of future academic performance prediction.

### Key Learning

A high accuracy score does not automatically indicate that a model is useful for real-world prediction.

The relationship between model inputs and target variables must be examined to ensure that the evaluation reflects the intended prediction task.

---

## 17. Key Project Findings

### Finding 1: Reading Has the Highest Average Score

The average reading score is approximately 69.17, followed by writing at 68.05 and mathematics at 66.09.

### Finding 2: Mathematics Has the Lowest Average Score

Among the three subjects, mathematics has the lowest observed average.

### Finding 3: Most Students Exceed the Selected Score Thresholds

A large proportion of students score above 50 in individual subjects.

The notebook also records that 93.9% score above 40 in all three subjects.

### Finding 4: High Scores Across All Subjects Are Less Common

Only 2.3% of students score above 90 in mathematics, reading and writing simultaneously.

### Finding 5: Student Attributes Support Group-Based Analysis

Gender, race/ethnicity group, parental education, lunch category and test preparation completion provide useful dimensions for comparing academic outcomes.

These comparisons are descriptive and do not establish causal relationships.

### Finding 6: Engineered Academic Grades Can Be Classified

The classification models demonstrate how supervised machine learning can be applied to grade labels derived from examination scores.

### Finding 7: Target Leakage Affects Model Interpretation

The Random Forest's perfect recorded accuracy highlights the importance of understanding how a target variable is generated before interpreting model performance.

---

## 18. Challenges and Limitations

### 18.1 Limited Dataset Scope

The dataset contains 1,000 records and does not include several potentially important variables such as attendance, study hours, previous academic achievement, teaching quality, or school characteristics.

### 18.2 Observational Data

The analysis identifies patterns and associations but does not establish causal relationships.

### 18.3 Derived Target Variable

The academic grade target is calculated from existing examination scores.

This means the classification task is different from predicting future examination results.

### 18.4 Target Leakage

Some model inputs contain information directly related to the target grade.

This can produce unusually high classification accuracy and limit the practical meaning of the model evaluation.

### 18.5 Categorical Encoding

Label encoding introduces numerical representations for categorical values.

For nominal variables, these representations may create artificial ordering that does not exist in the original data.

### 18.6 Model Evaluation Scope

Accuracy is useful but does not provide a complete assessment of multiclass classification performance.

Precision, recall, F1-score and confusion matrices would provide additional information.

### 18.7 Notebook Compatibility

Some code uses older library syntax and environment-specific paths.

Minor updates may be needed to run the notebook successfully in newer Python environments.

---

## 19. Future Enhancements

### Improved Preprocessing

Use a Scikit-learn preprocessing pipeline to separate training and testing transformations and reduce the risk of data leakage.

### Leakage-Free Prediction

Redesign the machine learning task to predict an outcome that is not directly calculated from the input features.

For example, a future dataset containing attendance, study hours and previous academic scores could support prediction of later examination performance.

### Additional Classification Metrics

Evaluate models using:

- Precision
- Recall
- F1-score
- Confusion matrix
- Cross-validation

### Additional Machine Learning Models

Explore algorithms such as Decision Trees, Support Vector Machines, Gradient Boosting and other appropriate classification methods.

### Hyperparameter Optimization

Apply GridSearchCV or RandomizedSearchCV to identify suitable model configurations.

### Interactive Dashboard

Develop an interactive dashboard using Power BI, Tableau, or Streamlit to visualize student performance patterns.

### Reproducible Project Structure

Improve reproducibility by organizing preprocessing, feature engineering, model training and evaluation into reusable Python modules.

### Model Explainability

Explore feature importance and explainability techniques to understand the relationships learned by classification models.

---

## 20. Project Structure

```text
student-performance-analysis-and-prediction/
│
├── StudentsPerformance.csv
│
├── student-performance-analysis.ipynb
│
├── README.md
│

```

### File Descriptions

**StudentsPerformance.csv**

Contains the original student examination records.

**student-performance-analysis.ipynb**

Contains the project implementation, including statistical analysis, exploratory data analysis, visualizations, feature engineering and machine learning.

**README.md**

Documents the project objectives, dataset, methodology, findings, results, limitations and execution instructions.


## 21. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-analysis-and-prediction.git
```

### Step 2: Navigate to the Project Directory

```bash
cd student-performance-analysis-and-prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 7: Open the Notebook

Open:

`student-performance-analysis.ipynb`

### Step 8: Verify the Dataset Path

Ensure the notebook reads the CSV file from the repository directory:

```python
df = pd.read_csv("StudentsPerformance.csv")
```

Replace any environment-specific dataset paths if necessary.

### Step 9: Execute the Notebook

Run the notebook cells sequentially to reproduce the data exploration, statistical analysis, visualizations, feature engineering and model training.

Some older plotting or library syntax may require compatibility updates.

---

## 22. Skills Demonstrated

This project demonstrates practical knowledge of several important data science concepts.

**Data Analysis**

Data inspection, filtering, grouping, aggregation and transformation using Pandas.

**Statistics**

Descriptive statistics, sampling, probability calculations and confidence intervals.

**Exploratory Data Analysis**

Identifying score distributions and relationships between student attributes and academic outcomes.

**Data Visualization**

Creating and interpreting statistical visualizations using Matplotlib and Seaborn.

**Feature Engineering**

Transforming examination scores into academic performance indicators and grade categories.

**Machine Learning**

Implementing supervised classification models using Scikit-learn.

**Model Evaluation**

Comparing model predictions and interpreting classification accuracy.

**Analytical Thinking**

Identifying limitations, interpreting statistical results carefully and recognizing potential target leakage.

---

## 23. Project Conclusion

This project presents an end-to-end exploration of student academic performance using Python and machine learning.

The analysis begins with understanding a dataset containing 1,000 student examination records and progresses through descriptive statistics, inferential statistical calculations, exploratory data analysis, visualization, feature engineering and classification modeling.

The statistical findings show that reading has the highest average score among the three academic subjects, while mathematics has the lowest.

The exploratory analysis provides a structured approach to examining academic outcomes across demographic and educational categories.

Feature engineering transforms examination scores into additional performance indicators, including total marks, percentages, passing status and academic grades.

Logistic Regression and Random Forest classifiers are implemented to explore academic grade classification, producing recorded accuracies of 76.8% and 100%, respectively.

The project also demonstrates an important machine learning principle: model accuracy must be interpreted in the context of how the target variable is constructed and which features are available to the model.

Overall, this project showcases foundational skills in Python programming, statistical analysis, exploratory data analysis, feature engineering, data visualization and supervised machine learning.

It also provides a foundation for future work involving improved data pipelines, leakage-free prediction, advanced model evaluation and interactive educational analytics dashboards.

This project is intended for educational and portfolio purposes.

The analysis describes patterns within the provided dataset and should not be used to make high-stakes decisions about individual students.

The machine learning results demonstrate classification of score-derived academic grades rather than validated prediction of future student achievement.

Any real-world educational application would require additional data, validation, fairness assessment and appropriate human oversight.
