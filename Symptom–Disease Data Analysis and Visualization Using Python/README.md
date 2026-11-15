# Symptom–Disease Data Analysis and Visualization Using Python

An exploratory healthcare data analysis project focused on understanding medical symptoms, disease information, diagnosis identifiers and symptom–disease 
relationships using Python, Pandas, NumPy, Matplotlib and Jupyter Notebook.

## Project Overview

Healthcare datasets contain valuable information about symptoms, diseases and their relationships. Organizing and analyzing this information can help researchers
and data analysts better understand how medical information is structured and identify patterns that may support further research.

This project explores a collection of CSV datasets containing medical symptoms, disease descriptions, diagnosis identifiers and weighted relationships between symptoms and diseases.

The primary objective is to examine the structure of the datasets, understand their contents and explore suitable visualization techniques using Python.

The project includes two Jupyter notebooks that demonstrate data loading, inspection and exploratory visualization.

**Project Category:** Healthcare Data Analytics / Exploratory Data Analysis (EDA)

**Programming Language:** Python

**Development Environment:** Jupyter Notebook

**Core Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn

## Project Objectives

The main objectives of this project are:

1. Explore structured healthcare datasets containing symptoms and disease information.
2. Understand how symptoms and diagnoses are represented using unique identifiers.
3. Examine the relationships between symptoms and diseases.
4. Inspect dataset dimensions, column names and sample records.
5. Develop reusable Python functions for exploratory data visualization.
6. Explore distributions, correlations and scatterplot relationships where the data supports them.
7. Identify data-quality considerations that may affect future analysis.
8. Establish a foundation for further healthcare analytics and symptom–disease modeling.

## Dataset Description

The project contains eight CSV datasets. Each dataset serves a different purpose in representing medical symptoms, diseases, or relationships between them.

### Dataset Summary

| Dataset | Rows | Columns | Description |
|---|---:|---:|---|
| `dia_3(1).csv` | 110 | 2 | Selected diagnosis identifiers and descriptions |
| `dia_t.csv` | 1,166 | 2 | Disease and diagnosis reference information |
| `diagn_title(1).csv` | 1,314 | 2 | Diagnosis identifiers and corresponding titles |
| `diffsydiw(1).csv` | 5,569 | 3 | Weighted symptom–disease relationships |
| `sym_3(1).csv` | 131 | 2 | Selected symptoms and identifiers |
| `sym_dis_matrix(1).csv` | 131 | 110 | Symptom–disease relationship matrix |
| `sym_t(1).csv` | 272 | 2 | Symptom reference information |
| `symptoms2(1).csv` | 389 | 9 | Detailed symptom and chief-complaint information |

*The row counts above represent the uploaded CSV files, including records containing missing values.*

### 1. Disease Dataset — `dia_3(1).csv`

This dataset contains a selected collection of diagnosis records.

**Columns:**
- `_id`: Diagnosis identifier
- `diagnose`: Diagnosis name or description

The dataset contains 110 rows and includes medical conditions such as ACE inhibitor-induced cough, acute angle-closure glaucoma and alcoholism.

It can be used to identify a subset of diagnoses for focused analysis and to map diagnosis identifiers to descriptive names.

### 2. Disease Reference Dataset — `dia_t.csv`

This dataset contains 1,166 diagnosis records.

**Columns:**
- `did`: Disease or diagnosis identifier
- `diagnose`: Disease name or description

It provides a broader reference list of medical conditions and supports interpreting diagnosis identifiers in other datasets.

### 3. Diagnosis Titles — `diagn_title(1).csv`

This dataset contains 1,314 rows of diagnosis titles.

**Columns:**
- `id`: Diagnosis identifier
- `title`: Diagnosis title

It provides an additional diagnosis reference table. The data can be used to examine diagnosis terminology and compare identifiers across the available disease datasets.

### 4. Symptom–Disease Weighted Relationships — `diffsydiw(1).csv`

This dataset contains 5,569 rows representing symptom–disease associations.

**Columns:**
- `syd`: Symptom identifier
- `did`: Disease identifier
- `wei`: Numerical relationship weight

Each populated record connects a symptom identifier to a disease identifier with an associated weight.

This dataset is particularly relevant for studying symptom–disease relationships. The exact clinical meaning and calculation method of the weights are not documented in the supplied files, so they should not automatically be interpreted as probabilities or diagnostic confidence scores.

### 5. Selected Symptoms — `sym_3(1).csv`

This dataset contains 131 symptom records.

**Columns:**
- `_id`: Symptom identifier
- `symptom`: Symptom description

Examples include upper abdominal pain, lower abdominal pain and alcohol abuse.

It provides a smaller symptom reference set that can be used alongside the selected diagnosis data.

### 6. Symptom–Disease Matrix — `sym_dis_matrix(1).csv`

This dataset represents symptom–disease relationships in matrix form.

**Dimensions:** 131 rows × 110 columns

- The first column, `eye`, contains symptom identifiers.
- The remaining columns are labeled using diagnosis identifiers.
- The numerical values represent relationships or weights associated with symptom–diagnosis combinations.

Many entries contain zero, while others contain nonzero values.

This structure is useful for exploring associations between symptoms and diseases and may support future matrix-based analytics.

The meaning of the numerical values should be verified before applying clinical or predictive interpretations.

### 7. Symptom Reference Dataset — `sym_t(1).csv`

This dataset contains 272 symptom records.

**Columns:**
- `syd`: Symptom identifier
- `symptom`: Symptom description

Examples include upper abdominal pain, lower abdominal pain and abscess.

The dataset provides symptom names that can be associated with symptom identifiers used in other files.

### 8. Detailed Symptom Information — `symptoms2(1).csv`

This dataset contains 389 rows and nine columns.

| Column | Description |
|---|---|
| `_id` | Record identifier |
| `chief_complaint_id` | Chief complaint identifier |
| `chief_complaint_name_id` | Chief complaint name identifier |
| `name` | Symptom or complaint name |
| `reason_code` | Associated reason code |
| `life_threatening` | Recorded life-threatening indicator |
| `msp_enabled` | Recorded MSP-enabled indicator |
| `primary_complaint` | Recorded primary-complaint indicator |
| `diagnosis_count` | Associated diagnosis count |

This dataset provides additional metadata about medical symptoms and chief complaints.

Some fields contain missing values and the meaning of the coded indicators should be confirmed against the original data source before interpretation.

## Project Structure

The uploaded files can be organized into the following GitHub repository structure:

```text
Symptom-Disease-Data-Analysis/
│
├── Data/
│   ├── dia_3(1).csv
│   ├── dia_t.csv
│   ├── diagn_title(1).csv
│   ├── diffsydiw(1).csv
│   ├── sym_3(1).csv
│   ├── sym_dis_matrix(1).csv
│   ├── sym_t(1).csv
│   └── symptoms2(1).csv
│
├── Symptom Disease Sorting.ipynb
├── Symptoms diseases sorting-checkpoint.ipynb
│
└── README.md
```

**Note:** The original notebooks reference filenames such as `Data/dia_3.csv` and `Data/diagn_title.csv`. These do not exactly match all uploaded filenames. Update the file paths in the notebooks or rename the CSV files consistently before running the project.

## Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Data analysis and processing |
| Pandas | Reading and exploring CSV datasets |
| NumPy | Numerical data operations |
| Matplotlib | Data visualization |
| Scikit-learn | StandardScaler import in the checkpoint notebook |
| Jupyter Notebook | Interactive analysis and documentation |

Although Scikit-learn is imported in the checkpoint notebook, the provided notebooks do not demonstrate a trained machine-learning model.

## Project Implementation

### Step 1: Importing Required Libraries

The notebooks begin by importing Python libraries used for numerical operations, data manipulation and visualization.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
```

The checkpoint notebook also imports `StandardScaler` from Scikit-learn and `Axes3D` from Matplotlib.

### Step 2: Loading Healthcare Datasets

The notebooks use Pandas to read CSV files.

Example:

```python
df_1 = pd.read_csv("Data/dia_3.csv")

df_1.dataframeName = "dia_3.csv"
```

This loads diagnosis information into a Pandas DataFrame.

The notebooks demonstrate loading three datasets:

- `dia_3.csv`
- `dia_t.csv`
- `diagn_title.csv`

The other five CSV datasets are included in the project files but are not explicitly loaded in the supplied notebook code.

### Step 3: Inspecting Dataset Dimensions

The project uses the DataFrame `shape` attribute to examine the number of rows and columns.

```python
df_1.shape
```

Example output:

```text
(110, 2)
```

This indicates that the selected diagnosis dataset contains 110 rows and two columns.

For the larger datasets, the notebooks demonstrate reading the first 1,000 rows using the `nrows` parameter.

### Step 4: Previewing Dataset Records

The notebooks use the Pandas `head()` function to display sample records.

```python
df_1.head()
```

This allows users to inspect the structure and contents of a dataset before conducting further analysis.

### Step 5: Defining Data Visualization Functions

The notebooks contain reusable functions designed to support exploratory visualization.

**Column Distribution Visualization**

The `plot_per_col_dist()` function is designed to create histograms or bar charts for columns with suitable numbers of unique values.

Its purpose is to help identify how frequently values occur within a dataset.

**Correlation Matrix Visualization**

The `plot_corr_mat()` function is designed to examine correlations among suitable numerical columns.

The function checks whether enough nonconstant numerical columns are available before creating a correlation plot.

**Scatter and Density Visualization**

The `plot_scatter_mat()` function is designed to generate scatterplot and density-based visualizations for numerical data.

It includes logic to filter numerical columns and remove unsuitable data before visualization.

These functions are exploratory tools and do not independently establish medical or clinical relationships.

### Step 6: Exploring Data Distributions

The notebooks call distribution-plotting functions on the loaded datasets.

Example:

```python
plot_per_col_dist(df_2, 10, 5)
```

The function attempts to visualize suitable columns based on their number of unique values.

For the diagnosis reference datasets, the recorded notebook outputs show empty figure results for some distribution calls because the data does not meet the function's plotting criteria.

### Step 7: Examining Correlations

The main notebook includes the following correlation analysis call:

```python
plot_corr_mat(df_1, 8)
```

The recorded output indicates that there are not enough suitable nonconstant numerical columns to create a correlation plot.

This demonstrates an important limitation of applying numerical correlation techniques directly to datasets containing identifiers and descriptive text.

## Data Quality Observations

Inspection of the uploaded CSV files identified several considerations for further analysis.

**Missing values**

Some datasets contain missing information, including the diagnosis title, symptom reference, detailed symptom and weighted relationship files.

**Duplicate records**

The weighted symptom–disease relationship dataset and diagnosis title dataset contain duplicate rows.

**Text formatting**

Some medical descriptions contain embedded control characters or line separators that may require cleaning for consistent display and comparison.

**Identifier consistency**

Multiple files use different column names for symptom and diagnosis identifiers. These identifiers should be standardized or carefully mapped before combining datasets.

**Numerical interpretation**

Relationship weights and coded flags require source documentation before being used in medical decision-making or predictive modeling.

The provided notebooks focus on exploration; they do not document a completed end-to-end cleaning pipeline for all eight datasets.

## Current Project Results

Based on the notebook code and recorded outputs, the project demonstrates:

- Successful loading of selected diagnosis CSV files using Pandas.
- Inspection of dataset dimensions and sample records.
- Creation of reusable functions for exploratory visualization.
- Attempts to visualize column distributions.
- Correlation-analysis checks that identify unsuitable data structures.
- Organization of additional symptom, diagnosis and relationship datasets for future analysis.

## Potential Future Improvements

The current project provides a starting point for more advanced healthcare analytics.

Potential extensions include:

1. **Data Cleaning:** Handle missing values, duplicate records and inconsistent medical descriptions.
2. **Dataset Integration:** Join symptom and diagnosis reference tables using their identifiers.
3. **Relationship Visualization:** Create heatmaps and network visualizations from the symptom–disease matrix.
4. **Exploratory Statistics:** Analyze symptom frequencies, disease associations and relationship-weight distributions.
5. **Feature Engineering:** Convert symptom–disease associations into structured features for analytical models.
6. **Machine Learning:** Investigate disease-classification approaches after validating data quality and obtaining suitable labels.
7. **Interactive Dashboard:** Develop a Streamlit or similar interface for exploring symptoms and disease relationships.
8. **Model Evaluation:** If predictive modeling is implemented, evaluate performance using appropriate metrics and validation datasets.

These are proposed enhancements, not features currently implemented in the notebooks.

## Applications

The datasets and exploratory techniques in this project may be useful for:

- Healthcare data exploration
- Medical terminology and reference-data analysis
- Symptom–disease relationship research
- Educational data science projects
- Data visualization practice
- Preliminary feature engineering for healthcare machine-learning research

## Limitations and Medical Disclaimer

This project is intended for educational, research and exploratory data analysis purposes only.

The datasets have not been independently verified for clinical completeness, accuracy, or suitability for medical decision-making.

The project does not provide validated diagnoses, treatment recommendations, or clinical predictions.

Any future predictive system would require appropriate data validation, medical expertise, rigorous evaluation and consideration of privacy, safety and regulatory requirements.

## Conclusion

The **Symptom–Disease Data Analysis and Visualization** project demonstrates how Python and Jupyter Notebook can be used to explore structured healthcare datasets.

By examining diagnosis records, symptom information and available relationship data, the project establishes a foundation for understanding medical dataset organization and preparing for more advanced analytical work.

The current implementation emphasizes data inspection and exploratory visualization, while the additional datasets provide opportunities for future symptom–disease relationship analysis and machine-learning research.

## Tools Used:

**Project:** Symptom–Disease Data Analysis and Visualization Using Python

**Domain:** Healthcare Analytics | Data Science | Exploratory Data Analysis

**Tools:** Python, Pandas, NumPy, Matplotlib, Jupyter Notebook

---

