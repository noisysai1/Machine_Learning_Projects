# PySpark DataFrame Operations & Stock Data Analysis

## 📌 Project Overview

This project demonstrates the fundamentals of **Apache PySpark DataFrame operations** through practical data-processing examples.

The notebook explores how PySpark can be used to load, inspect, filter, transform, aggregate, clean, and analyze structured datasets. Apple historical stock-market data is used for several of the examples, making the project a practical introduction to working with tabular data using **Apache Spark**.

The project covers important PySpark concepts including:

* Creating a Spark session
* Reading CSV datasets
* Understanding DataFrame schemas
* Selecting columns
* Filtering records
* Working with Spark Rows
* GroupBy operations
* Aggregate functions
* Sorting data
* Handling missing values
* Working with dates and timestamps
* Creating new columns
* Renaming columns
* Formatting numerical results

---

## 🎯 Project Objective

The main objective of this project is to develop a strong understanding of **PySpark DataFrame manipulation and data-processing techniques**.

Instead of performing analysis only with traditional Python libraries such as Pandas, this project demonstrates how similar operations can be performed using Spark's distributed DataFrame API.

The project provides a foundation for working with larger datasets and building scalable data-processing pipelines.

---

## 🛠️ Technologies Used

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| Python                 | Programming language                |
| Apache Spark           | Distributed data-processing engine  |
| PySpark                | Python API for Apache Spark         |
| Spark SQL / DataFrames | Structured data manipulation        |
| Jupyter Notebook       | Interactive development environment |
| CSV                    | Input data format                   |

---

## 📂 Project Structure

```text
PySpark-DataFrame-Operations/
│
├── PySpark_Basic_DataFrame_Operations.ipynb
├── apple.csv
└── README.md
```

### Files

**`PySpark_Basic_DataFrame_Operations.ipynb`**

Contains the PySpark implementation and examples for DataFrame operations, filtering, aggregation, missing-value handling, and date/time processing.

**`apple.csv`**

Contains historical Apple stock-market information used for stock-related DataFrame analysis.

The uploaded Apple dataset contains **1,762 rows and 13 columns**, including:

```text
Date
Open
High
Low
Close
Volume
Ex-Dividend
Split Ratio
Adj. Open
Adj. High
Adj. Low
Adj. Close
Adj. Volume
```

---

# 🚀 Project Workflow

## 1. Creating the Spark Session

The first step is initializing Apache Spark using `SparkSession`.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('ops') \
    .getOrCreate()
```

`SparkSession` provides the main entry point for working with structured data through PySpark.

---

## 2. Loading CSV Data into PySpark

The dataset is loaded into a Spark DataFrame using:

```python
df = spark.read.csv(
    'appl_stock.csv',
    inferSchema=True,
    header=True
)
```

### Important Parameters

**`header=True`**

Uses the first row of the CSV file as column names.

**`inferSchema=True`**

Allows Spark to automatically determine appropriate data types for the columns.

---

## 3. Exploring the DataFrame

Before performing analysis, it is important to understand the structure and contents of the dataset.

### Retrieve Records

```python
df.take(2)
```

This returns the first two rows as Spark Row objects.

### View the Schema

```python
df.printSchema()
```

This helps identify column names and their corresponding data types.

### Display Records

```python
df.show(3)
```

This displays the first three records in a tabular format.

---

# 🔎 4. Filtering Data

Filtering is one of the most important operations when analyzing large datasets.

The notebook demonstrates multiple PySpark filtering techniques.

### Filter Using SQL-Style Syntax

```python
df.filter('close < 500') \
  .select(['High', 'Low']) \
  .show()
```

This retrieves rows where the closing stock price is below 500 and displays only the `High` and `Low` columns.

### Filter Using DataFrame Syntax

```python
df.filter(df['close'] < 500) \
  .select(['High', 'Low']) \
  .show()
```

This produces the same type of filtering using PySpark column expressions.

---

## 5. Applying Multiple Conditions

More complex conditions can be combined using logical operators.

```python
df.filter(
    (df['Open'] > 200) &
    ~(df['Close'] < 200)
).show()
```

This demonstrates how multiple business or analytical conditions can be applied to a Spark DataFrame.

---

# 📥 6. Collecting Filtered Results

PySpark results can be collected from the distributed DataFrame into Python objects.

```python
result = df.filter(df['Low'] == 197.16).collect()
```

A particular row can then be accessed:

```python
row = result[0]
```

The Spark Row can also be converted into a dictionary:

```python
row.asDict()['Volume']
```

This demonstrates the relationship between Spark DataFrames, Spark Rows, and regular Python objects.

---

# 📊 7. GroupBy Operations

The notebook also explores grouping data using a dataset containing company and sales information.

```python
df.groupBy('Company')
```

Grouping enables records belonging to the same category to be analyzed together.

---

## Average by Group

```python
df.groupBy('Company').mean().show()
```

Calculates mean values for each company.

---

## Sum by Group

```python
df.groupBy('Company').sum().show()
```

Calculates totals for each company.

---

## Minimum by Group

```python
df.groupBy('Company').min().show()
```

Finds minimum values within each company group.

---

## Count Records

```python
df.groupBy('Company').count().show()
```

Counts the number of records associated with each company.

---

# 🧮 8. Aggregate Functions

PySpark provides several built-in aggregation functions.

### Minimum Sales

```python
df.agg({'Sales': 'min'}).show()
```

### Total Sales

```python
df.agg({'Sales': 'sum'}).show()
```

Aggregations can also be performed after grouping.

```python
group_data = df.groupBy('Company')

group_data.agg({'Sales': 'sum'}).show()
```

This is useful when calculating business metrics across categories.

---

# 📈 9. PySpark SQL Functions

Several functions from `pyspark.sql.functions` are explored.

```python
from pyspark.sql.functions import countDistinct, avg, stddev
```

## Count Distinct Values

```python
df.select(countDistinct('Sales')).show()
```

## Calculate Average

```python
df.select(
    avg('Sales').alias('Average Sales')
).show()
```

## Count Unique Companies

```python
df.select(countDistinct('Company')).show()
```

## Standard Deviation

```python
df.select(stddev('Sales')).show()
```

These functions demonstrate how descriptive statistics can be calculated directly within Spark.

---

# 🔢 10. Formatting Numerical Results

PySpark can format numerical results for easier interpretation.

```python
from pyspark.sql.functions import format_number

sales_std = df.select(
    stddev('Sales').alias('std')
)

sales_std.select(
    format_number('std', 2).alias('std')
).show()
```

The result is formatted to two decimal places.

---

# ↕️ 11. Sorting Data

Sorting is demonstrated using `orderBy()`.

### Ascending Order

```python
df.orderBy('Sales').show()
```

### Descending Order

```python
df.orderBy(
    df['Sales'].desc()
).show()
```

Sorting is especially useful for identifying highest or lowest values in a dataset.

---

# 🧹 12. Handling Missing Values

Real-world datasets frequently contain missing or null values.

The notebook explores several approaches for handling them.

---

## Drop Rows Containing Null Values

```python
df.na.drop().show()
```

---

## Threshold-Based Null Removal

```python
df.na.drop(thresh=2).show()
```

This keeps rows that contain at least the required number of non-null values.

---

## Drop Completely Empty Rows

```python
df.na.drop(how='all').show()
```

---

## Drop Null Values from Specific Columns

```python
df.na.drop(subset='Sales').show()
```

and:

```python
df.na.drop(subset='Name').show()
```

This provides more control over data-cleaning operations.

---

# 🩹 13. Filling Missing Values

Instead of deleting records, missing values can also be replaced.

### Fill Numeric Null Values

```python
df.na.fill(0).show()
```

### Fill String Null Values

```python
df.na.fill('fill value').show()
```

### Fill a Specific Column

```python
df.na.fill(
    'no name',
    subset='Name'
).show()
```

---

# 📊 14. Mean Imputation

A more meaningful approach for missing numerical values is replacing them with the column mean.

First, calculate the mean:

```python
from pyspark.sql.functions import mean

mean_val = df.select(
    mean(df['Sales'])
).collect()
```

Extract the calculated value:

```python
mean_sales = mean_val[0][0]
```

Then replace missing Sales values:

```python
df.na.fill(
    mean_sales,
    subset='Sales'
).show()
```

This demonstrates a basic **data preprocessing and imputation technique** using PySpark.

---

# 📅 15. Date and Timestamp Operations

The Apple stock dataset is loaded again to demonstrate date-based operations.

```python
df = spark.read.csv(
    'appl_stock.csv',
    inferSchema=True,
    header=True
)
```

The notebook imports several useful date functions:

```python
from pyspark.sql.functions import (
    dayofmonth,
    hour,
    dayofyear,
    month,
    year,
    weekofyear,
    format_number,
    date_format
)
```

---

## Extract Day

```python
df.select(
    dayofmonth(df['Date'])
).show(5)
```

## Extract Month

```python
df.select(
    month(df['Date'])
).show(5)
```

## Extract Year

```python
df.select(
    year(df['Date'])
).show(5)
```

These functions are particularly useful for time-series and historical data analysis.

---

# ➕ 16. Creating a New Column

A new `Year` column is created from the stock date.

```python
new_df = df.withColumn(
    'Year',
    year(df['Date'])
)
```

`withColumn()` is one of the most commonly used PySpark transformation methods.

It can be used to:

* Create new columns
* Transform existing columns
* Apply calculations
* Generate derived features

---

# 📈 17. Average Apple Closing Price by Year

One of the final analyses calculates the average closing stock price for each year.

```python
result = new_df \
    .groupBy('Year') \
    .mean() \
    .select(['Year', 'avg(Close)'])
```

The resulting column is renamed:

```python
new_result = result.withColumnRenamed(
    'avg(Close)',
    'Average Closing Price'
)
```

Finally, the value is formatted to two decimal places:

```python
new_result.select(
    [
        'Year',
        format_number(
            'Average Closing Price',
            2
        ).alias('Average Closing Price')
    ]
).show()
```

This combines several PySpark concepts:

**Date Extraction → Feature Creation → Grouping → Aggregation → Column Renaming → Formatting**

---

# 💡 Key Concepts Demonstrated

This project provides hands-on experience with the following PySpark concepts:

### Data Loading

* Reading CSV files
* Header detection
* Automatic schema inference

### Data Exploration

* `take()`
* `show()`
* `printSchema()`

### Data Selection

* `select()`

### Data Filtering

* `filter()`
* Conditional expressions
* Multiple conditions

### Data Aggregation

* `groupBy()`
* `mean()`
* `sum()`
* `min()`
* `count()`
* `agg()`

### Statistical Functions

* `avg()`
* `stddev()`
* `countDistinct()`

### Data Cleaning

* `na.drop()`
* `na.fill()`
* Mean-value imputation

### Sorting

* `orderBy()`
* Ascending and descending sorting

### Date Processing

* `dayofmonth()`
* `month()`
* `year()`
* `hour()`

### Data Transformation

* `withColumn()`
* `withColumnRenamed()`

### Formatting

* `format_number()`

---

# 📊 Dataset Insights

The Apple stock dataset provides historical market information containing variables such as:

* Opening price
* Highest daily price
* Lowest daily price
* Closing price
* Trading volume
* Adjusted prices
* Dividend information
* Stock split information

The notebook uses this data primarily to demonstrate PySpark DataFrame operations rather than to build a predictive machine-learning model.

One particularly useful analysis is the calculation of the **average Apple closing stock price by year**, demonstrating how Spark can transform date information and aggregate historical financial data.

---

# 🧠 What I Learned

Through this project, I developed practical experience with:

* Building Spark sessions
* Loading structured datasets into Spark
* Understanding Spark DataFrame schemas
* Filtering large datasets using multiple conditions
* Selecting relevant features
* Performing grouped calculations
* Calculating descriptive statistics
* Sorting records
* Detecting and handling missing values
* Performing basic mean imputation
* Manipulating dates and timestamps
* Creating derived columns
* Aggregating time-series information
* Formatting analytical results

The project strengthened my understanding of how **Apache Spark processes structured data through distributed DataFrame operations**.

---

# 🌍 Real-World Applications

The techniques demonstrated in this project can serve as building blocks for larger data-engineering and analytics workflows, including:

* Large-scale data preprocessing
* ETL pipelines
* Financial data analysis
* Transaction processing
* Log analysis
* Customer analytics
* Data quality pipelines
* Data transformation
* Time-series aggregation
* Feature engineering
* Business intelligence data preparation

---

# ⚙️ Installation

Install PySpark using:

```bash
pip install pyspark
```

To work with the notebook:

```bash
pip install jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
PySpark_Basic_DataFrame_Operations.ipynb
```

---

# ▶️ How to Run the Project

1. Clone or download the repository.
2. Install Python and PySpark.
3. Ensure Java is installed and configured for your Spark environment.
4. Place the required CSV datasets in the appropriate project directory.
5. Start Jupyter Notebook.
6. Open the PySpark notebook.
7. Execute the cells sequentially.
8. Review the Spark DataFrame transformations and outputs.

> **Note:** The notebook currently references filenames such as `appl_stock.csv`, `sales_info.csv`, and `ContainsNull.csv`. Make sure those files exist with the expected names before executing the corresponding sections. If using the provided `apple.csv`, update the Apple-stock file path in the notebook accordingly.

---

# 🔮 Future Improvements

The project can be expanded further by adding:

* Spark SQL queries
* Window functions
* User Defined Functions (UDFs)
* Advanced DataFrame joins
* Data visualization
* Stock-price trend analysis
* Moving averages
* Additional financial KPIs
* Spark MLlib machine-learning models
* Parquet file processing
* Partitioning and caching
* Performance optimization
* Integration with cloud storage
* Larger distributed datasets

---

# 📌 Conclusion

This project provides a practical introduction to **Apache PySpark DataFrame operations** using structured datasets and historical Apple stock-market data.

It demonstrates the complete basic workflow of:

```text
Data Loading
     ↓
Schema Inspection
     ↓
Data Selection
     ↓
Filtering
     ↓
Grouping
     ↓
Aggregation
     ↓
Missing Value Handling
     ↓
Date Processing
     ↓
Feature Creation
     ↓
Analytical Results
```

The project establishes a solid foundation for progressing toward more advanced topics in 

**Big Data Analytics, Data Engineering, Apache Spark, and distributed data processing**.

---

