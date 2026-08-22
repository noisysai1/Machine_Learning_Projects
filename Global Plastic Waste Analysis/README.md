# 🌍 Global Plastic Waste Analysis

## 📌 Project Overview

Plastic pollution is one of the major environmental challenges affecting countries around the world. 
The amount of plastic waste generated and the way that waste is managed can vary significantly depending on population, economic development, infrastructure and waste-management practices.

This project performs an **Exploratory Data Analysis (EDA)** of global plastic waste data, with a primary focus on the year **2010**.

The analysis combines information about:

* Plastic waste generated per person
* Mismanaged plastic waste per person
* GDP per capita
* Country population
* Geographic continent
* Total estimated annual plastic waste
* Total estimated annual mismanaged plastic waste

The project also investigates whether a country's **GDP per capita is related to the amount of plastic waste generated or mismanaged per person**.

---

# 🎯 Project Objectives

The major objectives of this project are:

1. Load and explore global plastic waste datasets.
2. Clean missing and incomplete records.
3. Filter the data for the year **2010**.
4. Combine plastic-waste and mismanaged-waste information.
5. Calculate total annual plastic waste for each country.
6. Calculate total annual mismanaged plastic waste.
7. Analyze the relationship between **GDP per capita and plastic waste generation**.
8. Analyze the relationship between **GDP per capita and mismanaged plastic waste**.
9. Visualize the relationships using scatter plots and regression lines.
10. Evaluate the hypotheses proposed in the analysis.

---

# 📂 Project Structure

```text
Global-Plastic-Waste-Analysis/
│
├── global-plastic-waste-2010.ipynb
│
├── datasets/
│   ├── global-plastics-production.csv
│   ├── mismanaged-waste-global-total.csv
│   ├── plastic-waste-per-capita.csv
│   ├── per-capita-plastic-waste-vs-gdp-per-capita.csv
│   └── per-capita-mismanaged-plastic-waste-vs-gdp-per-capita.csv
│
└── README.md
```

---

# 📊 Datasets

The project contains five datasets related to global plastic production and waste.

## 1. Global Plastics Production

**File:** `global-plastics-production.csv`

Contains historical information about global plastic production.

Important columns:

* `Entity`
* `Code`
* `Year`
* `Global plastics production (million tonnes)`

The available data covers the period from **1950 to 2015**.

---

## 2. Mismanaged Waste — Global Total

**File:** `mismanaged-waste-global-total.csv`

Contains each country's contribution to globally mismanaged plastic waste for **2010**.

Important columns:

* `Entity`
* `Code`
* `Year`
* `Mismanaged waste (% global total)`

---

## 3. Plastic Waste Per Capita

**File:** `plastic-waste-per-capita.csv`

Contains plastic waste generated per person per day for countries in **2010**.

Important columns:

* `Entity`
* `Code`
* `Year`
* `Per capita plastic waste (kg/person/day)`

---

## 4. Plastic Waste vs GDP Per Capita

**File:** `per-capita-plastic-waste-vs-gdp-per-capita.csv`

This is one of the primary datasets used in the notebook.

It combines:

* Country
* Country code
* Year
* Plastic waste generated per person
* GDP per capita
* Total population
* Continent

Important variables include:

```text
Entity
Code
Year
Per capita plastic waste (kg/person/day)
GDP per capita, PPP (constant 2011 international $)
Total population (Gapminder, HYDE & UN)
Continent
```

---

## 5. Mismanaged Plastic Waste vs GDP Per Capita

**File:** `per-capita-mismanaged-plastic-waste-vs-gdp-per-capita.csv`

This dataset contains information about the amount of plastic waste that is mismanaged per person along with economic and population information.

Important variables include:

```text
Entity
Code
Year
Per capita mismanaged plastic waste
GDP per capita, PPP (constant 2011 international $)
Total population (Gapminder, HYDE & UN)
Continent
```

---

# 🛠️ Technologies Used

The project was developed using **Python** and Jupyter Notebook.

### Python Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Main Tools

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Data analysis                  |
| Pandas           | Data manipulation and cleaning |
| NumPy            | Numerical calculations         |
| Matplotlib       | Data visualization             |
| Seaborn          | Statistical visualization      |
| Jupyter Notebook | Interactive analysis           |

---

# 🔄 Project Workflow

The analysis follows the workflow below:

```text
Raw Plastic Waste Data
        ↓
Load Datasets
        ↓
Inspect Variables
        ↓
Rename Columns
        ↓
Handle Missing Data
        ↓
Filter 2010 Data
        ↓
Add Continent Information
        ↓
Prepare Waste Dataset
        ↓
Prepare Mismanaged Waste Dataset
        ↓
Merge Datasets
        ↓
Calculate Annual Waste
        ↓
GDP vs Waste Analysis
        ↓
Data Visualization
        ↓
Hypothesis Evaluation
        ↓
Insights & Conclusions
```

---

# 🔍 Step 1 — Import Required Libraries

The first step is importing the libraries required for analysis.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

These libraries are used for loading data, manipulating DataFrames, performing calculations, and creating visualizations.

---

# 📥 Step 2 — Load Plastic Waste Data

The primary plastic-waste dataset is loaded using Pandas.

```python
df = pd.read_csv(
    '../input/plastic-datasets/per-capita-plastic-waste-vs-gdp-per-capita.csv'
)
```

The dataset contains plastic-waste, GDP, population, and geographic information.

---

# ✏️ Step 3 — Rename Columns

Some of the original column names are long, so they are renamed to make the analysis easier to understand.

For example:

```text
GDP per capita, PPP (constant 2011 international $)
                        ↓
GDP per capita in PPP
```

```text
Total population (Gapminder, HYDE & UN)
                        ↓
Total Population
```

```text
Per capita plastic waste (kg/person/day)
                        ↓
Waste per person(kg/day)
```

This improves readability throughout the notebook.

---

# 🧹 Step 4 — Data Cleaning

Real-world datasets frequently contain missing information.

The notebook identifies entities where both population and GDP information are missing.

```python
incomplete_data_index = df[
    (df['Total Population'].isna()) &
    (df['GDP per capita in PPP'].isna())
].index

df.drop(incomplete_data_index, inplace=True)
```

Removing incomplete records helps ensure that subsequent analysis is based on usable observations.

---

# 📅 Step 5 — Filter Data for 2010

The main analysis focuses on **2010**.

```python
data = df[df['Year'] == 2010]
```

This creates a consistent snapshot for comparing countries during the same year.

---

# 🌎 Step 6 — Add Continent Information

Continent information is retrieved from available records and added to the 2010 dataset.

Rows without usable continent information are subsequently removed.

This allows countries to retain useful geographic information in the final dataset.

---

# 🗑️ Step 7 — Remove Missing Plastic Waste Values

Countries without plastic-waste-per-person values are excluded.

```python
data = data[
    data['Waste per person(kg/day)'].notna()
]
```

The cleaned result is stored as:

```python
waste_gener
```

This represents the prepared **plastic waste generation dataset**.

---

# ♻️ Step 8 — Prepare Mismanaged Plastic Waste Data

The second major dataset contains **mismanaged plastic waste per person**.

The relevant columns are renamed:

```text
Per capita mismanaged plastic waste
                    ↓
Mismanaged waste per person(kg/day)
```

The analysis again filters the records for:

```python
Year == 2010
```

Rows without mismanaged-waste values are removed.

The resulting DataFrame is stored as:

```python
waste_misma
```

---

# 🔗 Step 9 — Merge the Datasets

The cleaned plastic-waste and mismanaged-waste datasets are combined.

```python
plastic_waste = pd.merge(
    waste_gener,
    waste_misma,
    how='inner'
)
```

The merged dataset allows waste generation, waste management, GDP, population and geographic information to be analyzed together.

---

# 📋 Final Data Structure

The columns are rearranged into a logical structure:

```python
col_list = [
    'Entity',
    'Code',
    'Year',
    'Waste per person(kg/day)',
    'Mismanaged waste per person(kg/day)',
    'GDP per capita in PPP',
    'Total Population',
    'Continent'
]
```

The per-person waste values are also rounded for easier interpretation.

---

# 🧮 Step 10 — Calculate Total Annual Plastic Waste

The original data provides plastic waste in:

```text
kg / person / day
```

To estimate the total waste generated by a country, the analysis uses:

```text
Waste per person per day
        ×
Total population
        ×
365 days
```

The notebook implements:

```python
plastic_waste['Total waste(kgs/year)'] = (
    plastic_waste['Waste per person(kg/day)']
    * plastic_waste['Total Population']
    * 365
)
```

Therefore:

### Total Annual Plastic Waste

```text
Total Waste =
Per-Capita Daily Waste × Population × 365
```

---

# ♻️ Step 11 — Calculate Total Mismanaged Plastic Waste

The same approach is applied to mismanaged plastic waste.

```python
plastic_waste['Total waste mismanaged(kgs/year)'] = (
    plastic_waste['Mismanaged waste per person(kg/day)']
    * plastic_waste['Total Population']
    * 365
)
```

Therefore:

### Total Annual Mismanaged Waste

```text
Total Mismanaged Waste =
Mismanaged Waste per Person per Day
× Population
× 365
```

These calculated variables make it possible to move beyond per-capita measurements and estimate annual country-level waste.

---

# 🧪 Hypothesis Analysis

A major objective of the project is investigating the relationship between economic development and plastic waste.

Two hypotheses are explored.

## Hypothesis 1

> Higher GDP per capita is associated with higher mismanaged plastic waste per person.

Variables:

```text
GDP per capita
        ↓
Mismanaged plastic waste per person
```

---

## Hypothesis 2

> Higher GDP per capita is associated with higher plastic waste generation per person.

Variables:

```text
GDP per capita
        ↓
Plastic waste generated per person
```

---

# 📈 Visualization 1 — GDP vs Mismanaged Plastic Waste

A scatter plot and regression line are used to investigate the relationship between:

```text
GDP per capita
        vs
Mismanaged waste per person
```

The notebook uses both **Matplotlib** and **Seaborn** for the visualization.

```python
sns.regplot(
    x='GDP per capita in PPP',
    y='Mismanaged waste per person(kg/day)',
    data=plastic_waste
)
```

### Observation

The visualization indicates that **mismanaged plastic waste does not increase as GDP per capita increases**.

Therefore, the notebook concludes that the evidence does **not support the first hypothesis**.

### Interpretation

A country having a larger GDP per capita does not automatically mean that it mismanages more plastic waste per person.

---

# 📈 Visualization 2 — GDP vs Plastic Waste Generation

The second visualization investigates:

```text
GDP per capita
        vs
Plastic waste generated per person
```

The regression visualization is created using:

```python
sns.regplot(
    x=plastic_waste['GDP per capita in PPP'],
    y=plastic_waste['Waste per person(kg/day)']
)
```

### Observation

The analysis shows that plastic waste generated per person **tends to increase with GDP per capita**.

Therefore, the notebook concludes that the evidence **supports the second hypothesis**.

---

# 🔑 Key Findings

The analysis produces two major findings:

### 1. GDP vs Mismanaged Plastic Waste

The data does **not** show that mismanaged plastic waste per person increases with GDP per capita.

Therefore:

```text
Higher GDP
   ≠
Higher Mismanaged Waste Per Person
```

The first proposed relationship is rejected based on the visual evidence presented in the notebook.

---

### 2. GDP vs Plastic Waste Generation

Plastic waste generation per person **tends to increase as GDP per capita increases**.

Therefore:

```text
Higher GDP
    →
Tendency Toward Higher
Plastic Waste Generation Per Person
```

The second proposed relationship is supported by the analysis.

---

# 💡 Project Insights

This project demonstrates an important distinction between:

**generating plastic waste** and **mismanaging plastic waste**.

Countries with greater GDP per capita may generate more plastic waste per person, but this does not necessarily mean that they mismanage more plastic waste per person.

This makes it important to study both indicators separately rather than treating total plastic waste generation and mismanaged plastic waste as the same environmental measure.

---

# 🧠 Skills Demonstrated

This project demonstrates practical skills in:

### Data Analysis

* Exploratory Data Analysis
* Dataset inspection
* Data filtering
* Data transformation
* Multi-dataset analysis

### Data Cleaning

* Missing-value detection
* Missing-value removal
* Column renaming
* Data-type conversion
* Data preparation

### Pandas

* `read_csv()`
* `rename()`
* `drop()`
* `isna()`
* `notna()`
* Boolean filtering
* `reset_index()`
* `merge()`
* Column selection
* Data-type conversion

### NumPy

* Numerical transformations
* Rounding numerical values

### Data Visualization

* Scatter plots
* Regression plots
* Matplotlib
* Seaborn
* Relationship analysis

### Analytical Thinking

* Hypothesis formulation
* Variable comparison
* Trend interpretation
* Data-driven conclusions

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

## 2. Navigate to the Project

```bash
cd Global-Plastic-Waste-Analysis
```

## 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the Notebook

Open:

```text
global-plastic-waste-2010.ipynb
```

and execute the cells sequentially.

> **Note:** If you change the folder structure from the original notebook environment, update the CSV paths in `pd.read_csv()` so they point to your local `datasets/` directory.

---

# 📦 Requirements

A basic `requirements.txt` for this project can contain:

```text
pandas
numpy
matplotlib
seaborn
jupyter
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

# 🔮 Future Improvements

The project can be extended with additional analysis such as:

* Compare plastic waste across continents
* Identify countries with the highest waste generation
* Identify countries with the highest mismanaged waste
* Analyze plastic waste as a percentage of global totals
* Study historical global plastic production
* Compare high-income and low-income countries
* Add correlation coefficients to quantify relationships
* Build interactive visualizations
* Create a Power BI or Tableau dashboard
* Extend the analysis beyond 2010 where compatible data is available
* Develop predictive models for plastic-waste indicators

These additions could turn the project into a broader global environmental data-analysis portfolio project.

---

# 📌 Conclusion

This project provides a data-driven exploration of **global plastic waste in 2010** by integrating plastic waste generation, mismanaged waste, GDP per capita, population, and geographic information.

The analysis finds that:

* **Mismanaged plastic waste per person does not increase with GDP per capita** in the relationship visualized in the notebook.
* **Plastic waste generated per person tends to increase with GDP per capita**.

The project demonstrates a complete foundational data-analysis workflow, including **data loading, cleaning, filtering, merging, feature calculation, visualization, hypothesis evaluation, and interpretation**.

---

# 📁 Repository Files

```text
global-plastic-waste-2010.ipynb
mismanaged-waste-global-total.csv
global-plastics-production.csv
plastic-waste-per-capita.csv
per-capita-plastic-waste-vs-gdp-per-capita.csv
per-capita-mismanaged-plastic-waste-vs-gdp-per-capita.csv
README.md
