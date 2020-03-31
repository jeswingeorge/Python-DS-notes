# Steps of Data Exploration and Preparation:
1. Variable Identification
2. Univariate Analysis
3. Bi-variate Analysis
4. Missing values treatment
5. Outlier treatment
6. Variable transformation
7. Variable creation


***

Code to import libraies:

```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(rc={'figure.figsize':(11,9)})
from pandas.api.types import CategoricalDtype

# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
```


Loading dataset:

```
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("Training data shape: ", train.shape)
print("Testing data shape: ", test.shape)
train.head()
```

***

## 1. Variable Identification
Identify the target variable and the predictor varaibles. 

To find the data types of columns:

```
train.info()
``` 

Determining data types of columns

```
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
boolean_dtypes = ['bool']
print("Number of numerical data types: ", len(train.select_dtypes(include = numeric_dtypes).columns))
print("Number of boolean data types: ", len(train.select_dtypes(include = 'bool').columns))
print("Number of object data types: ", len(train.select_dtypes(include = 'object').columns))
```

```
numerical_columns = list(train.select_dtypes(include = numeric_dtypes).columns)
string_columns = list(train.select_dtypes(include = 'object').columns)
```

## 2. Univariate Analysis

### Univariate Analysis for numerical variables

Function to find the skewness and kurtosis

```
global measure_columns_spread_df
measure_columns_spread_df = pd.DataFrame(columns = ['col_name', 'skewness', 'kurtosis', 'transformation_reqd'])
def collect_calculate_col_spread(df, col):
    global measure_columns_spread_df
    temp = pd.DataFrame(pd.Series([col, df[col].skew(), df[col].kurt(), np.nan])).T
    temp.columns = ['col_name', 'skewness', 'kurtosis', 'transformation_reqd']
    condition = (df[col].skew() < -1) | (df[col].skew() > 1) | (df[col].kurt() < -1) | ((df[col].kurt() > 1))
    temp.loc[0,'transformation_reqd']=np.where(condition, 'Yes', 'No')
    measure_columns_spread_df = pd.concat([measure_columns_spread_df, temp], ignore_index = True)
```


Plotting a histogram
```
col = '....'
plt.figure(figsize=(10,6));
sns.distplot(train[col]);
```
To find skewness and kurtosis of the numerical variable-

```
collect_calculate_col_spread(train, col)
print("Skewness {}: {}".format(col, train[col].skew()))
print("Kurtosis {}: {}".format(col, train[col].kurt()))
```

Plotting a boxplot 
```
plt.figure(figsize=(9,6));
sns.boxplot(train[col]);
```


Identify the numerical variables that can be converted to nominal category and note them down in a list col_names-

```
col_names = [.........]
for col in col_names:
	train[col] = train[col].astype('category')
	test[col] = test[col].astype('category')
```

For each ordinal category data type
```
cat_type = CategoricalDtype([.........],ordered = True)
train[col] = train[col].astype(cat_type)
train[col] = train[col].astype(cat_type)
```

COlumns having month, year as its values
[.......]

Columns with kde bw error with `sns.distplot()`
[......]


### Univariate Analysis for categorical variables

`string_columns = string_columns + [newly created categorical columns from numerical variables]`

```
col = ''
train[col].value_counts(dropna = False)
```

```
(train[col].value_counts(dropna = False, normalize = True)*100).plot(kind = 'barh');
plt.xlabel('Count %ge');
plt.ylabel(col);
```

Check for NA and for those values which have the maximum frequency




