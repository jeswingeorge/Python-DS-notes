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
sns.set()
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
bool_columns = list(train.select_dtypes(include = 'bool').columns)
```

## 2. Univariate Analysis

### Univariate Analysis for numerical variables

Function to find the skewness and kurtosis and presence of null values of columns of dataframe

```
global measure_columns_spread_df
measure_columns_spread_df = pd.DataFrame(columns = ['col_name', 'skewness', 'kurtosis', 'transformation_reqd', 'null_values'])
def collect_calculate_col_spread(df, col):
    global measure_columns_spread_df
    temp = pd.DataFrame(pd.Series([col, df[col].skew(), df[col].kurt(), np.nan, np.nan])).T
    temp.columns = ['col_name', 'skewness', 'kurtosis', 'transformation_reqd', 'null_values']
    condition = (df[col].skew() < -1) | (df[col].skew() > 1) | (df[col].kurt() < -1) | ((df[col].kurt() > 1))
    temp.loc[0,'transformation_reqd']=np.where(condition, 'Yes', 'No')
    temp.loc[0, 'null_values'] = np.where(df[col].isnull().sum(), 'Present', 'Absent')
    print("Count of null values in train dataset: ", df[col].isnull().sum())
    measure_columns_spread_df = pd.concat([measure_columns_spread_df, temp], ignore_index = True)
```

Function to automate the univariate analysis of numerical variable 

```
def univariate_numerical_analysis(df, col):
    collect_calculate_col_spread(df, col)
    fig, axs = plt.subplots(2,1, figsize = (10,9));
    sns.distplot(train[col], ax = axs[0]);
    axs[0].set_title("Histogram of " + col);
    axs[1].set_title("Boxplot of " + col);
    sns.boxplot(train[col], ax = axs[1]);
    title=("Skewness of {}: ".format(col) + "{0:.2f}".format(train[col].skew()) + " and " + "Kurtosis of {}: ".format(col) 
           +"{0:.2f}".format(train[col].kurt()))
    fig.suptitle(title, y = 1.01);
    plt.tight_layout(); 
    plt.show();
```







```
measure_columns_spread_df.to_pickle('numerical_spread.pkl')
```


Identify the numerical variables that can be converted to nominal category and note them down in a list col_names-
1st fill all the NA values

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

Columns having month, year as its values
[.......]

Columns with kde bw error with `sns.distplot()`
[......]


### Univariate Analysis for categorical variables

Make a list of object/category columns where NA implies a category in itself eg: No road, No pool, etc, replace NA here by 'None'.

`string_columns = string_columns + [newly created categorical columns from numerical variables]`

```
col = ''
(train[col].value_counts(dropna = False, normalize = True)*100).plot(kind = 'barh');
plt.xlabel('Count %ge');
plt.ylabel(col);
```

```
train[col].value_counts(dropna = False)
```

Check for NA and for those values which have the maximum frequency and Nominal and ordinal category type.

***

# Data Wrangling

#### Make a list of object columns where NA implies a category in itself eg: No road, No pool, etc, replace NA here by 'None'. THEN convert to category type.

(Try to fix this issue in Univariate analysis itself if not possible use this: )
One way to make a list of such columns - `col_fillna`

Then from the columns obtained use data description to get columns whose NA values implies None. To remove those columns which do not satisfy the NA value as None condition remove that particular columns  from the list.

```
all_object_category_columns = train.select_dtypes(include = ['object', 'category']).columns
col_fillna = list(set(test[all_object_category_columns].isnull().sum()[test[all_object_category_columns].isnull().sum()!=0].index.tolist() + 
train[all_object_category_columns].isnull().sum()[train[all_object_category_columns].isnull().sum()!=0].index.tolist()))

# list of columns which doesnt have NA as category and to be deleted from obj_categoryc_col_with_na
del_list = []
for col in del_list:
    col_fillna.remove(col)
    
col_fillna  # columns where NaN values have meaning e.g. no pool etc.
```

```
# replace 'NaN' with 'None' in these columns
for col in cols_fillna:
    train[col].fillna('None',inplace=True)
    test[col].fillna('None',inplace=True)
```







# Data Wrangling - Convert object to category column type only after filling all the NA values





## 3. Bivariate Analysis


