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
sns.set(style="ticks", rc={'figure.figsize':(11,9)})
sns.set_context(rc = {"font.size":15, "axes.labelsize":15}, font_scale=2)
sns.set_palette('colorblind');
from pandas.api.types import CategoricalDtype
# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

import math
from collections import Counter
from scipy import stats

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")


import statsmodels.api as sm
from statsmodels.formula.api import ols   # To perform ANOVA

# f - To find F-statistic for ANOVA
from scipy.stats import chi2_contingency, chi2, f

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

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
global measure_columns_spread_df, list_of_columns_with_null_values
list_of_columns_with_null_values = []
measure_columns_spread_df = pd.DataFrame(columns = ['col_name', 'skewness', 'kurtosis', 'transformation_reqd', 'null_values'])
def collect_calculate_col_spread(col, df = train):
    global measure_columns_spread_df
    temp = pd.DataFrame(pd.Series([col, df[col].skew(), df[col].kurt(), np.nan, np.nan])).T
    temp.columns = ['col_name', 'skewness', 'kurtosis', 'transformation_reqd', 'null_values']
    condition = (df[col].skew() < -1) | (df[col].skew() > 1) | (df[col].kurt() < -1) | ((df[col].kurt() > 1))
    temp.loc[0,'transformation_reqd']=np.where(condition, 'Yes', 'No')
    temp.loc[0, 'null_values'] = np.where(df[col].isnull().sum(), 'Present', 'Absent')
    if (df[col].isnull().sum()):
        list_of_columns_with_null_values.append(col)
    print("Count of null values in train dataset: ", df[col].isnull().sum())
    measure_columns_spread_df = pd.concat([measure_columns_spread_df, temp], ignore_index = True)
```

Function to automate the univariate analysis of numerical variable 

```
def univariate_numerical_analysis(col, df = train):
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


For some of the numerical variable this error might come up - [seaborn: Selected KDE bandwidth is 0. Cannot estimate density](https://stackoverflow.com/questions/60596102/seaborn-selected-kde-bandwidth-is-0-cannot-estimate-density). Then use the following code for analysis:

```
col = ''
train[col].plot(kind= 'hist');
```


Identify the numerical variables that can be converted to nominal category and note them down in a list col_names-
1st fill all the NA values


```
for col in list_of_columns_with_null_values:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())
```

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



CHeck for presence of 

```
measure_columns_spread_df.to_pickle('numerical_spread.pkl')
```




### Univariate Analysis for categorical variables

Make a list of object/category columns where NA implies a category in itself eg: No road, No pool, etc, replace NA here by 'None'.

```
object_category_columns = ['category', 'object']
train.select_dtypes(include = object_category_columns).columns
```

```
global list_of_object_category_columns_with_null_values
list_of_object_category_columns_with_null_values = []
def univariate_category_analysis(train, col):
    print('Number of null values in {} is: {}'.format(col, train[col].isnull().sum()))
    if (train[col].isnull().sum()):
        list_of_object_category_columns_with_null_values.append(col)
    (train[col].value_counts(dropna = False, normalize = True)*100).plot(kind = 'barh');
    plt.xlabel('Count %ge');
    plt.ylabel(col);
    plt.show();
```



Check for NA and for those values which have the maximum frequency and Nominal and ordinal category type.




Create two lists in which ordinal and nominal columns' names are appended. 

```
ordinal_columns_list = []
nominal_columns_list = []
```

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


#### a. Categorical and continuous variables

To find association between a categorical and continuous(target) value using ANOVA
```
global categ_columns_with_high_association, categ_columns_with_low_association
categ_columns_with_high_association = []
categ_columns_with_low_association = []
def perform_anova_and_its_results(categ_col, num_col=target_numerical_col, df = train):
    df_sst = len(df[num_col])-1
    df_ssb = df[categ_col].nunique() - 1
    df_ssw = df_sst - df_ssb
    F_critical = f.ppf(0.95, df_ssb, df_ssw)
    print("F_Critical: {0:.3f}".format(F_critical))
    results = ols('{} ~{}'.format(num_col, categ_col), data = train).fit()
    aov_table = sm.stats.anova_lm(results, typ = 1)  
    F_stat = aov_table.loc[categ_col, 'F']
    print("F_statistic: {0:.3f}".format(F_stat))
    if (F_stat > F_critical):
        print("F-statistic is more than F-critical")
        print("There is an association between {} and {}".format(categ_col,num_col))
        categ_columns_with_high_association.append(categ_col)
    else:
        print("F-statistic is less than F-critical")
        print("There is no association between {} and {}".format(categ_col,num_col))
        categ_columns_with_low_association.append(categ_col)
    print('-'*30)
```

```

for col in train.select_dtypes(include = ['category', 'object']).columns.to_list():
    perform_anova_and_its_results(col)
```


### b. categorical-categorical variable relationship

First need to check if the 2 categorical variables involved are dependant or not this can be done with the help of chi-squared test.

```
dependant_category_cols = []
def chi_square_test(a,b, df = X_train):
    # a and b are the column names of dataframe - pandas series
    two_way_table = pd.crosstab(X_train[a],X_train[b])
    p_value = chi2_contingency(two_way_table)[1]
    if (p_value < 0.05):
      #  print("Null hypothesis is rejected. The variables {} and {} are dependent.".format(a,b))
        dependant_category_cols.append((a,b))
   # else:
       # print("The variables {} and {} are independent.".format(a,b))

```

```
# To create a dataframe with columns which are dependant
df = pd.DataFrame(dependant_category_cols, columns =['col1', 'col2']) 
```



#### To find association i.e., phi-coefficient between 2 binary categorical variables:
```
def phi_coefficient(a,b):
	# a and b are 2 binary columns/series of dataframe
    # At least one variable a or b is a nominal variable.
    temp = pd.crosstab(a,b)
    return(((temp.iloc[1,1] * temp.iloc[0,0]) - (temp.iloc[0,1]*temp.iloc[1,0]))/
          np.sqrt(np.prod(temp.apply(sum, axis = 'index').to_list()) * np.prod(temp.apply(sum, axis = 'columns').to_list())))

```

#### To find association between nominal categorical variables

1. Using Crammer's V (Gives correlation between the 2 nominal categorical variables assume symmetry i.e, V(x,y)=V(y,x))

```
def cramers_v(a,b):
    crosstab = pd.crosstab(a,b)
    chi2 = chi2_contingency(crosstab)[0]  # chi-squared value
    n = crosstab.sum().sum()
    phi2 = chi2/n
    r, k = crosstab.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return(np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))))
```


2.  Using Theil’s U (Gives correlation between the 2 nominal categorical variables doesnt consider Symmetry i.e, U(x,y)!=U(y,x))

```
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy


def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = stats.entropy(p_x)
    if s_x == 0:
        return(1)
    else:
        return((s_x - s_xy)/s_x)

```