

Code to import libraies:

```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set(style="ticks", rc={'figure.figsize':(9,8)})
sns.set_context(rc = {"font.size":15, "axes.labelsize":15}, font_scale=2)
sns.set_palette('colorblind');
from pandas.api.types import CategoricalDtype
# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
np.set_printoptions(precision=4)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

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


## Important functions:

```
def generate_heatmap(df):
    # Generate a heatmap with the upper triangular matrix masked
    # Compute the correlation matrix
    corr = df.corr(method="spearman")
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    plt.figure(figsize = (15,9));
    # Draw the heatmap with the mask 
    sns.heatmap(corr, mask=mask, cmap='coolwarm', fmt = '.2f', linewidths=.5, annot = True);
    plt.title("Correlation heatmap");
    return
```





## 3. Bivariate Analysis


#### a. Categorical and continuous (target) variables

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
    results = ols('{} ~{}'.format(num_col, categ_col), data = df).fit()
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
    temp = pd.crosstab(a,b)
    nr = (temp.iloc[1,1] * temp.iloc[0,0]) - (temp.iloc[0,1]*temp.iloc[1,0])
    dr = np.sqrt(np.product(temp.apply(sum, axis = 'index')) * np.prod(temp.apply(sum, axis = 'columns')))
    return(nr/dr)
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