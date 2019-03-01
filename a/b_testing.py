#Analyze A/B Test Results

#Table of Contents

#Introduction

#Part I - Probability

#Part II - A/B Test

#Part III - Regression

#Introduction

#A/B tests are very commonly performed by data analysts and data scientists.

#Part I - Probability

#To get started, let's import our libraries.

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#Setting the seed
random.seed(42)

#Now, reading the ab_data.csv data. Storing it in df.

#Reading the dataset and having a look at the top few rows here:

df=pd.read_csv('ab_data.csv')
df.head()

#Number of rows in the dataset.

df.shape[0]

#Number of unique users in the dataset.

df.user_id.nunique()

#The proportion of users converted.

k=df.groupby(['user_id'])['converted'].mean()
t=pd.DataFrame(k)
t.mean()

#The number of times the new_page and treatment don't match.

df[((df['group'] == 'treatment') != (df['landing_page'] == 'new_page')) == True].shape[0]


#Do any of the rows have missing values?

df.info()

#For the rows where treatment does not match with new_page or control does not match with old_page, we cannot be sure if this row truly received the new or old page.

df.drop(df.query("group == 'treatment' and landing_page == 'old_page'").index, inplace=True)
df.drop(df.query("group == 'control' and landing_page == 'new_page'").index, inplace=True)
df.to_csv('ab_edited.csv', index=False)
df2 = pd.read_csv('ab_edited.csv')

# Double Check all of the correct rows were removed - this should be 0

df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


#User_ids in df2

df2.user_id.nunique()

#User_id repeated in df2.

df2.user_id.value_counts()

#Row information for the repeat user_id

df2.query('user_id == 773192')

#Removing one of the rows with a duplicate user_id, but keeping dataframe as df2.

df2.drop_duplicates('user_id',inplace=True)

# Checking for above operation

df2.query('user_id == 773192')

#What is the probability of an individual converting regardless of the page they receive?

df2['converted'].mean()

#Given that an individual was in the control group, what is the probability they converted?

df_grp = df2.groupby('group')
df_grp.describe()

#What is the probability that an individual received the new page?

df2.query('landing_page == "new_page"').user_id.nunique()/df2.user_id.nunique()

#Conclusion

#No, there is insufficient evidence that new treatment leads to more conversions as the results obtained are reverse.


#Part II - A/B Test

#The conversion rate for 𝑝𝑛𝑒𝑤 under the null

p_new=df2['converted'].mean()
