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

#The conversion rate for  𝑝𝑜𝑙𝑑  under the null

p_old=df2['converted'].mean()

#𝑛𝑛𝑒𝑤 , the number of individuals in the treatment group

n_new=df2.query('group =="treatment"').user_id.nunique()

#𝑛𝑜𝑙𝑑 , the number of individuals in the control group

n_old=df2.query('group =="control"').user_id.nunique()

#Simulating  𝑛𝑛𝑒𝑤  transactions with a conversion rate of  𝑝𝑛𝑒𝑤  under the null and storing these  𝑛𝑛𝑒𝑤  1's and 0's in new_page_converted.

new_page_converted = np.random.choice([0,1],n_new, p=(1-p_new,p_new))

#Simulate  𝑛𝑜𝑙𝑑  transactions with a conversion rate of  𝑝𝑜𝑙𝑑  under the null storing these  𝑛𝑜𝑙𝑑  1's and 0's in old_page_converted.

old_page_converted = np.random.choice([0,1],n_old, p=(1-p_old,p_old))

#𝑝𝑛𝑒𝑤  - 𝑝𝑜𝑙𝑑 simulated values

obs_diff=new_page_converted.mean()-old_page_converted.mean()

#Creating 10,000  𝑝𝑛𝑒𝑤  -  𝑝𝑜𝑙𝑑  values using the same simulation process used above. Storing all 10,000 values in a NumPy array called p_diffs.

p_diffs=[]
size=df.shape[0]

for i in range(10000):
    samp=df2.sample(size,replace=True)
    old_samp_conv=np.random.choice([0,1],n_old, p=(p_old,1-p_old))
    new_samp_conv= np.random.choice([0,1],n_new, p=(p_new,1-p_new))
    p_diffs.append(new_samp_conv.mean()-old_samp_conv.mean())

#A histogram of the p_diffs.This plot looks like what we expected.

p_diffs=np.array(p_diffs)
plt.hist(p_diffs)
plt.show()

#Proportion of the p_diffs are greater than the actual difference observed in ab_data.csv

convert_new = df2.query('converted == 1 and landing_page == "new_page"')['user_id'].nunique()
convert_old = df2.query('converted == 1 and landing_page == "old_page"')['user_id'].nunique()
actual_cvt_new = float(convert_new)/ float(n_new)
actual_cvt_old = float(convert_old)/ float(n_old)
obs_diff = actual_cvt_new - actual_cvt_old
null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)
plt.hist(null_vals)


#Plot vertical line for observed statistic

plt.axvline(x=obs_diff,color ='red')
(null_vals > obs_diff).mean()

#Type I error rate of 5%, and Pold > Alpha, we fail to reject the null.

#Therefore, the data show, with a type I error rate of 0.05, that the old page has higher probablity of convert rate than new page.

#P-Value: The probability of observing our statistic or a more extreme statistic from the null hypothesis.

#We could also use a built-in to achieve similar results. Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance.Calculate the number of conversions for each page, as well as the number of individuals who received each page. Let n_old and n_new refer the number of rows associated with the old page and new pages, respectively.
