#Analyzing A/B Test Results

#Table of Contents

#Introduction

#Part I - Probability

#Part II - A/B Test

#Part III - Regression

#Introduction

#A/B tests are very commonly performed by data analysts and data scientists.

#For this project, we will be working to understand the results of an A/B test run by an e-commerce website. Our goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.


#Part I - Probability

#To get started, let's import our libraries.

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline

#Setting the seed

random.seed(42)

# Now, reading in the ab_data.csv data. Store it in df.

# Read in the dataset and take a look at the top few rows here:

df=pd.read_csv('ab_data.csv')
df.head()

#the number of rows in the dataset.

df.shape[0]

#number of unique users in the dataset.

df.user_id.nunique()

#The proportion of users converted.

k=df.groupby(['user_id'])['converted'].mean()
t=pd.DataFrame(k)
t.mean()
