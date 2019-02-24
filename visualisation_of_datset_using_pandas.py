#Project: Analysing presence of Patients on Appointment
#Table of Contents
#Introduction
#Data Wrangling
#Exploratory Data Analysis
#Predictive Analysis
#Conclusions

#Introduction
#This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.

#ScheduledDay tells us on what day the patient set up their appointment. Neighborhood indicates the location of the hospital.

#Scholarship indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.

#Encoding of the last column: it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up.

#Is there any significant trend regarding presence of patients depending upon Gender,Age etc ? Is there some disparity among patients who did not show up at the time of appoitnment based on some disease or gender?

#Performing predictive analysis has been performed below after the analysis part and the accuracy of both testing dataset and training dataset have been checked

# Importing packages planned to use for data analysis and predictive analysis.

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline

#Data Wrangling
#Accessing and cleaning data

#General Properties
# Loading and printing out a few lines. Perform operations to inspect data

df=pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head(3)

#Performing operations to inspect data types and look for instances of missing or possibly errant data.

df.info()

#Data Cleaning
#Here I have converted the two columns of Appointment Day and Scheduled Day to date time format.Renamed columns as per requirement

#Descriptive statistic of each column has been accessed so as to ensure that there is not any discrepancy in data and if found that particular row has been removed.

#The entire dataset has been filtered in two parts as per their presence on the appointment day.

# Converting ScheduledDay and AppointmentDay to datetime format and further checking for clarity.

df['ScheduledDay']=pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay']=pd.to_datetime(df['AppointmentDay'])
df.info()

# Renaming columns as per requirement

df=df.rename(columns ={'No-show':'No_show'})

# To get descriptive statistics for each column

df.describe()

#It is clear from above descriptive statistics that majority of the patients included in the dataset

#Did not receive Scholarship.

#Don't suffer from Hipertension.

#Don't suffer from Diabetes.

#Don't suffer from Alcoholism.

#Aren't Handicapped.

#Received SMS for Appointment.

# As it is clear from above data that the minimum age is negative which is not possible so dropping

# that partcular row

df.query('Age <0')

# Dropping that particular row

df.drop(df.index[99832],inplace=True)

# Checking the result

df.query('Age <0')

#Filtering the dataset on the basis of patients who did show up at the appointment and those who
# didn't, further dividing them into two different datasets.


df1=df.query('No_show == "No" ')
df2=df.query('No_show == "Yes" ')
df1.describe()

#From the above descriptive statistics of df1 it is clear that among the patients who did attend the appointment

#Majority were not suffering from any type of disease mentioned in the dataset viz Hipertension,Diabetes etc.

#Majority did not receive the Scholarship.

#Majority of them did receive the SMS.

df2.describe()

#From the above descriptive statistics of df2 it is clear that among the patients who did not attend the appointment

#Majority were not suffering from any type of disease mentioned in the dataset viz Hipertension,Diabetes etc.

#Majority did not receive the Scholarship.

#Majority of them did receive the SMS.

#As it can be seen above there is a wide variation in the age of the patients present so adding a new column Age_group which will show in which category does the patients belong.

#Category is divided as

#Age <18 Minor

#18<Age<30 Adult

#30<Age<60 Mature

#60<Age Senior_Citizen

# Adding new column and intialising it

df1.loc[:,'Age_group']=" "
df2.loc[:,'Age_group']=" ";


# Checking the results

df1.head(1)


# Filling the new column with values as defined above

w=df1['Age']
for i,c in enumerate(w):
    if (c<=18):
        df1.iloc[i,-1]='Minor'
    elif(c>18 and c<=30):
        df1.iloc[i,-1]='Adult'
    elif(c>30 and c<=60):
        df1.iloc[i,-1]='Mature'
    else:
        df1.iloc[i,-1]='Senior_Citizen'


# Similarly for df2 dataframe

w1=df2['Age']
for i,c in enumerate(w1):
    if (c<=18):
        df2.iloc[i,-1]='Minor'
    elif(c>18 and c<=30):
        df2.iloc[i,-1]='Adult'
    elif(c>30 and c<=60):
        df2.iloc[i,-1]='Mature'
    else:
        df2.iloc[i,-1]='Senior_Citizen'

# Checking for results

df1.head(2)
# Similarly checking for df2

df2.head(2)


#Exploratory Data Analysis
#Is there any difference in age group distribution for those who did show up at the appointment and those who didn't?
# Using pie-chart to answer the above question for both sections od dataset df1 and df2.
# For df1 the Pie-chart is

age_dist=df1['Age_group'].value_counts()
age_dist.plot(kind='pie',figsize=(20,10));

# Similarly for df2 dataset
age_dist1=df2['Age_group'].value_counts()
age_dist1.plot(kind='pie',figsize=(20,10));


#From the above two charts it can be concluded that the proportion of Mature and Minor age group in both df1 and df2 datasets are approximately same,but the proportions for Adult and Senior Citizen age groups are different for each subset of dataframe.
#What proportion of patients received scholarship who did attend the appointment(trying to get an insight whether scholarship does impact their presence for the appointment)?
# Here bar charts have been used to compare the proportions of students who received scholarships
# Comparing the proportions for those who did show up at the appointment


c1=df1['Scholarship'].value_counts()
k=["No","Yes"]
plt.bar(k,[c1[0]/(c1[0]+c1[1]),c1[1]/(c1[0]+c1[1])])
plt.title("Distribution of patients having received scholarships who did show up at the Appointment")
plt.xlabel("Scholarship status")
plt.ylabel("Number of Patients");


# Similarly comparing the proportions for those who did not show up at the appointment
c2=df2['Scholarship'].value_counts()
plt.bar(k,[c2[0]/(c2[0]+c2[1]),c2[1]/(c2[0]+c2[1])])
plt.title("Distribution of patients having received scholarships who did not show up at the Appointment")
plt.xlabel("Scholarship status")
plt.ylabel("Number of Patients");


#From the above two Bar charts it is clear that the proportions of students who did receive the scholarship for both sub-datasets df1 and df2 are approximately same.
#Thus it can be concluded that Scholarships doesn't affect the presence of patients at the Appointment.
#Distribution of patients on gender basis who did not show up at the appointment but are suffering from Alcoholism
# First dividing the the df2 dataset into two groups of males and females


df2_m=df2.query('Gender == "M"')
df2_f=df2.query('Gender == "F"')

# Now plotting the proportions of males and females who are suffereing from alcoholism but did not show up
# at the Appointment


count_m=df2_m['Alcoholism'].value_counts()
k1=["Males","Females"]
count_f=df2_f['Alcoholism'].value_counts()
plt.bar(k1,[count_m[1]/(count_m[1]+count_m[0]),count_f[1]/(count_f[0]+count_f[1])])
plt.title("Distribution of patients on gender who did not attend the appointment but are suffering from Alcoholism")
plt.xlabel("Gender")
plt.ylabel("Number of Patients");

#It can be concluded from above visualisation that the proportion of males suffering from alcoholism are more than that of women in the group of not attending the appointment.
