##Linking to a database in python

#Importing the sql library in python environment

from sqlalchemy import create_engine

engine=create_engine('sqlite:///sample.db')

#Storing pandas dataframe to a mater table in the database

df.to_sql('master',engine,index=False)

#Reading database data into a pandas DataFrame

df_gather=pd._readsql('Select * from master',engine)
