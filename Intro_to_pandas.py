# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:45:55 2018

@author: victzuan
"""
#Importpandas library for Machin learning:
import pandas as pd
pd.__version__
#Import NumPy library, useful for scientific computing:
import numpy as np

#Create a Serie.
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacamento'])
population = pd.Series([852469, 1015785, 485199])

#Create a Dataframe, which is a table of several Series
cities = pd.DataFrame({'City name': city_names, 'Population': population})

#Ading data to Dataframes.
cities['Area square miles']= pd.Series([46.87, 176.53, 97.92])
cities['Population density']= cities['Population']/cities['Area square miles']



#Examples of data manipulation:
print (cities['City name'])
print (cities['City name'][1])
print (cities[0:2])
#Basic arithmetic operations are applied to the whole series:
population/1000
np.log(population)
population.apply(lambda val: val>1000000)

#Return the indexes of the DataFrame or Serie.
city_names.index
cities.index

#Manually sorts the DataFrame in the inputed order.
cities.reindex([2,0,1])

#Shuffle the DataFrame:
cities.reindex(np.random.permutation(cities.index))




#Import data from a document:
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
#.describe(): gives some interesting statistics about the data.
california_housing_dataframe.describe()
#.head(): return the contentant of the 5 first lines of the document.
california_housing_dataframe.head()
#.hist(index): return the histogram of one of the collumns of the data:
california_housing_dataframe.hist('housing_median_age')


