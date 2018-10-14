# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:35:31 2018

@author: victzuan
"""

#Exercise #1
#Modify the cities table by adding a new boolean column that is True if and 
#only if both of the following are True:
#
#The city is named after a saint.
#The city has an area greater than 50 square miles.
#Note: Boolean Series are combined using the bitwise, rather than the 
#traditional boolean, operators. For example, when performing logical and, use & instead of and.
#
#Hint: "San" in Spanish means "saint."

#Importpandas library for Machin learning:
import pandas as pd
pd.__version__

#Create a Serie.
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacamento'])
population = pd.Series([852469, 1015785, 485199])

#Create a Dataframe, which is a table of several Series
cities = pd.DataFrame({'City name': city_names, 'Population': population})

#Ading data to Dataframes.
cities['Area square miles']= pd.Series([46.87, 176.53, 97.92])
cities['Population density']= cities['Population']/cities['Area square miles']

#cities['Named after a Saint'] = cities['City name'].apply(lambda name: 'San' in name)
#cities['Area > 50 square miles'] = cities['Area square miles'].apply(lambda val: val>50)

cities['Saint & Area > 50'] = cities['City name'].apply(lambda name: 'San' in name) & cities['Area square miles'].apply(lambda val: val>50)

#Proposed solution in the tutorial:
#cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))

print(cities)