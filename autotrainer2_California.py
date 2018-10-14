# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:51:06 2018

@author: victzuan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:24:12 2018

@author: victzuan
"""

import california_housing_training_model as cal

import math
import random

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

class Individual (object):
    '''
    '''
    
    def __init__(self, learn_rate, n_steps, batch_size, error=0.0):
        self.learn_rate = learn_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.error = error
        
        

# file_path stores the address of the data to be analized.
file_path ="https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
# Calls function Load_and_Suffle to load the data and suffle it.
california_housing_dataframe = cal.Load_and_Shuffle(file_path)

#Scale median_house_value to be in units of thousands, so it can be learned a 
#little more easily with learning rates in a range that we usually use.
cal.Reescale(california_housing_dataframe,"median_house_value",factor = 1000.0)

#Print a quick summary of useful statistics:
#Std, mean, max, min...
#print(california_housing_dataframe.describe())

num_individuals = 5
error_target = 90
individual = []
for i in range(num_individuals):
    j = i % 10 
    individual.append(Individual(
            learn_rate = (random.randint(1,10))/(10**(j+1)),
            n_steps = (random.randint(j+1,1+ 10**j)),#
            batch_size = random.randint(1,1+j**3)#
            ))

IF = "median_income"
P = 10
DL = "median_house_value"
error_history = []

good_enough = False
iterations = 1

while not good_enough:
    error=[]
    for i in range(num_individuals):
        print ('parameters', i,': LR ', individual[i].learn_rate, ' ST ', individual[i].n_steps,' BS ', individual[i].batch_size)
        error.append(cal.train_model(
            dataframe = california_housing_dataframe,
            learning_rate = individual[i].learn_rate, 
            steps = individual[i].n_steps, 
            batch_size = individual[i].batch_size, 
            input_feature = IF, 
            periods = P, 
            data_label = DL,
            show_sample = 300
            ))
        individual[i].error = error[i]
        
    
    #Store the errors:
    error_history.append(error)
    champion = error.index(min(error))
        
    #Improve the model:
    partial_ok = False
    if min(error)[-1]< error_target:
        good_enough = True
    else:
        for i in range(num_individuals):
            if i != champion:
                if random.randint(0,10) != 7:
                    individual[i].learn_rate = (individual[i].learn_rate+individual[champion].learn_rate*(1+random.randint(-1,1)/random.randint(9,11)))/2
                    individual[i].n_steps = 1+ int((individual[i].n_steps + individual[champion].n_steps*(1+random.randint(-1,1)/random.randint(9,11)))/2)
                    individual[i].batch_size =1+  int((individual[i].batch_size + individual[champion].batch_size*(1+random.randint(-1,1)/random.randint(9,11)))/2)
                else:
                    individual[i].learn_rate = (random.randint(1,10))/(10**(random.randint(0,5)))
                    individual[i].n_steps = 1+ int(individual[i].n_steps/random.randint(1,10))
                    individual[i].batch_size = 1+ int(individual[i].batch_size/random.randint(1,10))

    print ('iteration number: ', iterations)       
    print ('Individual', champion)
    print ('learning rate:', individual[champion].learn_rate)
    print ('steps:', individual[champion].n_steps)
    print ('batch size:', individual[champion].batch_size)
    iterations += 1
    if iterations > 20:
        good_enough = True

print ('Finished')
print ('Error:', error[champion])
print ('Individual', champion)
print ('learning rate:', individual[champion].learn_rate)
print ('steps:', individual[champion].n_steps)
print ('batch size:', individual[champion].batch_size)