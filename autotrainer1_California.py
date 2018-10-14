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
LR = (random.randint(0,100))*0.0000001
ST = (random.randint(100,10000))
BS = random.randint(1,100)
IF = "median_income"
P = 10
DL = "median_house_value"
error_history = []

not_good_enough = True
iterations = 1

while not_good_enough:
    print ('parameters: LR ', LR, ' ST ', ST,' BS ', BS)
    error = cal.train_model(
            dataframe = california_housing_dataframe,
            learning_rate = LR, 
            steps = ST, 
            batch_.size = BS, 
            input_feature = IF, 
            periods = P, 
            data_label = DL,
            show_sample = 300
            )
    
    error_history.append(error[-1])
    partial_ok = False
    if iterations >= 2:
        #Check if error is descent.
        for i in range(len(error)-1):
            if error[i]>error[i+1]:
                partial_ok = True
            else:
                partial_ok = False
        #Check if error is smaller than the last one
        if partial_ok & (error_history[-2]>error_history[-1]*1.5):
            not_good_enough = False
        else:
            #This model does not converge
            LR = math.fabs(LR*(1+random.randint(-1,1)*(error[random.randint(0,len(error)-1)]/LR)))
            ST = 1+ int(math.fabs(ST*int(1+random.randint(-1,1)*(error[random.randint(0,len(error)-1)]/ST))))
            BS = 1+ int(math.fabs(BS*int(1+random.randint(-1,1)*(error[random.randint(0,len(error)-1)]/BS))))
    else:
        print (LR)
        LR = math.fabs(LR*(1+random.randint(-1,1)*(error[random.randint(0,len(error)-1)]/LR)))
        print (LR)
    iterations += 1
    if iterations > 100:
        not_good_enough = False

print ('done')