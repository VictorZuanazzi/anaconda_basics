# -*- coding: utf-8 -*-
"""
Created on Fri May 18 07:02:39 2018

@author: VICTZUAN

source: https://colab.research.google.com/notebooks/mlcc/validation.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=validation-colab&hl=en#scrollTo=J2ZyTzX0HpCc
"""

import math

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

def preprocess_features(california_housing_dataframe):
    '''Prepares input features from California housing data set.
    
    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data from the California housing dataset.
    
    Returns:
        A DataFrame that contains the features to be used for the model, including synthetic features.
    '''
    
    selected_features = california_housing_dataframe[
            ["latitude",
             "longitude",
             "housing_median_age",
             "total_rooms",
             "total_bedrooms",
             "population",
             "households",
             "median_income"]]
    processed_features = selected_features.copy()
    #Create a synthetic feature
    processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"]/
            california_housing_dataframe["population"])
    return processed_features

def preprocess_targets(california_housing_dataframe):
    """Pepares target features (i.e., labels) from California housing data set.
    
    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain
            data from the California housing data set.
    Returns:
        A DataFrame that contains the target feature.
    """
    
    output_targets = pd.DataFrame()
    #Scale the target to be in units of thousands of dollars.
    output_targets["median_house_falue"] = (
            california_housing_dataframe["median_house_value"]/1000.0)
    return output_targets


california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
#For the training set, we'll choose the first 12000 examples, out of the total 
#of 17000.
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_examples.describe()
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
training_targets.describe()

































