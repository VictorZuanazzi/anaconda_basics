# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:13:57 2018

@author: victzuan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:43:07 2018

@author: VICTZUAN
"""

#Import necessary libraries:
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

def Load_and_Shuffle(file_path, separator = ','):
    '''Load and pre-process the data:
        
    '''
    #Load dataset:
    load_file = pd.read_csv(file_path, sep = separator)
    #Randomize the data, just to be sure not to get any pathological ordering 
    #effects that might harm the performance of Stochastic Gradient Descent.
    load_file = load_file.reindex(np.random.permutation(load_file.index))

    return load_file

def Reescale (dataframe, column, factor = 1):
    '''(list of list, str, number) -> NoneType
        
        Re-escale the *column* in the *dataframe* by deviding all its values 
        by the *factor*.
        
        Args:
            dataframe: pandas dataframe.
            column: str, name of the column to be re-escaled.
            factor: float, number that will devide all values in the specified
                column.
    
    '''
    if factor != 0:
        dataframe[column] /= factor
    
def Optimizer_Gradient_Descent(learning_rate):
    '''(number) -> tensorflow.contrib.estimator.python.estimator.extenders._TransformGradients
        
        Train this model using the GradientDescentOptimizer, which implements 
            Mini-Batch Stochastic Gradient Descent (SGD). Uses gradient clipping 
            to the optimizer via clip_gradients_by_norm. 
            #Gradient clipping ensures the magnitude of the gradients do not 
            become too large during training, which can cause gradient 
            descent to fail.
        
        Args:
            learning_rate: The learning_rate argument controls the size of the
                gradient step.
        Return: 
    '''
    #Use gradient descent as the optimizer for training the model.
    my_optimizer= tf.train.GradientDescentOptimizer(learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    return my_optimizer

#Step4: Define the Input Function:
def Input_1_Feature(features, targets, batch_size=1, shuffle=True, num_epochs = None):
    '''Trains a linear regression model of one feature.

    Args: 
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: Ture or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. 
            None = repeat indefinetely.
    Returns:
        Tuple of (features, labels) for next data batch
    '''
    
    #Define an input function, which instructs TensorFlow how to preprocess 
    #the data, as well as how to batch, shuffle, and repeat it during model 
    #training.
    
    #Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key, value in dict(features).items()}
    
    #Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets)) #warning: 2GB limit.
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    #Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
        
    #Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def Predict_On_Training_Data(prediction_input_fn, linear_regressor):
    ''' (function, function) -> generator object Estimator.predict
    
        Make predictions on training data, to see how well our model fit it 
        during training.
        Note: Training error measures how well your model fits the training data, 
        but it does not measure how well your model generalizes to new data.
        
        args:
            prediction_input_fn: _____
            linear_regressor: ____
        
        return:
            list with the data the model is tring to predict
    
    '''
    #Call predict() on the linear_rergessor to make predictions.
    predictions = linear_regressor.predict(input_fn = prediction_input_fn)
    
    #Format predictions as Numpy array, so we can calculate error metrics.
    predictions = np.array([item['predictions'][0] for item in predictions])
    return predictions

def compute_loss_RMSE(predictions, targets, root_mean_squared_errors):
    '''
    '''
    #Compute loss. <- THAT CAN HAPPEN SOMEWHERE ELSE
    root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets)
            )
    #Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    return root_mean_squared_error

def train_model(
        dataframe,
        learning_rate = 0.00002001, 
        steps = 500, 
        batch_size = 5, 
        input_feature = "total_rooms", 
        periods = 10, 
        data_label = "median_house_value",
        show_sample = 300
        ):
    '''Trains a linear regression model of one feature.
    
        Args:
            learning_rate: A 'float', the learning rate.
            steps: A non-zero 'int', the total number of training steps. A training 
                step consists of a forward and backward pass using a single batch.
            batch_size: A non-zero 'int', the batch size.
            input_feature: A 'string' specifying a column from 'california_housing_datarfame'
                to use as input feature.
            periods: number of iterations in between two plots.
        Return:
            root_mean_squared_errors: list of the RMSE of each period.
    
    '''
    #In this function, we'll proceed in 10 evenly divided periods so that we 
    #can observe the model improvement at each period.   
    steps_per_period = steps/periods
    my_feature_data = dataframe[[input_feature]]
    #Define the target.
    targets = dataframe[data_label]
    #Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(input_feature)] 
    
    #Create input functions.
    training_input_fn = lambda: Input_1_Feature(
            my_feature_data, 
            targets, 
            batch_size=batch_size
            )
    prediction_input_fn= lambda: Input_1_Feature(
            my_feature_data, 
            targets, 
            num_epochs=1,
            shuffle=False
            )
    #Configure the LinearRegressor.
    my_optimizer = Optimizer_Gradient_Descent(learning_rate)
    #Configure the linear regression model with our feature columns and optimizer.
    linear_regressor = tf.estimator.LinearRegressor(
            feature_columns = feature_columns,
            optimizer = my_optimizer
            )
    
    #CAN THAT HAPPEN SOMEWHERE ELSE?
    #Set up to plot the state of our model's line each period.
    plt.figure(figsize = (15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(data_label)
    plt.xlabel(input_feature)
    sample = dataframe.sample(n=show_sample)
    plt.scatter(sample[input_feature], sample[data_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    
    #Train the model, but do so inside a loop so that we can periodically asses 
    # loss metrics.
    print ('Training model...')
    print ("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range (0, periods):
        #Tran the model, starting from the prior state.
        linear_regressor.train(
                input_fn = training_input_fn,
                steps= steps_per_period
                )
        #Take a brake and comput predictions.
        predictions = Predict_On_Training_Data(prediction_input_fn, linear_regressor)
        
        root_mean_squared_error = compute_loss_RMSE(predictions, targets, root_mean_squared_errors)
        #Occasionally print the current loss.
        print ('period %02d: %0.2f' % (period, root_mean_squared_error))
        
        #Track the weights and biases over time.
        #Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[data_label].max()])
        
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias =  linear_regressor.get_variable_value('linear/linear_model/bias_weights')
        
        x_extents = (y_extents - bias)/weight
        x_extents = np.maximum(np.minimum(x_extents, 
                                          sample[input_feature].max()), 
                                        sample[input_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color = colors[period])
    print ('Model training finished')
    
    #For each period, we'll compute and graph training loss andalso plot the 
    #feature weight and bias term values learned by the model over time. 
    #This may help you judge when a model is converged, or if it needs more 
    #iterations.
    #Output a graph of loss metrics over periods.
    plt.subplot(1,2,2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    
    #Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())
    
    print ("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
    
    return root_mean_squared_errors
   
def california_housing_example(
        learning_rate = 0.00002072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value"):
    '''Exapme Function of how it works to fit the data.
    '''
    # file_path stores the address of the data to be analized.
    file_path ="https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
    # Calls function Load_and_Suffle to load the data and suffle it.
    california_housing_dataframe = Load_and_Shuffle(file_path)
    
    #Scale median_house_value to be in units of thousands, so it can be learned a 
    #little more easily with learning rates in a range that we usually use.
    Reescale(california_housing_dataframe,"median_house_value",factor = 1000.0)
    
    #Creates a Synthetic feature.
    california_housing_dataframe["rooms_per_person"]= (california_housing_dataframe["total_rooms"]/california_housing_dataframe["population"])
    
    #Select outliers and clip them from the dataframe.
    california_housing_dataframe["rooms_per_person"] = (
            california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

    _ = california_housing_dataframe["rooms_per_person"].hist()
    #Print a quick summary of useful statistics:
    #Std, mean, max, min...
    print(california_housing_dataframe.describe())
    
    _ = train_model(
            dataframe = california_housing_dataframe,
            learning_rate = 0.00002072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )

###
