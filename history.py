# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Fri Apr  6 15:50:25 2018)---
runfile('C:/Users/victzuan/.spyder-py3/hello.py', wdir='C:/Users/victzuan/.spyder-py3')
import panda as pd
import pandas as pd
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()
california_housing_dataframe.head()
dalifornia_housing_dataframe.hist('hosing_median_age')
california_housing_dataframe.hist('hosing_median_age')
california_housing_dataframe.hist('housing_median_age')
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacamento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({'City name': city names, 'Population': population})
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print type(cities['Cty name'])
print type(cities['City name'])
print (cities['City name'])
print (cities['City name'][1])
print (cities[0:2])
population/1000
cities['City name']
import numpy as np
np.log(population)
population.apply(lambda val: val > 1000000)
runfile('C:/Users/victzuan/.spyder-py3/Intro_to_pandas_Exercise_1.py', wdir='C:/Users/victzuan/.spyder-py3')
cities
runfile('C:/Users/victzuan/.spyder-py3/Intro_to_pandas_Exercise_1.py', wdir='C:/Users/victzuan/.spyder-py3')
cities
runfile('C:/Users/victzuan/.spyder-py3/Intro_to_pandas_Exercise_1.py', wdir='C:/Users/victzuan/.spyder-py3')
city_names.index
cities.index
cities.reindex([2,0,1])
cities.reindex(np.random.permutation(cities.index))
cities.reindex([2,5,1])
cities
cities.reindex([2,5,1,0])
clear
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

import tensorflow as tf
from tensorflow.python.data import Dataset
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

pd.options.display.max_rows = 10
pd.option.display.float_format = '{:.1f}'.format

import TensorFlow

## ---(Mon Apr  9 07:50:26 2018)---
import tensorflow as tf
C:\> pip3 install --upgrade tensorflow-gpu
import tensorflow as tf

## ---(Mon Apr  9 17:45:17 2018)---
import tensorflow
which python
runfile('C:/Users/victzuan/.spyder-py3/Intro_to_pandas.py', wdir='C:/Users/victzuan/.spyder-py3')
import tensorflow

## ---(Mon Apr  9 18:08:11 2018)---
import tensorflow
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')
california_housing_dataframe.describe()
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')
print ('Tess makes me so happy!')
print "Min jesuis"
d= 1.23456789
print(d)
print ("d= %0.3f" % d)
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')
clear
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

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

#Randomize the data, just to be sure not to get any pathological ordering effects that might harm the performance of Stochastic Gradient Descent.
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

#Scale median_house_value to be in units of thousands, so it can be learned a little more easily with learning rates in a range that we usually use.
california_housing_dataframe["median_house_value"] /= 1000.0

california_housing_dataframe.describe()
my_feature = california_housing_dataframe[["total_rooms"]]
#Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]


targets = california_housing_dataframe["median_house_value"]
my_optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
my_optimizer = tf.contrib.Estimator.clip_gradients_by_norm(my_optimizer, 5.0)
my_optimizer = tf.contrib.tensorflowestimator.clip_gradients_by_norm(my_optimizer, 5.0)
help (tensorflow.contrib)
help (tensorflow.contrib.estimator)
help (tensorflow.contrib.tpu)
help (tensorflow.contrib.timeseries)
help (tensorflow.contrib.timeseries())
import tensorflow as tf
my_optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
help (tensorflow.contrib)
help (tensorflow.contrib.boosted_trees)

## ---(Tue Apr 10 08:05:50 2018)---
import tensorflow as tf
my_optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
help(tensorflow)
help(tf)
help(tf.tools)

## ---(Tue Apr 10 08:12:01 2018)---
import tensorflow as tf
my_optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.00000001)

## ---(Tue Apr 10 12:06:00 2018)---
import tensorflow as tf
my_optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

## ---(Tue Apr 10 16:09:03 2018)---
import tensorflow as tf
my_optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')
runfile('C:/Users/victzuan/.spyder-py3/testensor.py', wdir='C:/Users/victzuan/.spyder-py3')
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')
help(tensorflow.python.data)
import tensorflow
help(tensorflow.python)
import tensorflow as tf
runfile('C:/Users/victzuan/.spyder-py3/testensor.py', wdir='C:/Users/victzuan/.spyder-py3')
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')
import tensorflow
tensorflow.data
runfile('C:/Users/victzuan/.spyder-py3/testensor.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Tue Apr 10 16:54:57 2018)---
runfile('C:/Users/victzuan/.spyder-py3/testensor.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Tue Apr 10 17:04:19 2018)---
runfile('C:/Users/victzuan/.spyder-py3/testensor.py', wdir='C:/Users/victzuan/.spyder-py3')
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Tue Apr 10 17:11:00 2018)---
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')
runfile('C:/Users/victzuan/.spyder-py3/Training_model.py', wdir='C:/Users/victzuan/.spyder-py3')
my_feature = california_housing_dataframe[["total_rooms"]]
my_feature
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
feature_columns
printf(feature_columns)
print(feature_columns)
learning_rate = 0.00000001
my_optimizer= tf.train.GradientDescentOptimizer(learning_rate)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
my_optimizer
type(my_optimizer)
runfile('C:/Users/victzuan/.spyder-py3/Training_model.py', wdir='C:/Users/victzuan/.spyder-py3')
prediction_input_fn = lambda: Input_1_Feature(my_feature, targets, num_epochs=1, shuffle=False)
prediction_input_fn
predictions = linear_regressor.predict(input_fn = prediction_input_fn)
predictions
predictions = np.array([item['predictions'][0] for item in predictions])
runfile('C:/Users/victzuan/.spyder-py3/Training_model.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Wed Apr 11 16:14:55 2018)---
runfile('C:/Users/victzuan/.spyder-py3/First_steps_with_tensor_flow.py', wdir='C:/Users/victzuan/.spyder-py3')
f=lambda: x^2
f(2)
f=lambda: x^2 for x in range(10)
f=lambda x: x^2
f(2)
2^2
f=lambda x: x**2
f(2)
def x2(g):
    return g+2

x2(6)
kl=lambda: x2(k)
kl(5)
kl=lambda: x2(2)
kl
kl(2)
runfile('C:/Users/victzuan/.spyder-py3/Training_model.py', wdir='C:/Users/victzuan/.spyder-py3')
root_mean_squared_errors
runfile('C:/Users/victzuan/.spyder-py3/Training_model.py', wdir='C:/Users/victzuan/.spyder-py3')
LR = 0.0001
import Training_model
runfile('C:/Users/victzuan/.spyder-py3/Training_model.py', wdir='C:/Users/victzuan/.spyder-py3')
import Training_model
import pandas as pd
pd.__version__
#Import NumPy library, useful for scientific computing:
import numpy as np

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacamento'])
population = pd.Series([852469, 1015785, 485199])

#Create a Dataframe, which is a table of several Series
cities = pd.DataFrame({'City name': city_names, 'Population': population})

#Ading data to Dataframes.
cities['Area square miles']= pd.Series([46.87, 176.53, 97.92])
cities['Population density']= cities['Population']/cities['Area square miles']

_ = train_model(
    dataframe = cities,
    learning_rate=0.001,
    setps = 100,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    

_ = train_model(
    dataframe = cities,
    learning_rate=0.001,
    steps = 100,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=0.101,
    steps = 100,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=0.101,
    steps = 100,
    batch_size = 2,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=1.101,
    steps = 100,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=0.501,
    steps = 100,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=0.501,
    steps = 200,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=0.7501,
    steps = 200,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=0.7501,
    steps = 200,
    batch_size = 3,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=10.7501,
    steps = 200,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = cities,
    learning_rate=0.87501,
    steps = 200,
    batch_size = 1,
    input_feature = 'Population',
    periods = 10,
    data_label = 'Area square miles',
    show_sample = 3
    )
    
a = pd.Series([1,2,3,4,5,6,7,8,9,10])
b = pd.Series([1,2,3,4,5,6,7,8,9,10])

ab = pd.DataFrame({'a': a, 'b': b})
_ = train_model(
    dataframe = ab,
    learning_rate=0.0000001,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 3
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.0000011,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.0000111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.0001111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.0011111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.1011111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.0511111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.02511111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.01511111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.012511111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.0112511111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.01052511111,
    steps = 100,
    batch_size = 1,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.01052511111,
    steps = 100,
    batch_size = 5,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.01052511111,
    steps = 300,
    batch_size = 5,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.01052511111,
    steps = 300,
    batch_size = 10,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.02052511111,
    steps = 300,
    batch_size = 10,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
_ = train_model(
    dataframe = ab,
    learning_rate=0.05052511111,
    steps = 300,
    batch_size = 10,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
a = pd.Series([1,2,3,4,5,6,7,8,9,10])
b = pd.Series([5,6,7,8,9,10,11,12,13,14,15])

ab = pd.DataFrame({'a': a, 'b': b})

_ = train_model(
    dataframe = ab,
    learning_rate=0.05052511111,
    steps = 300,
    batch_size = 10,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
a = pd.Series([1,2,3,4,5,6,7,8,9,10])
b = pd.Series([5,6,7,8,9,10,11,12,13,14])

ab = pd.DataFrame({'a': a, 'b': b})

_ = train_model(
    dataframe = ab,
    learning_rate=0.05052511111,
    steps = 300,
    batch_size = 10,
    input_feature = 'a',
    periods = 10,
    data_label = 'b',
    show_sample = 10
    )
    
runfile('C:/Users/victzuan/.spyder-py3/Synthetic_Feature_Example.py', wdir='C:/Users/victzuan/.spyder-py3')
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
            
california_housing_dataframe["rooms_per_person"]= (california_housing_dataframe["total_rooms"]/california_housing_dataframe["population"])
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
            
_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.00022072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.00222072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.02222072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.22222072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.12222072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.05222072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.02222072, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.020, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
sample = dataframe.sample(n=show_sample)
sample = california_housing_.sample(n=1000)
sample = california_housing_dataframe.sample(n=1000)
plt.scatter(sample["rooms_per_person"], sample["median_house_value"])
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(calibration_data["predictions"], calibration_data["targets"])

plt.subplot(1, 2, 2)
_ = california_housing_dataframe["rooms_per_person"].hist()

california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

_ = california_housing_dataframe["rooms_per_person"].hist()

_ = train_model(

            dataframe = california_housing_dataframe,
            learning_rate = 0.020, 
            steps = 1000, 
            batch_size = 5, 
            input_feature = "rooms_per_person", 
            periods = 10, 
            data_label = "median_house_value",
            show_sample = 300
            )
            
_ = plt.scatter(calibration_data["predictions"], calibration_data["targets"])
_ = california_housing_dataframe["media_income"].hist()
_ = california_housing_dataframe["median_income"].hist()
runfile('C:/Users/victzuan/.spyder-py3/california_housing_training_model.py', wdir='C:/Users/victzuan/.spyder-py3')
california_housing_example(
        learning_rate = 0.00002072, 
            steps = 500, 
            batch_size = 5, 
            input_feature = "median_income", 
            periods = 10, 
            data_label = "median_house_value")
            
runfile('C:/Users/victzuan/.spyder-py3/california_housing_training_model.py', wdir='C:/Users/victzuan/.spyder-py3')
runfile('C:/Users/victzuan/.spyder-py3/autotrainer1_California.py', wdir='C:/Users/victzuan/.spyder-py3')
error
len(error)
runfile('C:/Users/victzuan/.spyder-py3/autotrainer1_California.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Fri Apr 13 10:34:09 2018)---
runfile('C:/Users/victzuan/.spyder-py3/autotrainer1_California.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Fri Apr 13 15:49:27 2018)---
runfile('C:/Python34/Employee_class.py', wdir='C:/Python34')
runfile('C:/Python34/customer_class_example.py', wdir='C:/Python34')
class Customer(object):
    '''A customer of Zuanazzi Bank with a checking account. Customers hav the following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.

    '''

    def __init__(self, name, balance = 0.0):
        '''Return a Customer object whose name is *name* and starting balance is *balance*.
        '''
        self.name = name
        self.balance = balance

    
    def withdraw(self, amount):
        '''Return the balance remaining after withdrawing *amount* dollars.
        '''
        if amount > self.balance:
            raise RuntimeError('Amont greater than available balance.')
        self.balance -= amount
        return self.balance

    def deposit(self, amount):
        '''Return the balance remaining after depositing *amount* dollars.
        '''
        self.balance += amount
        return self.balance
    
c1 = Customer
c1.name = 'jose'
c1
c1.name 
c1.balance
c1.balance = 100
c1.balance
c1.dicksize = 53
c1.dicksize
c1.withdraw(10)
withdraw(c1,10)
c1.withdraw(c1, 10)
Customer.withdraw(c1, 10)
c2 = Customer('maria', 30)
c2.name
c2.balance
c[] = Customer.withdraw
c = []
c.append(Customer('a', 10)

)
c.append(Customer('b', 20)
)
c[0]
c[0].name
c.append(Customer('c'))
c[2].name
c[2].balance
11%2
11//2
12//2
13//2
1//5
2//5
1%5
2%5
12%5
j = []
j.append(3)
j
j[1].append(3)
j = []
j.append([])
j
j[1] =3
j.append(3)
j
j[0]
j[0].append(5)
j
j[0].append(6)
j
a = []
a.append(Customer('a', 10))
a.append(Customer('b', 20))
a.append(Customer('c', 30))
a.error
a[0].name
a.name
a[:].name
a[1].name
a[:].name
a[0:-1].name
g = [1,2,3]
g.find(1)
g.index(1)
runfile('C:/Users/victzuan/.spyder-py3/autotrainer2_California.py', wdir='C:/Users/victzuan/.spyder-py3')
min(error)
error.min
help(min)
min(error)[-1]
runfile('C:/Users/victzuan/.spyder-py3/autotrainer2_California.py', wdir='C:/Users/victzuan/.spyder-py3')
error.index(min(error))
min(error)
error
runfile('C:/Users/victzuan/.spyder-py3/autotrainer2_California.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Mon Apr 16 16:47:16 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
help (dot)
a = [[1,2],[3,4]]
b= [[1],[2]]
a.dot(b)
import numpy
a.dot(b)
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
# Create a new graph
Graph().as_default()
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
# Create a new graph

Graph().as_default()
# Create variables
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

A = Variable()
A.append([[1,0],[0,-1]])
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
# Create a new graph

Graph().as_default()
# Create variables
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

# Create placeholder
x = placeholder()
# Create hidden node y
y = matmul(A, x)
A
print(A)
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
# Create a new graph

Graph().as_default()
# Create variables
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

# Create placeholder
x = placeholder()
# Create hidden node y
y = matmul(A, x)
y
print(y)
# Create output node z
z = add(y, b)
z
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
session = Session()
output = session.run(z, {x: [1,2]})
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
session = Session()
output = session.run(z, {x: [1,2]})
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
Graph().as_default()
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])
x = placeholder()
y = matmul(A, x)
z = add(y, b)
session = Session()
output = session.run(z, {x: [1,2]})
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
debugfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
debugfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Wed Apr 18 17:33:18 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')

## ---(Thu Apr 19 16:23:14 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/open web browser.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/rename_files.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import os
file_name = "48saopaulo.jpg"
file_name.translate(None, "0123456789")
file_name.translate( "0123456789")
file_name.translate(None, "0123456789")
help(file_name.translate())
help(file_name.translate(""))
import string
file_name.translate(None, "0123456789")
from string import maketrans
intab = "aeiou"
outtab = "12345"
trantab = string.maketrans(intab, outtab)
intab.translate(str.maketrans('a', 'b'))
file_name 
file_name.translate(str.(maketrans(None, '0123456789')))
file_name.translate(str.maketrans(None, '0123456789'))
file_name.translate(str.maketrans('0123456789', None))
file_name.translate(str.maketrans('0123456789', ""))
file_name.translate("", "0123456789")
file_name.translate("""""""""""""""""""", "0123456789")
file_name.translate(str.maketrans('0123456789', """"""""""""""""""""))
file_name.translate(str.maketrans('0''1''2''3''4''5''6''7''8''9', """"""""""""""""""""))
file_name.translate(str.maketrans('0''1''2''3''4''5''6''7''8''9', ''''''''''''''''''''))
file_name
remove_digits = str.maketrans('','', '0123456789')
new_name = file_name.translate(remove_digits))
new_name = file_name.translate(remove_digits)
os.getcwd()
new_name
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/rename_files.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
debugfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/rename_files.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/rename_files.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
folder_path = input('please paste the folder adress:\n')
file_list = os.listdir(folder_path)
len(file_list)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/alphabet_files.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')

## ---(Fri Apr 20 10:25:40 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/draw_squares.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')

## ---(Fri Apr 20 10:45:31 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/draw_squares.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
clear
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/draw_squares.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/make_drawings.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
draw_pointy_flower()
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/make_drawings.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/ProgramData/Anaconda3/lib/turtle.py', wdir='C:/ProgramData/Anaconda3/lib')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/make_drawings.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/send_text.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/find_curse_words.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
x= [lambda a: a in range(10)]
x
x[1]
x= [lambda a: for i in range(10) a= i]
x= [lambda a: for i in range(10): a= i]
x= [lambda a: for a in range(10)]
x
x[0]
a
print (x)
print(x[0])
len(x)
x = []
for i in range(10):
    x.append(i)
    
x
x = []
for i in range(1,11):
    x.append(i)
    
x
y = [lambda a: x**2]
y
y = map([lambda a: x**2])
y = map(lambda a: a**2, x)
y
print(y)
nums = range(2,50)
nums
nums[0]
nums[1]
for in in range(2,8):
for i in range(2,8):
    nums= filter(lambda a: x == i or a%i, nums)
    
print (nums)
nums[0]
def f(x) : return x**2
print(f(9))
g = lambda x: x**2
g(9)
def make_incrementor(n): return lambda x: x+n
f = make_incrementor(2)
f
f(2)
g = make_incrementor(6)
g(2)
foo = [1,2,3,4,5,6,7,8,9]
print (filter(lambda x: x%3 == 0))
print (filter(lambda x: x%3 == 0, foo))
soma = lambda x,y : x+y
soma(3,4)
def fah(T):
    return ((float(9)/5)*T+32)

def cel(T):
    return (float(5)/9)*(T-32)

temp = (36.5, 37, 37.5, 38, 39)
F = map(fah, temp)
F
C = map(cel, F)
temp_in_fah = list(map(fah, temp))
temp_in_cel = list(map(cel, temp))
temp_in_fah
temp_in_cel
temp_in_cel = list(map(cel, temp_in_fah))
temp_in_cel
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
lambda_sum(3,4)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
lambda_sum(3,4)
x= [1,2,3]
y= [4,5,6]
lambda_sum(x,y)
y= 4,5,6
x= 1,2,3
lambda_sum(x,y)
x = 'come meu cu'
y = ' bem gostosinho'
lambda_sum(x,y)
x= 1,2,3
lambda_sum(x,y)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
T = [30, 35, 40, 32, 25, -10]
lambda_map.no_lambda(T)
lambda_map().no_lambda(T)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
T
lambda_map("no_lambda", T)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
lambda_map("no_lambda", T)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
lambda_map("no_lambda", T)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
lambda_map("no_lambda", T)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
lambda_map("no_lambda", T)
lambda_map("whith_lambda", T)
lambda_map("with_lambda", T)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
iterate_function(1)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
iterate_function(1)
iterate_function(2)
iterate_function(2, value= [pi, -pi, 0])
x = range(10)
x
x = list(range(10))
x
for i in range(10)
for i in range(1,10):
    x[0] = 0
    x = x[i] + x[i-1]
    
x = []
x.append(0)
x.append(1)
for i in range(2,10):
    x.append(x[i-1] + x[i-2])
    
x
x = [0,1]
lambda x: x.append(x[i-1]+x[i-2]) for i in range(2,10)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
create_fibonacci_list(2)
create_fibonacci_list(10)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
create_fibonacci_list(10)
a_list = create_fibonacci_list(11)
a_list
odd_numbers = list(filter(lambda x: x%2, a_list))
odd_numbers
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
sum_reduce()
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
sum_reduce()
create_fibonacci_list(5)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
sum_reduce()
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
max_reduce()7
max_reduce()
x = [0,1]
t = lambda a: a= x.append(x[i-1]+x[i-2]) for i in range(2,7)
t = lambda a,b: a+b if a>b
t = lambda(a,b: a+b if (a>b))
t = lambda(a,b: a+b)
t = lambda(a,b : a + b)
t = lambda a,b : a + b
t = (lambda a,b : a + b)
t = (lambda a,b : a + b if a> b)
t = (lambda a,b : a + b if (a> b))
t = (lambda a,b : a + b (if a> b))
t = (lambda a,b : a + b if a > b)
t = (lambda a,b : a + b if a >= b)
t = (lambda a,b: a + b if a >= b)
t = (lambda a,b: a + b if a >= b else a)
t(1,2)
t = (lambda x: list(range(0,x)))
t(2)
import functools
u = list(functools.reduce(lambda a, b: a+b, t(2)))
t(10)
u = lambda x: x[i] = x[i-2] +x[i-1] for i in range(2, len(x))
def fibo(a):
    for i in range(2, len(a)):
        a[i] = a[i-1] + a[i-2]
        
a = t(10)
fibo(a)
a
fib = lambda n, x=0, y=1 : x if not n else fib(n-1, y, x+y)
fib(3)
u = list(map(lambda (lambda n, x=0, y=1 : x if not n else fib(n-1, y, x+y)), a))
u = list(map(lambda n, x=0, y=1 : x if not n else fib(n-1, y, x+y), a))
u
t(10)
u = list(map(lambda n, x=0, y=1 : x if not n else fib(n-1, y, x+y), t(10)))



u
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
fibonacci_lambda(10)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
fibonacci_lambda(10)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/find_curse_words.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import profanity
help(profanity)
dir(profanity)
help(profanity.__builtins__)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/find_curse_words.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')

## ---(Mon Apr 23 18:20:56 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/read_website.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
debugfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/read_website.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
read_website("")
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/read_website.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/entretainment_center.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import turtle
turtle.Turtle.__doc__
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/entretainment_center.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import media_OOP as m
m.Movie.__dict__
m.Movie.__str__
m.Movie.__name__
m.Movie.__module__
from media_OOP import Movie as M
M.__module__
M.title = 'rabiola'
M.storyline = 'pra cima e pra baixo'
M.rate = M.
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/entretainment_center.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/distance_to_origin_inheritance_example.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import distance_to_origin_inheritance_example as dist
dist.Point(3,4)
dist.Point(x=3,y=4)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/distance_to_origin_inheritance_example.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import distance_to_origin_inheritance_example as dist
dist.Point(x=3,y=4)
dist.Cartesian(3,4)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/distance_to_origin_inheritance_example.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import distance_to_origin_inheritance_example as dist
dist.Point(x=3,y=4)
dist.Cartesian(3,4)
print(dist.Cartesian(3,4))
z = dist.Cartesian(3,4)
z
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/distance_to_origin_inheritance_example.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import distance_to_origin_inheritance_example as dist
z = dist.Cartesian(3,4)
z
import distance_to_origin_inheritance_example as dist
z = dist.Cartesian(3,4)
z
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/distance_to_origin_inheritance_example.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import distance_to_origin_inheritance_example as dist
z = dist.Cartesian(3,4)
z
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/inheritance.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
victor_zuanazzi = Parent("Zuanazzi", "brown")
print(victor_zuanazzi)
print(victor_zuanazzi.last_name)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/inheritance.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/distance_to_origin_inheritance_example.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
import math
math.abs(-3)
math.fabs(-3)
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/distance_to_origin_inheritance_example.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
runfile('C:/Users/victzuan/.spyder-py3/Computational graphs.py', wdir='C:/Users/victzuan/.spyder-py3')
import numpy as np
np.ones((50,2))
np.random.randn(50,2)
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')

## ---(Thu Apr 26 18:51:24 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
import numpy as np
red_points = np.random.randn(50,2) + [2,-2]*np.ones((50,2))
red_points
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
red_data
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
red_data
red_data.color
red_data.center
red_data.nop
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
red_data.points
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
red_data.points
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
red_data.points
p = np.random.randn(2,2)
p
p + [1,1]
p + [[-1,-1],[0,0]]
p
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
red_data.points
blue_data.points
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
division_line.color
division_line.b
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron_motivation_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
sigmoid_plot()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example_2()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example_2()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')

## ---(Fri Apr 27 15:28:47 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example_2()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
certanty
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example_3([0,0])
example_3([0,1])
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example_3([-1, 1])

## ---(Mon Apr 30 22:46:52 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/Perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/binary_data_generator.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi_class_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi_class_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi_class_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi_class_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi_class_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
red_data
red_data.points
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi_class_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi_class_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example5°
example()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi_class_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
example_2()
X[0]
X
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/training_criterion.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
output_probabilities

## ---(Fri May  4 16:03:51 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
W_value
b_value
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/computational_graphs.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')

## ---(Sat May  5 14:26:50 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training&computational_graphs.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training&computational_graphs.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training&computational_graphs.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training&computational_graphs.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training&computational_graphs.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/perceptron_training.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/training_criterion.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
center = [[-2, -2],[2, 2]]
for i in center:
    print (i)
    np.random.randn(2)
    
for i in center:
    print (i)
    print (np.random.randn(2))
    
for i in center:
    print (i)
    print (np.random.randn(2)+i)
    
for i in center:
    print (i)
    print (np.random.randn(10, 2)+i)
    
data = []
for i in center:
    print (i)
    data.append(np.random.randn(10, 2)+i)
    print (data)
    
data
data = []
for j in range(10):
    for i in center:
        print (i)
        data.append(np.random.randn(2)+i)
        print (data)
        
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
blue_data
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
clear
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
generate_data()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
generate_data()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
generate_data()
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')

## ---(Mon May  7 17:18:46 2018)---
d = array(2)
d = list(2)
d = []*2
d
d = []**2
for i in range(10):
    print(i)
    
len(range(10))
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptrion_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptrion_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
debugfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptrion_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptrion_2.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
ml = np.random.randn(4,2)
ml
ml = np.random.randn(4)
ml
X = []
X = [lambda x: i for i in range(10)]
X
X[0]
X[0][0]
X[0]
X = list(map(lambda x: x, range(10)))
X
X = list(map(lambda x: cg.placeholder, range(10)))
X
y = lambda x: x**2
y(2)
X = list(map(lambda x: 2 if x == 0 else y(X[x-1]) for x in range(10)))
X = list(map(lambda x: 2 if x == 0 else y(X[x-1]), for x in range(10)))
X = list(lambda x: x, for x in range(10))
X = list(map(lambda x: cg.placeholder, range(10)))
X[0]
X[0](2)
X(2)
X = list(map(lambda x: x**2, range(10)))
X[0]
X[2]
X = list(map(lambda x: 0 if x == 0 else x-1, range(10)))
X
X = list(map(lambda x: 0 if x == 0 else X[x-1], range(10)))
X
clear
X = list(map(lambda x: 0 if x == 0 else X[x-1], range(10)))
X
X = []
X = list(map(lambda x: 0 if x == 0 else X[x-1], range(10)))
X = [0]
X = list(map(lambda x: 0 if x == 0 else X[x-1], range(10)))
p = []
for i in range(10):
    p.append(i)
    
p
for i in range(10):
    p.append([i, i+1, i+2])
    
p
p = []
for i in range(10):
    
    p.append([i, i+1, i+2])

p
p = [1,2,3]
for i in range(10):
    p.append([i, i+1, i+2])
    
p
p[0]
p = [[1,2,3]]
for i in range(10):
    p.append([i, i+1, i+2])
    
p
p[0]
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron_3.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')

## ---(Tue May  8 16:53:30 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron_3.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
u = cg.Variable(np.random.randn(10,2))
u
u[0]
u = list(map(lambda: x= cg.Variable(np.random.randn(2,2)), range(10)))
u = list(map(lambda x: cg.Variable(np.random.randn(2,2)), range(10)))
u[0]
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron_3.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
session.run(red_blue.w)
session.run(red_blue.w[0])
red_blue.w[0]
session.run(p[0])
session.run(red_blue.p[0])
session.run(red_blue.p)
session.run(cg.log(red_blue.p[0]))
cg.log(red_blue.p[0])
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron_3.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
blue_data
b = [[1,1],[2,2],[3,3]]
r = [[4,4]]
b+r
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron_3.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
data = list(map(lambda x: [x, x**2], range(10)))
data
data[1]+data[2]
list(data[1]) + list(data[2])
np.concatenate((data[1],data[2]))
w= []
w.append(data[1])
w
w.append(data[2])
x
w
w[0]
w[0][0]
w= [[]]
w.append(data[1])
w
W = []
w = []
w.append(data[1])
w.append(data[2])
w
W.append(w)
W
w= []
w.append(data[3])
w.append(data[4])
w
W.append(w)
W
W[0]
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron_3.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
np.random.randn()
np.random.randn()**2
np.random.randn()**10
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/multi-layer_perceptron_3.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')

## ---(Fri May 18 06:41:40 2018)---
runfile('C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch/two_layer_perceptron_tf.py', wdir='C:/Users/victzuan/.spyder-py3/Deep Learning from Scratch')
runfile('C:/Users/victzuan/.spyder-py3/autotrainer2_California.py', wdir='C:/Users/victzuan/.spyder-py3')
runfile('C:/Users/victzuan/.spyder-py3/multi-parameter_california_housing_model.py', wdir='C:/Users/victzuan/.spyder-py3')
training_examples.describe()
training_targets.describe()

## ---(Fri Jun 22 17:45:50 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/ccxt trial 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
'ethbtc' in hitbtc_markets
'ETH/BTC' in hitbtc_markets
hitbtc_markets['ETH/BTC']
hitbtc_markets.index('ETH/BTC')
help (dict)))
help (dict)
hitbtc_markets.index('ETH/BTC')
a = {'a': 0; 'b': 1; 'c': 2}
a = {'a': 0, 'b': 1, 'c': 2}
a.index('a')
a.get('a'))
a.get('a')
a = {'a': 10, 'b': 11, 'c': 12}
a.get('a')
print(hitbtc.fetch_order_book(hitbtc('ETH/BTC'))
)
clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/ccxt trial 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
print(hitbtc.fetch_order_book(hitbtc.symbols[0]))
print(hitbtc.fetch_order_book(hitbtc.symbols[80]))

## ---(Sat Jun 23 09:25:13 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kraken ccxt trial 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kraken ccxt candle chart.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
print(exchange.fetch_ohlcv("BTC/USD", '1d')[-1])
runfile('C:/Users/victzuan/.spyder-py3/ccxt/initialize exchange.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
okcoin_example()
runfile('C:/Users/victzuan/.spyder-py3/ccxt/initialize exchange.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
okcoin_example()
runfile('C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master/main.py', wdir='C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master')
import os
c_dir = os.path.dirname(__file__)
with open(os.path.join(c_dir, "config/secrets.txt")) as key_file:
        api_key, secret, telegram_tkn, user_id = key_file.read().splitlines()
        
runfile('C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master/main.py', wdir='C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master')
print(exchange)
print (ccxt_ex)
print(ccxt_ex)
runfile('C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master/main.py', wdir='C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master')
debugfile('C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master/main.py', wdir='C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master')
runfile('C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master/core/telegrambot.py', wdir='C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master/core')
runfile('C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master/main.py', wdir='C:/Users/victzuan/.spyder-py3/Telegram trading bot/Auto-Trade-Crypto-Bot-master')
from telegram.ext import Updater

## ---(Mon Jun 25 18:20:57 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kraken ccxt trial 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kraken ccxt candle chart.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
kraken = ccxt.kraken({
       'apiKey': 'JVfh8ajAL357p1CKput2btQRgy9crL2naMZf0zTWRtPRsUwcsi0nG/0r',
       'secret': 'ptmYjw0k0i6M9C7517TRqY+Gve+Ltvn6Q1yod6QteoRC5Y4tW0Cp42HPf/wyxgR3u2fxsuXmMOuLJxQU3KE2ag==',
   })
   print('--------------- MARKETS --------------')
   kraken_markets = kraken.load_markets()
kraken = ccxt.kraken({
        'apiKey': 'JVfh8ajAL357p1CKput2btQRgy9crL2naMZf0zTWRtPRsUwcsi0nG/0r',
        'secret': 'ptmYjw0k0i6M9C7517TRqY+Gve+Ltvn6Q1yod6QteoRC5Y4tW0Cp42HPf/wyxgR3u2fxsuXmMOuLJxQU3KE2ag==',
    })
    
kraken_markets = kraken.load_markets()
print(kraken.fetch_order_book(kraken.symbols[0]))
print(kraken.fetch_ticker('BTC/EUR'))
print(kraken.fetch_balance())
print(kraken.fetch_order_book(kraken.symbols[0]))
kraken.symbols
print(kraken.fetch_order_book('BCH/BTC'))
kraken.fetch_order_book('BCH/BTC')[0]
kraken.fetch_order_book('BCH/BTC'){'datetime'}
kraken.fetch_order_book('BCH/BTC')('datetime')
kraken.fetch_order_book('BCH/BTC')[0]
kraken.fetch_order_book('BCH/BTC')['datetime']
kraken.fetch_order_book('BCH/BTC')['ask']
kraken.fetch_order_book('BCH/BTC')['asks']
kraken.fetch_order_book('BCH/BTC')['asks'][0]
kraken.symbols
base = 'BTC', LiquidCurrency = "ETH", WeakCurrency = "ETC"
base = 'BTC'
LiquidCurrency = "ETH"
WeakCurrency = "ETC"
pair = []
pair.append(LiquidCurrency+'/'+base) #ETH/BTC in the example
pair.append(WeakCurrency+'/'+base) #ETC/BTC in the example
pair.append(WeakCurrency+'/'+LiquidCurrency) #ETC/ETH in the example

for i in range(3):print(kraken.fetch_order_book('BCH/BTC'))
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kraken ccxt trial 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Tue Jun 26 09:07:43 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kraken ccxt trial 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Tue Jun 26 15:51:06 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
main()
exchange.simbols
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange = ccxt.kraken()
exchange.symbols
pairs = exchange.symbols
pairs
exchange = ccxt.kraken()
exchange.symbols
print(exchange.symbols)
market = exchange.load_markets()
exchange.symbols
pairs = exchange.symbols
a = pairs[0]
base = a[0:2]
base = a[0:3]
pair = ['', '', '']
pair = ['', '5', '^^']
pair[0]
a[-3:]
pairs = exchange.symbols
    pair = ['', '', '']
    coin = ['', '', '']
    pairs = exchange.symbols
    pair = ['', '', '']
    coin = ['', '', '']

for i in pairs:
        pair[0] = i
        coin[0] = i[0:3]
        for j in pairs:
            if (i != j) and (coin[0] in j):
                pair[1] = j
                coin[1] = j[-3:]
                for k in pairs:
                    if (k != i) and (k != j):
                        if coin[0] in k and coin[1] in k:
                            pair[2]= k
                            
pairs = exchange.symbols
pair = ['', '', '']
coin = ['', '', '']
trair = []

for i in pairs:
        pair[0] = i
        coin[0] = i[0:3]
        for j in pairs:
            if (i != j) and (coin[0] in j):
                pair[1] = j
                coin[1] = j[-3:]
                for k in pairs:
                    if (k != i) and (k != j):
                        if coin[0] in k and coin[1] in k:
                            pair[2]= k
                            trair.append(pair[0], pair[1], pair[2])
                            
for i in pairs:
        pair[0] = i
        coin[0] = i[0:3]
        for j in pairs:
            if (i != j) and (coin[0] in j):
                pair[1] = j
                coin[1] = j[-3:]
                for k in pairs:
                    if (k != i) and (k != j):
                        if coin[0] in k and coin[1] in k:
                            pair[2]= k
                            trair.append([pair[0], pair[1], pair[2]])
                            
debugfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange = ccxt.kraken()
market = exchange.load_markets()
find_trairs(exchange)
h = 'batata'
bat in h
'bat' in h
debugfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
clear
debugfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
pairs
p = lambda(x: i if not '.d' in i for i in range(pair))
p= list(map(lambda x: x if not '.id' in x, pair))


p= list(map(lambda x: x if not '.id' in x , pair))
p= list(map(lambda x: x , pair))
p= list(map(lambda x: x , pairs))
p= list(map(lambda x: x if '.d' in x , pairs))
p= list(map(lambda x: '.d' in x , pairs))
p
p = filter(lambda x: 'id' in x, pairs)
p
p = list(filter(lambda x: 'id' in x, pairs))
p
p = list(filter(lambda x: '.d' in x, pairs))
p
p = list(filter(lambda x: not '.d' in x, pairs))
p
runfile('C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon/lambda_examples.py', wdir='C:/Users/victzuan/.spyder-py3/Programing foundations with phtyon')
filter_elements(pairs, '.d', contain = False)
a = ['e','r','t']
len(a)
a[3]
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
kraken = ccxt.kraken({
    'apiKey': 'JVfh8ajAL357p1CKput2btQRgy9crL2naMZf0zTWRtPRsUwcsi0nG/0r',
    'secret': 'ptmYjw0k0i6M9C7517TRqY+Gve+Ltvn6Q1yod6QteoRC5Y4tW0Cp42HPf/wyxgR3u2fxsuXmMOuLJxQU3KE2ag==',
})
kraken_markets = kraken.load_markets()
print(kraken.fetch_ticker('BTC/EUR'))
print(kraken.fetch_order_book(kraken.symbols[0]))
print(kraken.fetch_order_book('BTC/EUR'))
print(kraken.fetch_order_book('BTC/EUR'))['ask']
print(kraken.fetch_order_book('BTC/EUR')['ask'])
print(kraken.fetch_order_book('BTC/EUR')['asks'])
print(kraken.fetch_order_book('BTC/EUR')['asks'][0])
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange = ccxt.kraken()
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
pairs[0].find('/')
trair[0].find('/')
trair
exchange = ccxt.kraken()
market = exchange.load_markets()
exchange_pairs = exchange.symbols #loads all pairs from the exchange.
    pairs = list(filter(lambda x: not '.d' in x, exchange_pairs)) #filters off the darkpool pairs.
    pair = ['', '', '']
    coin = ['', '', '']
    trair = []
    
    #Semi-optimized loop to look for triplece of pairs.
    #Example:['BCH/BTC', 'BCH/EUR', 'BTC/EUR']
    for i in range(len(pairs)-3):
        pair[0] = pairs[i]
        coin[0] = pairs[i][0:3]
        coin[1] = pairs[i][-3:]
        for j in range(i+1, len(pairs)-2):
            if (coin[0] in pairs[j]):
                pair[1] = pairs[j]
                coin[2] = pairs[j][-3:]
                for k in range(j+1, len(pairs)-1):
                    if coin[1] in pairs[k] and coin[2] in pairs[k]:
                        pair[2]= pairs[k]
                        trair.append([pair[0], pair[1], pair[2]])
                        break
    exchange_pairs = exchange.symbols #loads all pairs from the exchange.
    pairs = list(filter(lambda x: not '.d' in x, exchange_pairs)) #filters off the darkpool pairs.
    pair = ['', '', '']
    coin = ['', '', '']
    trair = []
    
    #Semi-optimized loop to look for triplece of pairs.
    #Example:['BCH/BTC', 'BCH/EUR', 'BTC/EUR']
    for i in range(len(pairs)-3):
        pair[0] = pairs[i]
        coin[0] = pairs[i][0:3]
        coin[1] = pairs[i][-3:]
        for j in range(i+1, len(pairs)-2):
            if (coin[0] in pairs[j]):
                pair[1] = pairs[j]
                coin[2] = pairs[j][-3:]
                for k in range(j+1, len(pairs)-1):
                    if coin[1] in pairs[k] and coin[2] in pairs[k]:
                        pair[2]= pairs[k]
                        trair.append([pair[0], pair[1], pair[2]])
                        break
                
trair
trair[0]
trair[0][0]
trair[0][0].find('/')
len(trair)
trair[9][0].find('/')
trair[0][0][3]
trair[0][0][4:]
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
pairs[0]
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange_pairs = exchange.symbols #loads all pairs from the exchange.
    pairs = list(filter(lambda x: not '.d' in x, exchange_pairs)) #filters off the darkpool pairs.
    pair = ['', '', '']
    coin = ['', '', '']
    trair = []
    exchange_pairs = exchange.symbols #loads all pairs from the exchange.
    pairs = list(filter(lambda x: not '.d' in x, exchange_pairs)) #filters off the darkpool pairs.
    pair = ['', '', '']
    coin = ['', '', '']
    trair = []

for i in range(len(pairs)-3):
    pair[0] = pairs[i]
    #not all coins are 3 digits long, we must find the slash that separetes
    #each coin in order to have a robust algorithm.
    slash_position = pairs[i].find('/') 
    coin[0] = pairs[i][0:slash_position]
    coin[1] = pairs[i][slash_position+1:]
    for j in range(i+1, len(pairs)-2):
        if (coin[0] in pairs[j]):
            pair[1] = pairs[j]
            slash_position = pairs[j].find('/') 
            coin[2] = pairs[j][slash_position+1:]
            for k in range(j+1, len(pairs)-1):
                if coin[1] in pairs[k] and coin[2] in pairs[k]:
                    pair[2]= pairs[k]
                    trair.append([pair[0], pair[1], pair[2]])
                    break
                
trair
len(trair)
for i in range(len(pairs)-3):
    #not all coins are 3 digits long, we must find the slash that separetes
    #each coin in order to have a robust algorithm.
    slash_position = pairs[i].find('/') 
    coin[0] = pairs[i][0:slash_position]
    coin[1] = pairs[i][slash_position+1:]
    for j in range(i+1, len(pairs)-2):
        if (coin[0] in pairs[j]):
            slash_position = pairs[j].find('/') 
            coin[2] = pairs[j][slash_position+1:]
            for k in range(j+1, len(pairs)-1):
                if coin[1] in pairs[k] and coin[2] in pairs[k]:
                    trair.append([pairs[i], pairs[j], pairs[k]])
                    break
                
trair
trair = []
for i in range(len(pairs)-3):
    #not all coins are 3 digits long, we must find the slash that separetes
    #each coin in order to have a robust algorithm.
    slash_position = pairs[i].find('/') 
    coin[0] = pairs[i][0:slash_position]
    coin[1] = pairs[i][slash_position+1:]
    for j in range(i+1, len(pairs)-2):
        if (coin[0] in pairs[j]):
            slash_position = pairs[j].find('/') 
            coin[2] = pairs[j][slash_position+1:]
            for k in range(j+1, len(pairs)-1):
                if coin[1] in pairs[k] and coin[2] in pairs[k]:
                    trair.append([pairs[i], pairs[j], pairs[k]])
                    break
                
trair
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
pair = trair[0]
pair
'BCH' in pair
trairs
trair
interesting_coins = ['BTC', 'EUR', 'ETH']
for pairs in trair:
        coin_of_interest = ''
        i = 0
        while (coin_of_interest == '') and (i <3):
            for j in pairs:
                if interesting_coins[i] in j:
                    coin_of_interest = interesting_coins[i]
            i = i+1
            
coin_of_interest = []
interesting_coins = ['BTC', 'EUR', 'ETH']
for pairs in trair:
    coin_of_interest.append('')
    i = 0
    while (coin_of_interest == '') and (i <3):
        for j in pairs:
            if interesting_coins[i] in j:
                coin_of_interest = interesting_coins[i]
        i = i+1
        
coin_of_interest = []
interesting_coins = ['BTC', 'EUR', 'ETH']
for pairs in range(len(trair)):
    coin_of_interest.append('')
    i = 0
    while (coin_of_interest[pairs] == '') and (i <3):
        for j in pairs:
            if interesting_coins[i] in j:
                coin_of_interest[pairs] = interesting_coins[i]
        i = i+1
        
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
# 1- Load exchange.
    exchange = ccxt.kraken()
    market = exchange.load_markets()
    # 1- Load exchange.
    exchange = ccxt.kraken()
    market = exchange.load_markets()

import ccxt
    # 1- Load exchange.
    exchange = ccxt.kraken()
    market = exchange.load_markets()

exchange_pairs = exchange.symbols #loads all pairs from the exchange.
    pairs = list(filter(lambda x: not '.d' in x, exchange_pairs)) #filters off the darkpool pairs.
    pair = ['', '', '']
    coin = ['', '', '']
    trair = []
    
    #Semi-optimized loop to look for triplece of pairs.
    #Example:['BCH/BTC', 'BCH/EUR', 'BTC/EUR']
    for i in range(len(pairs)-3):
        #not all coins are 3 digits long, we must find the slash that separetes
        #each coin in order to have a robust algorithm.
        slash_position = pairs[i].find('/') 
        coin[0] = pairs[i][0:slash_position]
        coin[1] = pairs[i][slash_position+1:]
        for j in range(i+1, len(pairs)-2):
            if (coin[0] in pairs[j]):
                slash_position = pairs[j].find('/') 
                coin[2] = pairs[j][slash_position+1:]
                for k in range(j+1, len(pairs)-1):
                    if coin[1] in pairs[k] and coin[2] in pairs[k]:
                        trair.append([pairs[i], pairs[j], pairs[k]])
                        break
    exchange_pairs = exchange.symbols #loads all pairs from the exchange.
    pairs = list(filter(lambda x: not '.d' in x, exchange_pairs)) #filters off the darkpool pairs.
    pair = ['', '', '']
    coin = ['', '', '']
    trair = []
    
    #Semi-optimized loop to look for triplece of pairs.
    #Example:['BCH/BTC', 'BCH/EUR', 'BTC/EUR']
    for i in range(len(pairs)-3):
        #not all coins are 3 digits long, we must find the slash that separetes
        #each coin in order to have a robust algorithm.
        slash_position = pairs[i].find('/') 
        coin[0] = pairs[i][0:slash_position]
        coin[1] = pairs[i][slash_position+1:]
        for j in range(i+1, len(pairs)-2):
            if (coin[0] in pairs[j]):
                slash_position = pairs[j].find('/') 
                coin[2] = pairs[j][slash_position+1:]
                for k in range(j+1, len(pairs)-1):
                    if coin[1] in pairs[k] and coin[2] in pairs[k]:
                        trair.append([pairs[i], pairs[j], pairs[k]])
                        break
                
trair
coin_of_interest = []
interesting_coins = ['BTC', 'EUR', 'ETH']
for pairs in range(len(trair)):
    coin_of_interest.append('')
    i = 0
    while (coin_of_interest[pairs] == '') and (i <3):
        for j in pairs:
            if interesting_coins[i] in j:
                coin_of_interest[pairs] = interesting_coins[i]
        i = i+1
        
coin_of_interest = []
interesting_coins = ['BTC', 'EUR', 'ETH']
for p in range(len(trair)):
    coin_of_interest.append('')
    i = 0
    while (coin_of_interest[p] == '') and (i <3):
        for j in pairs:
            if interesting_coins[i] in j:
                coin_of_interest[p] = interesting_coins[i]
        i = i+1
        
coin_of_interest = []
interesting_coins = ['BTC', 'EUR', 'ETH']
for p in range(len(trair)):
    coin_of_interest.append('')
    i = 0
    while (coin_of_interest[p] == '') and (i <3):
        for j in trair[p]:
            if interesting_coins[i] in j:
                coin_of_interest[p] = interesting_coins[i]
        i = i+1
        
trair
coin_of_interest
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Wed Jul  4 17:08:20 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
import intertools
import itertools
currency_pairs = ['BCH/BTC', 'BCH/ETH', 'DASH/USD', 'BTC/USDT', 'ETH/BTC']
set_triplets = set()
for triplet in itertools.permutations(currency_pairs, 3):
    c1, c2 = triplet[0].split('/')
    if (c1 in triplet[1] or c1 in triplet[2]) and (c2 in triplet[1] or c2 in triplet[2]):
        set_triplets.add(tuple(sorted(triplet)))
for triplet in set_triplets:
    print(triplet)
    
exchange = ccxt.kraken()
exchange_pairs = exchange.symbols #loads all pairs from the exchange.
    pairs = list(filter(lambda x: not '.d' in x, exchange_pairs))
    exchange_pairs = exchange.symbols #loads all pairs from the exchange.
    pairs = list(filter(lambda x: not '.d' in x, exchange_pairs))

exchange_pairs = exchange.symbols #loads all pairs from the exchange.
pairs = list(filter(lambda x: not '.d' in x, exchange_pairs))
exchange = ccxt.kraken()
exchange.symbols
exchange_pairs = exchange.symbols
market = exchange.load_markets()
exchange_pairs = exchange.symbols
currency_pairs = exchange_pairs
set_triplets = set()
for triplet in itertools.permutations(currency_pairs, 3):
    c1, c2 = triplet[0].split('/')
    if (c1 in triplet[1] or c1 in triplet[2]) and (c2 in triplet[1] or c2 in triplet[2]):
        set_triplets.add(tuple(sorted(triplet)))
for triplet in set_triplets:
    print(triplet)
    
currency_pairs= list(filter(lambda x: not '.d' in x, currency_pairs))
set_triplets = set()
for triplet in itertools.permutations(currency_pairs, 3):
    c1, c2 = triplet[0].split('/')
    if (c1 in triplet[1] or c1 in triplet[2]) and (c2 in triplet[1] or c2 in triplet[2]):
        set_triplets.add(tuple(sorted(triplet)))
for triplet in set_triplets:
    print(triplet)
    
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 3.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange = ccxt.kraken()market = exchange.load_markets()
exchange = ccxt.kraken() market = exchange.load_markets()
exchange = ccxt.kraken()
market = exchange.load_markets()

trair = find_trairs(exchange)
interesting_coins = ['BTC', 'EUR', 'ETH']
coins_of_interest = define_coins_of_interest(exchange, trair, interesting_coins)
trair = sort_pairs(trair, coins_of_interest)
trair[0][0]
profit_coin = 'BTC'
profit_coin_position_start = trair[0][0].find(profit_coin)
profit_coin = 'BCH'
profit_coin_position_start = trair[0][0].find(profit_coin)
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 3.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
for i in range(len(trair[0])):
        ask[i] = exchange.fetch_order_book(pairs[i])['asks'][0] #price to buy
        bid[i] = exchange.fetch_order_book(pairs[i])['bids'][0] #price to sell
        
for i in range(len(trair[0])):
        ask[i] = exchange.fetch_order_book(trair[0][i])['asks'][0] #price to buy
        bid[i] = exchange.fetch_order_book(trair[0][i])['bids'][0] #price to sell
        
ask = ['','','']
bid = ['','','']

for i in range(len(trair[0])):
    
        ask[i] = exchange.fetch_order_book(trair[0][i])['asks'][0] #price to buy
        bid[i] = exchange.fetch_order_book(trair[0][i])['bids'][0] #price to sell

runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 3.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
l = [0,0]
l[0]
l = [0,1]
l[0]
l[1]
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 3.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 4.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
t = [[,],[,]]
ask[i] = exchange.fetch_order_book(pairs[i])['asks'][0] #price to buy
bid[i] = exchange.fetch_order_book(pairs[i])['bids'][0] #price to sell

exchange = ccxt.kraken()
market = exchange.load_markets()

# 2- Find possible tri-pairs (trairs) within the exchange.
trair = find_trairs(exchange)

interesting_coins = ['BTC', 'EUR', 'ETH']
coins_of_interest = define_coins_of_interest(exchange, trair, interesting_coins)

trair = sort_pairs(trair, coins_of_interest)
pairs= trair[0]
for i in range(len(pairs)):
    ask[i] = exchange.fetch_order_book(pairs[i])['asks'][0] #price to buy
    bid[i] = exchange.fetch_order_book(pairs[i])['bids'][0] #price to sell
    
pairs
bid[0][0]*bid[0][1]
ask[0][1]*ask[0][0]

## ---(Thu Jul  5 17:44:20 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 4.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
trair=[['ETC/ETH', 'ETC/USD', 'ETH/USD'],['EOS/BTC', 'EOS/ETH', 'ETH/BTC'],['EOS/EUR', 'EOS/ETH', 'ETH/EUR']]
d_trair={'ETH': trair[0], 'BTC': trair[1], 'EUR': trair[2]}
d_trair
trair
main_coin = ['BTC','ETH','EUR']
dic_trair = {main_coin: trair}
dic_trair = dict()
dic_trair
dic_track[main_coin] = trair
dic_trair[main_coin] = trair
dic_trair = dic(main_coin, trair)
dic_trair = dict(main_coin, trair)
dic_trair = dict(main_coin = trair)
dic_trair
main_coin
{x: x**2 for x in (2, 4, 6)}
{x: trair for x in main_coin}
{x: trair[y] for x in main_coin for y in range(len(main_coi))}
{x: trair[y] for x in main_coin for y in range(len(main_coin))}
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
import time
import datetime
datetime.date
datetime.time
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
file = open('gfd.txt', w)
file = open('gfd.txt', 'w')
file.write('### 123 456 789 ###')
file.close()
file
file = open('gfd.txt', 'w')
file
file.close()
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
done = False 
while not done
    try:
        while True:
            exchange = ccxt.kraken()
            market = exchange.load_markets()
            done = True
    except RequestTimeout:
        pass
done = False 
while not done:
    try:
        while True:
            exchange = ccxt.kraken()
            market = exchange.load_markets()
            done = True
    except RequestTimeout:
        pass
    
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
done = False 
while not done:
    try:
        while True:
            exchange = ccxt.kraken()
            market = exchange.load_markets()
            done = True
    except:
        pass
    

def do_it:
    try:
        while True:
            exchange = ccxt.kraken()
            market = exchange.load_markets()
    except:
        do_it()


def do_it():
    try:
        while True:
            exchange = ccxt.kraken()
            market = exchange.load_markets()
    except:
        do_it()
        
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
import time
import datetime
datetime.datetime.now().timestamp()
datetime.datetime.now()
ts = datetime.datetime.now()
ts
datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
ts = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
ts
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
if yyy:
    print('yy')
else:
    print ('xxx')
    
x = []
if x[-1]
if x[-1]:
    print('x')
else:
    print('no')
    
if x:
    print('x')
else:
    print('no')
    
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 7.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance 7.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-1.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
c= [['R/ETH', 'TRAC/ETH', 'UKG/ETH'],['EOS/ETH', 'EOS/NEO', 'NEO/ETH'],['LA/ETH', 'LALA/ETH', 'LEND/ETH'],['LA/BTC', 'PLAY/BTC', 'POLL/BTC'],['ABT/BTC', 'ABT/ETH', 'EBTC/ETH'],['CS/BTC', 'CS/KCS', 'KCS/BTC']]
pairs = c
pair = ['', '', '']
    coin = ['', '', '']
    trair = []
pair = ['', '', '']
coin = ['', '', '']
trair = []

pair[0].find('/')
pairs[0].find('/')
pairs[0][0].find('/')
slash_position = pairs[0][0].find('/')
coin[0] = pairs[0][0][0:slash_position]
coin[1] = pairs[0][0][slash_position+1:]
slash_position = pairs[0][1].find('/')
coin[2] = pairs[0][1][slash_position+1:]
exchange = ccxt.kucoin()
market = exchange.load_markets()
exchange = ccxt.kucoin()
import ccxt
exchange = ccxt.kucoin()
market = exchange.load_markets()
exchange_pairs = exchange.symbols
pairs = list(filter(lambda x: not '.d' in x, exchange_pairs))
pairs
pair = ['', '', '']
coin = ['', '', '']
trair = []

runfile('C:/Users/victzuan/.spyder-py3/ccxt/kucoin sandbox 0.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
trairs[0]
trairs[0][0]
trairs[0][0][0]
rformat = []
for i in rtrairs:
    for j in i:
        rformat.append(j[0]+'/'+j[1])
        

        
rformat = []
for i in trairs:
    for j in i:
        rformat.append(j[0]+'/'+j[1])
        
rformat = []
t_trair = []
for i in range(len(trairs)):
    for j in range(len(trairs[i])):
        t_trair.append(trairs[i][j][0]+'/'+trairs[i][j][1])
    rformat.append(t_trair)

        
rformat = []
for i in range(len(trairs)):
    t_trair = []
    for j in range(len(trairs[i])):
        t_trair.append(trairs[i][j][0]+'/'+trairs[i][j][1])
    rformat.append(t_trair)
    
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kucoin sandbox 0.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
trairs[0]
trairs[0][0]
len(trairs[0])
trairs[0][0][0]
trairs[0][0][1]
t_trair = []
t_trair.append(trairs[0][0]+'/'+trairs[0][0][1])
t_trair.append(trairs[0][0][0]+'/'+trairs[0][0][1])
t_trair.append(trairs[0][1][0]+'/'+trais[0][1][1])
t_trair.append(trairs[0][1][0]+'/'+trairs[0][1][1])
t_trair.append(trairs[0][2][0]+'/'+trairs[0][2][1])
trair = []
trair.append(t_trair)
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kucoin sandbox 0.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kucoin sandbox 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/kucoin sandbox 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
debugfile('C:/Users/victzuan/.spyder-py3/ccxt/kucoin sandbox 2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange = ccxt.kucoin()
market = exchange.load_markets()
exchange_pairs = exchange.symbols

import ccxt
from collections import Counter
from itertools import combinations, chain

exchange = ccxt.kucoin()
market = exchange.load_markets()
exchange_pairs = exchange.symbols

raw_pairs = list(filter(lambda x: not '.d' in x, exchange_pairs))
pairs = map(parse_pair, raw_pairs)
trair_candidates = combinations(pairs, r=3)
#filter the actual trairs from all trair_candidates
u_trair = list(filter(is_trair,trair_candidates)) #unformated trairs.

def parse_pair(raw_pair):
    return raw_pair.split('/')

def is_trair(trair_candidate):
    # assuming that all currencies appear twice means we have a trair
    currency_counts = Counter(chain.from_iterable(trair_candidate))
    return set(currency_counts.values()) == {2}

pairs = map(parse_pair, raw_pairs)
trair_candidates = combinations(pairs, r=3)
#filter the actual trairs from all trair_candidates
u_trair = list(filter(is_trair,trair_candidates)) #unformated trairs.

runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
try
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-2.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
x = {'a': 2, 'b': 3
}
x
y= {'a': 2, 4, 'b': 3, 4}
y= {'a': [2, 4], 'b': [3, 4]}
y
y['a']
y[0]
y['a'][0]
t = [x: x in range(10)]
t = lambda(x: x in range(10))
t = lambda[x: x in range(10)]
t = lambda x: x in range(10)
t
t = [0:10]
t = [0,1,2,3,4,5,6,7,8,9]
t[:]
t = [t,t,t]
t
t[:][0]
t[:]
t[0][:]
t[:][1]
t[0] = t[0]+2
t[0][:] = t[0][:] +2
a = [0,1,2,3]
a = 2*[a]
a
a = [[0,1,2],[3,4,5],[6,7,8]]
a
a[:]
a[:][0]
a[:][1]
a[0]
a[0][:]
a.find(4)
a[0].find(0)
a.pop(1)
a[0].pop(1)
a[:].pop(1)
a
a.pop(1)
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-3.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
f = 'sdf/wxc'
f.split('/')
f.find('sdf')
r = 's/wxc'
r.find('s')
r = 'sdf/w'
r.fnd('sdf')
r.find('sdf')
if r.find('sdf'):
    print('=)')
    
r
r.find('w')
if r.find('w'):
    print('=)')
    
a = [[1,2],[3,4],[5,6]]
a[0][0:0]
a[0][0]
a[0][0:1]
u = {'s': 2, 'r': -1}
u
for f in u:
    print (f)
    
for f in u:
    print (u[f])
    
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-4.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange, market = load_exchange()
trair = find_trairs(exchange)
interesting_coins = ['BTC', 'EUR', 'ETH', 'BTC', 'USDT']
coins_of_interest = define_coins_of_interest(exchange, trair, interesting_coins)
trair = sort_pairs(trair, coins_of_interest)
trade_path = generate_sequence(trair, coins_of_interest)
oportunities = []
ask, bid = load_order_book(exchange, pairs)
pairs = trair[0]
ask, bid = load_order_book(exchange, pairs)
unbalance_factor = {
            'straight': calculate_factor(ask, bid, trade_path), 
            'inverse': calculate_factor(ask, bid, trade_path, mode = 'inverse')}
            
unbalance_factor = 1
trade = trade_mode('straight')
unbalance_factor = unbalance_factor/ask[0][0]
ask[0][0]
ask
ask = ['','','']
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-4.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
l = [1,2,3,4,5,6,7,8,9]
for i in range(len(l)):
    if l[i] == 3
for i in range(len(l)):
    print (l[i], '--', i)
    if l[i] == 3:
        i = i -1
        print ('in',l[i], '--', i)
        
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-6.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-7.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
v = [1,2,3]
min(v)
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-8.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
a = [1,2,3]
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-9.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-9.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
try:
    exchange = ccxt.kraken()
    market = exchange.load_markets()
except as e:
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-10.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-10.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-9.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Mon Jul  9 18:45:22 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-10.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-10.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-10.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-10.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
v = [1,2,3]
v.del(2)
v.pop(2)
v
v[0].pop
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-11.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-11.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-11.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
d = {}
d['piroca'] = 20
d['bolas'] = 2
d['peso'] = 0.5
d['densidade'] = d['piroca']/d['peso']
len(d)
sum(d)
d['bolas'] = 30
d['bolas'] = [2,1,2]
sum(d['bolas'])
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
v = {'a': [20,30], 'b':[1,2]}
v = {'a': 20, 'b':1}
b = {'a': 20, 'b':1}
v.append(b)
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
v = {'a':1, 'b':2}
b = {'a':11, 'b':22}
z=[]
z.append(v)
z.append(b)
z[0]
z[0][0]
z[:]['a']
z[0]['a']
z = {'a' = [1,2], 'b' = [3,4]}
z = {'a': [1,2], 'b': [3,4]}
x = {'a': 5, 'b': 6}
for y in z:
    print y
for y in z:
    print (y)
    
for y in z:
    z[y].append(x[y])
    
for y in x:
    z[y].append(x[y])
    
def sss(a, b):
    for y in a:
    b[y].append(a[y])
def sss(a, b):
    for y in a:
        b[y].append(a[y])
        
def mmm():
    f = {}
    x = {'a':1, 'b':2}
    sss(x, f)
    print ('x',x,'\nf',f)
    
mmm
mmm()
def mmm():
    f = {}
    x = {'a':1, 'b':2}
    print ('x',x,'\nf',f)
    sss(x, f)
    print ('x',x,'\nf',f)
    
mmm()

    f = {}
    x = {'a':1, 'b':2}
    print ('x',x,'\nf',f)


f = {}
x = {'a':1, 'b':2}
print ('x',x,'\nf',f)

for y in x:
    f[y].append(x[y])
    
f['a'].append(x['a'])
f['a'] = 1
f['a'].append(x['a'])
list(f['a']).append(x['a'])
f = {'a' = [], 'b'=[]}
f = {'a':[], 'b':[]}
f['a'].append(x['a'])
f['b'].append(x['b'])
def sss(a, b):
    for y in a:
        b[y].append(a[y])
        
sss(b,f)
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Tue Jul 10 11:29:13 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
a = ['a','b','c','a','b','a']
list(filter(a, 'a'))
list(filter('a', a))
list(filter(lambda x: 'a', a))
list(filter(lambda x: x=='a', a))
a.find('a')
help(a)
a.index('a')
a.count('a')
b = {'a':{'b':1, 'c':2}, 'b':{'d':3,'e':4}}"
b = {'a':{'b':1, 'c':2}, 'b':{'d':3,'e':4}}
b['a']
b['a']['b']
c = {'a':[1, 2, 3, 4, 5], 'b':['a','b','a','b','c']}
bs = list(filter(lambda x: x=='a', c['b']))
for i in range c['b']:
counter = 0
for i in range(len(c['b'])):
    if c['b'][i] == 'a'
        counter = counter + c['b'][i]
for i in range(len(c['b'])):
    if c['b'][i] == 'a':
        counter = counter + c['b'][i]
        
for i in range(len(c['b'])):
    if c['b'][i] == 'a':
        counter = counter + c['a'][i]
        
 counter = {}
counter['a'] = counter['a'] +1
counter = {interesting_coin}
counter = {a}
counter = dict(a)
a = ['a','b','c']
counter = {}
for i in a:
    counter[a] = 0
    for j in c['b']:
        if j == i:
            counter[a] = counter[a]+c['a']
            

counter = {}
for i in a:
    counter[i] = 0
    for j in c['b']:
        if j == i:
            counter[i] = counter[i]+c['a']
            
counter = {}
for i in a:
    counter[i] = 0
    for j in range(len(c['b'])):
        if c['b'][j] == i:
            counter[i] = counter[i]+c['a'][j]
        
counter = {}
for i in a:
    counter[i] = 0
    
for i in c['b'] and j in c['a']:
    counter[i] = counter[i] + j
    
for i in range(len(c['b'])):
    counter[c['b'][i]] = counter[c['b'][i]] + c['a'][i]
    
o = 1
o++
o += 1
o
counter = {}
counter = dict(map(lambda x: x =0, a))
counter = dict(map(lambda x: 0, a))
map(lambda x: 0, a)
counter = map(lambda x: 0, a)
counter = map(lambda x: counter[x] = 0, a)
t= map(lambda x: counter[x] = 0, a)
g ={}
g['z']=c
g
list(filter(c['b'], 'a'))
list(filter('a', c['b']))
list(filter(lambda x: x=='a', c['b']))
len(list(filter(lambda x: x=='a', c['b'])))
help(a)
c['b'].count('a')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
o = {'a':[],'b':[]}
o['a']
o['a'][0]
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange, market = load_exchange()
trair = find_trairs(exchange)
interesting_coins = ['BTC','EUR', 'ETH', 'USDT', 'KCS']
coins_of_interest = define_coins_of_interest(exchange, trair, interesting_coins)
trair = sort_pairs(trair, coins_of_interest)
trade_path = generate_sequence(trair, coins_of_interest)
oportunities = create_dict_oportunities()
file_name = save_oportunities(action = 'new file')

trair[0]
pairs = trair[0]
ask = ['','','']
    bid = ['','','']
ask = ['','','']
bid = ['','','']

len(pairs)
len(ask)
for i in range(len(pairs)):
    ask[i] = exchange.fetch_order_book(pairs[i])['asks'][0]
    
for i in range(len(pairs)):
    ask[i] = exchange.fetch_order_book(pairs[i])['asks'][0] #price to buy
    bid[i] = exchange.fetch_order_book(pairs[i])['bids'][0] #price to sell
    
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
    exchange, market = load_exchange()
    
    # 2- Find possible tri-pairs (trairs) within the exchange.
    trair = find_trairs(exchange)
    
    # Define the coin of interest for each trair.
    # The coin of interest is the coin we want to profit on with the trade.
    interesting_coins = ['BTC','EUR', 'ETH', 'USDT', 'KCS']
    # define_coins_of_insterest, set one coin of interest for each trair.
    coins_of_interest = define_coins_of_interest(exchange, trair, interesting_coins)

    trair = sort_pairs(trair, coins_of_interest)
    trade_path = generate_sequence(trair, coins_of_interest)
    # 3. Seek for unbalances in the trairs.
    oportunities = create_dict_oportunities()
    file_name = save_oportunities(action = 'new file')

t : 0
t = 0
unbalance = find_unbalance(
                        exchange, 
                        trair[t], 
                        coins_of_interest[t], 
                        trade_path[t])
                        
unbalance = find_unbalance(
                        exchange, 
                        trair[t], 
                        coins_of_interest[t], 
                        trade_path[t])
                        
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-13.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange = ccxt.kucoin()
exchange.apiKey = '5b4781fa91ed2941605c7f0d'
exchange.secret = '09b20a05-f889-4f12-a1b2-1a0e6bba7b4d'

market = exchange.load_markets()
balance = exchange.fetch_balance()
balance['info'][0]
balance['info']['free']
balanc['free']
balance['free']
balance['free'][0]
balance['free'][BTC]
balance['free']['BTC']
market = exchange.load_market()
market = exchange.load_markets()
balance = exchange.fetch_balance
balance = exchange.fetch_balance()
balance = exchange.fetch_balance('BTC')
balance = exchange.fetch_balance('free','BTC')
balance['free']['BTC']
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
a = 2
is int
a is int
a is int()
int(2)
int(a)
a = [2,3]
int(a)
type(a)
b = 2
type(b)
'abc'
a = 'abc'
list(a)
a = {}
a = {'a':1; 'b':2}
a = {'a':1, 'b':2}
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
print('Triangulo Amoroso')
    exchange, market = load_exchange()
    
    # 2- Find possible tri-pairs (trairs) within the exchange.
    trair = find_trairs(exchange)
    
    # Define the coin of interest for each trair.
    # The coin of interest is the coin we want to profit on with the trade.
    interesting_coins = ['BTC','EUR', 'ETH', 'USDT', 'KCS']
    print('Triangulo Amoroso')
    exchange, market = load_exchange()
    
    # 2- Find possible tri-pairs (trairs) within the exchange.
    trair = find_trairs(exchange)
    
    # Define the coin of interest for each trair.
    # The coin of interest is the coin we want to profit on with the trade.
    interesting_coins = ['BTC','EUR', 'ETH', 'USDT', 'KCS']

coin_of_interest = interesting_coins
balances = exchange.fetch_balance()
kucoin_key = 'free'
return_balance = balances[kucoin_key]
return_balance[0] = balances[coin_of_interest['free'][0]]
return_balance = balances[coin_of_interest['free']['BTC']]
return_balance = 0
return_balance = balances['free']['BTC']
for one_coin in coin_of_interest:
    if one_coin in balances[kucoin_key]:
        return_balance[one_coin] = balances[kucoin_key][one_coin]
        
return_balance = {}
for one_coin in coin_of_interest:
    if one_coin in balances[kucoin_key]:
        return_balance[one_coin] = balances[kucoin_key][one_coin]
        
return_balance = {}
for one_coin in coin_of_interest:
    if one_coin in balances[kucoin_key]:
        return_balance[one_coin] = balances[kucoin_key][one_coin]
    else:
        return_balance[one_coin] = -1
        
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Sun Jul 15 14:11:05 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-12.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
a = [1,2,3]
b = [2,3,4]
a*b
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-15.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange, market = load_exchange()
trair = find_trairs(exchange)
interesting_coins = ['BTC','EUR', 'ETH', 'USDT', 'KCS']
balances = exchange.fetch_balance()
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-14.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-15.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Sun Jul 15 14:42:22 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-15.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-16.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-16.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-16.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-16.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-16.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-16.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-16.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Tue Jul 17 17:48:54 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-17.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-17.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-17.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
v = [[0,0,0],[0,0,0],[0,0,0]]
v
for i in range(3):
    for j in range(3):
        v[i][j] = i*j
        
for i in range(3):
    for j in range(3):
        v[i][j] = i+j
    
vi = min(v[:][0])
vi = v.index(min(v[:][0]))
vi = v[:][0].index(min(v[:][0]))
v[:][0]
v[vi]
vi = v[:][0].index(max(v[:][0]))
v[vi]
trair = ['BCH/BTC', 'BCH/ETH', 'ETH/BTC']
i_coin = 'BTC'
path = ['bid', 'ask', 'ask']
ask = [[0.2,1],[2,1],[0.1,1]]
bid = [[0.19,1],[2,1],[0.09,1]]
ask = [[0.2,1],[2.1,1],[0.1,1]]
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
balances = exchange.fetch_balance()
exchange, market = load_exchange()
balances = exchange.fetch_balance()
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
c = [[],[],[]]
v
c
c[1]
c[1] = [2,3]
c

## ---(Thu Jul 19 17:29:21 2018)---
w= lambda x: x**2
w
w(2)4
w(2)
y = lambda x: x-2
y
y(2)
d ={'w':w, 'y', y}
d ={'w':w, 'y': y}
d['w']
d['w'](2)
x = 2
def change_x(a):
    a = a**2
    
change_x(x)
x = [2,3,4]
def change_x(a):
    a = a*2
    
change_x(x)
x = x*2
change_x(x)
def change_x(a):
    a = a*2
    return a

change_x(x)
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange, market = load_exchange()
trair = find_trairs(exchange)
    
    # Define the coin of interest for each trair.
    # The coin of interest is the coin we want to profit on with the trade.
    interesting_coins = ['BTC','EUR', 'ETH', 'USDT', 'KCS']
    trair = find_trairs(exchange)
    
    # Define the coin of interest for each trair.
    # The coin of interest is the coin we want to profit on with the trade.
    interesting_coins = ['BTC','EUR', 'ETH', 'USDT', 'KCS']

balance = load_balance(exchange, interesting_coins)
coins_of_interest = define_coins_of_interest(exchange, trair, interesting_coins)
trair = sort_pairs(trair, coins_of_interest)
trade_path = generate_sequence(trair, coins_of_interest)

    coins_of_interest = define_coins_of_interest(exchange, trair, interesting_coins)

    trair = sort_pairs(trair, coins_of_interest)
    trade_path = generate_sequence(trair, coins_of_interest)

oportunities = create_dict_oportunities()
file_name = save_oportunities(action = 'new file')
ask, bid = load_order_book(exchange, pairs)

pairs = trair[0]
ask, bid = load_order_book(exchange, pairs)
unbalance_factor = {
        'straight': calculate_factor(ask, bid, trade_path), 
        'inverse': calculate_factor(ask, bid, trade_path, mode = 'inverse')}
        
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')

## ---(Fri Jul 20 08:49:05 2018)---
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-18.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
b = load_balance(exchange, 'BTC')
exchange, market = load_exchange()
balance = load_balance(exchange, 'BTC')
balance
balance['BTC']
balance = load_balance(exchange, 'BTC')['BTC']
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
import ccxt
exchange, market = load_exchange()
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
exchange, market = load_exchange()
market['BTC']
market['ETH/BTC']
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
%clear
runfile('C:/Users/victzuan/.spyder-py3/ccxt/unbalance Kukoin 7-19.py', wdir='C:/Users/victzuan/.spyder-py3/ccxt')
f = 1.23456789
t = '%.5f'%f
c = 5
t = '%.'+str(c)+%f
import math
math.trunc(1.23456789, 5)
math.trunc(1.23456789)
math.trunc(12.3456789)
strformnum = "{0:."+str(c)+"f}"
trunc_num = float(strformnum.float(f))
strformnum = "{0:."+str(c+5)+"f}"
trunc_num = float(strformnum.float(f)[:-5])
trunc_num = float(strformnum.format(f)[:-5])
market