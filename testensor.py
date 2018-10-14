# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:24:58 2018

@author: VICTZUAN
"""

#import tensorflow as tf
#hello=tf.constant('Hello, tensorflow')
#sess=tf.Session()
#print(sess.run(hello))

# Import `tensorflow` 
import tensorflow as tf
from tensorflow.python.data import Dataset

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run(result))

# Close the session
sess.close()