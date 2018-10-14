# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:29:37 2018

@author: victzuan
"""

import tensorflow as tf

#Set up a linear classifier.
classifier = tf.estimator.LinearClassifier()

#Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps = 2000)

predictions = classifier.predict(input_fn = predict_input_fn)
