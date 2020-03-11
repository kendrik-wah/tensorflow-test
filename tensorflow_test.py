import tensorflow as tf

####################
# Simple operation #
####################

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0, dtype=tf.float32)

total = a+b
total_2 = tf.add(a, b)

##print(total)
##print(total_2)

#####################
# Matrix operations #
#####################

A = tf.constant([[10,10],[11.,1.]])
I = tf.constant([[1.,0.],[0.,1.]]) # notice that I is the identity matrix
B = tf.Variable(12.)

y = tf.matmul(A, I) + B

##print(y)
##print(y.numpy())

'''

1) Notice that A.shape gives a TensorShape object.
2) Using y.numpy will give a method.However, giving y.numpy() will return value.
3) Need to find out how to use matrix operations in TensorFlow2
4) What is the difference between tf.Variable and tf.constant?

'''

'''

############################################
# General procedure for any data retrival: #
############################################

1) Import pandas, numpy, train_test_split (matplotlib and scipy if required as well):
=======================================================================================
    a) Use pandas to extract the csv data. In particular, use read_csv().
    
    b) Use data.head() to obtain the first 5 lines.
    
    c) Inspect the columns. Use data.columns. Oh yeah, understand the data.
            Note: This is important. Identify between discrete, continuous and result.
            
    d) Extract the relevant features, and make sense if these values are discrete or
       continuous.

2) After extracting the dependent and independent variables, do the following:
===============================================================================
    a) Use train_test_split(X, Y, test_size=k), where 0 <= k <= 1
            - There are 4 features to look at:
                1) train_features (this is X_train)
                2) test_features  (this is X_test)
                3) train_labels   (this is Y_train)
                4) test_labels    (this is Y_test)

    b) Establish the 4 features in 2a) as Tensorflow constants.
            - Observe their datatypes clearly.

    c) Create a dataset using the training features and labels. Notice that we want
       to preserve the dimensionality of both data structures. Refer to the link below:

       Link: https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#from_tensor_slices

            - Usually, the following initialization will do:
       dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))

    d) dataset has multiple functions. The following are generally quite recommended:
            - .repeat(number_of_epochs)
            - .batch(batch_size)
            - .shuffle(n)
       these initializations help us to store and batch data efficiently.

3) Determine the type of algorithm that needs to be used:
==========================================================
NOTE: The following procedure details the algorithms learned in both:
            - EE4305 Fuzzy and Neural Systems for Intelligent Robotics
            - CS3244 Machine Learning

    a) Are we trying to predict the value of something, given a set of inputs?
    
        - Linear Regression
            1) The mean squared error will be needed.
            2) The derivative of the mean squared error will also be needed.
                   NOTE: we want to ensure that the mean squared error is 0.
            3) Set a few things:
                   a) number of epochs
                   b) number of samples
                   c) batch size -> determines learning procedure.
                   d) learning rate (eta) -> set small
            4) Initialize the weight and bias. Knowing dimensions may be helpful here.
            5) For each epoch, do the following:
                   a) obtain the batch and find the hypothesized values with it.
                   b) calculate the MSE of the output.
                   c) find the rate of change of loss relative to weight (dL_dw)
                   d) find the rate of change of loss relative to bias (dL_db)
                   e) Note that w_nxt = w_curr - (eta*dL_dw). This is weight update.
                   f) In a similar fashion, b_nxt = b_curr - (eta*dL_db). This is bias update.
        
        - Classification

    b) Are we somehow trying to optimize something?

        - Function Approximation

        - Locating Global Minima

'''

#####################
# Linear Regression #
#####################

'''

I will be using the Graduate Admission 2 Data here.

Below is the Kaggle link to obtain the required data:
https://www.kaggle.com/mohansacharya/graduate-admissions/data

NOTE: This method of linear regression is done without using LinearRegressor.
The code below is derived based on the tutorial in the link below:

https://towardsdatascience.com/get-started-with-tensorflow-2-0-and-linear-regression-29b5dbd65977

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("Admission_Predict_Ver1.1.csv")
data.head()

features = list(data.columns)[1:7]
continuous_features = data[features].values/100
research_features = data[['Research']].values

X = np.concatenate([continuous_features, research_features], axis = 1)
Y = data[['Chance of Admit ']].values

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2)

X_train = tf.constant(train_features, dtype=tf.float32)
Y_train = tf.constant(train_labels, dtype=tf.float32)

X_test = tf.constant(test_features, dtype=tf.float32)
Y_test = tf.constant(test_labels, dtype=tf.float32)

def mse(y, y_pred):
    return tf.reduce_mean(tf.square(y_pred-y))

def mse_deriv(y, y_pred):
    return tf.reshape(tf.reduce_mean(2*(y_pred-y)), [1, 1])

def hypothesized_values(x, w, b): # based on a single perceptron.    
    return tf.tensordot(x, w, axes=1) + b

def lin_reg_scratch_eg():
    epochs = 20
    batch_size = 10
    eta = 0.001

    num_samples = X_train.shape[0]               # number of rows
    num_features = X_train.shape[1]              # number of columns
    weights = tf.random.normal((num_features,1)) # linear algebra
    bias = 0
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    dataset = dataset.shuffle(500).repeat(epochs).batch(batch_size)
    iterator = dataset.__iter__()

    epochs_plot = list()
    loss_plot = list()

    for i in range(0, epochs):

        epoch_loss = list()
        for j in range(int(num_samples/batch_size)):
            x_batch, y_batch = iterator.get_next()

            output = hypothesized_values(x_batch, weights, bias)
            loss = epoch_loss.append(mse(y_batch, output))

            # This part is a whole lot of calculus, perceptron learning algo
            dL_dh = mse_deriv(y_batch, output)
            dh_dw = x_batch
            dL_dw = tf.reduce_mean(dL_dh * dh_dw) # chain rule, updates weight
            dL_db = tf.reduce_mean(dL_dh)         # chain rule, updates bias
            
            weights = weights - (eta * dL_dw)
            bias = bias - (eta * dL_db)

            loss = np.array(epoch_loss).mean() # calculate the avg loss per batch
            epochs_plot.append(i+1)
            loss_plot.append(loss)

    output = hypothesized_values(X_test, weights, bias)
    labels = Y_test
    accuracy = tf.metrics.MeanAbsoluteError()
    accuracy.update_state(labels, output)
    print('Mean Absolute Error = {}'.format(accuracy.result().numpy()))
    print('Accuracy = {}'.format(1-accuracy.result().numpy()))
