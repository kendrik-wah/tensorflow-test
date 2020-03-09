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

#####################
# Linear Regression #
#####################

'''

This method uses the hands-on approach

'''

from tensorflow import keras as ks

x = tf.constant([[1.0], [1.9], [2.4], [2.6], [2.9]])
y_true = tf.constant([[100],[250],[275],[200],[300]], dtype=tf.float32)
linear_model = ks.layers.Dense(units=1)
w = tf.constant([[0],[0],[0],[0],[0]])
iterations = 1000

##y_pred = linear_model(x)
##loss = ks.losses.MSE(y_true, y_pred)

for i in range(0, iterations):
    y_pred = ks.layers.Dot(x, w)
    loss = ks.losses.MSE(y_true, y_pred)
    if (

    print("y_pred: ", y_pred)
    print("MSE: ", loss)

##print(y_pred)
##print(loss)
