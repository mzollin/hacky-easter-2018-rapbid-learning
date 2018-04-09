#!/usr/bin/env python3

# Copyright (c) 2018 Marco Zollinger <marco@freelabs.space>

#import json
import requests
import tensorflow as tf
import numpy as np

base_url = 'http://whale.hacking-lab.com:2222'
train_samples = 10


# get as much training data as we like
print('collecting training samples...')

for sample in range(train_samples):
    train_req = requests.get(base_url + '/train')
    if train_req.status_code == requests.codes.ok:
        print('success')
    try:
        print(train_req.json())
        #TODO: save JSON samples for later
    except ValueError as e:
        print("JSON decoder value error: {}".format(e))


# build tensorflow model
print('building tensorflow model...')

# model parameters
b = tf.get_variable('bias', [1])
a1 = tf.get_variable('weight_gender', [1])
a2 = tf.get_variable('weight_age', [1])
a3 = tf.get_variable('weight_color', [1])
a4 = tf.get_variable('weight_weight', [1])
a5 = tf.get_variable('weight_length', [1])
a6 = tf.get_variable('weight_spoon', [1])
a7 = tf.get_variable('weight_tail', [1])

# model input and output
x1 = tf.placeholder(tf.float32, shape=[None], 'input_gender')
x2 = tf.placeholder(tf.float32, shape=[None], 'input_age')
x3 = tf.placeholder(tf.float32, shape=[None], 'input_color')
x4 = tf.placeholder(tf.float32, shape=[None], 'input_weight')
x5 = tf.placeholder(tf.float32, shape=[None], 'input_length')
x6 = tf.placeholder(tf.float32, shape=[None], 'input_spoon')
x7 = tf.placeholder(tf.float32, shape=[None], 'input_tail')
y = tf.placeholder(tf.float32, shape=[None], 'output')

# define logistic regression model, loss function and optimizer
logistic_model = tf.sigmoid(a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5 + a6*x6 + a7*x7 + b)
#TODO: use cross entropy instead of least squares here
loss = tf.reduce_sum(tf.square(logistic_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01) #TODO: explain this number
train = optimizer.minimize(loss)

#TODO: assign training data arrays here

# run training loop
print('running training loop...')
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
for i in range(1000): #TODO: explain this number
    session.run(train, feed_dict={x1: x1_train, x2: x2_train, x3: x3_train, x4: x4_train,
                                  x5: x5_train, x6: x6_train, x7: x7_train, y: y_train})

# next up?
