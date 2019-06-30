#!/usr/bin/env python3

# Copyright (c) 2018 Marco Zollinger <marco@freelabs.space>

import requests
import tensorflow as tf
import numpy as np

base_url = 'http://whale.hacking-lab.com:2222'
train_samples = 100
train_runs = 10000

# get as much training data as we like
print('collecting training samples...')

# TODO: revert to np.empty()?
x1_train = np.array(np.zeros([train_samples]))
x2_train = np.array(np.zeros([train_samples]))
x3_train = np.array(np.zeros([train_samples]))
x4_train = np.array(np.zeros([train_samples]))
x5_train = np.array(np.zeros([train_samples]))
x6_train = np.array(np.zeros([train_samples]))
x7_train = np.array(np.zeros([train_samples]))
y_train = np.array(np.zeros([train_samples]))

s = requests.Session()
for sample in range(train_samples):
    train_req = s.get(base_url + '/train')
    if (sample % (train_samples / 100) == 0):
        print('run ' + str(round(sample / train_samples * 100)) + ' of 100')
    if train_req.status_code == requests.codes.ok:
        try:
            # save training data
            # TODO: get the required number of samples, even after exceptions
            json_sample = train_req.json()
            gender_ints = {'male': 0, 'female': 1}
            color_ints = {'red': 0, 'green': 1, 'blue': 2, 'black': 3,
                          'brown': 4, 'grey': 5, 'white': 6, 'purple': 7}
            np.put(x1_train, sample, gender_ints[json_sample['g3nd3r']])
            np.put(x2_train, sample, json_sample['ag3'])
            #TODO: is mapping colors to numbers useful for learning?
            np.put(x3_train, sample, color_ints[json_sample['c0l0r']])
            np.put(x4_train, sample, json_sample['w31ght'])
            np.put(x5_train, sample, json_sample['l3ngth'])
            np.put(x6_train, sample, json_sample['sp00n'])
            np.put(x7_train, sample, json_sample['t41l'])
            np.put(y_train, sample, int(json_sample['g00d']))
        except ValueError as e:
            print("JSON decoder value error: {}".format(e))
    else:
        print('bad HTTP request: error ' + str(train_req.status_code))


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
x1 = tf.placeholder(tf.float32, shape=[None], name='input_gender')
x2 = tf.placeholder(tf.float32, shape=[None], name='input_age')
x3 = tf.placeholder(tf.float32, shape=[None], name='input_color')
x4 = tf.placeholder(tf.float32, shape=[None], name='input_weight')
x5 = tf.placeholder(tf.float32, shape=[None], name='input_length')
x6 = tf.placeholder(tf.float32, shape=[None], name='input_spoon')
x7 = tf.placeholder(tf.float32, shape=[None], name='input_tail')
y = tf.placeholder(tf.float32, shape=[None], name='output')

# define logistic regression model, loss function and optimizer
logistic_model = tf.sigmoid(a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5 + a6*x6 + a7*x7 + b)
#TODO: use cross entropy instead of least squares here
loss = tf.reduce_sum(tf.square(logistic_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.001) #TODO: explain this number
train = optimizer.minimize(loss)

# run training loop
print('running training loop...')
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
for i in range(train_runs):
    if (i % (train_runs / 100) == 0):
        print('run ' + str(round(i / train_runs * 100)) + ' of 100')
    session.run(train, feed_dict={x1: x1_train, x2: x2_train, x3: x3_train, x4: x4_train,
                                  x5: x5_train, x6: x6_train, x7: x7_train, y: y_train})

# evaluate training accuracy
#print('a1: {0} a2: {0} b: {1} loss: {2}'.format(curr_a1, curr_a2, curr_b, curr_loss))
#print('wrong classifications: {0}'.format(np.sum(y_out != y_train)))

goodtail, a1c, a2c, a3c, a4c, a5c, a6c, a7c, bc, lossc = session.run([logistic_model, a1, a2, a3, a4, a5, a6, a7, b, loss],
feed_dict={x1: x1_train, x2: x2_train, x3: x3_train, x4: x4_train, x5: x5_train, x6: x6_train, x7: x7_train, y: y_train})
print('loss: ', end='')
print(lossc)
print('test data: ', end='')
print(y_train)
print('result data: ', end='')
print(np.around(goodtail))
print('output vector: ', end='')
print(goodtail)

correct_tails = np.sum(np.around(goodtail) == y_train)
print('percent correct: ', end='')
print(correct_tails / train_samples * 100)

print(a1c, a2c, a3c, a4c, a5c, a6c, a7c, bc)

#TODO: remove hardcoded coefficients
#a1c = -1.24541
#a2c = 0.18913
#a3c = 0.29244
#a4c = 6.38998
#a5c = 0.20899
#a6c = -1.25035
#a7c = -1.57856
#bc = 1.16029
#lossc = 0.00733

# TODO: explain this number
if lossc < 1:
    s = requests.Session()
    test_req = s.get(base_url + '/gate')
    if train_req.status_code == requests.codes.ok:
        try:
            json_test = test_req.json()['data']
            #classifications = np.array(np.empty(len(json_test)))
            classifications = np.array([])
            for i in json_test:
                x1t = gender_ints[i[1]]
                x2t = i[2]
                x3t = color_ints[i[3]]
                x4t = i[4]
                x5t = i[5]
                x6t = i[6]
                x7t = int(i[7])
                outp = a1c*x1t + a2c*x2t + a3c*x3t + a4c*x4t + a5c*x5t + a6c*x6t + a7c*x7t + bc
                result = np.around(1 / (1 + np.exp(-outp)))
                #np.put(classifications, i, result)
                classifications = np.append(classifications, result)
            classifications = classifications.astype(int)
            json_data = classifications.tolist()
            irritating_cookie = test_req.cookies['session_id']
            irritating_dict = {}
            irritating_dict['session_id'] = irritating_cookie
            sub_req = s.post(base_url + '/predict', json = json_data, cookies = irritating_dict)
            print(sub_req.status_code)
            print(sub_req.text)
            print(irritating_cookie)
            # get reward!
        except ValueError as e:
            print("JSON decoder value error: {}".format(e))
    else:
        print('bad HTTP request: error ' + str(train_req.status_code))
else:
    print("training failed. testing aborted")
