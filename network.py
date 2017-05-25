#============================================================#
#                                                            #
#   Artificial Neural Network                                #
#   Using the adult data to predict income                   #
#   - Matthew Lee-Mattner                                    #
#                                                            #
#============================================================#

import tensorflow as tf
import numpy as np
from adult_data import Adult_Data

# import data
data = Adult_Data()

#constants
features = 14
hl_1 = 500
hl_2 = 500
hl_3 = 500
output_nodes = 1
epochs = 30
batch_size = 100

#hyperparameters
lr = 0.001

# placholders
x = tf.placeholder('float', [None, features])
y = tf.placeholder('float', [None, output_nodes])

# return an object with weights and biases
def layer_setup(inputs, outputs):
    layer = {
        'weights': tf.Variable(tf.truncated_normal([inputs, outputs], stddev=0.1)),
        'biases': tf.constant(0.1, shape=[outputs])
    }
    return layer

def network_setup(x):
    # setup each layer
    hidden_layer_1 = layer_setup(features, hl_1)
    hidden_layer_2 = layer_setup(hl_1, hl_2)
    hidden_layer_3 = layer_setup(hl_2, hl_3)
    output = layer_setup(hl_3, output_nodes)
    # forward prop
    hl_1_result = tf.matmul(x, hidden_layer_1['weights']) + hidden_layer_1['biases']
    hl_1_result = tf.nn.sigmoid(hl_1_result)
    hl_2_result = tf.matmul(hl_1_result, hidden_layer_2['weights']) + hidden_layer_2['biases']
    hl_2_result = tf.nn.sigmoid(hl_2_result)
    hl_3_result = tf.matmul(hl_2_result, hidden_layer_3['weights']) + hidden_layer_3['biases']
    hl_3_result = tf.nn.sigmoid(hl_3_result)
    result = tf.matmul(hl_3_result, output['weights']) + output['biases']
    result = tf.nn.sigmoid(result) # reduce to value between 0 and 1
    return result

def train_network(x):
    prediction = network_setup(x)
    with tf.name_scope("Optimization"):
        cost = tf.reduce_mean( tf.squared_difference(y, prediction))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(data.get_len("train") / batch_size)):
                epoch_x, epoch_y = data.get_data("train", i, batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: np.reshape(epoch_y, [batch_size, 1])})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)
            #Test epoch
            # Compare the predicted outcome against the expected outcome

            correct = tf.equal(tf.round(prediction), y)
            # Use the comparison to generate the accuracy
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            test_batch_amount = int(data.get_len("test") / batch_size)
            final_accuracy = 0
            for i in range(test_batch_amount):
                epoch_x, epoch_y = data.get_data("test", i, batch_size) # Magically gets the next batch
                final_accuracy += accuracy.eval(feed_dict={x: epoch_x, y: np.reshape(epoch_y, [batch_size, 1])})
            print("test accuracy %", final_accuracy / test_batch_amount * 100)

train_network(x)
