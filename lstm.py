# -*- coding: utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import tensorflow as tf
from tensorflow.contrib import rnn
import helper
import os
import numpy as np
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 50
# Network Parameters
n_input = 100 # MNIST data input (img shape: 28*28)
n_steps = 30 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.int32, [None, n_steps, n_input])
x1 = tf.placeholder(tf.int32, [None, n_steps])
y = tf.placeholder(tf.int32, [None,2])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
train_path = 'train.2'
char2id_file = 'char2id'
label2id_file = 'label2id'
save_path = './'
emb_dim = '100'
X_train, y_train, X_val, y_val = helper.getTrain(train_path=train_path, val_path=None, seq_max_len=n_steps,
                                                 char2id_file=char2id_file, label2id_file=label2id_file)
sh_index = np.arange(len(X_train))
np.random.shuffle(sh_index)
X_train = X_train[sh_index]
y_train = y_train[sh_index]

char2id, id2char = helper.loadMap(char2id_file)
label2id, id2label = helper.loadMap(label2id_file)
num_chars = len(id2char.keys())  # vocabulary大小
num_classes = len(id2label.keys())  # 标注类别数
emb_path = None
if emb_path != None:
    embedding_matrix = helper.getEmbedding(emb_path, char2id_file)
    # print len([_ for _ in np.sum(embedding_matrix,axis=1) if _ != 0])
    np.savetxt(os.path.join(save_path, "embedding_matrix"), embedding_matrix)
    num_chars = embedding_matrix.shape[0]  # vocabulary大小
else:
    embedding_matrix = None

# char embedding
if embedding_matrix is not None:
    embedding = tf.Variable(embedding_matrix, trainable=True, name="emb", dtype=tf.float32)
else:
    embedding = tf.get_variable("emb", [num_chars, emb_dim])
inputs_emb = tf.nn.embedding_lookup(embedding, x1)  # ??

print "building model"

def last_relevant(output, length):
    output = tf.transpose(output, [1,0,2])
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def RNN( x,weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    length = tf.reduce_sum(tf.sign(x1), reduction_indices=1)
    length = tf.cast(length, tf.int32)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])

    # # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])

    # # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32 )
    # outputs = last_relevant(outputs, length)
    # Linear activation, using rnn inner loop last output
    return tf.matmul( outputs[-1], weights['out'] ) + biases['out'], outputs

pred,outputs = RNN( inputs_emb, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # Keep training until reach max iterations
    while step  < training_iters:
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x, batch_y = helper.nextRandomBatch(X_train, y_train,batch_size=125)
        batch_y = batch_y[:, 2]
        tmp_y = []
        for i in range(0, batch_y.shape[0]) :
            if batch_y[i] == 2 :
                tmp_y.append([0,1])
            else:
                tmp_y.append([1,0])
        tmp_y = np.array(tmp_y)
        batch_y = tmp_y
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x1: batch_x, y: batch_y})

        if step % display_step == 0:
            # Calculate batch accuracy
            acc,pre,output = sess.run([accuracy,pred,outputs],feed_dict={x1: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x1: batch_x, y: batch_y})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            # print pred
        step += 1
    print("Optimization Finished!")
    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data,test_label = helper.nextRandomBatch(X_val,y_val,test_len)
    test_label = test_label[:, 2]
    tmp_y = []
    for i in range(0, test_label.shape[0]):
        if test_label[i] == 2:
            tmp_y.append([0, 1])
        else:
            tmp_y.append([1, 0])
    tmp_y = np.array(tmp_y)
    test_label = tmp_y
    print("Testing Accuracy:", )
    acc, pre = sess.run([accuracy, pred], feed_dict={x1: test_data, y: test_label})
    print acc
    pre_pos_count = 0
    test_pos_count = 0
    pos_count = 0
    for i in range(0, len(pre)) :
        print pre[i],tmp_y[i]
        if pre[i][0] < pre[i][1] and tmp_y[i][1] == 1 :
            pos_count += 1
        if tmp_y[i][1] == 1 :
            test_pos_count += 1
        if pre[i][0] < pre[i][1] :
            pre_pos_count += 1
    print test_pos_count, pos_count, pre_pos_count
    print pos_count*1.0/test_pos_count, pos_count*1.0/pre_pos_count
