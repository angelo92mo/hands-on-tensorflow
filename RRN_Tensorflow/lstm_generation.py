import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10

hidden_size_1 = 28*4
hidden_size_2 = 28

batch_size = 256
training_epochs = 1000
display_step = 100
learning_rate = 0.01

tf.reset_default_graph()

x_images = tf.placeholder(dtype=tf.float32,shape=[None,IMAGE_SIZE*IMAGE_SIZE])
y_images = tf.placeholder(dtype=tf.float32,shape=[None,NUM_CLASSES])

y = tf.reshape(tf.tile(input=y_images,multiples=[1,IMAGE_SIZE]),shape=[-1, IMAGE_SIZE, NUM_CLASSES])

lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(hidden_size_1,activation=tf.nn.tanh)
outputs_1,st1 = tf.nn.dynamic_rnn(lstm_cell_1,y,dtype=tf.float32,scope="ltsm1")

lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(hidden_size_2,activation=tf.nn.sigmoid)
outputs_2,st2 = tf.nn.dynamic_rnn(lstm_cell_2,outputs_1,dtype=tf.float32,scope="ltsm2")

out_images = tf.reshape(outputs_2,shape=[-1, IMAGE_SIZE*IMAGE_SIZE])

loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(out_images,x_images),axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

def fill_feed_dict(input_plc, output_plc, X, y, batch_size):
    batch_index = np.random.choice(X.shape[0], batch_size, replace=False)
    return {input_plc:X[batch_index], output_plc:y[batch_index]}

def generate_test_dict():
    gen = np.reshape(np.diag(np.ones(10)),newshape=[10,10])
    return {y_images:gen}

with tf.Session() as sess:

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train,y_train = mnist.train.images, mnist.train.labels
    X_test,y_test =  mnist.test.images, mnist.test.labels
    X_val,y_val = mnist.validation.images, mnist.validation.labels

    init = tf.global_variables_initializer()
    sess.run(init)

    for step in xrange(training_epochs):

        start_time = time.time()
        feed_dict = fill_feed_dict(x_images,y_images,X_train,y_train,batch_size)
        sess.run(train_op,feed_dict=feed_dict)
        current_loss = sess.run(loss,feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % display_step == 0:
            print 'Step {}: accuracy = {} ({} sec)'.format(step, current_loss, duration)

    ten_digits = sess.run(outputs_2,feed_dict=generate_test_dict())
    #"""
    for i in range(0,10):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,4))
        im1 = ten_digits[i,:,:]
        ax1.imshow(im1,cmap="gray")
        ax2.imshow(np.reshape(X_train[y_train.argmax(axis=1) == i][0],newshape=(28,28)),cmap="gray")
    plt.show()
    #"""



