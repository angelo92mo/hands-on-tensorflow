import tensorflow as tf
import numpy as np
import time

from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10

hidden_size = 128
batch_size = 512
training_epochs = 2000
display_step = 100
learning_rate = 0.001

tf.reset_default_graph()

x_images = tf.placeholder(dtype=tf.float32,shape=[None,IMAGE_SIZE*IMAGE_SIZE])
X = tf.reshape(tf.to_float(x_images),shape=[-1, IMAGE_SIZE, IMAGE_SIZE])
y = tf.placeholder(dtype=tf.float32,shape=[None,NUM_CLASSES])

lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
outputs, states = tf.nn.dynamic_rnn(lstm_cell,X,dtype=tf.float32)

W1 = tf.Variable(tf.truncated_normal([hidden_size,hidden_size/2],stddev=0.1,dtype=tf.float32))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden_size/2], dtype=tf.float32))
out1 = tf.nn.relu(tf.matmul(outputs[:,-1,:],W1) + b1)

W2 = tf.Variable(tf.truncated_normal([hidden_size/2,NUM_CLASSES],stddev=0.1,dtype=tf.float32))
b2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES], dtype=tf.float32))
logits = tf.matmul(out1,W2) + b2

y_ = tf.nn.softmax(logits=logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
eval_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

def fill_feed_dict(input_plc, output_plc, X, y, batch_size):
    batch_index = np.random.choice(X.shape[0], batch_size, replace=False)
    feed_dict = {input_plc:X[batch_index], output_plc:y[batch_index]}
    return feed_dict

with tf.Session() as sess:

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train,y_train = mnist.train.images, mnist.train.labels
    X_test,y_test =  mnist.test.images, mnist.test.labels
    X_val,y_val = mnist.validation.images, mnist.validation.labels

    init = tf.global_variables_initializer()
    sess.run(init)

    for step in xrange(training_epochs):

        start_time = time.time()
        feed_dict = fill_feed_dict(x_images,y,X_train,y_train,batch_size)
        _,acc = sess.run([train_op, eval_op],feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % display_step == 0:
            print 'Step {}: accuracy = {} ({} sec)'.format(step, acc, duration)

    feed_dict = fill_feed_dict(x_images,y,X_val,y_val,1024)
    acc = sess.run([eval_op],feed_dict=feed_dict)
    print 'Batch Accuracy val set = {}'.format(acc)

    feed_dict = fill_feed_dict(x_images,y,X_test,y_test,1024)
    acc = sess.run([eval_op],feed_dict=feed_dict)
    print 'Batch Accuracy test set = {}'.format(acc)
