import os
import time

import numpy as np
import tensorflow as tf

from CNN_Tensorflow.model.cnn_network import CNNNetwork

MODEL_FILE = 'model.ckpt'


class TrainingProcedure:

    def __init__(self,training_epochs,learning_rate,batch_size,display_step,log_dir):

        self.training_epochs = training_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.display_step = display_step
        self.log_dir = log_dir

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)

        tf.gfile.MakeDirs(self.log_dir)

    def get_train_op(self,loss):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def run(self,dataset):

        [X_train,y_train] = dataset.train()
        [X_val,y_val] = dataset.validation()
        accuracy_stats_train,accuracy_stats_val = [], []

        with tf.Graph().as_default():

            cnnnetw = CNNNetwork(dataset.IMAGE_SIZE,dataset.IMAGE_CHANNELS,dataset.NUM_CLASSES)

            loss_op = cnnnetw.loss()
            eval_op = cnnnetw.evaluation()

            train_op = self.get_train_op(loss_op)

            summary = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess = tf.Session()

            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            sess.run(init)

            # Start the training loop.
            for step in xrange(self.training_epochs):

                start_time = time.time()

                feed_dict = cnnnetw.fill_feed_dict(X_train,y_train,0.5,self.batch_size)

                _,loss_value = sess.run([train_op, loss_op],feed_dict=feed_dict)

                duration = time.time() - start_time

                if step % self.display_step == 0:
                    feed_dict = cnnnetw.fill_feed_dict(X_train,y_train,1.0,self.batch_size)
                    accuracy_stats_train.append(sess.run(eval_op,feed_dict=feed_dict))

                    feed_dict = cnnnetw.fill_feed_dict(X_val,y_val,1.0,self.batch_size)
                    accuracy_stats_val.append(sess.run(eval_op,feed_dict=feed_dict))

                    print 'Step {0:.2f}: loss = {1:.2f} ({2:.2f} sec)'.format(step, loss_value, duration)
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if (step + 1) % 1000 == 0 or (step + 1) == self.training_epochs:
                    checkpoint_file = os.path.join(self.log_dir,MODEL_FILE)
                    #saver.save(sess, checkpoint_file, global_step=step)
                    saver.save(sess, checkpoint_file)

        return accuracy_stats_train,accuracy_stats_val


class TestProcedure:

    def __init__(self,batch_size,log_dir):
        self.batch_size = batch_size
        self.log_dir = log_dir

    def run(self,dataset):

        [X_test,y_test] = dataset.test()
        accuracy = 0.0

        with tf.Graph().as_default():

            sess = tf.Session()
            init = tf.global_variables_initializer()

            sess.run(init)

            cnnnetw = CNNNetwork(dataset.IMAGE_SIZE,dataset.IMAGE_CHANNELS,dataset.NUM_CLASSES)
            eval_op = cnnnetw.evaluation()

            saver = tf.train.Saver()
            checkpoint_file = os.path.join(self.log_dir,MODEL_FILE)
            saver.restore(sess, checkpoint_file)
            print("Model restored.")

            feed_dict = cnnnetw.fill_feed_dict(X_test,y_test,1.0,self.batch_size)
            accuracy = sess.run(eval_op,feed_dict=feed_dict)

        return accuracy
