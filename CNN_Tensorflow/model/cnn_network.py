import tensorflow as tf
import numpy as np
from CNNHelper import CNNHelper

class CNNNetwork:

    def __init__(self,image_size,image_channels,num_classes):
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.inference()

    def placeholder_inputs(self):

        with tf.name_scope('input'):
            X = tf.placeholder(dtype=tf.float32,shape=(None,self.image_size * self.image_size),name="input")

        with tf.name_scope('output'):
            y = tf.placeholder(tf.float32,shape=[None,self.num_classes],name="output")

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32,name="dropout")

        return X, y, keep_prob

    def inference(self):

        X, y, keep_prob = self.placeholder_inputs()

        # input (32x32x1)
        size_map = self.image_size*self.image_size
        with tf.name_scope('conv1'):
            x_image = tf.reshape(tf.to_float(X),shape=[-1, self.image_size, self.image_size, self.image_channels])
            out_conv1 = CNNHelper.conv_layer(x_image,16)

        # input (14x14x16)
        size_map = size_map/4
        with tf.name_scope('conv2'):
            out_conv2 = CNNHelper.conv_layer(out_conv1,32)

        # input (7x7x32)
        size_map = size_map/4
        with tf.name_scope('dense1'):
            in_dense1 = tf.reshape(shape=[-1,(size_map)*32],tensor=out_conv2)
            out_dense1 = CNNHelper.dense_layer(in_dense1,512)
            out_dense1_drop = tf.nn.dropout(out_dense1, keep_prob)

        # input (512)
        with tf.name_scope('logits'):
            logits = CNNHelper.dense_layer(out_dense1_drop,self.num_classes,name="output")

        # input (NUM_CLASSES)
        with tf.name_scope('predictions'):
            y_ = tf.nn.softmax(logits=logits,name="output")

        return [X,y,keep_prob,logits,y_]

    def loss(self):
        logits = tf.get_default_graph().get_tensor_by_name("logits/output:0")
        labels = tf.get_default_graph().get_tensor_by_name("output/output:0")
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def evaluation(self):
        predictions = tf.get_default_graph().get_tensor_by_name("predictions/output:0")
        labels = tf.get_default_graph().get_tensor_by_name("output/output:0")
        correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def fill_feed_dict(self, X, y, keep_prob, batch_size):
        input_plc = tf.get_default_graph().get_tensor_by_name("input/input:0")
        output_plc = tf.get_default_graph().get_tensor_by_name("output/output:0")
        droput_plc = tf.get_default_graph().get_tensor_by_name("dropout/dropout:0")
        batch_index = np.random.choice(X.shape[0], batch_size, replace=False)
        feed_dict = {input_plc:X[batch_index], output_plc:y[batch_index], droput_plc:keep_prob}
        return feed_dict
