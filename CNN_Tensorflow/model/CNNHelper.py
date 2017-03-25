import tensorflow as tf

class CNNHelper:

    @staticmethod
    def weight_variable(num_in,num_out):
        shape = [num_in,num_out]
        initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32,name="weights")
        return tf.Variable(initial)

    @staticmethod
    def kernel_variable(num_channel_in,num_channel_out,W=3,H=3):
        shape = [W,H,num_channel_in,num_channel_out]
        initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32,name="filter")
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(num_out):
        shape = [num_out]
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32,name="bias")
        return tf.Variable(initial)

    @staticmethod
    def convolution(input_channel,kernel):
        return tf.nn.conv2d(input=input_channel,filter=kernel,strides=[1, 1, 1, 1], padding="SAME", name="conv")

    @staticmethod
    def max_pool_2x2(input_channel):
        return tf.nn.max_pool(input_channel,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME", name="maxpool2x2")

    @staticmethod
    def conv_layer(channel_in,num_channel_out,width=5,height=5):
        num_channel_in = channel_in.get_shape().as_list()[3]
        W_  =  CNNHelper.kernel_variable(num_channel_in=num_channel_in,num_channel_out=num_channel_out,W=width,H=height)
        b_  =  CNNHelper.bias_variable(num_out=num_channel_out)
        h_   =  tf.nn.relu(CNNHelper.convolution(channel_in,W_) + b_)
        return CNNHelper.max_pool_2x2(h_)

    @staticmethod
    def dense_layer(channel_in,num_channel_out,name="output"):
        num_channel_in = channel_in.get_shape().as_list()[1]
        W_ = CNNHelper.weight_variable(num_channel_in,num_channel_out)
        b_ = CNNHelper.bias_variable(num_channel_out)
        return tf.nn.relu(tf.add(tf.matmul(channel_in,W_),b_), name=name)

    @staticmethod
    def count_params():
        "print number of trainable variables"
        size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
        n = sum(size(v) for v in tf.trainable_variables())
        print "Model size: %dK" % (n,)
