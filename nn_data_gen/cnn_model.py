import tensorflow as tf

from model import Model


class CNN_Model(Model):

    def __init__(self, name, cnn_group_size=2, cnn_group_num=2, cnn_base_filters=64, dense_layer_num=1, dense_layer_units=256, dropout=0.2):
        Model.__init__(self, name)
        self.cnn_group_size = cnn_group_size
        self.cnn_group_num = cnn_group_num
        self.cnn_base_filters = cnn_base_filters
        self.dense_layer_num = dense_layer_num
        self.dense_layer_units = dense_layer_units
        self.dropout = dropout

    def inference(self, input, train=True):
        input = tf.multiply(input, 1)
        tf.add_to_collection("input_op", input)

        float_image = tf.image.convert_image_dtype(input, dtype=tf.float32)

        net = float_image

        conv_ind = 0
        for group in range(self.cnn_group_num):
            for layer in range(self.cnn_group_size):
                net = tf.layers.conv2d(net, pow(2, group) * self.cnn_base_filters, [3, 3], activation=tf.nn.relu, padding='same', name='conv{}'.format(conv_ind), reuse=tf.AUTO_REUSE)
                if (train):
                    tf.add_to_collection('VisOps', net)
                conv_ind +=1

            net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], name='maxpool{}'.format(group))
            if (train):
                tf.add_to_collection('VisOps', net)
            net = tf.layers.batch_normalization(net)
            if (train):
                tf.add_to_collection('VisOps', net)
            if(train):
                net = tf.layers.dropout(net, self.dropout)

        batch_size = tf.shape(net)[0]
        last_conv_shape = net.get_shape()

        net = tf.reshape(net, [batch_size, last_conv_shape[1] * last_conv_shape[2] * last_conv_shape[3]])

        for dense in range(self.dense_layer_num):
            net = tf.layers.dense(net, self.dense_layer_units, tf.nn.relu, name='fc{}'.format(dense), reuse=tf.AUTO_REUSE)
            if (train):
                tf.add_to_collection('VisOps', net)
            net = tf.layers.batch_normalization(net)
            if (train):
                tf.add_to_collection('VisOps', net)
            if(train):
                net = tf.layers.dropout(net, 0.2)

        net = tf.layers.dense(net, 10, name='fc_out', reuse=tf.AUTO_REUSE)
        if (train):
            tf.add_to_collection('VisOps', net)

        return net