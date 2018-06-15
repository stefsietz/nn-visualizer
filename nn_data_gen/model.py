import tensorflow as tf


class Model:

    def __init__(self, name):
        self.name = name

    def inference(self, input, train=True):
        pass

    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        return cross_entropy_mean

    def accuracy(self, logits, labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), tf.float32))
