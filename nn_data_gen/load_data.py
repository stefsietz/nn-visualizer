import numpy as np
import tensorflow as tf

def load_cifar10():
    b1 = np.load('cifar10/batch1.npy')
    b2 = np.load('cifar10/batch2.npy')
    b3 = np.load('cifar10/batch3.npy')
    b4 = np.load('cifar10/batch4.npy')
    b5 = np.load('cifar10/batch5.npy')

    b_test = np.load('cifar10/batch_test.npy')

    all_training_data = np.concatenate((b1, b2, b3, b4, b5))

    x = all_training_data[:, 1:]
    y = all_training_data[:, 0]

    x = np.reshape(x, [-1, 32, 32, 1])

    x_test = b_test[:, 1:]
    y_test = b_test[:, 0]

    x_test = np.reshape(x_test, [-1, 32, 32, 1])

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return x, y, x_test, y_test, 32, class_names

def load_mnist():

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    x = train_data[:]
    y = train_labels

    x = np.reshape(x, [-1, 28, 28, 1])

    x_test = eval_data[:]
    y_test = eval_labels

    x_test = np.reshape(x_test, [-1, 28, 28, 1])

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return x, y, x_test, y_test, 28, class_names

def load_dataset(dataset):
    if(dataset is 'cifar10'):
        return load_cifar10()
    else:
        return load_mnist()
