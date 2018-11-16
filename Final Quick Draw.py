# CS451 Final Project

# Michael Czekanski
# Elijah Peake

import pickle
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

def load_mnist_data():
    """
    Return the MNIST data as (train_data, valid_data): train_data contains
    50,000 tuples (x, y) and valid_data contains 10,000 tuples (x, y).
    In each tuple, x is a 784 x 1 numpy array of floats between 0 and 1
    representing  the pixels of the 28 x 28 input image of a hand-written
    digit (0.0=white, 1.0=black).  y is the label (0..9).
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = pickle.load(f, encoding='latin1')
    f.close()
    train_data = [(np.reshape(x, (784, 1)), y)
                       for x, y in zip(data[0][0], data[0][1])]
    valid_data = [(np.reshape(x, (784, 1)), y)
                       for x, y in zip(data[1][0], data[1][1])]
    return (train_data, valid_data)


x, _ = load_mnist_data()
im = x[1][0].reshape((28, 28))
plt.imshow(im)
height = 28
width = 28
m = len(x)
n = len(x[1][0])
batch_size = 30
n_batches = int(np.ceil(m / batch_size))

# conv1 params
conv1_n_fills = 69
conv1_kernel = 5
conv1_stride = 1
conv1_pad = "same"

# conv2 params
conv2_n_fills = 32
conv2_kernel = 5
conv2_stride = 1
conv2_pad = "valid"


X = tf.placeholder("float", [None, height, width, 64])   # [None, height, width, channels]
Y = tf.placeholder("float", [None, 5])       # [None, classes]

conv1 = tf.layers.conv2d(X, filters=conv1_n_fills, kernel_size = conv1_kernel,
                         strides = conv1_stride, padding=conv1_pad,
                         activation = tf.nn.relu, name="conv1")

conv2 = tf.layers.conv2d(conv1, filters=conv2_n_fills, kernel_size = conv2_kernel,
                         strides = conv2_stride, padding=conv2_pad,
                         activation = tf.nn.relu, name="conv1")


W2 = tf.get_variable("W2", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
init = tf.global_variables_initializer()



with tf.Session() as sess_test:
    init.run()
    print(W2.eval())


def class_to_one_hot(Y, n_classes):
    return np.eye(n_classes)[Y.ravel()]


# def get_batch(data, start_index, batch_size):





init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
