# CS451 Final Project

# Elijah Peake
# Michael Czekanski


import numpy as np
import tensorflow as tf
import os
import random
import math
import sys

# for debugging
tf.set_random_seed(1)
np.random.seed(1)


class csvManager:
    """
    Manager for csv files.  Can randomly read batches of entries from multiple csv files so we don't have to store
    entire csv files in memory
    """
    def __init__(self, filepath):
        """
        Constructor for csvManager.  Stores file names to be open in given filepath
        :param filepath: (string)
        """
        self.filepath = filepath
        self.files = os.listdir(filepath)
        self.files_opened = None

    def open_files(self):
        """
        Opens each of the files, stores them in a list, and removes the first line of colnames
        """
        if self.filepath[-1] != "/":   # so we can add the filename to the filepath
            self.filepath += "/"

        self.files_opened = [open(self.filepath + f) for f in self.files]
        for f in self.files_opened:
            next(f)  # skip colnames

    def close_files(self):
        """
        Closes all files.
        """
        if self.files_opened:
            for f in self.files_opened:
                f.close()

    def _try_line(self):
        """
        Returns a line if available.  Closes exhausted files along the way.
        :return: (string) line or None
        """
        if not self.files_opened:
            print("files exhausted")

        else:
            index = random.randint(0, len(self.files_opened) - 1)
            try:
                return next(self.files_opened[index])

            except StopIteration:  # file exhausted, try another
                self.files_opened[index].close()
                self.files_opened.pop(index)
                return self._try_line()

    def read_lines(self, n):
        """
        Read in n files from all stored, opened files.

        :param n: (int) number of files
        :return: (list of dictionaries) our entries
        """
        if not self.files_opened:
            print("files must first be opened")

        else:
            raw_lines = []
            for _ in range(n):
                line = self._try_line()   # need to see if has value or is none (files exhausted)
                if line:
                    raw_lines.append(line)
                else:
                    break

            lines = []
            for line in raw_lines:
                split_line = line.split('"')  # split line so we can extract points
                pre_points = eval(split_line[1])
                points = []
                for segment in pre_points:
                    points.append([(x, y) for x, y in zip(segment[0], segment[1])])
                part_of_line = split_line[2].split(",")
                recognized = eval(part_of_line[2])
                label = part_of_line[-1].strip()
                lines.append({"points": points, "recognized": recognized, "label": label})

            return lines


def get_pixels(points):
    """
    Calculates all the pixels needed to recreate an image based on a set of points and their connections

    :param points: (list of lists of tuples) each nested list represents a stroke and each tuple a point
    :return: (list of tuples) all pixels of the 256x256 image
    """
    pixels = []
    for group in points:
        for i in range(len(group)-1):

            # to avoid divide by 0 errors:
            if group[i][0] != group[i+1][0]:
                slope = (group[i][1] - group[i+1][1]) / (group[i][0] - group[i+1][0])
                b = group[i][1] - (slope * group[i][0])
                previous = min(group[i], group[i+1], key=lambda point: point[0])[1]   # y val of least x

                # look at corresponding y vals one x step at a time
                for x in range(min(group[i][0], group[i+1][0]), max(group[i][0], group[i+1][0]) + 1):
                    y = slope * x + b
                    rounded_y = int(y)

                    # draw down or up to connect the dots
                    for j in range(min(rounded_y, previous) + 1, max(rounded_y, previous)):
                        pixels.append((x, j))

                    pixels.append((x, rounded_y))
                    previous = rounded_y

            #  in this case, draw a line
            else:
                for y in range(min(group[i][1], group[i + 1][1]), max(group[i][1], group[i + 1][1]) + 1):
                    pixels.append((group[i][0], y))

    return pixels


def draw_picture(pixels):
    """
    Given a set of pixels for a 256x256 image, plots all pixels in a np.array

    :param pixels: (list of tuples)
    :return: (np.array) our picture
    """
    picture = np.zeros((256, 256))
    for pixel in pixels:
        picture[pixel] = 1

    return picture


# example:
FILEPATH = "/Users/michael/Desktop/cs451/fp"

cvm = csvManager(FILEPATH)
cvm.open_files()
file_batch = cvm.read_lines(BATCH_SIZE)
pictures = []
for file in file_batch:
    pixels = get_pixels(file["points"])
    pictures.append(draw_picture(pixels))

import matplotlib.pyplot as plt
plt.imshow(pictures[random.randint(0, BATCH_SIZE - 1)])












# TODO: Make run in command line
# if __name__ == "__main__":
#     filepath = sys.argv[1]


# TODO: Make CNN
# def class_to_one_hot(Y, n_classes):
#     return np.eye(n_classes)[Y.ravel()]
#
# height = 256
# width = 256
# m = None
# n = None
# batch_size = 30
# n_batches = int(np.ceil(m / batch_size))
#
# # conv1 params
# conv1_n_fills = 10
# conv1_kernel = 3
# conv1_stride = 1
# conv1_pad = "same"
#
# # conv2 params
# conv2_n_fills = 20
# conv2_kernel = 5
# conv2_stride = 2
# conv2_pad = "valid"
#
# # logits
# n_outputs = 5
#
# # optimization
# learning_rate = 0.009
#
#
# X = tf.placeholder("float", [None, height, width, 1])   # [None, height, width, channels]
# Y = tf.placeholder("float", [None, n_outputs])       # [None, classes]
#
# conv1 = tf.layers.conv2d(X, filters=conv1_n_fills, kernel_size=conv1_kernel,
#                          strides=conv1_stride, padding=conv1_pad,
#                          activation=tf.nn.relu, name="conv1")
#
# pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=1, padding='valid', name="pool1")
#
# conv2 = tf.layers.conv2d(pool1, filters=conv2_n_fills, kernel_size=conv2_kernel,
#                          strides=conv2_stride, padding=conv2_pad,
#                          activation=tf.nn.relu, name="conv2")
#
# pool2 = tf.layers.max_pooling2d(conv2, pool_size=(5, 5), strides=2, padding='valid', name="pool2")
#
# pre_fully_connected = tf.contrib.layers.flatten(pool2)
#
# fully_connected_1 = tf.layers.dense(pre_fully_connected, 64, activation=tf.nn.relu, name="fc1")
#
# logits = tf.layers.dense(fully_connected_1, n_outputs, activation=tf.nn.relu, name="fc2")
#
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="softmax"))
#
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
# correct = tf.nn.in_top_k(logits, Y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()


# TODO: Batch Optimization
# n_epochs = 10
# batch_size = 100
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for iteration in range(m // batch_size):
#             # this cycle is for dividing step by step the heavy work of each neuron
#             X_batch = x[iteration * batch_size:iteration * batch_size + batch_size, 1:]
#             y_batch = x[iteration * batch_size:iteration * batch_size + batch_size, 0]
#             sess.run(optimizer, feed_dict={X: X_batch, Y: y_batch})
#         acc_train = accuracy.eval(feed_dict={X: X_batch, Y: y_batch})
#         acc_test = accuracy.eval(feed_dict={X: test_images, y: test_labels})
#         print("Epoch:", epoch + 1, "Train accuracy:", acc_train, "test accuracy:", acc_test)
#
#         save_path = saver.save(sess, "./my_fashion_model")

