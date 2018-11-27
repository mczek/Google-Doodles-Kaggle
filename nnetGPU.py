# CS451 Final Project
# Elijah Peake and Michael Czekanski

# CS 451 Final Project
# Michael Czekanski and Elijah Peake


#TensorFlow Neural Network Implementation

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import random
import csv
import tensorflow.keras.backend as K

os.chdir("/Users/michael/Desktop/cs451/fp")


def load_csv(filename):
    '''
    reads data from csv
    '''
    print("loading: " + filename)
    lines = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def assembleData(objects):
    '''
    assembles data from multiple files
    '''
    print("Assembling Data")
    data = []
    decode = {}
    data1 = []
    data2 = []
    encodings = {}
    nObjects = len(objects)
    i = 0
    while i < nObjects:
        print(str(i) + " / " + str(nObjects))
        data1.extend(load_csv(objects[i] + ".csv"))
        encodings[objects[i]] = i
        decode[i] = objects[i]
        if (i+1) < nObjects:
            data2.extend(load_csv(objects[i + 1] + ".csv"))
            encodings[objects[i + 1]] = i + 1
            decode[i + 1] = objects[i + 1]
        i += 2
    # print(encodings)
    data.extend(data1)
    data.extend(data2)
    return np.array(data), encodings, decode


def assembleFullObjectData():
    '''
    gets data for every object in the training set
    '''
    objects = os.listdir(".")
    objects.remove('train_simplified.zip')
    objects.remove('.DS_Store')
    objects.remove('test_simplified.csv')
    objects.remove('GitHub')
    print("Assembling FullData")
    fullData = []
    data1 = []
    data2 = []
    encodings = {}
    nObjects = len(objects)
    i = 0
    while i <= nObjects:
        print(str(i) + " / " + str(nObjects))
        data1.extend(load_csv(objects[i]))
        data2.extend(load_csv(objects[i + 1]))
        encodings[objects[i]] = i
        encodings[objects[i+1]] = i+1
        i += 2
    # print(encodings)
    data.extend(data1)
    data.extend(data2)

    return np.array(data), encodings


def cleanData(data, encodings):
    '''
    cleans assembled data
    '''
    # print("Cleaning Data")
    # data = [data]
    labels = []
    drawings = []
    keys = []
    rec = []
    countries = []
    for elt in data:
        # print(elt["word"])
        labels.append(encodings[elt["word"]])
        drawings.append(eval(elt["drawing"]))
        keys.append(elt["key_id"])
        rec.append(elt["recognized"])
        countries.append(elt["countrycode"])
    return (drawings, labels, keys, rec, countries)


import math


def createPixels(point_1, point_2):
    '''
    figures out which pixels are colored between two points
    '''
    pt1x = point_1[0]
    pt1y = point_1[1]
    pt2x = point_2[0]
    pt2y = point_2[1]
    pixels = [(pt1x, [pt1y]), (pt2x, [pt2y])]

    if pt2x != pt1x:
        slope = (pt2y - pt1y) / (pt2x - pt1x)
        #if pt2y != pt2y:
        for y in range(min(pt2y, pt1y) + 1, max(pt2y, pt1y)):
            x = []
            if slope == 0:
                xval = (y - pt1y + pt1x)
            else:
                xval = (y - pt1y + pt1x * slope) / slope
            #xval = min(xval, 255)
            #xval = max(xval, 0)
            x.extend((math.floor(xval), math.ceil(xval)))
            pixels.append((x, y))

        for x in range(min(pt2x, pt1x) + 1, max(pt2x, pt1x)):
            y = []
            yval = slope * (x - pt1x) + pt1y
            #yval = min(yval, 255)
            #yval = max(yval, 0)
            y.extend((math.floor(yval), math.ceil(yval)))
            pixels.append((x, y))
    else:  # pt1x == pt2x
        for y in range(min(pt2y, pt1y) + 1, max(pt2y, pt1y)):
            pixels.append(([pt1x], y))
    return pixels


def cleanPoints(pointTuples):
    '''
    clean up the output of createPixels
    '''
    totalPts = []
    for pt in pointTuples:
        x = pt[0]
        y = pt[1]
        # print(x)
        # print(y)
        if isinstance(x, int):
            # print("Y:")
            # print(y)
            for elty in y:
                # print("elt y " + str(elty))
                assert(int(x) <= 255 and int(x) >=0 )
                assert (int(elty) <= 255 and int(elty) >=0)
                totalPts.append((int(x), int(elty)))
        else:
            for eltx in x:
                # print("elt x " + str(eltx))
                assert (int(eltx) <= 255 and int(eltx) >= 0)
                assert (int(y) <= 255 and int(y) >= 0)
                totalPts.append((int(eltx), int(y)))

    return totalPts


def createDrawing(drawing):
    pixels = []
    # drawing = drawing[0]
    for stroke in drawing:
        # print("printing stroke")
        # print(stroke)
        # print(len(stroke))
        for i in range(len(stroke[0]) - 1):
            # print(i)
            pixel1 = (stroke[0][i], stroke[1][i])
            pixel2 = (stroke[0][i + 1], stroke[1][i + 1])
            dirtyPixels = createPixels(pixel1, pixel2)
            cleanPixels = cleanPoints(dirtyPixels)
            pixels.extend(cleanPixels)
    return (pixels)


def plotLines(pointTuples, show=False):
    '''
    plot lines generated from cleanPoints
    '''
    canvas = np.zeros((256, 256))
    for pt in pointTuples:
        #print(pt)
        canvas[pt] = 1

    if show:
        plt.imshow(canvas)
    return canvas


def assembleDrawings(objects, frac=1):
    fullRawData, encodings, decode= assembleData(objects)
    random.shuffle(fullRawData)
    fullRawData = fullRawData[1:round(frac * len(fullRawData))]
    drawings, labels, keys, rec, countries = cleanData(fullRawData, encodings)
    pics = []
    nDrawings = len(drawings)
    for ctr in range(nDrawings):
        print(str(ctr) + " / " + str(nDrawings))
        canvas = plotLines(createDrawing(drawings[ctr]))
        pics.append(canvas)
    return np.array(pics), np.array(labels), keys, rec, countries


def sampleData(dirtyData, encodings, sampleSize):
    m = len(dirtyData)
    samp = random.sample(range(m), sampleSize)
    dataSubset = dirtyData[samp]
    drawings, labels, keys, rec, countries = cleanData(dataSubset, encodings)
    pics = []
    for ctr in range(sampleSize):
        # print(str(ctr) + " / " + str(m))
        canvas = plotLines(createDrawing(drawings[ctr]))
        pics.append(canvas.reshape((1, 256, 256, 1)))
    return np.array(pics), np.array(labels), keys, rec, countries


#from keras import backend as K
sess = tf.Session()
K.set_session(sess)

arch = [1]
objects = ["airplane"]
batch_size = 10

fullRawData, encodings, decode = assembleData(objects)
model = keras.Sequential()
model.add(keras.layers.Conv2D(5, kernel_size=5, activation='relu', input_shape=(256, 256, 1)))
model.add(keras.layers.MaxPool2D(pool_size=(5, 5)))
model.add(keras.layers.Conv2D(5, kernel_size=7, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(7, 7)))
model.add(keras.layers.Flatten())
for num in arch:
    model.add(keras.layers.Dense(num, activation=tf.nn.relu))
    model.add(keras.layers.Dense(len(objects), activation=tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#with tf.device('/cpu:0'):
    #x = tf.placeholder(tf.float32, shape=(256, 256, 1))
#    y = tf.placeholder(tf.int64)

with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float64, shape=(1, 256, 256, 1))
    #y = tf.placeholder(tf.int64)
    features, labels, keys, rec, countries = sampleData(fullRawData, encodings, batch_size)
    #xclean = tf.reshape(x
    # , [256, 256, 1])
    #yhat = model.predict(x, steps = 1)
    pred = model.predict(x, steps = batch_size)
    cost = tf.metrics.average_precision_at_k(y, pred, 3)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #x = model.train_on_batch(features, tf.reshape(labels, [batch_size, 1]))



with tf.device('/cpu:0'):
    #features, labels, keys, rec, countries = sampleData(fullRawData, encodings, batch_size)
    model_output = sess.run([xclean, yhat, cost, optimizer], feed_dict={x: features, y:labels})