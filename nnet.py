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
        pics.append(canvas.reshape((256, 256, 1)))
    return np.array(pics), np.array(labels), keys, rec, countries


#from keras import backend as K




#objects = ["bird", "airplane", "The Mona Lisa"]
#cleanDrawings, labels, keys, rec, country = assembleDrawings(objects, 0.001)
#labels = np.array(labels)


def createRegNetwork(arch, nClasses):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape = (256, 256)))
    for num in arch:
        model.add(keras.layers.Dense(num, activation = tf.nn.relu))
    model.add(keras.layers.Dense(nClasses, activation = tf.nn.softmax))
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', top_3_accuracy])
    return(model)

def createCNN(arch, nClasses):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(5, kernel_size=5, activation= 'relu', input_shape = (256,256,1)))
    model.add(keras.layers.MaxPool2D(pool_size = (5,5)))
    model.add(keras.layers.Conv2D(5, kernel_size=7, activation= 'relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(7, 7)))
    model.add(keras.layers.Flatten())
    for num in arch:
        model.add(keras.layers.Dense(num, activation = tf.nn.relu))
    model.add(keras.layers.Dense(nClasses, activation = tf.nn.softmax))
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return(model)

def top_3_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)



def generator(dirtyData, encodings, batch_size):
    m = len(dirtyData)
    while True:
        features, labels, keys, rec, countries = sampleData(dirtyData, encodings, batch_size)
        #print(labels)
        yield features, labels

def runRegNet(arch, objects, numSteps, numEpochs):
    fullRawData, encodings, decode = assembleData(objects)
    model = createRegNetwork(arch, len(objects))
    K.set_session(tf.Session(config = tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': 3})))
    model.fit_generator(generator(fullRawData, encodings, 10), steps_per_epoch=numSteps, epochs=numEpochs)
    return (model)

def runCNN(arch, objects, numEpochs):
    fullRawData, encodings, decode = assembleData(objects)
    #print(len(fullRawData))
    numSteps = len(fullRawData)//10
    model = createCNN(arch, len(objects))
    K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count={'CPU': 3})))
    model.fit_generator(generator(fullRawData, encodings, 10), steps_per_epoch=numSteps, epochs=numEpochs)
    return(model)


#fullRawData, encodings = assembleData(objects)
#model = createRegNetwork([100], len(objects))

#model = createCNN(len(objects))
#model.fit_generator(generator(fullRawData, 10), steps_per_epoch= 41000, epochs = 1)

#model.fit(X_train, y_train)

#fullRawData, encodings, decode = assembleData(fullObjects)
m = runCNN(arch = [1], objects = ["airplane", "The Mona Lisa", "train"], numEpochs = 1)

#img = tf.placeholder(tf.float32, shape=(256, 256))
#arch = [1]
#objects = "airplane"
#with tf.device('/cpu:0'):
#    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
#    model = createCNN(arch, len(objects))
#    model.fit_generator(generator(fullRawData, encodings, 10), steps_per_epoch=numSteps, epochs=numEpochs)
    #y = LSTM(32)(x)  # all ops in the LSTM layer will live on GPU:0

#features, labels, keys, rec, countries = sampleData(fullRawData[1:100], encodings, 10)
#preds = m.predict(features[1:4])
#preds = np.argmax(preds, axis = -1)
#decode = dict((v,k) for k,v in encodings)
#decode[str(preds[0])]
#x = assembleFullObjectData()
#fullRawData = x[0]
#econdings = x[1]
#model = createCNN([2], len(objects))
#model.fit_generator(generator(fullRawData, encodings, 10), steps_per_epoch=numSteps, epochs=numEpochs)

fullObjects = ['line',
 'bucket',
 'bus',
 'cello',
 'ocean',
 'truck',
 'camouflage',
 'harp',
 'telephone',
 'stairs',
 'star',
 'guitar',
 'sandwich',
 'sun',
 'feather',
 'leaf',
 'toilet',
 'strawberry',
 'waterslide',
 'bottlecap',
 'coffee cup',
 'banana',
 'dresser',
 'house plant',
 'skateboard',
 'skyscraper',
 'pizza',
 'hammer',
 'teapot',
 'giraffe',
 'underwear',
 'snowman',
 'monkey',
 'computer',
 'pencil',
 'shovel',
 'necklace',
 'compass',
 'bat',
 'bicycle',
 'teddy-bear',
 'scorpion',
 'hot dog',
 'fish',
 'see saw',
 'rain',
 'snail',
 'sink',
 'belt',
 'speedboat',
 'pants',
 'trombone',
 'crocodile',
 'broccoli',
 'hedgehog',
 'rainbow',
 'fork',
 'bulldozer',
 'sock',
 'snake',
 'paper clip',
 'bear',
 'marker',
 'birthday cake',
 'saxophone',
 'rake',
 'stethoscope',
 'broom',
 'crown',
 'square',
 'fire hydrant',
 'jail',
 'donut',
 'oven',
 'beard',
 'yoga',
 'The Eiffel Tower',
 'camera',
 'purse',
 'ice cream',
 'pig',
 'trumpet',
 'table',
 'bush',
 'rollerskates',
 'goatee',
 'cup',
 'anvil',
 'suitcase',
 'chair',
 'drill',
 'peanut',
 'squirrel',
 'matches',
 'sword',
 'cat',
 'toe',
 'snorkel',
 'pond',
 'calculator',
 'airplane',
 'squiggle',
 'blackberry',
 'ear',
 'frying pan',
 'chandelier',
 'lollipop',
 'binoculars',
 'garden',
 'basket',
 'penguin',
 'washing machine',
 'canoe',
 'screwdriver',
 'beach',
 'eyeglasses',
 'mouse',
 'apple',
 'van',
 'grapes',
 'grass',
 'watermelon',
 'floor lamp',
 'moon',
 'zigzag',
 'nail',
 'leg',
 'smiley face',
 'octagon',
 'dumbbell',
 'sweater',
 'stitches',
 'tractor',
 'foot',
 'helmet',
 'basketball',
 'crab',
 'clock',
 'diamond',
 'car',
 'axe',
 'traffic light',
 'sleeping bag',
 'baseball',
 'eye',
 'flower',
 'hot air balloon',
 'tree',
 'wine bottle',
 'hot tub',
 'peas',
 'door',
 'calendar',
 'wine glass',
 'stove',
 'hockey stick',
 'toothpaste',
 'moustache',
 'mountain',
 'tooth',
 'cannon',
 'firetruck',
 'shorts',
 'stereo',
 'cloud',
 'paintbrush',
 'pear',
 'dishwasher',
 'laptop',
 'frog',
 'vase',
 'diving board',
 'backpack',
 'lobster',
 'golf club',
 'garden hose',
 'hexagon',
 'bird',
 'finger',
 'animal migration',
 'steak',
 'mailbox',
 'shark',
 'television',
 'mermaid',
 'cow',
 'crayon',
 'palm tree',
 'windmill',
 'cookie',
 'kangaroo',
 'blueberry',
 'tiger',
 'tennis racquet',
 'dragon',
 'cell phone',
 'pineapple',
 'candle',
 'sheep',
 'cactus',
 'angel',
 'mosquito',
 'church',
 'couch',
 'The Great Wall of China',
 'tornado',
 'jacket',
 'nose',
 'octopus',
 'motorbike',
 'bracelet',
 'brain',
 'The Mona Lisa',
 'toothbrush',
 'carrot',
 'barn',
 'microphone',
 'zebra',
 'map',
 'camel',
 'wheel',
 'bridge',
 'lighthouse',
 'spreadsheet',
 'hockey puck',
 'wristwatch',
 'helicopter',
 'swan',
 'flamingo',
 'eraser',
 'bee',
 'flashlight',
 'megaphone',
 'ladder',
 'shoe',
 'asparagus',
 't-shirt',
 'passport',
 'hand',
 'triangle',
 'lightning',
 'mug',
 'submarine',
 'violin',
 'owl',
 'scissors',
 'baseball bat',
 'string bean',
 'lantern',
 'house',
 'elbow',
 'power outlet',
 'stop sign',
 'bed',
 'school bus',
 'hamburger',
 'lipstick',
 'light bulb',
 'flip flops',
 'alarm clock',
 'ant',
 'face',
 'microwave',
 'hourglass',
 'panda',
 'pool',
 'circle',
 'onion',
 'raccoon',
 'bowtie',
 'umbrella',
 'butterfly',
 'fireplace',
 'skull',
 'train',
 'mouth',
 'hat',
 'drums',
 'book',
 'radio',
 'roller coaster',
 'snowflake',
 'piano',
 'rhinoceros',
 'cake',
 'toaster',
 'paint can',
 'knee',
 'spider',
 'tent',
 'rabbit',
 'clarinet',
 'whale',
 'boomerang',
 'hospital',
 'ceiling fan',
 'saw',
 'pillow',
 'fence',
 'dog',
 'duck',
 'parrot',
 'swing set',
 'owls',
 'spoon',
 'fan',
 'cruise ship',
 'picture frame',
 'mushroom',
 'headphones',
 'horse',
 'flying saucer',
 'lion',
 'postcard',
 'bench',
 'keyboard',
 'parachute',
 'streetlight',
 'arm',
 'police car',
 'sailboat',
 'cooler',
 'bathtub',
 'campfire',
 'hurricane',
 'soccer ball',
 'potato',
 'dolphin',
 'key',
 'elephant',
 'sea turtle',
 'popsicle',
 'envelope',
 'pickup truck',
 'remote control',
 'ambulance',
 'pliers',
 'bread',
 'castle',
 'river',
 'bandage']
