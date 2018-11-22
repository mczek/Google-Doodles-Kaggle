# CS451 Final Project

# Michael Czekanski
# Elijah Peake

import csv
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import random


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
    data1 = []
    data2 = []
    encodings = {}
    i = 0
    nObjects = len(objects)
    while i <= nObjects:
        elt1 = objects[i]
        data1.extend(load_csv(elt1 + ".csv"))
        encodings[i] = [elt1]
        elt2 = objects[i+1]
        data1.extend(load_csv(elt2 + ".csv"))
        encodings[i+1] = [elt2]
        i += 2
    # print(encodings)
    for elt in objects:
        data.extend(load_csv(elt + ".csv"))
        encodings[i] = [elt]
        i += 1
    #print(encodings)
    return np.array(data), encodings
    
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
        data2.extend(load_csv(objects[i+1]))
        encodings[objects[i]] = [i]
        encodings[objects[i+1]] = [i+1]
        i+=2
    # print(encodings)
    data.extend(data1)
    data.extend(data2)

    return np.array(data), encodings

def cleanData(data, encodings):
    '''
    cleans assembled data
    '''
    #print("Cleaning Data")
    #data = [data]
    labels = []
    drawings = []
    keys = []
    rec = []
    countries = []
    for elt in data:
        #print(elt["word"])
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
    pixels = [(pt1x, [pt1y]) , (pt2x, [pt2y])]
    
    if pt2x != pt1x:
        slope = (pt2y - pt1y)/(pt2x - pt1x)
        for y in range(min(pt2y, pt1y) + 1, max(pt2y, pt1y)):
            x = []
            xval = (y - pt1y + pt1x*slope)/slope
            x.extend((math.floor(xval), math.ceil(xval)))
            pixels.append((x, y))

        for x in range(min(pt2x, pt1x) + 1, max(pt2x, pt1x)):
            y = []
            yval = slope*(x - pt1x) + pt1y
            y.extend((math.floor(yval), math.ceil(yval)))
            pixels.append((x, y))
    else: #pt1x == pt2x
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
        #print(x)
        #print(y)
        if isinstance(x, int):
            #print("Y:")
            #print(y)
            for elty in y:
                #print("elt y " + str(elty))
                totalPts.append((x, elty))
        else:
            for eltx in x:
                #print("elt x " + str(eltx))
                totalPts.append((eltx, y))
                
    return totalPts
    
def createDrawing(drawing):
    pixels = []
    #drawing = drawing[0]
    for stroke in drawing:
       # print("printing stroke")
       # print(stroke)
       # print(len(stroke))
        for i in range(len(stroke[0]) - 1):
            #print(i)
            pixel1 = (stroke[0][i], stroke[1][i])
            pixel2 = (stroke[0][i+1], stroke[1][i+1])
            dirtyPixels = createPixels(pixel1, pixel2)
            cleanPixels = cleanPoints(dirtyPixels)
            pixels.extend(cleanPixels)
    return(pixels)
    
    
def plotLines(pointTuples, show = False):
    '''
    plot lines generated from cleanPoints
    '''
    canvas = np.zeros((256,256))
    for pt in pointTuples:
        #print(pt)
        canvas[pt] = 1
        
    if show:
        plt.imshow(canvas)
    return canvas
    

def assembleDrawings(objects, frac = 1):
    fullRawData, encodings = assembleData(objects)
    random.shuffle(fullRawData)
    fullRawData = fullRawData[1:round(frac*len(fullRawData))]
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
        #print(str(ctr) + " / " + str(m))
        canvas = plotLines(createDrawing(drawings[ctr]))
        pics.append(canvas.reshape((256,256,1)))
    return np.array(pics), np.array(labels), keys, rec, countries



objects = ["airplane"]
x = cleanPoints(createPixels((50,47), (50,147)))
y, encodings, decode = assembleData(objects)
drawings, labels, keys, rec, countries = cleanData(y[1:15], encodings)
a = createDrawing(drawings[5])
b = plotLines(a, show = True)
#if __name__ == "__main__":
#    objects = ["owl"]
#    cleanDrawings, labels, keys, rec, country = assembleDrawings(objects)
    
#y = np.zeros((256,256))
#x.reshape((2,len(x)/2))
#y[x] = 1

#for pt in x:
    #print(pt)

#plt.imshow(y)
#plt.show()
##from Kaggle Kernal How To Draw an Owl
#owls = pd.read_csv('owl.csv')
#owls = owls[owls.recognized]
#owls['timestamp'] = pd.to_datetime(owls.timestamp)
#owls = owls.sort_values(by='timestamp', ascending=False)[-100:]
#owls['drawing'] = owls['drawing'].apply(ast.literal_eval)
#
#owls.head()
#
#n = 10
#fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))
#for i, drawing in enumerate(owls.drawing):
#    ax = axs[i // n, i % n]
#    for x, y in drawing:
#        ax.plot(x, -np.array(y), lw=3)
#    ax.axis('off')
#fig.savefig('owls.png', dpi=200)
#plt.show();