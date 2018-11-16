# CS451 Final Project

# Michael Czekanski
# Elijah Peake

import csv
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy
import os

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
    for elt in objects:
        data.extend(load_csv(elt + ".csv"))
    return data
    
    
def cleanData(data):
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
        #print(elt)
        labels.append(elt["word"])
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
    fullRawData = assembleData(objects)
    fullRawData = fullRawData[1:round(frac*len(fullRawData))]
    drawings, labels, keys, rec, countries = cleanData(fullRawData)
    pics = []
    nDrawings = len(drawings)
    for ctr in range(nDrawings):
        print(str(ctr) + " / " + str(nDrawings))
        pics.append(createDrawing(drawings[ctr]))
    return pics,labels, keys, rec, countries





#objects = ["The Mona Lisa"]
#x = cleanPoints(createPixels((50,47), (150,167)))
#y = assembleData(objects)
#drawings, labels, keys, rec, countries = cleanData(y[1:15])
#a = createDrawing(drawings[4])
#b = plotLines(a, show = True)
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