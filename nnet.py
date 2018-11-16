# CS 451 Final Project
# Michael Czekanski and Elijah Peake


#TensorFlow Neural Network Implementation

import numpy



execfile("fp.py")

objects = ["owl"]
cleanDrawings, labels, keys, rec, country = assembleDrawings(objects, 0.001)

def oneHot(labels):
    encodings = []
    uniqueLabels = np.array(list(set(labels)))
    for label in labels:
        print(label)
        encodings.append(label == uniqueLabels)
    return encodings, uniqueLabels

encodings = oneHot(labels)



