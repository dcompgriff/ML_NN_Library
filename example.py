# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:39:35 2016

This program shows how to use the NN_library. it


@author: DAN
"""

import numpy as np
import pandas as pd
from NN_library import NNModel
import random
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def buildSmallExampleNet():
    # Build model.
    mModel = NNModel.Model()
    mModel.add(layer_size=2, learning_rate=1, isInput=True)
    mModel.add(layer_size=3, learning_rate=1, momentum_factor=.3)
    mModel.add(layer_size=2, learning_rate=1, momentum_factor=.3)
    print("Created Model.")

    # Train model.
    testData = np.array([[1,1]])
    labelData = np.array([[1,0]])
    mModel.train(testData, labelData, epochs=10000)
    # Predict data.
    output = mModel.predict(testData[0])
    print("Model output is: ")
    print(output)

def calculateAccuracy(ypredicted, yactual):
    metrics = {}
    metrics["tp"] = 0
    metrics["tn"] = 0
    metrics["fp"] = 0
    metrics["fn"] = 0
    for i in range(0, len(yactual)):
        if ypredicted[i] == 0 and yactual[i] == 0:
            metrics["tn"] += 1
        elif ypredicted[i] == 1 and yactual[i] == 0:
            metrics["fp"] += 1
        elif ypredicted[i] == 0 and yactual[i] == 1:
            metrics["fn"] += 1
        elif ypredicted[i] == 1 and yactual[i] == 1:
            metrics["tp"] += 1

    accuracy = (metrics["tp"] + metrics["tn"]) / (metrics["tp"] + metrics["tn"] + metrics["fp"] + float(metrics["fn"]))

    return accuracy

def labelToOneHotEncoding(labels):
    uniqueValues = sorted(list(set(labels)))
    newLabels = np.zeros((labels.shape[0], len(uniqueValues)))
    for label_index in range(0, len(labels[:])):
        value_index = uniqueValues.index(labels[label_index])
        # Flip the bit corresponding to the position of the element. Values are encoded in descending order.
        # Aka, smalles value is bit in first position, and largest value is bit in last position.
        newLabels[label_index, value_index] = 1
    return newLabels

def oneHotEncodingToLabels(labels):
    newLabels = np.zeros((labels.shape[0], 1))
    for index in range(0, labels.shape[0]):
        argMax = np.argmax(labels[index])
        newLabels[index] = np.array([argMax])
    return newLabels


def runNetTrial():
    # Build model.
    mModel = NNModel.Model()
    mModel.add(layer_size=2, learning_rate=.1, isInput=True)
    mModel.add(layer_size=20, learning_rate=.1)
    mModel.add(layer_size=2, learning_rate=.1)
    print("Created Model.")

    data = pd.read_table('./hw2_dataProblem.txt', sep=" +", engine='python')
    #Range scale the P data.
    data["P"] = data["P"].apply(lambda item: (item - data.P.min()) / (data.P.max() - data.P.min()))
    #Range scale the L data
    data["L"] = data["L"].apply(lambda item: (item - data.L.min()) / (data.L.max() - data.L.min()))

    #Split the data into training and test data sets.
    train0, test0 = train_test_split(data[data.D == 0].values, test_size = 0.2, random_state=random.randint(0, 100000))
    train1, test1 = train_test_split(data[data.D == 1].values, test_size = 0.2, random_state=random.randint(0, 100000))

    #Combine and shuffle the test and train examples.
    testSet = np.vstack((test0, test1))
    np.random.shuffle(testSet)
    trainSet = np.vstack((train0, train1))
    #trainSet = np.vstack((trainSet, train0))
    np.random.shuffle(trainSet)

    testSetData = testSet[:,0:2]
    testSetLabels = labelToOneHotEncoding(testSet[:,2])
    trainSetData = trainSet[:,0:2]
    trainSetLabels = labelToOneHotEncoding(trainSet[:,2])

    print("Starting training.")
    mModel.train(trainSetData, trainSetLabels, epochs=100)
    print("Training finished.")
    # Predict data.
    output = mModel.predict(trainSetData[0])
    output = oneHotEncodingToLabels(np.array([output]))
    print("Model output is: ")
    print(output)

    predictedLabels = mModel.predictAll(trainSetData)
    predictedLabels = oneHotEncodingToLabels(predictedLabels)

    accuracy = calculateAccuracy(predictedLabels, trainSet[:,2].reshape((len(trainSet), 1)))

    print("Predicted Labels:")
    print(predictedLabels)
    print("Accuracy on training set is: ")
    print(accuracy)


    print("Finished.")




def main():
    print("In main.")
    #buildSmallExampleNet()
    runNetTrial()




if __name__ == "__main__":
    main()
