# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:39:35 2016

This program shows how to use the NN_library. it


@author: DAN
"""

import numpy as np
import pandas as pd
from NN_library import NNModel

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

def labelToOneHotEncoding(labels):
    uniqueValues = sorted(list(set(labels.T[0])))
    newLabels = np.zeros((labels.shape[0], len(uniqueValues)))
    for label_index in range(0, len(labels[:, 0])):
        value_index = uniqueValues.index(labels[label_index, 0])
        # Flip the bit corresponding to the position of the element. Values are encoded in descending order.
        # Aka, smalles value is bit in first position, and largest value is bit in last position.
        newLabels[label_index, value_index] = 1


def runNetTrial():
    # Build model.
    mModel = NNModel.Model()
    mModel.add(layer_size=2, learning_rate=.1, isInput=True)
    mModel.add(layer_size=20, learning_rate=.1, momentum_factor=.3)
    mModel.add(layer_size=2, learning_rate=.1, momentum_factor=.3)
    print("Created Model.")

    data = pd.read_table('./hw2_dataProblem.txt', sep=" +", engine='python')
    labels = data["D"].values.reshape((300, 1))
    train_set = data[['L','P']].values


def main():
    print("In main.")
    #buildSmallExampleNet()
    runNetTrial()




if __name__ == "__main__":
    main()
