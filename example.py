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

    print("Saving model.")
    NNModel.save(mModel, "temp_model.p")
    print("Saved model.")
    print("Load model.")
    mModel2 = NNModel.load("temp_model.p")
    print("Model loaded.")
    output = mModel.predict(testData[0])
    print("Model output is: ")
    print(output)


def main():
    print("In main.")
    buildSmallExampleNet()



hey = 7
if __name__ == "__main__":
    main()
