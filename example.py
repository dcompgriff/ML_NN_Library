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
    mModel.add(layer_size=2, isInput=True)
    mModel.add(layer_size=3)
    mModel.add(layer_size=2)
    print("Created Model.")

    # Train model.

    # Predict data.
    testData = np.array([1, 1])
    output = mModel.predict(testData)
    print("Model output is: ")
    print(output)


def main():
    print("In main.")
    buildSmallExampleNet()



hey = 7
if __name__ == "__main__":
    main()
