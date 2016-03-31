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
    mModel = NNModel.Model()
    print("Loaded Model")

def main():
    print("In main.")
    buildSmallExampleNet()



hey = 7
if __name__ == "__main__":
    main()
