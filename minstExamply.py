



import numpy as np
import pandas as pd
from NN_library import NNModel
import random
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import *
import time


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

def calculateMetrics(ypredicted, yactual):
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

    metrics["sensitivity"] = float(metrics["tp"]) / (float(metrics["tp"]) + metrics["fn"])
    metrics["specificity"] = float(metrics["tn"]) / (float(metrics["tn"]) + metrics["fp"])
    metrics["ppv"] = float(metrics["tp"]) / (float(metrics["tp"]) + metrics["fp"])
    metrics["npv"] = float(metrics["tn"]) / (float(metrics["tn"]) + metrics["fn"])

    return metrics

def runNetTrial():
    # Build model.
    mModel = NNModel.Model()
    mModel.add(layer_size=784, learning_rate=.01, isInput=True)
    mModel.add(layer_size=200, learning_rate=.005, momentum_factor=0)
    #mModel.add(layer_size=100, learning_rate=.005, momentum_factor=0)
    mModel.add(layer_size=10, learning_rate=.005, momentum_factor=0)
    print("Created Model.")

    # Read data from file.
    xData = pd.read_csv('./MNISTnumImages5000.txt', sep='\t', header=None)
    yData = pd.read_csv('./MNISTnumLabels5000.txt', header=None, names=['labels'])
    xData['labels'] = yData.values
    # Break data into train and test sets.
    import random
    trainSet, testSet = train_test_split(xData, test_size=0.2, random_state=random.randint(0, 100000))

    # Break data into train and test sets.
    originalTrainLabels = trainSet['labels'].values
    originalTestLabels = testSet['labels'].values
    trainLabels = NNModel.labelToOneHotEncoding(originalTrainLabels)
    testLabels = NNModel.labelToOneHotEncoding(originalTestLabels)
    trainData = trainSet[trainSet.columns[:-1]].values
    testData = testSet[testSet.columns[:-1]].values

    print("Starting training.")
    trialWiseErrorList = mModel.train(trainData, trainLabels, validation_data_set=testData, validation_label_set=originalTestLabels, epochs=60)
    print("Training finished.")

    # Predict the test set metrics
    predictedLabels = mModel.predictAll(testData)
    predictedLabels = NNModel.oneHotEncodingToLabels(predictedLabels)
    accuracy = NNModel.calculateAccuracy(predictedLabels, originalTestLabels)
    #testSetMetrics = calculateMetrics(predictedLabels, originalTestLabels)
    testSetMetrics = {}
    testSetMetrics["accuracy"] = accuracy

    # Predict the train set metrics
    predictedLabels = mModel.predictAll(trainData)
    predictedLabels = NNModel.oneHotEncodingToLabels(predictedLabels)
    accuracy = NNModel.calculateAccuracy(predictedLabels, originalTrainLabels)
    #trainSetMetrics = calculateMetrics(predictedLabels, originalTrainLabels)
    trainSetMetrics = {}
    trainSetMetrics["accuracy"] = accuracy
    trainSetMetrics["accuracyList"] = trialWiseErrorList

    print("Orig labels: ")
    print(originalTrainLabels[0:20])
    print("Pred labels: ")
    print(predictedLabels[0:20])

    return mModel, trainSetMetrics, testSetMetrics


def plotMetrics(metricList, modelName="", numTrials=0):
    fig = plt.gcf()
    fig.set_size_inches(12, 7)
    red_patch = mpatches.Patch(color='red', label='Train')
    blue_patch = mpatches.Patch(color='blue', label='Test')

    #NN PLOTS.
    #Plot the sensitivity
    mplt = fig.add_subplot(2,2,1)
    xVals = np.arange(1, 10, 1)
    testMetrics = []
    trainMetrics = []
    for i in range(0, numTrials):
        testMetrics.append(metricList[i][2])
        trainMetrics.append(metricList[i][1])
    #Make perceptron plot.
    mplt.set_title(modelName + " Sensitivity")
    mplt.xaxis.set_ticks(xVals)
    mplt.bar(xVals - 0.1, [item["sensitivity"] for item in testMetrics], width=.2, color='b')
    mplt.bar(xVals + 0.1, [item["sensitivity"] for item in trainMetrics], width=.2, color='r')
    mplt.legend(handles=[red_patch, blue_patch], loc=3)

    #Plot the Specificity
    mplt = fig.add_subplot(2,2,2)
    #Make perceptron plot.
    mplt.set_title(modelName + " Specificity")
    mplt.xaxis.set_ticks(xVals)
    mplt.bar(xVals - 0.1, [item["specificity"] for item in testMetrics], width=.2, color='b')
    mplt.bar(xVals + 0.1, [item["specificity"] for item in trainMetrics], width=.2, color='r')
    mplt.legend(handles=[red_patch, blue_patch], loc=3)

    #Plot the ppv
    mplt = fig.add_subplot(2,2,3)
    #Make perceptron plot.
    mplt.set_title(modelName + " Ppv")
    mplt.xaxis.set_ticks(xVals)
    mplt.bar(xVals - 0.1, [item["ppv"] for item in testMetrics], width=.2, color='b')
    mplt.bar(xVals + 0.1, [item["ppv"] for item in trainMetrics], width=.2, color='r')
    mplt.legend(handles=[red_patch, blue_patch], loc=3)

    #Plot the npv
    mplt = fig.add_subplot(2,2,4)
    #Make plot.
    mplt.set_title(modelName + " Npv")
    mplt.xaxis.set_ticks(xVals)
    mplt.bar(xVals - 0.1, [item["npv"] for item in testMetrics], width=.2, color='b')
    mplt.bar(xVals + 0.1, [item["npv"] for item in trainMetrics], width=.2, color='r')
    mplt.legend(handles=[red_patch, blue_patch], loc=3)

    plt.show()

def averagePerformance(metricList, numTrials=0):
    testMetrics = []
    trainMetrics = []

    for i in range(0, numTrials):
        trainMetrics.append(metricList[i][1])
        testMetrics.append(metricList[i][2])

    #Metrics on the test set.
    sen = np.array([item["sensitivity"] for item in testMetrics])
    spec = np.array([item["specificity"] for item in testMetrics])
    ppv = np.array([item["ppv"] for item in testMetrics])
    npv = np.array([item["npv"] for item in testMetrics])
    print("Test metrics.")
    print("Sensitivity: " + str(sen.mean()) + ", " + str(sen.std()))
    print("Specificity: " + str(spec.mean()) + ", " + str(spec.std()))
    print("ppv: " + str(ppv.mean()) + ", " + str(ppv.std()))
    print("npv: " + str(npv.mean()) + ", " + str(npv.std()))
    print("\n")

    #Metrics on the train set.
    sen = np.array([item["sensitivity"] for item in trainMetrics])
    spec = np.array([item["specificity"] for item in trainMetrics])
    ppv = np.array([item["ppv"] for item in trainMetrics])
    npv = np.array([item["npv"] for item in trainMetrics])
    print("Train metrics.")
    print("Sensitivity: " + str(sen.mean()) + ", " + str(sen.std()))
    print("Specificity: " + str(spec.mean()) + ", " + str(spec.std()))
    print("ppv: " + str(ppv.mean()) + ", " + str(ppv.std()))
    print("npv: " + str(npv.mean()) + ", " + str(npv.std()))
    print("\n")

def plotTrialError(metricList, numTrials=0):
    plt.clf()
    trialAccuracyList = []
    # Get the accuracy list from the training metrics dict in the metricsList obj.
    for i in range(0, numTrials):
        trialAccuracyList.append(metricList[i][1]["accuracyList"])

    #Plot the trial wise accuracy over time.
    for i in range(0, numTrials):
        pltLabel = "Trial %s" % str(i)
        plt.plot(np.arange(0, len(trialAccuracyList[i])), trialAccuracyList[i], label=pltLabel )
    legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., framealpha=0.5)
    plt.title("Trial accuracy over time.")
    plt.show()

    #Plot the mean trial wise error, and plot the std dev as error bars.
    avgList = []
    stdList = []
    for i in range(0, len(trialAccuracyList[0])):
        temp = np.array(trialAccuracyList)
        avgList.append(temp[:, i].mean())
        stdList.append(temp[:, i].std())
    plt.title("Average trial-wise error plot.")
    plt.errorbar(np.arange(0, len(trialAccuracyList[0])), avgList, yerr=stdList)
    plt.show()

def plotNN(mModel):
    plt.clf()
    print("Beginning best knn...")
    #Create a grid to classify over.
    testSet = []
    for x in np.arange(0, 1, 0.02):
        for y in np.arange(0, 1, 0.02):
            testSet.append([x, y])
    testSet = np.array(testSet)

    #Classify over the grid.
    predictedLabels = mModel.predictAll(testSet)
    predictedLabels = NNModel.oneHotEncodingToLabels(predictedLabels)

    #Group together to be filtered by color.
    data = pd.DataFrame(testSet, columns=['L', 'P'])
    data['D'] = predictedLabels
    posData = data[data.D == 1]
    negData = data[data.D == 0]
    plt.scatter(posData.L, posData.P, color="red")
    plt.scatter(negData.L, negData.P, color="blue")
    plt.title("Best Classifier Decision Boundary.")
    plt.show()

def main():
    global mModel, trainMetrics, testMetrics
    print("In main.")

    # Run program for trial wise metrics.
    initTime = time.time()
    print("Training net...")
    mModel, trainMetrics, testMetrics = runNetTrial()
    print("Finished training!")
    finTime = time.time()
    print("Total time: " + str((finTime - initTime) / 60.0) + " min.")

    print('Train metrics:')
    print(trainMetrics)
    print('Test metrics: ')
    print(testMetrics)

    NNModel.save(mModel, './MINST_MODEL.pk')
    mModel = NNModel.load('./MINST_MODEL.pk')

    print("Saving metrics.")
    NNModel.save(trainMetrics, './MINST_MODEL_TRAIN_METRICS.pk')
    NNModel.save(testMetrics, './MINST_MODEL_TEST_METRICS.pk')
    trainMetrics = NNModel.load('./MINST_MODEL_TRAIN_METRICS.pk')
    testMetrics = NNModel.load('./MINST_MODEL_TEST_METRICS.pk')
    print("Saved and reloaded metrics.")

    print("End of main.")


if __name__ == "__main__":
    main()
