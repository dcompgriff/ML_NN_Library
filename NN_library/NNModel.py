'''
This library has some parameters indicating the possibility of implementing backpropagation
with alternate loss functions, activation functions, and optimizers. Currently, only
backpropagation with SGD and an activation of sigmoids is used.
'''
from functools import *
import numpy as np
import _pickle as pickle
import math

'''
This class is used to build a neural network model.
'''
class Model:
    def __init__(self):
        self.layers = []
        self.inputSize = 0

    '''
    Add a new fully connected layer to this model.
    '''
    def add(self, layer_size=1, learning_rate=0.1, momentum_factor=0, loss_function="lms", optimizer="sgd", isInput=False):
        # Set input size and return.
        if isInput:
            self.inputSize = layer_size
            return
        else:
            newLayer = Layer()
            if len(self.layers) == 0:
                # First layer, so use inputSize as input size value.
                newLayer.setParams(input_size=self.inputSize, size=layer_size)
            else:
                newLayer.setParams(input_size=self.layers[-1].size, size=layer_size)
            self.layers.append(newLayer)

    '''
    Using the training set of data, run through each data example, and backpropogate the errors.

    train_set: (m x k) numpy array with m examples of dimension k
    label_set: (m x o) numpy array with m outputs of dimension o
    '''
    def train(self, train_set, label_set, validation_data_set=None, validation_label_set=None, epochs=1):
        accuracyList = []

        for epoch in range(0, epochs):
            randomIndecies = np.random.random_integers(0, train_set.shape[0] - 1, 2000)#train_set.shape[0])
            mTrainSet = train_set[randomIndecies]
            mLabelSet = label_set[randomIndecies]
            actualLabels = oneHotEncodingToLabels(mLabelSet)

            print("Epoch: " + str(epoch))
            # Generate the trial wise error and add it to a list to return every 10th epoch.
            if epoch % 10 == 0:
                predictedLabels = self.predictAll(mTrainSet)
                predictedLabels = oneHotEncodingToLabels(predictedLabels)
                accuracy = calculateAccuracy(predictedLabels, actualLabels)
                if validation_label_set is None:
                    accuracyList.append(1.0 - accuracy)
                else:
                    # Calculate the accuracy of the network on the validation set.
                    predictedLabels = self.predictAll(validation_data_set)
                    predictedLabels = oneHotEncodingToLabels(predictedLabels)
                    validationAccuracy = calculateAccuracy(predictedLabels, validation_label_set)
                    accuracyList.append({'train_accuracy': accuracy, 'test_accuracy': validationAccuracy})
                    print("Accuracy: ")
                    print(accuracyList)


            for train_index in range(0, len(mTrainSet[:])):
                '''
                1) Predict current example.
                2) Calculate error for last level.
                3) For each level before last level:
                    (In vector form)
                    A) levelError = (levelOutput)*(1 - levelOutput)*(Sum of next level's weights * next level's errors)
                    B) Calculate delta wji (With momentum)
                    C) Update weight wji as wji(n) = wji(n-1) + delta wji
                '''
                prediction = self.predict(mTrainSet[train_index])
                # Calculate error term for every output neuron. Dims (1 x o)
                error = np.array([(mLabelSet[train_index] - prediction)])#np.array([(prediction)*(1.0 - prediction)*(label_set[train_index] - prediction)])
                errorMat = error
                # Backpropogate errors for each layer.
                for layer_index in range(len(self.layers) - 1, -1, -1):
                    error = self.layers[layer_index].backpropogateErrors(errorMat)
                    errorMat = self.layers[layer_index].generateErrorMat(error)
                # Continue to next example.

        print("Finished Training.")
        return accuracyList

    '''
    Generate an output prediction from the NN model using the training data and current network weights.
    The data should be an np array with dims (1 x k), where k is the number of inputs specified in the input
    layer when the model was being built.
    '''
    def predict(self, data):
        '''
        1) Set current data numpy matrix.
        2) For each layer in the net:
            Expand inputs so that they can be passed into the layer.
            Pass inputs to the layer.
            The layer will apply a dot product and activation function to generate the outputs, and store the output vector.
        3) Return the final output vector, and threshold if necessary.
        '''

        # The input data is not of the proper size, so error out.
        if len(data) < self.inputSize:
            raise AttributeError("Input data size not equal to weight input size.")

        # Initialize the pLayerOutput to the input data_set for the looped dot product code.
        pLayerOutput = data
        nLayerOutput = None
        # For each layer, compute the dot product of the pLayerOutput and the input weights of each neuron of the layer.
        for cLayer in self.layers:
            nLayerOutput = cLayer.generateOutput(pLayerOutput)
            pLayerOutput = nLayerOutput

        return pLayerOutput

    def predictAll(self, data):
        labels = []
        for entry in data[:]:
            labels.append(self.predict(entry))

        return np.array(labels)

'''
Convert a one-hot encoding of the classes to a numerical number from 0 to the number of classes - 1.
'''
def oneHotEncodingToLabels(labels):
    newLabels = np.zeros((labels.shape[0], 1))
    for index in range(0, labels.shape[0]):
        argMax = np.argmax(labels[index])
        newLabels[index] = np.array([argMax])
    return newLabels

'''
Convert a set of labels into a one-hot encoding with smallest number in bit position 0, and largest
number in the last bit position.
'''
def labelToOneHotEncoding(labels):
    uniqueValues = sorted(list(set(labels)))
    newLabels = np.zeros((labels.shape[0], len(uniqueValues)))
    for label_index in range(0, len(labels[:])):
        value_index = uniqueValues.index(labels[label_index])
        # Flip the bit corresponding to the position of the element. Values are encoded in descending order.
        # Aka, smalles value is bit in first position, and largest value is bit in last position.
        newLabels[label_index, value_index] = 1
    return newLabels

'''
Given a predicted and actual set of labels, determine the accuracy of the list.
'''
def calculateAccuracy(ypredicted, yactual):
    metrics = {}
    metrics["tp"] = 0
    metrics["tn"] = 0
    metrics["fp"] = 0
    metrics["fn"] = 0
    for i in range(0, len(yactual)):
        if ypredicted[i] == yactual[i]:
            metrics["tp"] += 1

    accuracy = (metrics["tp"]) / (len(ypredicted))

    return accuracy

'''

'''
class Layer:

    def __init__(self):
        # Each column represents the weights of a neuron. Column 0 are the input weights of neuron 0. Column 1 are the input weights
        # of neuron 1 and so on.
        self.input_weights = None
        self.input_weight_deltas = None
        self.output = None
        self.input = None
        self.size = 0
        self.momentum = 0
        self.learning_rate = 0.1

    def setParams(self, input_size, size, momemtum=0, learning_rate=0.1, activation_function='sigmoid'):
        # Weight matrix. (# weights or inputs, # neurons). (k x H).
        self.size = size
        posNegArr = [1, -1]
        initRandWeights = np.random.rand(input_size, size)
        for row in range(0, initRandWeights.shape[0]):
            for col in range(0, initRandWeights.shape[1]):
                initRandWeights[row, col] = initRandWeights[row, col]*posNegArr[np.random.randint(0, 2)]*0.05

        self.input_weights = initRandWeights#np.random.rand(input_size, size)
        self.input_weight_deltas = np.zeros((input_size, size))
        self.output = np.zeros((size, 1))
        self.momentum = momemtum
        self.learning_rate = learning_rate
        # TODO: Potentially allow an activation function to be passed, or set using the activation function param.

    '''
    1) Do dot product of input (1xk) and weight matrix (kxH)
    2) Store output as copy in layer output ndarray.
    3) Return output in the form or (1xH)
    '''
    def generateOutput(self, input):
        # Set the input for the backprop to use later. (1 x k) vector.
        # TODO: Make sure input isn't being changed by any other func.
        self.input = np.array([input])
        # Generate output.
        output = np.dot(input, self.input_weights)
        # Apply sigmoid function, and reset values of output so that mem doesn't have to be allocated.
        for col in range(0, len(output)):
            # Set the output with dim (1 x H) values to the layer's output var with dims (H X 1)
            output[col] = 1.0 / (1.0 + math.exp(-1.0 * output[col]))
            # Set the output to the output term that will be used later in backprop.
            self.output[col] = output[col]*(1.0 - output[col])

        return output

    '''
    Return the input weight vectors for this layer, stacked horizontally.
    '''
    def getInputWeights(self):
        return self.input_weights

    '''
    Return the input weight vector for a neuron in this layer, from neuron 0
    through neuron (layer_size - 1).
    '''
    def getInputWeightsForNeuron(self, neuron):
        pass

    '''
    The error matrix is the transpose of the input matrix, with each column multiplied by the
    error term for that output neuron. It has dims (1 x H)
    '''
    def backpropogateErrors(self, errorMat):
        # Calculate the new delta's for this layer. It should be (H x 1) * (1 x H).T
        # Note, these intermediate numpy arrays are necessary for the transpose operations to work.
        # mErrorMat = errorMat.reshape((len(errorMat), 1))
        #mInput = self.input.reshape((len(self.input), 1))
        error = self.output * errorMat.T
        # Calculate the new weight deltas along with momentum. Make input of form (k x 1) and error of form (1 x H)
        self.input_weight_deltas = (self.learning_rate * np.dot(self.input.T, error.T)) + (self.momentum*self.input_weight_deltas)
        # Update the weights. input_weight_deltas should still be a (k x H) weight matrix.
        self.input_weights = self.input_weights + self.input_weight_deltas

        # Return the error.
        return error.T

    def generateErrorMat(self, error):
        # Calculate the errorMat to use for backpropagation.
        errorMat = np.dot(error, self.input_weights.T)
        return errorMat
'''
Save the model in the specified file path as a pickled object.
'''
def save(model, file_path):
    f = open(file_path, 'wb')
    pickle.dump(model, f)
    f.close()

'''
Return the model saved in the specified pickled file.
'''
def load(file_path):
    f = open(file_path, 'rb')
    model = pickle.load(f)
    f.close()
    return model


