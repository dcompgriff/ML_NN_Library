from functools import *
import numpy as np
import pickle


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

    Note: Potentially backpropogate errors (Perform dot products) during training in parallel. In parallel for each level.
    '''
    def train(self, train_set, label_set):
        pass

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

'''

'''
class Layer:

    def __init__(self):
        # Each column represents the weights of a neuron. Column 0 are the input weights of neuron 0. Column 1 are the input weights
        # of neuron 1 and so on.
        self.input_weights = None
        self.output = None
        self.size = 0
        self.momentum = 0
        self.learning_rate = 0.1

    def setParams(self, input_size, size, momemtum=0, learning_rate=0.1):
        # Weight matrix. (# weights or inputs, # neurons). (k x H).
        self.size = size
        self.input_weights = np.ones((input_size, size))
        self.output = np.zeros((size, 1))
        self.momentum = momemtum
        self.learning_rate = learning_rate

    '''
    1) Do dot product of input (1xk) and weight matrix (kxH)
    2) Store output as copy in layer output ndarray.
    3) Return output in the form or (1xH)
    '''
    def generateOutput(self, input):
        # Generate output
        output = np.dot(input, self.input_weights)
        # Reset values of output so that mem doesn't have to be allocated.
        for col in range(0, len(output)):
            # Set the output with dim (1 x H) values to the layer's output var with dims (H X 1)
            self.output[col]

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

    def backpropogateErrors(self, e):
        pass

'''
Save the model in the specified file path as a pickled object.
'''
def save(model, file_path):
    f = open(file_path, 'w+')
    pickle.dumps(model, file_path)
    f.close()

'''
Return the model saved in the specified pickled file.
'''
def load(file_path):
    f = open(file_path, 'r+')
    model = pickle.loads(f)
    f.close()
    return model


