

'''
This class is used to build a neural network model.
'''
class Model:
    def __init__(self):
        self.layers = [Layer()]
        self.inputSize = 0

    '''
    Add a new fully connected layer to this model.
    '''
    def add(self, layer_size=1, learning_rate=0.1, momentum_factor=0, loss_function="lms", optimizer="sgd", isInput=False):
        # Set input size.
        if isInput:
            self.inputSize = layer_size
        



    '''
    Using the training set of data, run through each data example, and backpropogate the errors.

    Note: Potentially backpropogate errors (Perform dot products) during training in parallel. In parallel for each level.
    '''
    def train(self, train_set, label_set):
        pass

    '''
    Generate an output prediction from the NN model using the training data and current network weights.
    '''
    def predict(self, data_set):
        '''
        1) Set current data numpy matrix.
        2) For each layer in the net:
            Expand inputs so that they can be passed into the layer.
            Pass inputs to the layer.
            The layer will apply a dot product and activation function to generate the outputs, and store the output vector.
        3) Return the final output vector, and threshold if necessary.
        '''

        if len(data_set) < self.layers[0].getInputWeightsForNeuron(0):
            # The input data is not of the proper size, so error out.
            throw AttributeError("Input data size not equal to weight input size.")

        pass
'''

'''
class Layer:

    def __init__(self):
        self.input_weights = []
        self.output = 0
        self.size = 0
        self.momentum = 0
        self.learning_rate = 0.1

    def generateOutputs(self, inputs):
        pass

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