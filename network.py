import math
import numpy as np
import random


# runs a layer of value matrix over the sigmoid function to map it between -1 and 1
def sigmoidLayer(layer):
    vector = []
    for i in layer:
        vector.append(sigmoidFunction(i))
    return vector


# limits the max input -100 to 100 due to rounding errors??? need more understanding again on why i did this but there
# was a reason
def sigmoidFunction(z):
    maxSigInput = 100
    if -math.inf < z < -maxSigInput:
        z = -maxSigInput
    elif maxSigInput < z < math.inf:
        z = maxSigInput
    S = 1 / (1 + math.pow(math.e, -z))

    return S


outputScale = 1


def realSigmoidInverse(i):
    sigInverse = sigmoidFunction(i[0])
    return sigInverse * (1 - sigInverse)


# calc
def sigmoidInverse(layer):
    return np.multiply(layer, np.subtract(np.ones(len(layer)), layer))


class Network:
    learningRate = .05

    def __init__(self, framework):
        self.numCorrect = 0
        self.numZeros = 0
        self.expected = np.zeros(framework[len(framework) - 1])
        self.outputerror = np.zeros(framework[len(framework) - 1])
        self.layerWeightMatrix = [np.random.uniform(-1, 1, size=(framework[i], framework[i - 1])) for i in
                                  range(1, len(framework))]
        self.baisMatrix = [np.zeros(framework[i]) for i in range(1, len(framework))]
        self.valueMatrix = [np.zeros(framework[i]) for i in range(len(framework))]
        self.errorMatrix = [np.zeros(framework[i]) for i in range(len(framework))]

    def setInputs(self, inputs):
        if len(inputs[:-1]) == len(self.valueMatrix[0]):
            self.valueMatrix[0] = np.array(inputs[:-1])
            self.expected = np.array(inputs[len(inputs) - 1])
        else:
            raise Exception("failed to set inputs", inputs)
        print(self.valueMatrix)

    # feed forward works
    def feedForward(self):
        for l in range(len(self.valueMatrix[:-1])):
            # find next layer values by multiplying prev value layer by current weight matrix layer
            dot = np.dot(np.array(self.layerWeightMatrix[l]), self.valueMatrix[l])
            # apply sigmoid function to new layer to put it in bounds for activation
            self.valueMatrix[l + 1] = np.array(sigmoidLayer(dot))
        # if output is greater than 50% set to true, if less than set false
        self.valueMatrix[len(self.valueMatrix) - 1] = (self.valueMatrix[len(self.valueMatrix) - 1] > .5) * 1

    errorList = []

    def calcOutputError(self):
        # need to make this scalable to any size output layer!
        # finds the diffrence between the networks guess and the correct answer
        self.outputerror = (self.valueMatrix[len(self.valueMatrix) - 1][0] - self.expected)

        if self.outputerror == 0:
            self.numCorrect += 1
        if self.expected == 0:
            self.numZeros += 1
        self.errorMatrix[len(self.errorMatrix) - 1][0] = self.outputerror

    def gradient(self):
        dWeightList = []
        # once we have the error of the output layer we need to back calculate the error of the network to find the
        # gradient to find the optimal direction of movement for the network for that specific question
        for layer in range(len(self.valueMatrix) - 1, 0, -1):
            # since we are setting the output to 0 or 1 we need to do a proper sigmoid inverse rather then the fake one
            grad = realSigmoidInverse(self.valueMatrix[layer]) if layer == len(self.valueMatrix) - 1 else sigmoidLayer(
                self.valueMatrix[layer])
            # use this if output is sigmoided
            # grad = sigmoidLayer(self.valueMatrix[layer])

            # find the amount of contribution each node had to the prev layers error
            grad = np.multiply(self.errorMatrix[layer], grad)
            # scale the distance travled by the gradient so we dont overstep the problem
            grad = np.multiply(grad, self.learningRate)

            layerBackTrans = np.matrix(self.valueMatrix[layer - 1]).transpose()

            # this is the found change in weights we need to get closer to the correct answer
            dWeight = np.dot(layerBackTrans, grad)

            if layer != len(self.valueMatrix) - 1:
                dWeight = np.transpose(dWeight)

            dWeightList.append(dWeight)

            layerBackError = dWeight.transpose().dot(self.errorMatrix[layer][0])

            # weird rotation stuff cuz i dont understand numpy
            if layer == len(self.valueMatrix) - 1:
                layerBackError = np.transpose(layerBackError)

            self.errorMatrix[layer - 1] = np.array(layerBackError)
        dWeightList.reverse()
        self.updateWeights(dWeightList)

    def updateWeights(self, dWeight):
        if len(dWeight) == len(self.layerWeightMatrix) and len(dWeight[0]) == len(self.layerWeightMatrix[0]):
            for layers in range(len(dWeight)):
                self.layerWeightMatrix[layers] = np.subtract(self.layerWeightMatrix[layers], dWeight[layers])
        else:
            raise Exception("bad update weight")

    epochSize = 50000
    threshold = .95

    def train(self, inputs):
        percentError = 0
        while percentError < self.threshold:
            print("----------------------NEW----------------")
            for i in range(self.epochSize):
                rand = random.randint(0, len(inputs) - 1)
                self.setInputs(inputs[rand])
                self.feedForward()
                self.calcOutputError()
                self.gradient()
            percentError = self.numCorrect / self.epochSize
            print(percentError, self.numZeros / self.epochSize, "percent correct, percent zeros")
            if percentError > self.threshold - .1:
                print("update")
                self.learningRate = .02
            self.numCorrect = 0
            self.numZeros = 0
        return percentError

    def test(self, inputs):
        for i in range(40):
            rand = random.randint(0, len(inputs) - 1)
            input = inputs[rand]
            self.setInputs(input)
            self.feedForward()
            self.calcOutputError()
            print("Guess:", self.valueMatrix[len(self.valueMatrix) - 1], "Expected:", self.expected)


def testXor():
    inputs = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0],
              [1, 1, 1, 1]]
    #inputs = [[0,0,0],[0,1,1], [1,0,1],[1,1,0]]
    hiddenLayer1Length = 20
    hiddenLayer2Length = 10
    numOutputs = 1

    framework = [len(inputs[0]) - numOutputs, hiddenLayer1Length, hiddenLayer2Length, numOutputs]
    network = Network(framework)

    network.train(inputs)
    network.test(inputs)


testXor()
