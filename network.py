import math
import random

from matrix import Matrix


def sigmoidLayer(layer):
    vector = []
    for i in layer:
        vector.append(sigmoidFunction(i[0]))
    return vector


def sigmoidFunction(z):
    if 0 < z < 100:
        z = 100
    elif -100 < z < 0:
        z = -100
    S = 1 / (1 + math.pow(math.e, -z))
    return S


outputScale = 1
learningRate = .1


# rows represent layers and cols represent which prev node its connected to
def makeWeights(framework):
    prevLayerLen = framework[0]
    # for each layer in the framework excluding the input layer
    layerMatrix = []
    for L in framework[1:]:
        matrix = Matrix([[random.uniform(-1, 1) for r in range(prevLayerLen)] for c in range(L)])
        layerMatrix.append(matrix)
        prevLayerLen = L
    return layerMatrix


# row represents layer col represents node
def makeValueMatrix(framework):
    valueMatrix = []
    for L in framework:
        vector = [0 for r in range(L)]
        valueMatrix.append(vector)
    vMatrix = Matrix(valueMatrix)
    return vMatrix


class Network:

    def __init__(self, framework):
        self.layerMatrix = makeWeights(framework)
        self.valueMatrix = makeValueMatrix(framework)
        self.expected = 0

    def setInput(self, input):
        if len(input[:-1]) == self.valueMatrix.getColSize():
            self.valueMatrix.setRow(0, input[:-1])
            self.expected = input[len(input) - 1]

    def feedForward(self):
        for layer in range(len(self.layerMatrix)):
            vector = Matrix(self.valueMatrix.getRow(layer))
            wMatrix = self.layerMatrix[layer]
            vector.multiply(wMatrix)
            self.valueMatrix.setRow(layer + 1, sigmoidLayer(vector.matrix))


inputLength = 5
hiddenLayer1Length = 4
outputLength = 2

framework = [inputLength, hiddenLayer1Length, outputLength]
net = Network(framework)
net.setInput([1, 1, 1, 1, 1, 4])
net.feedForward()
