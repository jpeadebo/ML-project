import math
import random

from matrix import Matrix


def sigmoidFunction(z):
    if z < 100:
        z = 100
    elif -100 < z:
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
        matrix = Matrix()
        matrix.setMatrix([[random.uniform(-1, 1) for r in range(prevLayerLen)] for c in range(L)])
        layerMatrix.append(matrix)
        prevLayerLen = L
    return layerMatrix


# row represents layer col represents node
def makeValueMatrix(framework):
    valueMatrix = []
    for L in framework:
        vector = [0 for r in range(L)]
        valueMatrix.append(vector)
    vMatrix = Matrix()
    vMatrix.setMatrix(valueMatrix)
    return vMatrix


class Network:

    def __init__(self, framework):
        self.layerMatrix = makeWeights(framework)
        self.valueMatrix = makeValueMatrix(framework)
        self.feedForward()
        self.valueMatrix.printMatrix()

    def feedForward(self):
        for layer in range(len(self.layerMatrix)):
            vector = Matrix()
            valueVector = self.valueMatrix.getRow(layer)
            vector.vector(valueVector)
            vector.multiply(self.layerMatrix[layer])
            self.valueMatrix.setRow(layer + 1, vector.getRow(0))


inputLength = 5
hiddenLayer1Length = 4
outputLength = 2

framework = [inputLength, hiddenLayer1Length, outputLength]
net = Network(framework)
