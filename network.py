import math
import random

from matrix import Matrix

sign = lambda a: float((a > 0) - (a < 0))


def sigmoidLayer(layer):
    vector = []
    for i in layer:
        vector.append(sigmoidFunction(i[0]))
    return vector


def sigmoidFunction(z):
    maxSigInput = 10
    if -maxSigInput < z < maxSigInput:
        S = 1 / (1 + math.pow(math.e, -z))
    else:
        S = sign(z)
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
        self.errors = []
        self.framework = framework
        self.layerMatrix = makeWeights(framework)
        self.valueMatrix = makeValueMatrix(framework)
        self.biasMatrix = makeValueMatrix(framework)
        self.expected = [0.0, 0.0]

    def setInput(self, input):
        if len(input[:-1]) == self.valueMatrix.getColSize():
            self.valueMatrix.setRow(0, input[:-1])
            self.expected = input[len(input) - 1]

    def feedForward(self):
        for layer in range(len(self.layerMatrix)):
            vector = Matrix(self.valueMatrix.getRow(layer))
            bias = Matrix(self.biasMatrix.getRow(layer))
            wMatrix = self.layerMatrix[layer]
            vector.multiply(wMatrix)
            vector.add(bias)
            self.valueMatrix.setRow(layer + 1, sigmoidLayer(vector.matrix))

    def errorSquared(self):
        errorMatrix = makeValueMatrix(self.framework)
        prevError = self.expected
        for r in range(len(errorMatrix.matrix), 0, -1):
            length = len(errorMatrix.getRow(r - 1))
            if r == len(errorMatrix.matrix):
                error = [math.pow(self.valueMatrix.at([r - 1, i]) - prevError[i], 2) for i in range(length)] if isinstance(prevError, list) else math.pow(self.valueMatrix.at([r - 1, 0]) - prevError, 2)
            else:
                weightMatrix = self.layerMatrix[r - 1]
                weightMatrix.transpose()
                print(prevError, "prev")
                error = weightMatrix.scale(prevError) if isinstance(prevError, float) else weightMatrix.dotProduct(prevError)
            errorMatrix.setRow(r - 1, error)
            prevError = error
        self.errors.append(errorMatrix.getRow(errorMatrix.getRowSize() - 1))
        return self.deltaWeight(errorMatrix)

    def deltaWeight(self, errorMatrix):
        dWeightMatrix = makeWeights(self.framework)
        for r in range(1, len(dWeightMatrix)):
            errorRow = Matrix(errorMatrix.getRow(r)).scale(learningRate)
            errorRow = Matrix(errorRow)
            valueRow = Matrix(self.valueMatrix.getRow(r - 1))
            sigmoidInverse = self.valueMatrix.sigmoidInverse(r)
            elementWise = Matrix(errorRow.multiply(sigmoidInverse))
            valueRow.transpose()
            print(elementWise.matrix, valueRow.matrix, "element value")
            deltaWeights = elementWise.vectorVectorTranspose(valueRow)
            dWeightMatrix[r] = deltaWeights
        return dWeightMatrix

    def updateWeights(self, dWeightMatrix):
        for l in range(len(dWeightMatrix)):
            self.layerMatrix[l].subtract(dWeightMatrix[l])

    def train(self, inputs):
        for i in inputs:
            self.setInput(i)
            self.feedForward()
            print("feedForwardDone")
            self.updateWeights(self.errorSquared())
        print(self.errors)
