import math
import random
import csv

trainingData = 'C:\\Users\\andyd\\Downloads\\winequality-red.csv'
# testingData = 'C:\\Users\\andyd\\Downloads\\testsData.txt'
fileTrain = open(trainingData)
# fileTest = open(testingData)

csvreader = csv.reader(fileTrain)

inputVariableNames = []
inputVariableNames = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

inputs = [[float() for y in x] for x in rows]


def sigmoidFunction(z):
    S = 1 / (1 + math.pow(math.e, -z))
    return S


outputScale = 10
learningRate = .1


class Network:

    def __init__(self, frameWork):
        self.expected = 0
        self.framework = frameWork
        self.numLayers = len(self.framework)
        self.layers = []
        self.createFramework()
        self.setWeightsRandom()

    def outputLayer(self):
        return self.layers[self.numLayers - 1]

    def displayOutput(self):
        return self.outputLayer() * outputScale

    def printWeights(self):
        for i in self.layers:
            for j in i:
                print(j.getWeights())
            print("\n")

    def printValues(self):
        for i in self.layers:
            for j in i:
                print(j.getValue())
            print("\n")

    def printOutputs(self):
        for i in self.outputLayer():
            print(i.getValue())
        print("\n")

    def printError(self):
        for i in self.outputLayer():
            print(abs(i.getError()))

    def createFramework(self):
        for i in self.framework:
            layer = []
            for j in range(0, i):
                neuron = Neuron()
                layer.append(neuron)
            self.layers.append(layer)

    def setInputs(self, inputs):
        for i in range(len(self.layers[0])):
            self.layers[0][i].setValue(inputs[i])
        self.expected = inputs[len(inputs) - 1]

    def setWeightsRandom(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                randomList = []
                for k in range(len(self.layers[(i if i == 0 else i - 1)])):
                    randomList.append(0 if i == 0 else random.uniform(-1, 1))
                self.layers[i][j].setWeights(randomList)

    def getLayerValues(self, level):
        layerValues = []
        for i in range(len(self.layers[level])):
            layerValues.append(self.layers[level][i].getValue())
        return layerValues

    def calcOutputs(self):
        for i in range(len(self.layers)):
            if i != 0:
                prevLayerValues = self.getLayerValues(i - 1)
                for j in range(len(self.layers[i])):
                    self.layers[i][j].calcValue(prevLayerValues)

    # end of network setup
    # start of backprop
    def calcError(self):
        prevLayer = self.outputLayer()
        for i in prevLayer:
            i.setError(self.expected - i.getValue() * 10)
        # layers[1:-1] means remove input and output layers from being changed
        for j in reversed(self.layers[1:-1]):
            for i in range(len(j)):
                nodeError = 0
                for l in prevLayer:
                    strength = l.getWeights()[i]
                    nodeError += strength * l.getError()
                j[i].setError(nodeError)
            prevLayer = j
        return self.layers[self.numLayers - 1][0].getError()

    def calcDeltaWeights(self):
        # element wise multiplication
        prevLayer = self.layers[0]
        dArrayWeights = []
        for j in self.layers[1:]:
            dLayerWeight = []
            for i in range(len(j)):
                sigmoidInverse = j[i].getValue() * (1 - j[i].getValue())
                dWeight = learningRate * j[i].getError() * sigmoidInverse
                dLayerWeight.append(dWeight)
            layerWeightCol = []
            for c in dLayerWeight:
                layerWeightRow = []
                for r in prevLayer:
                    layerWeightRow.append(c * r.getValue())
                layerWeightCol.append(layerWeightRow)
            dArrayWeights.append(layerWeightCol)
            prevLayer = j
        return dArrayWeights

    def applyDeltaWeights(self):
        dArrayWeights = self.calcDeltaWeights()
        for i in range(len(dArrayWeights)):
            for j in range(len(dArrayWeights[i])):
                self.layers[i + 1][j].updateWeights(dArrayWeights[i][j])


class Neuron:

    def __init__(self):
        # weights is the list of connecting values between the current neuron and the previous layer
        self.weights = []
        self.value = 0
        self.bias = 0
        self.error = 0

    def getWeights(self):
        return self.weights

    def getValue(self):
        return self.value

    def getBias(self):
        return self.bias

    def getError(self):
        return self.error

    def setWeights(self, w):
        self.weights = w

    def setValue(self, v):
        self.value = v

    def setBias(self, b):
        self.bias = b

    def setError(self, e):
        self.error = e

    def updateWeights(self, dWeight):
        updatedWeights = []
        currentWeights = self.getWeights()
        for i in range(len(currentWeights)):
            updatedWeights.append(currentWeights[i] + dWeight[i])
        self.setWeights(updatedWeights)

    def calcValue(self, prevLayerValues):
        summation = 0.0
        for i in range(len(prevLayerValues)):
            summation += self.getWeights()[i] * prevLayerValues[i]
        z = summation + self.getBias()
        self.setValue(sigmoidFunction(z))


hiddenLayer1Length = 8
hiddenLayer2Length = 4
numOutputs = 1

n = Network([len(inputVariableNames), hiddenLayer1Length, hiddenLayer2Length, numOutputs])

for i in inputs:
    if (len(i) != 0):
        n.setInputs(i)
        n.calcOutputs()
        n.printError()
        n.calcError()
        n.applyDeltaWeights()



