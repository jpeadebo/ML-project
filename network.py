import math
import numpy as np
import random


def sigmoidLayer(layer):
    vector = []
    for i in layer:
        vector.append(sigmoidFunction(i))
    return vector


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


def sigmoidInverse(layer):
    return np.multiply(layer, np.subtract(np.ones(len(layer)), layer))


class Network:
    learningRate = .1

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

    # feed forward works
    def feedForward(self):
        for l in range(len(self.valueMatrix[:-1])):
            dot = np.dot(np.array(self.layerWeightMatrix[l]), self.valueMatrix[l])
            self.valueMatrix[l + 1] = np.array(sigmoidLayer(dot))
        self.valueMatrix[len(self.valueMatrix) - 1] = (self.valueMatrix[len(self.valueMatrix) - 1] > .5) * 1

    errorList = []

    def calcOutputError(self):
        # need to make this scalable to any size output!
        self.outputerror = (self.valueMatrix[len(self.valueMatrix) - 1][0] - self.expected)
        #self.errorList.append(self.outputerror)
        if abs(self.outputerror) == 0:
            self.numCorrect += 1
        if self.expected == 0:
            self.numZeros += 1
        self.errorMatrix[len(self.errorMatrix) - 1][0] = self.outputerror

    def gradient(self):
        dWeightList = []
        for layer in range(len(self.valueMatrix) - 1, 0, -1):
            # since we are setting the output to 0 or 1 we need to do a proper sigmoid inverse rather then the fake one
            grad = realSigmoidInverse(self.valueMatrix[layer]) if layer == len(self.valueMatrix) - 1 else sigmoidLayer(self.valueMatrix[layer])
            # use this if output is sigmoided
            #grad = sigmoidLayer(self.valueMatrix[layer])
            grad = np.multiply(self.errorMatrix[layer], grad)
            grad = np.multiply(grad, self.learningRate)

            layerBackTrans = np.matrix(self.valueMatrix[layer - 1]).transpose()

            dWeight = np.dot(layerBackTrans, grad)

            if layer != len(self.valueMatrix) - 1:
                dWeight = np.transpose(dWeight)

            dWeightList.append(dWeight)

            layerBackError = dWeight.transpose().dot(self.errorMatrix[layer][0])

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

    def sumLayerToNode(self, weightLayer, valueLayer):
        sum = 0
        print(weightLayer, valueLayer, "here")
        if len(weightLayer) == len(valueLayer):
            for i in range(len(weightLayer)):
                sum += weightLayer[i] * valueLayer[i]
        else:
            raise Exception("weightlayer and valuerLayer arent equal", len(weightLayer), len(valueLayer))
        return sum

    def checkFeedForward(self):
        layerWeight = self.layerWeightMatrix
        value = self.valueMatrix
        error = self.errorMatrix
        cValue = []
        cValue.append(value[0])
        for r in range(len(value[:-1])):
            rV = []
            for c in range(len(value[r + 1])):
                vVect = value[r]
                wL = layerWeight[r][c]
                rV.append(sigmoidFunction(self.sumLayerToNode(vVect, wL)))
            cValue.append(rV)

        toleranceFF = .25
        for r in range(len(value)):
            for c in range(len(value[r])):
                if not math.isclose(cValue[r][c], value[r][c], abs_tol=toleranceFF):
                    print("----------------error--------------")
                    print(cValue, "||", value, "cvalue, value")
                    print(layerWeight, "layer Weight")
                    raise Exception("feedForward Didnt work at r, c", r, c, cValue[r][c], value[r][c])

    def checkError(self):
        layerWeight = self.layerWeightMatrix
        value = self.valueMatrix
        error = self.errorMatrix
        cError = []
        eV = []

        for o in range(len(error[len(error) - 1])):
            eV.append(pow(self.expected - (value[len(error) - 1][o] * outputScale), 2))
        cError.append(eV)

        for r in range(len(value[:-1]), 0, -1):
            rV = []
            for c in range(len(value[r - 1])):
                sum = 0
                for n in range(len(value[r])):
                    sum += error[r][n] * layerWeight[r - 1][n][c]
                rV.append(sum)
            cError.append(rV)

        cError.reverse()

        # compare cError and error
        toleranceE = .25
        for r in range(len(value)):
            for c in range(len(value[r])):
                if not math.isclose(cError[r][c], error[r][c], abs_tol=toleranceE):
                    print("----------------error--------------")
                    print(cError, "||", error, "cvalue, value")
                    print(layerWeight, "layer Weight")
                    raise Exception("error Didnt work at r, c", r, c, cError[r][c], error[r][c])


def testXor():
    inputs = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0],[1, 1, 1, 1]]
    #inputs = [[0,0,0],[0,1,1], [1,0,1],[1,1,0]]
    hiddenLayer1Length = 20
    hiddenLayer2Length = 10
    numOutputs = 1

    framework = [len(inputs[0]) - 1, hiddenLayer1Length, hiddenLayer2Length, numOutputs]
    network = Network(framework)

    network.train(inputs)
    network.test(inputs)
