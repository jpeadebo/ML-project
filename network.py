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
learningRate = .1


def sigmoidInverse(layer):
    return np.multiply(layer, np.subtract(np.ones(len(layer)), layer))


class Network:

    def __init__(self, framework):
        self.expected = np.zeros(framework[len(framework) - 1])
        self.outputerror = np.zeros(framework[len(framework) - 1])
        self.layerWeightMatrix = [np.random.rand(framework[i], framework[i - 1]) for i in range(1, len(framework))]
        self.baisMatrix = [np.zeros(framework[i]) for i in range(len(framework))]
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
            self.valueMatrix[l + 1] = np.array(sigmoidLayer(np.dot(self.layerWeightMatrix[l], self.valueMatrix[l])))

    def calcError(self):
        for layer in range(len(self.valueMatrix), 0, -1):
            if layer == len(self.valueMatrix):
                self.outputerror = [(self.valueMatrix[layer - 1][0] * outputScale) - self.expected]
                self.errorMatrix[layer - 1] = np.power(self.outputerror, 1)
            else:
                self.errorMatrix[layer - 1] = np.dot(self.errorMatrix[layer], self.layerWeightMatrix[layer - 1])

    def gradient(self):
        dWeightList = []
        for layer in range(len(self.valueMatrix) - 1, 0, -1):
            grad = sigmoidInverse(self.valueMatrix[layer])
            grad = np.multiply(self.errorMatrix[layer], grad)
            grad = np.multiply(grad, learningRate)

            dWeight = grad[:, np.newaxis] * self.valueMatrix[layer - 1][np.newaxis, :]
            dWeightList.append(dWeight)
        dWeightList.reverse()

        self.updateWeights(dWeightList)

    def updateWeights(self, dWeight):
        if len(dWeight) == len(self.layerWeightMatrix) and len(dWeight[0]) == len(self.layerWeightMatrix[0]):
            for layers in range(len(dWeight)):
                self.layerWeightMatrix[layers] = np.add(self.layerWeightMatrix[layers], dWeight[layers])
        else:
            print("bad update weight")

    def train(self, inputs):
        for input in inputs:
            #print("_____________________________NEW____________________________")
            self.setInputs(input)
            self.feedForward()
            self.checkFeedForward()
            self.calcError()
            self.checkError()
            self.gradient()
            print(abs(self.outputerror[0]), self.valueMatrix[len(self.valueMatrix)-1])

    def test(self, inputs):
        self.setInputs(inputs)
        self.feedForward()
        self.calcError()
        print(self.outputerror)

    def sumLayerToNode(self, weightLayer, valueLayer):
        sum = 0
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
            for c in range(len(value[r+1])):
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

        for o in range(len(error[len(error)-1])):
            eV.append((value[len(error)-1][o] * outputScale) - self.expected)
        cError.append(eV)

        for r in range(len(value[:-1]),0,-1):
            rV = []
            for c in range(len(value[r-1])):
                sum = 0
                for n in range(len(value[r])):
                    sum += error[r][n] * layerWeight[r-1][n][c]
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

