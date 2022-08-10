import network
import numpy as np
import csv

trainingData = 'C:\\Users\\andyd\\git\\ML-project\\datasets\\ionosphereTrainingData.txt'
testingData = 'C:\\Users\\andyd\\git\\ML-project\\datasets\\ionosphereTestingData.txt'
fileTrain = open(trainingData)
fileTest = open(testingData)


def isNumeric(att):
    try:
        float(att)
    except ValueError:
        return False
    return True


def numericalCatigoricalData(inputs):
    understoodData = []
    catigoricalData = []

    for att in inputs[0]:
        if isNumeric(att):
            catigoricalData.append(-1)
        else:
            catigoricalData.append([att])

    for input in inputs:
        for att in range(len(input)):
            if catigoricalData[att] != -1 and not (input[att] in catigoricalData[att]):
                catigoricalData[att].append(input[att])

    for input in inputs:
        row = []
        for att in range(len(input)):
            attribute = input[att]
            if isNumeric(attribute):
                row.append(attribute)
            else:
                row.append(catigoricalData[att].index(attribute))
        understoodData.append(row)

    return allNumericalData(understoodData)


def allNumericalData(inputs):
    understoodData = [[float(attributes) for attributes in i] for i in inputs]
    return understoodData


def autoScaleAllRows(rows):
    maxs = [1] * len(rows[0])
    for r in rows:
        for c in range(len(r)):
            maxs[c] = max(maxs[c], abs(r[c]))

    for r in rows:
        for c in range(len(r)):
            r[c] /= maxs[c]


csvreader = csv.reader(fileTrain)

inputVariableNames = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

rows = numericalCatigoricalData(rows)


# rows = [[float(y) for y in x] for x in rows]


hiddenLayer1Length = 40
hiddenLayer2Length = 20
numOutputs = 1

framework = [len(rows[0]) - 1, hiddenLayer1Length, hiddenLayer2Length, numOutputs]
network = network.Network(framework)

network.train(rows)

print("-----------------------testing-----------------------")

csvreader = csv.reader(fileTest)

inputVariableNames = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

rows = numericalCatigoricalData(rows)

network.test(rows)
