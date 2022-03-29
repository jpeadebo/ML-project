import network
import csv

trainingData = 'C:\\Users\\andyd\\git\\ML-project\\datasets\\wineQualityTrainingData.txt'
testingData = 'C:\\Users\\andyd\\git\\ML-project\\datasets\\wineQualityTestingData.txt'
fileTrain = open(trainingData)
fileTest = open(testingData)


def understandData(inputs):
    wC = {"Private": 1, "Self-emp-not-inc": 2, "Self-emp-inc": 3, "Federal-gov": 4, "Local-gov": 5, "State-gov": 6,
          "Without-pay": 7, "Never-worked": 8, "?": 0}
    edu = {"Bachelors": 1, "Some-college": 2, "11th": 3, "HS-grad": 4, "Prof-school": 5, "Assoc-acdm": 6,
           "Assoc-voc": 7, "9th": 8, "7th-8th": 9, "12th": 10, "Masters": 11, "1st-4th": 12, "10th": 13,
           "Doctorate": 14, "5th-6th": 15, "Preschool": 16, "?": 0}
    mS = {"Married-civ-spouse": 1, "Divorced": 2, "Never-married": 3, "Separated": 4, "Widowed": 5,
          "Married-spouse-absent": 6, "Married-AF-spouse": 7, "?": 0}
    oc = {"Tech-support": 1, "Craft-repair": 2, "Other-service": 3, "Sales": 4, "Exec-managerial": 5,
          "Prof-specialty": 6, "Handlers-cleaners": 7, "Machine-op-inspct": 8, "Adm-clerical": 9, "Farming-fishing": 10,
          "Transport-moving": 11, "Priv-house-serv": 12, "Protective-serv": 13, "Armed-Forces": 14, "?": 0}
    rel = {"Wife": 1, "Own-child": 2, "Husband": 3, "Not-in-family": 4, "Other-relative": 5, "Unmarried": 6}
    rac = {"White": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4, "Black": 5, "?": 0}
    sex = {"Female": 1, "Male": 2, "?": 0}
    natCon = {"United-States": 1, "Cambodia": 2, "England": 2, "Puerto-Rico": 3, "Canada": 4, "Germany": 5,
              "Outlying-US(Guam-USVI-etc)": 6, "India": 7, "Japan": 8, "Greece": 9, "South": 10, "China": 11,
              "Cuba": 12, "Iran": 13, "Honduras": 14, "Philippines": 15, "Italy": 16, "Poland": 17, "Jamaica": 18,
              "Vietnam": 19, "Mexico": 20, "Portugal": 21, "Ireland": 22, "France": 23, "Dominican-Republic": 24,
              "Laos": 25, "Ecuador": 26, "Taiwan": 27, "Haiti": 28, "Columbia": 29, "Hungary": 30, "Guatemala": 31,
              "Nicaragua": 32, "Scotland": 33, "Thailand": 34, "Yugoslavia": 35, "El-Salvador": 36,
              "Trinadad&Tobago": 37, "Peru": 38, "Hong": 39, "Holand-Netherlands": 40, "?": 0}
    aboveBelow = {"<": 0, ">": 1, "?": -1}

    understoodData = []
    for input in inputs:
        dataSet = []
        # age
        dataSet.append(input[0])
        dataSet.append(wC[input[1]])
        # fnlwgt
        dataSet.append(input[2])
        dataSet.append(edu[input[3]])
        # education-num
        dataSet.append(input[4])
        dataSet.append(mS[input[5]])
        dataSet.append(oc[input[6]])
        dataSet.append(rel[input[7]])
        dataSet.append(rac[input[8]])
        dataSet.append(sex[input[9]])
        # capital-gain
        dataSet.append(input[10])
        # capital-gain
        dataSet.append(input[11])
        # hours-per-week
        dataSet.append(input[12])
        dataSet.append(natCon[input[13]])
        dataSet.append(aboveBelow[input[14]])
        understoodData.append(dataSet)

    understoodData = [[float(y) for y in x] for x in understoodData]
    return understoodData


csvreader = csv.reader(fileTrain)

inputVariableNames = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

#rows = understandData(rows)
rows = [[float(y) for y in x] for x in rows]

hiddenLayer1Length = 20
hiddenLayer2Length = 1
numOutputs = 1

framework = [len(rows[0]) - 1, hiddenLayer1Length, hiddenLayer2Length, numOutputs]
network = network.Network(framework)

network.train(rows)

