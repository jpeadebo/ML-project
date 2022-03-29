import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

trainingData = 'C:\\Users\\andyd\\git\\ML-project\\datasets\\irisTrainingData.txt'
fileTrain = open(trainingData)

csvreader = csv.reader(fileTrain)

rows = []
for row in csvreader:
    rows.append(row)

iris = {}
for input in rows:
    try:
        iris[input[len(input) - 1]].append(input[:-1])
    except KeyError:
        iris[input[len(input) - 1]] = [input[:-1]]

sepalLength = []
sepalWidth = []
for f in iris:
    print(f)
    for r in iris[f]:
        sepalLength.append(r[0])
        sepalWidth.append(r[1])

plt.scatter(sepalLength, sepalWidth)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Scatter plot on Iris dataset')
plt.show()
