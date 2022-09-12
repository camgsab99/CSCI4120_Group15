import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def loadDataset(filename, split):
    trainingSet = []
    testSet = []
    df = pd.read_csv(filename, header=None)
    array = df.to_numpy()
    random.shuffle(array)
    training_len = int(len(array) * split)
    trainingSet = array[:training_len]
    testSet = array[training_len:]
    return trainingSet, testSet


def getPredictionSkLearn(trainingSet, testSet, k, attributes):
    model = KNeighborsClassifier(n_neighbors=k)
    trainingData = np.array(trainingSet)
    trainingX = trainingData[:, 0:attributes]
    trainingY = trainingData[:, attributes]
    model.fit(trainingX, trainingY)

    testData = np.array(testSet)
    testX = testData[:, 0:attributes]
    testY = testData[:, attributes]
    predictions = model.predict(testX)
    accuracy = metrics.accuracy_score(testY, predictions)
    return predictions, accuracy


def main():
    url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
    split = 0.67
    trainingSet, testSet = loadDataset(url, split)
    accuracies = {}
    for r in range(5):
        for k in range(1, 21):
            predictions, accuracy = getPredictionSkLearn(trainingSet, testSet, k, 4)
            if k not in accuracies.keys():
                accuracies[k] = []
            accuracies[k].append(accuracy)
    averageAccuracies = {}
    for a in accuracies.keys():
        averageAccuracies[a] = sum(accuracies[a]) / len(accuracies[a])

    x = averageAccuracies.keys()
    y = averageAccuracies.values()
    plt.plot(x, y)
    xint = range(min(x), math.ceil(max(x)) + 1)
    plt.xticks(xint)
    plt.title('K Neighbors vs Average Accuracy')
    plt.xlabel('K Neighbors')
    plt.ylabel('Average Accuracy')
    plt.show()

main()