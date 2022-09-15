# Created by: Cameron Sabiston and Nick Terrell
# Created on: 09/15/2022

import math
import random
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# Loads the dataset and splits it into training and test sets
def load_dataset(file_name, split):
    training_set = []
    test_set = []
    df = pd.read_csv(file_name, header=None)
    array = df.to_numpy()
    random.shuffle(array)
    training_len = int(len(array) * split)
    training_set = array[:training_len]
    test_set = array[training_len:]
    return training_set, test_set


# Find the k nearest neighbors using KNeighborsClassifier and then find the accuracy
def get_prediction(training_set, test_set, k, attributes):
    model = KNeighborsClassifier(n_neighbors=k)
    training_data = np.array(training_set)
    training_x = training_data[:, 0:attributes]
    training_y = training_data[:, attributes]
    model.fit(training_x, training_y)

    test_data = np.array(test_set)
    test_x = test_data[:, 0:attributes]
    test_y = test_data[:, attributes]
    predictions = model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predictions)
    return predictions, accuracy


# Main function to run the program and create a line graph
def main():
    url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
    split = 0.67
    training_set, test_set = load_dataset(url, split)
    accuracies = {}
    for r in range(5):
        for k in range(1, 21):
            predictions, accuracy = get_prediction(
                training_set, test_set, k, 4)
            if k not in accuracies.keys():
                accuracies[k] = []
            accuracies[k].append(accuracy)
    average_accuracies = {}
    for a in accuracies.keys():
        average_accuracies[a] = sum(accuracies[a]) / len(accuracies[a])

    x = average_accuracies.keys()
    y = average_accuracies.values()
    plt.plot(x, y, color='green')
    xint = range(min(x), math.ceil(max(x)) + 1)
    plt.xticks(xint, color='red')
    plt.title('K Neighbors vs Average Accuracy', color='blue')
    plt.xlabel('K Neighbors', color='blue')
    plt.ylabel('Average Accuracy', color='blue')
    plt.show()


main()
