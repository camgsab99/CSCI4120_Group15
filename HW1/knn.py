# Created by: Cameron Sabiston and Nick Terrell
# Created on: 09/15/2022

from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier


def main():

    data = pd.read_csv(
        'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data', header=None)
    X = data.iloc[:, :4]
    y = data.iloc[:, 4]
    avg_test_accuracies = []
    for k in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=k)
        cross_val = cross_validate(
            estimator=model, X=X, y=y, scoring='accuracy', cv=5)
        avg_test_accuracy = cross_val['test_score'].mean()
        avg_test_accuracies.append(avg_test_accuracy)

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')
    k_values = [k for k in range(1, 21)]
    plt.plot(k_values, avg_test_accuracies, color='m')
    plt.xticks(k_values)
    plt.xlabel('k')
    plt.ylabel('Average Test Accuracy')
    plt.title('Average Test Accuracy vs. k')
    plt.show()


main()
