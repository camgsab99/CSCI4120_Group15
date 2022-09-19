from random import shuffle
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

# Find best K value for the data
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,20))

visualizer.fit(X)
visualizer.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

# Confusion Matrix
mat = confusion_matrix(y_true, y_kmeans)
plt.xlabel('y ground truth')
plt.ylabel('predicted y')

# Accuracy Score
score = accuracy_score(y_true, y_kmeans)
print('Accuracy Score: ' + repr(score))