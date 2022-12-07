import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import SpectralClustering, KMeans
import numpy as np


df = pd.read_csv('Country-data.csv')
print(df.head())

print('Пустих значень:')
print(df.isnull().sum())
del df['country']

F, p = stats.f_oneway(df['exports'], df['imports'])
F_test = stats.f.ppf((1-0.05), 4, 15)
print("Однофакторний аналіз для експорту та імпорту")
print('F значення = % .2F, значення p% .9f'%(F, p))

print("Двофакторний аналіз")
fa = FactorAnalyzer()
fa.fit(df)

ev, v = fa.get_eigenvalues()
print(ev)
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.xlabel('Фактори')
plt.ylabel('значення')
plt.grid()
plt.show()

print(df.describe())

plt.subplot(3,3,1)
plt.title('child_mort')
plt.boxplot(df['child_mort'])

plt.subplot(3,3,2)
plt.title('exports')
plt.boxplot(df['exports'])

plt.subplot(3,3,3)
plt.title('health')
plt.boxplot(df['health'])

plt.subplot(3,3,4)
plt.title('imports')
plt.boxplot(df['imports'])

plt.subplot(3,3,5)
plt.title('income')
plt.boxplot(df['income'])

plt.subplot(3,3,6)
plt.title('inflation')
plt.boxplot(df['inflation'])

plt.subplot(3,3,7)
plt.title('life_expec')
plt.boxplot(df['life_expec'])

plt.subplot(3,3,8)
plt.title('total_fer')
plt.boxplot(df['total_fer'])

plt.subplot(3,3,9)
plt.title('gdpp')
plt.boxplot(df['gdpp'])
plt.show()

X = pd.DataFrame(df, columns=['exports', 'inflation'])

kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
X.to_csv('kmeans.csv')

plt.scatter(df['exports'], df['inflation'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

X, y_true = make_blobs(n_samples=len(df['exports']), centers=4, cluster_std=0.70, random_state=0)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)
plt.show()


def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

X, y = make_moons(len(df['exports']), noise=.05, random_state=0)
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

print(df.describe())