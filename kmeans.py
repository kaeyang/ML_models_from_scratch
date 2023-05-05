import numpy as np
import random


def kmeans(X: np.ndarray, k: int, max_iter=30, tolerance=1e-2):
    # Randomly select starting k centroids
    centroids = X[random.sample(range(len(X)), k)]

    it = 0
    while it < 30:
        # assign data point to nearest centroid
        labels = np.argmin(
            np.sqrt(np.sum((X - centroids[:, np.newaxis])**2, axis=2)), axis=0)

        # update centroid with new clusters
        new_centroids = np.empty_like(centroids)
        for i in range(k):
            new_centroids[i] = np.mean(X[labels == i], axis=0)

        # check if centroid change is within tolerance
        if np.allclose(centroids, new_centroids, rtol=0, atol=tolerance):
            break

        centroids = new_centroids
        it += 1

    return centroids, labels


def kmeans_pp(X: np.ndarray, k: int, max_iter=30, tolerance=1e-2):
    # randomly select first centroid
    centroids = [random.choice(X)]

    # Select remaining k-1 centroids
    for i in range(k-1):
        # distance of every other point from first centroid
        distances = np.array([min([np.linalg.norm(point - c)
                                   for c in centroids]) for point in X])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        new_centroid_index = np.where(
            cumulative_probabilities >= random.random())[0][0]
        centroids.append(X[new_centroid_index])

    centroids = np.array(centroids)

    # perform k-means algorithm with the select centroids
    it = 0
    while it < max_iter:
        # assign data point to nearest centroid
        labels = np.argmin(
            np.sqrt(np.sum((X - centroids[:, np.newaxis])**2, axis=2)), axis=0)

        # update centroid with new clusters
        new_centroids = np.empty_like(centroids)
        for i in range(k):
            new_centroids[i] = np.mean(X[labels == i], axis=0)

        # check if centroid change is within tolerance
        if np.allclose(centroids, new_centroids, rtol=0, atol=tolerance):
            break

        centroids = new_centroids
        it += 1

    return centroids, labels

