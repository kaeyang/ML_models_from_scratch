import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path


def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))


def parse_spambase_data(filename):
    """ 
    Given a filename return X and Y numpy arrays
    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    X_list = []
    Y_list = []
    with open(filename, 'r') as f:
        data = f.read()
        data = data.split('\n')
        for row in data[:-1]:
            row = row.split(',')
            X_list.append(row[:-1])
            Y_list.append(row[-1])

    X = np.asarray(X_list, dtype='float')
    Y = np.asarray(Y_list, dtype='float')
    Y[Y == 0] = -1
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """
    Given an numpy matrix X, a array y and num_iter return trees and weights 
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = []
    N, _ = X.shape
    d = np.ones(N) / N  # initial weights

    for i in range(num_iter):
        tree_model = DecisionTreeClassifier(max_depth=max_depth)
        tree_model.fit(X, y, d)

        y_hat = tree_model.predict(X)
        err = np.sum(d * (y_hat != y))
        # added 0.000001 to prevent division by 0
        alpha = np.log((1-err+0.000001)/(err+0.000001))
        d = d * np.exp(alpha * (y_hat != y))
        d = d / np.sum(d)

        trees.append(tree_model)
        trees_weights.append(alpha)

    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ = X.shape
    y = np.zeros(N)

    for i in range(len(trees)):
        tree_model = trees[i]
        weights = trees_weights[i]
        y_hat = tree_model.predict(X)
        y += weights * y_hat

    return np.sign(y)
