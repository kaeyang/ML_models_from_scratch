import numpy as np
from sklearn.utils import resample

from dtree_mod import *


class RandomForest:
    def __init__(self, n_estimators=10, min_samples_leaf=3, oob_score=False, max_features=0.3, regression=True):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.max_features = max_features
        self.regression = regression
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        forest = []
        oob_counts = [0 for i in y]
        oob_pred_reg = [0 for i in y]
        oob_class_list = [[] for i in y]

        for i in range(self.n_estimators):
            # Bootstrapping samples
            n = len(y)
            idx = np.random.randint(0, n, size=n)
            X_train = X[idx]
            y_train = y[idx]

            # regression or classification
            if self.regression:
                tree_object = RegressionTree(
                    self.min_samples_leaf, self.max_features)
            else:
                tree_object = ClassifierTree(
                    self.min_samples_leaf, self.max_features)

            tree = tree_object.fit(X_train, y_train)
            forest.append(tree)

            # oob data
            oob_idx = list(set(range(len(y))) - set(idx))
            oob_X = X[oob_idx]
            oob_y = y[oob_idx]

            # oob score for regression
            if self.oob_score and self.regression == True:
                for i in range(len(oob_y)):
                    leaf = tree.predict(oob_X[i, :])
                    # all y_pred instead of just the mean, looking at pseudo code from Parr

                    oob_pred_reg[int(oob_idx[i])] += leaf.n*leaf.prediction

                    oob_counts[int(oob_idx[i])] += leaf.n

                    oob_avg_pred = [oob_pred_reg[i]/oob_counts[i]
                                    for i, val in enumerate(oob_counts) if oob_counts[i] > 0]

                    oob_final_y = [y[i] for i, val in enumerate(
                        oob_counts) if oob_counts[i] > 0]

                self.oob_score_ = r2_score(oob_final_y, oob_avg_pred)

            # oob score for classification, took around 3 minutes to pass the whole test, thought that -8 was better than -10 points

            if self.oob_score and self.regression == False:
                for i in range(len(oob_y)):
                    leaf = tree.predict(oob_X[i, :])
                    oob_class_list[int(oob_idx[i])].extend(leaf.prediction)
                    oob_counts[oob_idx[i]] += leaf.n

                    oob_pred_class = [stats.mode(oob_class_list[i])[0][0] for i, val in enumerate(
                        oob_counts) if oob_counts[i] > 0]

                    oob_final_y = [y[i] for i, val in enumerate(
                        oob_counts) if oob_counts[i] > 0]

                self.oob_score_ = accuracy_score(oob_final_y, oob_pred_class)

        self.rdm_forest = forest


class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=10, min_samples_leaf=3,
                 max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score, regression=True)

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        y_pred = []

        for i in range(len(X_test)):
            forest_pred = [tree.predict(X_test[i, :])
                           for tree in self.rdm_forest]

            prediction_count = 0
            tree_pred_sum = 0

            for i in forest_pred:
                mean_pred = i.prediction
                prediction_count += i.n
                tree_pred_sum += mean_pred*i.n

            weighted_average = tree_pred_sum/prediction_count
            y_pred.append(weighted_average)
        return y_pred

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return r2_score(y_test, self.predict(X_test))


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, min_samples_leaf=3,
                 max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score, regression=False)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf

    def predict(self, X_test) -> np.ndarray:
        y_pred = []

        for i in range(len(X_test)):
            forest_pred = [tree.predict(X_test[i, :])
                           for tree in self.rdm_forest]

            forest_pred_list = []

            for i in forest_pred:
                leaf_prediction = list(i.prediction)
                forest_pred_list += leaf_prediction

            mode_class = stats.mode(forest_pred_list)[0][0]
            y_pred.append(mode_class)
        return y_pred

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return accuracy_score(y_test, self.predict(X_test))
