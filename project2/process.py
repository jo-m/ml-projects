#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.covariance import EllipticEnvelope


import sklearn.cross_validation as skcv
import sklearn.metrics as skmet
from utils import *

cols = [
    "id",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7",
    "Y"
]

# X will have ID as the first column
def load_data(train=True):
    if train:
        fname = 'data/train.csv'
        names = cols
    else:
        fname = 'data/validate_and_test.csv'
        names = cols[:-1]

    data = pd.read_csv(fname,
                       index_col=None,
                       header=None,
                       names=names)

    if train:
        Y = data['Y'].as_matrix().astype(int)
        del data['Y']
    else:
        Y = None

    return data.as_matrix().astype(float), Y


# use clustering on all unsupervised data to produce one more feature
class ClusterTransform():
    def __init__(self, n_clusters=3, n_jobs=-1, thresh=1e-3, min_covar=1e-3, covariance_type='tied',
                 **kwaargs):
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.X_TEST, _ = load_data(False)

        xTrain = Xtrain[:, 1:]
        xTrain = DifferentTransforms().transform(xTrain)
        xTrain = Scaler.fit_transform(xTrain)
        cluster_means = np.array([xTrain[Ytrain == i].mean(axis=0)
                                  for i in xrange(n_clusters)])

        self.clusterizer = GMM(n_components=n_clusters, init_params='wc', params='wmc',
                               # allowing to adjust means helps to avoid overfitting
                               covariance_type=covariance_type, min_covar=min_covar, thresh=thresh)
        self.clusterizer.means_ = cluster_means

    def set_params(self, **params):
        self.clusterizer.set_params(**params)

    # use all x to train cluster
    def fit(self, X, _):
        xTest = DifferentTransforms().transform(self.X_TEST)
        xTest = xTest[:, 1:]
        xTest = Scaler.fit_transform(xTest)

        Xtotal = np.vstack((X, xTest))
        self.clusterizer.fit(Xtotal)
        return self

    # add the predicted cluster labels as a feature vector
    # binarize the labels
    def transform(self, X):
        xFeat = self.clusterizer.predict(X)
        xFeat = np.atleast_2d(xFeat).T
        xFeat = OneHotEncoder(sparse=False).fit_transform(xFeat)
        xRes = np.hstack((X, xFeat))
        return xRes

    def predict(self, X):
        return self.clusterizer.predict(X)

    # not sure why this is needed
    def score(self, X, Y):
        return score(self.predict(X), Y)

    def get_params(self, deep=True):
        return self.clusterizer.get_params(deep=deep)


class DifferentTransforms():
    def __init__(self, featureDel = 4, **kwargs):
        self.featureDel = featureDel

    def fit(self, X, Y):
        return self

    def transform(self, X):
        # delete useless frequency
        return np.delete(X, self.featureDel, 1)

    def get_params(self, **kwargs):
        return dict({'featureDel': self.featureDel})

    def set_params(self, **params):
        for key in params:
            if key == 'featureDel':
                self.featureDel = params[key]
                break

# delete outliers, boundaries were defined visually
def delOutliers(Xtrain, Ytrain):
    outliers = []
    bounds = [100, 4.5, 4, 3.8, 5, 6, 5, 7]
    xTrain = Scaler.fit_transform(Xtrain)
    for row in range(0, xTrain.shape[0]):
        for col in range(1, xTrain.shape[1]):  # zeros column ids
            if xTrain[row, col] > bounds[col]:
                outliers.append(row)
                break

    print Ytrain.shape
    print outliers
    Xtrain = np.delete(Xtrain, outliers, 0)
    Ytrain = np.delete(Ytrain, outliers, 0)

    print Ytrain.shape
    return Xtrain, Ytrain


def score(Ytruth, Ypred):
    Ytruth = Ytruth.ravel()
    Ypred = Ypred.ravel()
    if Ytruth.ndim != 1:
        raise Exception('Ytruth has invalid shape!')
    if Ypred.ndim != 1:
        raise Exception('Ypred has invalid shape!')

    sum = (Ytruth == Ypred).astype(float).sum().sum()
    return sum / np.product(Ytruth.shape)


def run_crossval(X, Y, model):
    scorefun = skmet.make_scorer(score)
    scores = skcv.cross_val_score(model, X[:, 1:], Y, scoring=scorefun, cv=4)
    print 'C-V score =', np.mean(scores), '+/-', np.std(scores)


def run_split(X, Y, model):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.8)
    Xtrain, Xtest = Xtrain[:, 1:], Xtest[:, 1:]
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    print "Split-score = %f" % score(Ypred, Ytest)


def write_Y(Y):
    if Y.shape[1] != 2:
        raise 'Y has invalid shape!'
    np.savetxt('results/Ypred.csv', Y,
               fmt='%d', delimiter=',', header='Id,Label', comments='')


def run_validate(Xtrain, Ytrain, model):
    model.fit(Xtrain[:, 1:], Ytrain)

    Xvalidate, _ = load_data(train=False)

    Xvalidate_ids = Xvalidate[:, 0]
    Yvalidate = model.predict(Xvalidate[:, 1:])
    ret = np.vstack((Xvalidate_ids, Yvalidate)).T
    write_Y(ret)


def run_gridsearch(X, Y, model):
    parameters = {
        'reg__n_estimators': range(50, 1000, 200),  # the greater C the harder is SVM

    }

    grid = GridSearchCV(model, parameters, verbose=1, n_jobs=-1)
    grid.fit(X[:, 1:], Y)

    for p in parameters.keys():
        print 'Gridseach: param %s = %s' % (
            p, str(grid.best_estimator_.get_params()[p]))
    return grid.best_estimator_


def build_pipe():
    trans = DifferentTransforms()
    scaler = Scaler
    cluster = ClusterTransform()
    regressor = RandomForestClassifier()
    return Pipeline([
        ('scaler', scaler),
        ('trans', trans),
        ('cls', cluster),
        ('reg', regressor),
    ])


Xtrain, Ytrain = load_data()

Scaler = StandardScaler()  # minmax is better for svm and Kmeans but worse for GMM

# delete outliers from the train set
Xtrain, Ytrain = delOutliers(Xtrain, Ytrain)

pipe = build_pipe()
pipe = run_gridsearch(Xtrain, Ytrain, pipe)
run_crossval(Xtrain, Ytrain, pipe)
run_split(Xtrain, Ytrain, pipe)
run_validate(Xtrain, Ytrain, pipe)
