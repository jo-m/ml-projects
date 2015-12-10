#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GMM

from datetime import datetime

import xgboost as xgb

import sklearn.cross_validation as skcv
import sklearn.metrics as skmet
from utils import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def load_data(train=True):
    if train:
        fname = 'data/train.csv'
    else:
        fname = 'data/test_validate.csv'

    dataT = pd.read_csv(fname,
                       index_col=None,
                       header=None)

    dataT = dataT.as_matrix().astype(float)
    if train:
        dataT = dataT[:, :-1]
    if train:
        name = 'data/train_labels.csv'
        Y = pd.read_csv(name,
                       index_col=None,
                       header=None)
        # labels also have index as the first column
        Y = Y[1].as_matrix().astype(int)
    else:
        Y = None
    return dataT, Y


# unsupervised classifier
# use clustering on all unsupervised data to produce one more feature
class ClusterTransform():
    def __init__(self, n_clusters=3, n_jobs=-1, tol=1e-3, min_covar=1e-3, covariance_type='tied',
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
                   covariance_type=covariance_type, min_covar=min_covar, thresh=tol)
        self.clusterizer.means_ = cluster_means

    def set_params(self, **params):
        self.clusterizer.set_params(**params)

    # use all x to train cluster
    def fit(self, X, _):
        xTest = self.X_TEST
        xTest = xTest[:, 1:]
        xTest = DifferentTransforms().transform(xTest)
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


# supervised classifier
class StackInstance():
    def __init__(self, **kwargs):
        self.classifier = kwargs.get('classifier')

    def set_params(self, **params):
        params.pop('classifier', None)
        self.classifier.set_params(**params)

    def fit(self, X, Y):
        self.classifier.fit(X, Y)
        return self

    # add the predicted labels as a feature vector
    # binarize the labels
    def transform(self, X):
        xFeat = self.classifier.predict(X)
        xFeat = np.atleast_2d(xFeat).T
        xFeat = OneHotEncoder(sparse=False).fit_transform(xFeat)
        xRes = np.hstack((X, xFeat))
        return xRes

    def predict(self, X):
        return self.classifier.predict(X)

    def score(self, X, Y):
        return score(self.predict(X), Y)

    def get_params(self, deep=True):
        dic = self.classifier.get_params(deep=deep)
        dic['classifier'] = self.classifier
        return dic


class DifferentTransforms():
    def __init__(self, featureDel=4, **kwargs):
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

    Xtrain = np.delete(Xtrain, outliers, 0)
    Ytrain = np.delete(Ytrain, outliers, 0)

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
    scores = skcv.cross_val_score(model, X[:, 1:], Y, scoring=scorefun, cv=3, n_jobs=1)
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
    np.savetxt('results/Ypred{0}.csv'.format(datetime.now().strftime('%Y-%m-%d,%H:%M:%S')), Y,
               fmt='%d', delimiter=',', header='Id,Label', comments='')


def run_validate(Xtrain, Ytrain, model):
    model.fit(Xtrain[:, 1:], Ytrain)

    Xvalidate, _ = load_data(train=False)

    Xvalidate_ids = Xvalidate[:, 0]
    Yvalidate = model.predict(Xvalidate[:, 1:])
    ret = np.vstack((Xvalidate_ids, Yvalidate)).T
    write_Y(ret)
    print 'wrote validate'

def run_gridsearch(X, Y, model):
    parameters = {
        'reg__n_estimators': [1000, 2000, 4000, 5000],
        'reg__learning_rate': [0.005, 0.008, 0.01, 0.015],
        'reg__max_depth': [2, 3, 4],
        'reg__subsample': [0.5, 0.6, 0.7, 0.8, 1],
        'selector__k': [10, 30, 60, 120, 300, 400, 600]
    }


    grid = GridSearchCV(model, parameters, verbose=1, n_jobs=1, cv=3)
    grid.fit(X[:, 1:], Y)

    for p in parameters.keys():
        print 'Gridseach: param %s = %s' % (
            p, str(grid.best_estimator_.get_params()[p]))
    return grid.best_estimator_


def build_pipe():
    scaler = Scaler

    selector = SelectKBest(chi2)

    regressor = xgb.XGBClassifier()

    return Pipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('reg', regressor),
    ])


Xtrain, Ytrain = load_data()

Scaler = StandardScaler(with_mean=False)  # do not delete mean in order to have only positive numbers,
                                          # required by chi2 score

pipe = build_pipe()
pipe = run_gridsearch(Xtrain, Ytrain, pipe)
run_validate(Xtrain, Ytrain, pipe)
run_split(Xtrain, Ytrain, pipe)
# run_crossval(Xtrain, Ytrain, pipe)