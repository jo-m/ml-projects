#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from datetime import datetime

import xgboost as xgb

import sklearn.cross_validation as skcv
import sklearn.metrics as skmet
from utils import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.cross_validation import KFold


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
    scores = []
    kf = KFold(X.shape[0], n_folds=10)
    for train, test in kf:
        model.fit(X[train], Y[train])
        Ypred = model.predict(X[test])
        sc = score(Y[test], Ypred)
        scores.append(sc)
    print 'C-V score %s' % (str(np.mean(scores)))
    print 'std %s' % str(np.std(scores))


def run_split(X, Y, model):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.9)
    Xtrain, Xtest = Xtrain[:, 1:], Xtest[:, 1:]
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    scored = score(Ypred, Ytest)
    print "Split-score = %f" % scored
    return scored


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
        'reg__n_estimators': [300, 500, 1250, 1500, 1750, 2500, 3000],
        'reg__learning_rate': [0.001, 0.003, 0.005, 0.006, 0.01],
        'reg__max_depth': [3, 5, 7, 9],
        'reg__subsample': [0.5, 0.7, 0.9],
        'selector__k': [100, 120, 150, 200, 300, 400, 'all'],
    }

    grid = GridSearchCV(model, parameters, verbose=1, n_jobs=-1, cv=5)
    grid.fit(X[:, 1:], Y)

    for p in parameters.keys():
        print 'Gridseach: param %s = %s' % (
            p, str(grid.best_estimator_.get_params()[p]))
    return grid.best_estimator_


def build_pipe():
    scaler = Scaler

    selector = SelectKBest(chi2, k=120)
    regressor = xgb.XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=5, subsample=0.5)

    return Pipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('reg', regressor),
    ])


Xtrain, Ytrain = load_data()

Scaler = StandardScaler(with_mean=False)  # do not subtract the mean,
                                            # chi2 does not accept negative numbers
pipe = build_pipe()
# pipe = run_gridsearch(Xtrain, Ytrain, pipe)
run_crossval(Xtrain, Ytrain, pipe)
run_split(Xtrain, Ytrain, pipe)
run_validate(Xtrain, Ytrain, pipe)

