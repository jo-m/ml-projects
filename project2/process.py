#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
    scores = skcv.cross_val_score(model, X[:,1:], Y, scoring=scorefun, cv=4)
    print 'C-V score =', np.mean(scores), '+/-', np.std(scores)

def run_split(X, Y, model):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.8)
    Xtrain, Xtest = Xtrain[:,1:], Xtest[:,1:]
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    print "Split-score = %f" % score(Ypred, Ytest)

def write_Y(Y):
    if Y.shape[1] != 2:
        raise 'Y has invalid shape!'
    np.savetxt('results/Ypred.csv', Y,
               fmt='%d', delimiter=',', header='Id,Label', comments='')

def run_validate(Xtrain, Ytrain, model):
    model.fit(Xtrain[:,1:], Ytrain)

    Xvalidate, _ = load_data(train=False)

    Xvalidate = np.delete(Xvalidate, [5], 1)
    Xvalidate_ids = Xvalidate[:,0]
    Yvalidate = model.predict(Xvalidate[:,1:])
    ret = np.vstack((Xvalidate_ids, Yvalidate)).T
    write_Y(ret)

def run_gridsearch(X, Y, model):
    parameters = {
        'reg__C': range(2180, 2200, 1),  # the greater C the harder is SVM
        'reg__gamma': np.arange(0.3, 0.5, 0.01),
    }

    grid = GridSearchCV(model, parameters, verbose=1, n_jobs=-1)
    grid.fit(X[:,1:], Y)
    for p in parameters.keys():
        print 'Gridseach: param %s = %s' % (
            p, str(grid.best_estimator_.get_params()[p]))
    return grid.best_estimator_

def build_pipe():
    scaler = MinMaxScaler()
    regressor = SVC()
    return Pipeline([
        ('scaler', scaler),
        ('reg', regressor),
    ])

Xtrain, Ytrain = load_data()

# also could delete 3 and 6, minimal score drop, try on the final model
Xtrain = np.delete(Xtrain, [5], 1)

pipe = build_pipe()
pipe = run_gridsearch(Xtrain, Ytrain, pipe)
run_crossval(Xtrain, Ytrain, pipe)
run_split(Xtrain, Ytrain, pipe)
run_validate(Xtrain, Ytrain, pipe)
