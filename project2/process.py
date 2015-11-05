#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn.cross_validation as skcv
import sklearn.metrics as skmet

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
    Xvalidate_ids = Xvalidate[:,0]
    Yvalidate = model.predict(Xvalidate[:,1:])
    ret = np.vstack((Xvalidate_ids, Yvalidate)).T
    write_Y(ret)

def run_gridsearch(X, Y, model):
    parameters = {
    # C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200
        # 'reg__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50],
        # 'reg__penalty': ['l1', 'l2'],
        # 'reg__alpha': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        # 'reg__epsilon': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        # 'reg__C': [0.5, .6, .7, .8, 1, 1.5, 2, 2.5, 3],
        # 'reg__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        # 'reg__degree': [2, 3, 4, 5],
        # 'reg__gamma': [0.0, 0.1, 0.01, 0.001, 0.0001, 0.2, 0.02, 0.002, 0.0002, 0.3, 0.03, 0.003, 0.0003, 0.05, 0.15, 0.08, 0.13],
        'reg__gamma': np.linspace(0.05, 0.15),
        # 'reg__probability': [True, False],
        # 'reg__shrinking': [True, False],
        # 'reg__multi_class': ['ovr', 'multinomial'],
        # 'reg__loss' : ['deviance', 'exponential'],
    }

    grid = GridSearchCV(model, parameters, verbose=1, n_jobs=-1)
    grid.fit(X[:,1:], Y)
    for p in parameters.keys():
        print 'Gridseach: param %s = %s' % (
            p, str(grid.best_estimator_.get_params()[p]))
    return grid.best_estimator_

def build_pipe():
    scaler = StandardScaler()
    regressor = SVC(kernel='rbf', probability=True, shrinking=True)
    return Pipeline([
        ('scaler', scaler),
        ('reg', regressor),
    ])

Xtrain, Ytrain = load_data()
pipe = build_pipe()
pipe = run_gridsearch(Xtrain, Ytrain, pipe)
run_crossval(Xtrain, Ytrain, pipe)
run_split(Xtrain, Ytrain, pipe)
run_validate(Xtrain, Ytrain, pipe)
