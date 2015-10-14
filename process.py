#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from utils import *

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_regression
import sklearn.cross_validation as skcv
import sklearn.metrics as skmet


#  0 A Width - 2,4,6,8
#  1 B ROB size - 32 to 160
#  2 C IQ size - 8 to 80
#  3 D LSQ size - 8 to 80
#  4 E RF sizes - 40 to 160
#  5 F RF read ports - 2 to 16
#  6 G RF write ports - 1 to 8
#  7 H Gshare size -  1K to 32K
#  8 I BTB size - 256 to 1024
#  9 J Branches allowed - 8,16,24,32
# 10 K L1 Icache size - 64 to 1024
# 11 L L1 Dcache size - 64 to 1024
# 12 M L2 Ucache size- 512 to 8K
# 13 N Depth - 9 to 36

cols = [
    "id",
    "Width", "ROB", "IQ", "LSQ", "RF", "RF read",
    "RF write", "Gshare", "BTB", "Branches",
    "L1Icache", "L1Dcache", "L2Ucache", "Depth",
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

    # data['Gshare'] = np.log(data['Gshare'])
    # data['BTB'] = np.log(data['BTB'])
    if train:
        Y = np.log(data['Y'].as_matrix())
        del data['Y']
    else:
        Y = None

    return data.as_matrix(), Y


def apply_polynominals(X, column, p=30):
    for i in range(2, p + 1):
        X['%s^%d' % (column, i)] = np.power(X[column], i)

def apply_mult(X, column1, column2, p=0):
    X['%s_mul_%s' % (column1,column2)] = \
        X[column1] * X[column2]
    if (p>0):
        apply_polynominals(X, '%s_mul_%s' % (column1,column2),p )

def transform_features(X):
    # map categorical features to [0...n_values]
    for index in [1, 10]:
        values = np.sort(list(set(X[:, index])))
        for i in range(0, X.shape[0]):
            X[i, index] = np.where(X[i, index] == values)[0]
    return X

def score(Ypred, Yreal):
    return skmet.mean_squared_error(np.exp(Ypred), np.exp(Yreal)) ** 0.5

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
               fmt='%d', delimiter=',', header='Id,Delay', comments='')

def run_validate(Xtrain, Ytrain, model):
    model.fit(Xtrain[:,1:], Ytrain)

    Xvalidate, _ = load_data(train=False)
    Xvalidate = transform_features(Xvalidate)
    Xvalidate_ids = Xvalidate[:,0]
    Yvalidate = np.exp(model.predict(Xvalidate[:,1:]))
    ret = np.vstack((Xvalidate_ids, Yvalidate)).T
    write_Y(ret)

def run_gridsearch(X, Y, model):
    parameters = {
        'reg__kernel': ['rbf'],
        'reg__C': np.arange(2.1, 2.7, 0.01),
        'reg__gamma': np.arange(0.01, 0.05, 0.01),
        'selector__k': [9]
    }

    grid = GridSearchCV(model, parameters, verbose=1, n_jobs=-1)
    grid.fit(X[:,1:], Y)
    for p in parameters.keys():
        print 'Gridseach: param %s = %s' % (
            p, str(grid.best_estimator_.get_params()[p]))
    return grid.best_estimator_

def build_pipe():
    scaler = StandardScaler(with_mean=False)
    encoder = OneHotEncoder(categorical_features=[0, 9], sparse=False)
    regressor = SVR(gamma=0.04, kernel='rbf', C=2.69)
    selector = SelectKBest(f_regression, k=9)
    return Pipeline([
        ('encoder', encoder),
        ('scaler', scaler),
        ('selector', selector),
        ('reg', regressor),
    ])

Xtrain, Ytrain = load_data()
Xtrain = transform_features(Xtrain)
pipe = build_pipe()
# pipe = run_gridsearch(Xtrain, Ytrain, pipe)
run_crossval(Xtrain, Ytrain, pipe)
run_split(Xtrain, Ytrain, pipe)
run_validate(Xtrain, Ytrain, pipe)
