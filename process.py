#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.cross_validation as skcv
import sklearn.metrics as skmet


# Width - 2,4,6,8
# ROB size - 32 to 160
# IQ size - 8 to 80
# LSQ size - 8 to 80
# RF sizes - 40 to 160
# RF read ports - 2 to 16
# RF write ports - 1 to 8
# Gshare size -  1K to 32K
# BTB size - 256 to 1024
# Branches allowed - 8,16,24,32
# L1 Icache size - 64 to 1024
# L1 Dcache size - 64 to 1024
# L2 Ucache size- 512 to 8K
# Depth - 9 to 36

cols = [
    "id",
    "Width", "ROB", "IQ", "LSQ", "RF", "RF read",
    "RF write", "Gshare", "BTB", "Branches",
    "L1Icache", "L1Dcache", "L2Ucache", "Depth",
    "Y"
]

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

    data['L1Icache'] = np.log(data['L1Icache'])
    data['L1Dcache'] = np.log(data['L1Dcache'])
    data['L2Ucache'] = np.log(data['L2Ucache'])

    if train:
        Y = data['Y'].as_matrix()
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

def transformFeatures(X):
    apply_polynominals(X, 'A', 5)
    apply_mult(X, 'hour', 'A', 2)

    return X

def score(Ypred, Yreal):
    return skmet.mean_squared_error(Ypred, Yreal) ** 0.5

def reg_crossval(X, Y, regressor):
    scorefun = skmet.make_scorer(score)
    scores = skcv.cross_val_score(regressor, X[:,1:], Y, scoring=scorefun, cv=5)
    print 'C-V score =', np.mean(scores), '+/-', np.std(scores)

def reg_split(X, Y, regressor):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.8)
    Xtrain, Xtest = Xtrain[:,1:], Xtest[:,1:]
    pipe.fit(Xtrain, Ytrain)
    Ypred = pipe.predict(Xtest)
    print "Split-score = %f" % score(Ypred, Ytest)

def write_Y(Y):
    if Y.shape[1] != 2:
        raise 'Y has invalid shape!'
    np.savetxt('results/Ypred.csv', Y,
               fmt='%d', delimiter=',', header='Id,Delay', comments='')

def reg_validate(Xtrain, Ytrain, regressor):
    pipe.fit(Xtrain[:,1:], Ytrain)

    Xvalidate, _ = load_data(train=False)
    Xvalidate_ids = Xvalidate[:,0]
    Yvalidate = pipe.predict(Xvalidate[:,1:])
    ret = np.vstack((Xvalidate_ids, Yvalidate)).T
    write_Y(ret)


def build_pipe():
    scaler = StandardScaler()
    filter_ = SelectKBest(f_regression, k=10)
    regressor = Lasso(alpha=2)
    return Pipeline([('scaler', scaler), ('filter', filter_), ('reg', regressor)])

Xtrain, Ytrain = load_data()
pipe = build_pipe()
reg_crossval(Xtrain, Ytrain, pipe)
reg_split(Xtrain, Ytrain, pipe)
reg_validate(Xtrain, Ytrain, pipe)
