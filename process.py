#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from utils import *

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import sklearn.cross_validation as skcv
import sklearn.metrics as skmet


# no Col      Name    Values
#      Importance
#  0 A 0.2715 Width - 2,4,6,8
#  1 B 0.0334 ROB size - 32 to 160
#  2 C 0.0251 IQ size - 8 to 80
#  3 D 0.1067 LSQ size - 8 to 80
#  4 E 0.0286 RF sizes - 40 to 160
#  5 F 0.0564 RF read ports - 2 to 16
#  6 G 0.0202 RF write ports - 1 to 8
#  7 H 0.0195 Gshare size -  1K to 32K
#  8 I 0.0091 BTB size - 256 to 1024
#  9 J 0.0143 Branches allowed - 8,16,24,32
# 10 K 0.0146 L1 Icache size - 64 to 1024
# 11 L 0.0142 L1 Dcache size - 64 to 1024
# 12 M 0.0151 L2 Ucache size- 512 to 8K
# 13 N 0.3707 Depth - 9 to 36

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

    if train:
        Y = np.log(data['Y'].as_matrix())
        del data['Y']
    else:
        Y = None

    return data.as_matrix().astype(float), Y

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
    Xvalidate_ids = Xvalidate[:,0]
    Yvalidate = np.exp(model.predict(Xvalidate[:,1:]))
    ret = np.vstack((Xvalidate_ids, Yvalidate)).T
    write_Y(ret)

def run_gridsearch(X, Y, model):
    parameters = {
        'reg__n_estimators': [100, 150, 200, 250, 500, 1000],
    }

    grid = GridSearchCV(model, parameters, verbose=1, n_jobs=-1)
    grid.fit(X[:,1:], Y)
    for p in parameters.keys():
        print 'Gridseach: param %s = %s' % (
            p, str(grid.best_estimator_.get_params()[p]))
    return grid.best_estimator_

def build_pipe():
    scaler = StandardScaler(with_mean=False)
    regressor = RandomForestRegressor(n_estimators=200)
    selector = SelectKBest(f_regression, k=9)
    return Pipeline([
        ('scaler', scaler),
        ('selector', selector),
        ('reg', regressor),
    ])

Xtrain, Ytrain = load_data()
pipe = build_pipe()
# pipe = run_gridsearch(Xtrain, Ytrain, pipe)
run_crossval(Xtrain, Ytrain, pipe)
run_split(Xtrain, Ytrain, pipe)
run_validate(Xtrain, Ytrain, pipe)
