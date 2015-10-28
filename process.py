#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from utils import *

from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_regression
import sklearn.cross_validation as skcv
import sklearn.metrics as skmet

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


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

    if train:
        Y = np.log(data['Y'].as_matrix())
        del data['Y']
    else:
        Y = None

    return data.as_matrix(), Y

def score(Ypred, Yreal):
    return skmet.mean_squared_error(np.exp(Ypred), np.exp(Yreal)) ** 0.5

def write_Y(Y):
    if Y.shape[1] != 2:
        raise 'Y has invalid shape!'
    np.savetxt('results/Ypred.csv', Y,
               fmt='%d', delimiter=',', header='Id,Delay', comments='')

def build_ann(num_features):
    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense1', DenseLayer),
               ('output', DenseLayer)]

    params = dict(
        layers=layers0,

        dropout0_p=0.1,

        dense0_num_units=num_features*1,
        dense1_num_units=num_features*2,

        input_shape=(None, num_features),
        output_num_units=1,

        update=nesterov_momentum,
        update_learning_rate=0.003,
        update_momentum=0.8,

        # verbose=1,
        max_epochs=300,
        regression=True
    )

    return NeuralNet(**params)

def build_pipe():
    encoder = OneHotEncoder(categorical_features=[0, 9],
                            sparse=False)
    scaler = StandardScaler()
    # originally we have 14 features, after trans. we have 20
    ann = build_ann(20)
    return Pipeline([
        ('encoder', encoder),
        ('scaler', scaler),
        ('ann', ann),
    ])

def run_split(X, Y, model):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=.9)
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    print "Split-score = %f" % score(Ypred, Ytest)

def run_validate(Xtrain, Ytrain, model):
    model.fit(Xtrain, Ytrain)

    Xvalidate, _ = load_data(train=False)

    Xvalidate_ids = Xvalidate[:,0]
    Xvalidate_ids.shape = (Xvalidate_ids.shape[0], 1)

    Xvalidate = Xvalidate[:,1:]

    Yvalidate = np.exp(model.predict(Xvalidate))
    ret = np.hstack((Xvalidate_ids, Yvalidate))
    write_Y(ret)

Xtrain, Ytrain = load_data()
Xtrain = Xtrain[:, 1:]  # cut away Ids
pipe = build_pipe()

print "run split score"
run_split(Xtrain, Ytrain, pipe)
print "run validation"
run_validate(Xtrain, Ytrain, pipe)
