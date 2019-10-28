#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:16:24 2019

@author: hliang
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
#import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score
from scipy import stats
#from minepy import MINE
from numpy import array
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.utils import class_weight
from sklearn.utils.class_weight import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from scipy import optimize
from scipy import stats

def huber_loss(theta, x, y, delta=2): 
    diff = abs(y-np.dot(x,theta.T)) 
    return ((diff < delta)*diff**2/2+(diff >= delta)*delta*(diff-delta/2)).sum()

dfX_train=pd.read_csv('train.csv')
h,w=dfX_train.shape
#print(h)

dx_train = dfX_train.iloc[:,list(range(2,w))].values
y_train = dfX_train.iloc[:,1].values
x1=dx_train
x2=np.power(dx_train,2)
x3=np.exp(dx_train)
x4=np.cos(dx_train)
#x5=np.ones((h,1), dtype=int)

x_train=np.concatenate((x1,x2,x3,x4),axis=1)
#print(x_train.shape)
X,y = shuffle(x_train,y_train)
normalized_data=stats.zscore(X,axis=0,ddof=1)
outlier_row,outlier_column=np.where(abs(normalized_data)>5)
outlier_row=np.unique(outlier_row)
Xinlier=np.delete(X,outlier_row,axis=0)
yinlier=np.delete(y,outlier_row,axis=0)
x5=np.ones((Xinlier.shape[0],1), dtype=int)
Xinlier=np.concatenate((Xinlier,x5),axis=1)

x0=np.zeros(21)
w = optimize.fmin(huber_loss, x0, args=(Xinlier,yinlier), ftol=0.00005, maxiter=100000,disp=False)
w = np.array(w, dtype=np.float64)
result = np.array([w]).T
np.savetxt("result.csv", result, delimiter=",")
'''
kf = KFold(n_splits=10)
X,y=Xinlier,yinlier
print(X.shape)
alphas = np.logspace(-5, 2, 60)
ratios = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]
enet = ElasticNet(max_iter=10000)

largest_error=-1
for ratio in ratios:
    enet.set_params(l1_ratio=ratio)
    train_errors = list()
    val_errors = list()
    for alpha in alphas:
        enet.set_params(alpha=alpha)
        train_error=list()
        val_error=list()
        for train_index, test_index in kf.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]
            
            enet.fit(X_train, y_train)
            train_error.append(enet.score(X_train, y_train))
            val_error.append(enet.score(X_val, y_val))
        train_errors.append(np.array(train_error).mean())
        val_errors.append(np.array(val_error).mean())
    error=np.max(val_errors)
    if largest_error<error:
        largest_error=error
        alpha_optim = alphas[np.argmax(val_errors)]
        ratio_optim = ratio
print("Optimal regularization parameter : %s" % alpha_optim)
print("Optimal regularization parameter : %s" % ratio_optim)
print('error: %s'% np.max(val_errors))

# Estimate the coef_ on full data with optimal regularization parameter
enet.set_params(l1_ratio=ratio_optim, alpha=alpha_optim)
w= enet.fit(X, y).coef_
w= np.array(w, dtype=np.float64)
result = np.array([w]).T
np.savetxt("result.csv", result, delimiter=",")
'''