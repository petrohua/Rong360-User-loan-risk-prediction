# coding=utf-8
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import csv
import os
import pickle
import cPickle
from math import ceil
from function import mape,evalerror
from sklearn import preprocessing
import math

train = pd.read_csv('train_new.csv') # 训练集
test = pd.read_csv('test_new.csv') # 预测集

'''xgb模型'''
import xgboost as xgb
params={'booster':'gbtree',
        'objective': 'rank:pairwise',
        # 'objective': 'binary:logistic',
        'learning_rate': 0.01,
        'eval_metric':'auc',
        'gamma':0.1,
        'min_child_weight':5,
        'max_depth':5,
        'lambda':10,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'colsample_bylevel':0.7,
        'eta': 0.01,
        'nthread':4,
        'tree_method':'exact',
        'seed':0,    
        }

plst = params.items()

# 预测
col = list(train.columns)[1:][:-1]
dtrain = xgb.DMatrix( train[col].as_matrix() , train['label'].as_matrix() ) #训练集的所有特征列，训练集的“要预测的那一列
print('begin to build model')
watchlist = [(dtrain,'train')]

# bst = xgb.train(params, dtrain, num_boost_round=350,feval=evalerror,evals=watchlist)
bst = xgb.train(params, dtrain, num_boost_round=3500,evals=watchlist)
print('build model over')

prediction = bst.predict( xgb.DMatrix(test[col].as_matrix()) ).clip(0, 1) 

submit = DataFrame()
submit['userid'] = test['user_id']
submit['probability'] = prediction

submit.to_csv('pred.csv' , index = False)






