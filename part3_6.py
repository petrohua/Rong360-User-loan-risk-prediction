# -*- coding: utf-8 -*-
# 导入所需的python库
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import csv
import os
import pickle
import cPickle
from math import ceil
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------------------------------------*- user_info -*--------------------------------------------------------------------
'''第一步处理数据'''
# 读取数据集
print('user_info')
user_info_train = pd.read_csv('../data/train/user_info_train_3_6.csv')
user_info_test = pd.read_csv('../data/test/user_info_test_3_6.csv')
# 合并train、test
user_info = pd.concat([user_info_train, user_info_test]) # 将 user_info_train user_info_test 两个文件行连接
# 5列特征进行 one-hot 编码
dummy_sex = pd.get_dummies(user_info['sex'],prefix='sex')
user_info = pd.concat([user_info , dummy_sex] , axis=1)
user_info.drop(['sex'] , axis = 1 , inplace = True)
dummy_occupation = pd.get_dummies(user_info['occupation'],prefix='occupation')
user_info = pd.concat([user_info , dummy_occupation] , axis=1)
user_info.drop(['occupation'] , axis = 1 , inplace = True)
dummy_education = pd.get_dummies(user_info['education'],prefix='education')
user_info = pd.concat([user_info , dummy_education] , axis=1)
user_info.drop(['education'] , axis = 1 , inplace = True)
dummy_marrige_state = pd.get_dummies(user_info['marrige_state'],prefix='marrige_state')
user_info = pd.concat([user_info , dummy_marrige_state] , axis=1)
user_info.drop(['marrige_state'] , axis = 1 , inplace = True)
dummy_marital = pd.get_dummies(user_info['marital'],prefix='marital')
user_info = pd.concat([user_info , dummy_marital] , axis=1)
user_info.drop(['marital'] , axis = 1 , inplace = True)

user_info.to_csv('../data/part3_6/data1.csv' , index = False , encoding="utf-8" , mode='a')

# -----------------------------------------------------------*- loan_time -*--------------------------------------------------------------------
print('loan_time')
loan_time_train = pd.read_csv('../data/train/loan_time_train.csv' , header = None)
loan_time_test = pd.read_csv('../data/test/loan_time_test.csv' , header = None)
loan_time = pd.concat([loan_time_train, loan_time_test])
loan_time.columns = ['user_id', 'loan_time']

# 将 user_info和 loan_time meger 起来（添加进来的这一列 不能当做特征 最后肯定是要删除掉的）
user_info = pd.merge(user_info , loan_time , on='user_id' , how='inner')

user_info.to_csv('../data/part3_6/data2.csv' , index = False , encoding="utf-8" , mode='a')

# -----------------------------------------------------------*- overdue_train -*--------------------------------------------------------------------
print('overdue_train')
target = pd.read_csv('../data/train/overdue_train.csv' , header = None)
target.columns = ['user_id', 'label']

user_info = pd.merge(user_info , target , on='user_id' , how='inner')
print(user_info.head(20))
user_info.to_csv('../data/part3_6/train.csv' , index = False , encoding="utf-8" , mode='a') # data6.csv是这一部分用户的训练集（在这个程序里面一块做出来）


