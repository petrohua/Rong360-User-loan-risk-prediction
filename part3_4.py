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
user_info_train = pd.read_csv('../data/train/user_info_train_3_4.csv')
user_info_test = pd.read_csv('../data/test/user_info_test_3_4.csv')
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

user_info.to_csv('../data/part3_4/data1.csv' , index = False , encoding="utf-8" , mode='a')

# -----------------------------------------------------------*- loan_time -*--------------------------------------------------------------------
print('loan_time')
loan_time_train = pd.read_csv('../data/train/loan_time_train.csv' , header = None)
loan_time_test = pd.read_csv('../data/test/loan_time_test.csv' , header = None)
loan_time = pd.concat([loan_time_train, loan_time_test])
loan_time.columns = ['user_id', 'loan_time']

# 将 user_info和 loan_time meger 起来（添加进来的这一列 不能当做特征 最后肯定是要删除掉的）
user_info = pd.merge(user_info , loan_time , on='user_id' , how='inner')

user_info.to_csv('../data/part3_4/data2.csv' , index = False , encoding="utf-8" , mode='a')

# -----------------------------------------------------------*- browse_history -*--------------------------------------------------------------------
print('browse_history')
browse_history_train = pd.read_csv('../data/train/browse_history_train.csv' , header = None)
browse_history_test = pd.read_csv('../data/test/browse_history_test.csv' , header = None)
col_names = ['user_id', 'timestamp', 'browse_data', 'child_numbers']
browse_history_train.columns = col_names
browse_history_test.columns = col_names
browse_history = pd.concat([browse_history_train, browse_history_test])

# 1：借款时间之前 （前缀是：bef_loan_）
#  1：“浏览行为数据”有多少条（bef_loan_borwse_num1）？多少种(unique)（bef_loan_borwse_num2）
#  2：每一种“浏览行为数据”有多少条（）
#  3：“浏览子行为编号”有多少条（bef_loan_borwse_num4）？多少种(unique)（bef_loan_borwse_num5）
#  4：每一种“浏览子行为编号”有多少条（）
# 2：借款时间之后 aft_loan_
#  1：“浏览行为数据”有多少条（aft_loan_borwse_num1）？多少种(unique)（aft_loan_borwse_num2）
#  2：每一种“浏览行为数据”有多少条（）
#  3：“浏览子行为编号”有多少条（aft_loan_borwse_num4）？多少种(unique)（aft_loan_borwse_num5）
#  4：每一种“浏览子行为编号”有多少条（）

# 将上面的特征分成 两部分，这是第一部分
print('the first_part feature')
browse_fea = ['bef_loan_borwse_num1' , 'bef_loan_borwse_num2' , 'bef_loan_borwse_num4' , 'bef_loan_borwse_num5' ,\
              'aft_loan_borwse_num1' , 'aft_loan_borwse_num2' , 'aft_loan_borwse_num4' , 'aft_loan_borwse_num5']
for i in browse_fea: # 添加8个特征
    user_info[i] = 0
for i in user_info['user_id'].unique():
    print(i) 
    data = browse_history[browse_history['user_id'] == i]
    # 将data数据分成两部分，划分依据是 时间戳timestamp 是否 大于 借款时间 loan_time
    time_1 = user_info[user_info['user_id'] == i]['loan_time'][user_info[user_info['user_id'] == i]['loan_time'].index[0]]
    '''先来做 大于 借款时间（借款之后）'''
    data_1 = data[data['timestamp'] >= time_1]
    if(len(data_1) > 0):
        user_info.loc[user_info['user_id'] == i,'aft_loan_borwse_num1'] = len(data_1)
        user_info.loc[user_info['user_id'] == i,'aft_loan_borwse_num2'] = len(data_1['browse_data'].unique())
        user_info.loc[user_info['user_id'] == i,'aft_loan_borwse_num4'] = len(data_1)
        user_info.loc[user_info['user_id'] == i,'aft_loan_borwse_num5'] = len(data_1['child_numbers'].unique())
    '''再来做 小于 借款时间（借款之前）'''
    data_2 = data[data['timestamp'] < time_1]
    if(len(data_2) > 0):
        user_info.loc[user_info['user_id'] == i,'bef_loan_borwse_num1'] = len(data_2)
        user_info.loc[user_info['user_id'] == i,'bef_loan_borwse_num2'] = len(data_2['browse_data'].unique())
        user_info.loc[user_info['user_id'] == i,'bef_loan_borwse_num4'] = len(data_2)
        user_info.loc[user_info['user_id'] == i,'bef_loan_borwse_num5'] = len(data_2['child_numbers'].unique())
        
# 下来是第二部分，先添加这些特征列，全部初始化成0
print('the second_part feature')
for i in browse_history['browse_data'].unique(): # 添加特征列（借款之前）
    user_info['bef_loan_browse_data_' + str(i)] = 0
for i in browse_history['child_numbers'].unique():
    user_info['bef_loan_child_numbers_' + str(i)] = 0

for i in browse_history['browse_data'].unique(): # 添加特征列（借款之后）
    user_info['aft_loan_browse_data_' + str(i)] = 0
for i in browse_history['child_numbers'].unique():
    user_info['aft_loan_child_numbers_' + str(i)] = 0
# 用程序的方法，将其赋成正确的值   
for i in user_info['user_id'].unique():
    print(i)
    data = browse_history[browse_history['user_id'] == i]
    # 将data数据分成两部分，划分依据是 时间戳timestamp 是否 大于 借款时间 loan_time
    time_1 = user_info[user_info['user_id'] == i]['loan_time'][user_info[user_info['user_id'] == i]['loan_time'].index[0]]
    '''先来做 大于 借款时间（借款之后）'''
    data_1 = data[data['timestamp'] >= time_1]
    for j in data_1['browse_data']:
        user_info.loc[user_info['user_id'] == i , 'aft_loan_browse_data_' + str(j)] += 1
    for j in data_1['child_numbers']:
        user_info.loc[user_info['user_id'] == i , 'aft_loan_child_numbers_' + str(j)] += 1
        
    '''再来做 小于 借款时间（借款之前）'''
    data_2 = data[data['timestamp'] < time_1]
    for j in data_2['browse_data']:
        user_info.loc[user_info['user_id'] == i , 'bef_loan_browse_data_' + str(j)] += 1
    for j in data_2['child_numbers']:
        user_info.loc[user_info['user_id'] == i , 'bef_loan_child_numbers_' + str(j)] += 1
               
user_info.to_csv('../data/part3_4/data3.csv' , index = False , encoding="utf-8" , mode='a')

# -----------------------------------------------------------*- overdue_train -*--------------------------------------------------------------------
print('overdue_train')
target = pd.read_csv('../data/train/overdue_train.csv' , header = None)
target.columns = ['user_id', 'label']

user_info = pd.merge(user_info , target , on='user_id' , how='inner')
print(user_info.head(20))
user_info.to_csv('../data/part3_4/train.csv' , index = False , encoding="utf-8" , mode='a') # data6.csv是这一部分用户的训练集（在这个程序里面一块做出来）


