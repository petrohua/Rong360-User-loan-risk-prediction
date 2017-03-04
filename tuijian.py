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
from sklearn import preprocessing
import sklearn.preprocessing as preprocessing
from sklearn import  linear_model
from sklearn.ensemble import ExtraTreesRegressor #tree回归
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

'''阶段一：因为[1, 2,'3_1','3_2','3_3','3_4','3_5','3_6']这8个文件中，1已经是全部特征的了，所以需要对应将其余所有的几个文件的特征补齐'''
'''补齐方式用的是：先找出来最相近的一批用户，然后缺失的数据用这批用户的中位数填充'''
'''但是在实际测试中，这种方式效果并不好，所以没有使用'''

# # 加载“所有用户（包含训练集 预测集）”的用户信息表，然后合二为一
# user_info_train = pd.read_csv('../data/train/user_info_train.csv' , header = None)
# user_info_train.columns = ['user_id', 'sex', 'occupation', 'education', 'marrige_state', 'marital']
# user_info_test = pd.read_csv('../data/test/user_info_test.csv')
# user_info_test.columns = ['user_id', 'sex', 'occupation', 'education', 'marrige_state', 'marital']
# user_info = pd.concat([user_info_train, user_info_test])
# 
# # 加载 user_info_train_1 user_info_test_1 这两个文件，连为一体（这两个“用户信息表”中的用户，是含有3个文件所有特征的 “用户信息表”用户）
# # 一会就是要在这个 用户表 中找出来 “要填充的每一个用户 也就是在train test中的用户” 最相近的一个用户
# user_info_train_1 = pd.read_csv('../data/train/user_info_train_1.csv')
# user_info_test_1 = pd.read_csv('../data/test/user_info_test_1.csv')
# user_info_1 = pd.concat([user_info_train_1, user_info_test_1])
# 
# # 上面对应“3个文件特征”都含有的用户--对应的特征（全）
# user_info_train = pd.read_csv('../data/new1/train.csv')
# user_info_train_col = user_info_train.columns # 将一个正确的列名顺序排好
# user_info_test = pd.read_csv('../data/new1/test.csv')
# user_info_fea_all_1 = pd.concat([user_info_train, user_info_test])
# user_info_fea_all_1 = user_info_fea_all_1[user_info_train_col] #调整好列名顺序
# 
# list_file = [2,'3_1','3_2','3_3','3_4','3_5','3_6']
# for i in list_file:
#     # 加载数据(要添加“列特征”的 train test)
#     file_name_train = '../data/new' + str(i) + '/train.csv'
#     file_name_test = '../data/new' + str(i) + '/test.csv'
#     train = pd.read_csv(file_name_train) # 训练集
#     test = pd.read_csv(file_name_test) # 预测集
#     # 由于训练集和预测集 的特征列顺序必须要是一样，所有我新建一个空的 DataFrame，然后把列名顺序弄得和 user_info_fea_all_1 一样
#     # 再用原先有的特征列数值（也就是train test中已有的数值）来初始化其中一些列，这些列就算是完成了，，其余没有的“先”初始化为0，之后用代码改正
#     train_new = DataFrame()
#     test_new = DataFrame()
#     for ii in list(user_info_train.columns): # 将“完整的列名”添加进去(包括label列)
#         train_new[ii] = 0
#     for ii in list(user_info_test.columns): # 将“完整的列名”添加进去(不包括label列)
#         test_new[ii] = 0
#     for ii in list(train.columns): # 将能赋值的特征列，全部赋值
#         train_new[ii] = train[ii]
#     for ii in list(test.columns):
#         test_new[ii] = test[ii]
#     
#     # 找出来user_info_1用户中 和train中每一个用户最相近的一个用户，然后将 user_info_fea_all_1 对应用户特征中 在 train中没有的添加进去
#     for j in train['user_id'].unique():
#         print('train_new' + str(i) + '--' + str(j))
#         # 找出来这个用户 j 的5个特征
#         j_sex = user_info[user_info['user_id'] == j]['sex'][user_info[user_info['user_id'] == j]['sex'].index[0]]
#         j_occupation = user_info[user_info['user_id'] == j]['occupation'][user_info[user_info['user_id'] == j]['occupation'].index[0]]
#         j_education = user_info[user_info['user_id'] == j]['education'][user_info[user_info['user_id'] == j]['education'].index[0]]
#         j_marrige_state = user_info[user_info['user_id'] == j]['marrige_state'][user_info[user_info['user_id'] == j]['marrige_state'].index[0]]
#         j_marital = user_info[user_info['user_id'] == j]['marital'][user_info[user_info['user_id'] == j]['marital'].index[0]] 
#         '''用户相似度代码--最终的 shengyu 就是 j 用户最相近的一个或者一批用户'''
#         if(len(user_info_1[user_info_1['sex'] == j_sex]) != 0):
#             shengyu = user_info_1[user_info_1['sex'] == j_sex]
#             if(len(shengyu[shengyu['occupation'] == j_occupation]) != 0):
#                 shengyu = shengyu[shengyu['occupation'] == j_occupation]
#                 if(len(shengyu[shengyu['education'] == j_education]) != 0):
#                     shengyu = shengyu[shengyu['education'] == j_education]
#                     if(len(shengyu[shengyu['marrige_state'] == j_marrige_state]) != 0):
#                         shengyu = shengyu[shengyu['marrige_state'] == j_marrige_state]
#                         if(len(shengyu[shengyu['marital'] == j_marital])):
#                             shengyu = shengyu[shengyu['marital'] == j_marital]
#         del shengyu['sex']
#         del shengyu['occupation']
#         del shengyu['education']
#         del shengyu['marrige_state']
#         del shengyu['marital']
#         shengyu = pd.merge(user_info_fea_all_1 , shengyu , on='user_id' , how='inner')
#         
#         '''这个用户的训练集现在为 train_new ， 预测集现在为 test_new，这两个文件现状是这样子的，有的列已经赋值好了，
#            对于那些为NaN的列得整好--使用 user_info_fea_all_1 中的数据就好！'''
#         # 先找出来要赋值的列，思路很简单--train_new和train两个文件“特征列”的差集
#         list_will_fill = list(set(list(train_new.columns)).difference(set(list(train.columns))))
#         # 训练集
#         for k in list_will_fill:
#             train_new.loc[train_new['user_id'] == j , k] = shengyu[k].median()
#         
#           
#     # 找出来user_info_1用户中 和test中每一个用户最相近的一个用户，然后将 user_info_fea_all_1 对应用户特征中 在 test中没有的添加进去
#     for j in test['user_id'].unique():
#         print('test_new' + str(i) + '--' + str(j))
#         # 找出来这个用户 j 的5个特征
#         j_sex = user_info[user_info['user_id'] == j]['sex'][user_info[user_info['user_id'] == j]['sex'].index[0]]
#         j_occupation = user_info[user_info['user_id'] == j]['occupation'][user_info[user_info['user_id'] == j]['occupation'].index[0]]
#         j_education = user_info[user_info['user_id'] == j]['education'][user_info[user_info['user_id'] == j]['education'].index[0]]
#         j_marrige_state = user_info[user_info['user_id'] == j]['marrige_state'][user_info[user_info['user_id'] == j]['marrige_state'].index[0]]
#         j_marital = user_info[user_info['user_id'] == j]['marital'][user_info[user_info['user_id'] == j]['marital'].index[0]] 
#         '''用户相似度代码--最终的 shengyu 就是 j 用户最相近的一个或者一批用户'''
#         if(len(user_info_1[user_info_1['sex'] == j_sex]) != 0):
#             shengyu = user_info_1[user_info_1['sex'] == j_sex]
#             if(len(shengyu[shengyu['occupation'] == j_occupation]) != 0):
#                 shengyu = shengyu[shengyu['occupation'] == j_occupation]
#                 if(len(shengyu[shengyu['education'] == j_education]) != 0):
#                     shengyu = shengyu[shengyu['education'] == j_education]
#                     if(len(shengyu[shengyu['marrige_state'] == j_marrige_state]) != 0):
#                         shengyu = shengyu[shengyu['marrige_state'] == j_marrige_state]
#                         if(len(shengyu[shengyu['marital'] == j_marital])):
#                             shengyu = shengyu[shengyu['marital'] == j_marital]
#         del shengyu['sex']
#         del shengyu['occupation']
#         del shengyu['education']
#         del shengyu['marrige_state']
#         del shengyu['marital']
#         shengyu = pd.merge(user_info_fea_all_1 , shengyu , on='user_id' , how='inner')
#         
#         '''这个用户的训练集现在为 train_new ， 预测集现在为 test_new，这两个文件现状是这样子的，有的列已经赋值好了，
#            对于那些为NaN的列得整好--使用 user_info_fea_all_1 中的数据就好！'''
#         # 先找出来要赋值的列，思路很简单--train_new和train两个文件“特征列”的差集
#         list_will_fill = list(set(list(test_new.columns)).difference(set(list(test.columns))))
#         # 训练集
#         for k in list_will_fill:
#             test_new.loc[test_new['user_id'] == j , k] = shengyu[k].median()
# 
#     file_name_train_new = '../data/new' + str(i) + '/train_new' + '.csv'
#     file_name_test_new = '../data/new' + str(i) + '/test_new' + '.csv'
#     
#     train_new.to_csv(file_name_train_new , index = False)
#     test_new.to_csv(file_name_test_new , index = False)

'''阶段一：结束'''



'''上面 和 下面 这两段程序 仅仅是做我的第一批特征的'''
'''阶段一：因为[1, 2,'3_1','3_2','3_3','3_4','3_5','3_6']这8个文件中，1已经是全部特征的了，所以需要对应将其余所有的几个文件的特征补齐'''
'''补齐方式用的是：因为几乎都是 数值型特征，所以用 99 来填充'''

# 这是含有全部特征的训练集、预测集
train_1 = pd.read_csv('../data/new1/train.csv')
train_1_col = train_1.columns # 训练集列名
test_1 = pd.read_csv('../data/new1/test.csv')
test_1_col = test_1.columns # 预测集列名

list_file = [2,'3_1','3_2','3_3','3_4','3_5','3_6']
for i in list_file:
    # 加载数据(要添加“列特征”的 train test)
    file_name_train = '../data/new' + str(i) + '/train.csv'
    file_name_test = '../data/new' + str(i) + '/test.csv'
    train = pd.read_csv(file_name_train) # 训练集
    test = pd.read_csv(file_name_test) # 预测集
    # 新建两个空的DataFrame()，先添加完整的列 并赋值为99，然后再将这个文件已经有的列名重新赋值
    train_new = train.copy()
    test_new = test.copy()
    # train_new.drop(list(train_new.columns),axis=1)
    # test_new.drop(list(test_new.columns),axis=1)

    for ii in train_1_col:
        train_new[ii] = 99
    for ii in test_1_col:
        test_new[ii] = 99

    for ii in list(train.columns):
        train_new[ii] = train[ii]
    for ii in list(test.columns):
        test_new[ii] = test[ii]

    for ii in train_1_col:
        train_new[ii] = 99
    for ii in test_1_col:
        test_new[ii] = 99

    for ii in list(train.columns):
        train_new[ii] = train[ii]
    for ii in list(test.columns):
        test_new[ii] = test[ii]  # 必须要这样做两次，否则会有问题，问题是什么我忘了

    train_new = train_new[train_1_col]
    test_new = test_new[test_1_col]

    file_name_train_new = '../data/new' + str(i) + '/train_new_999' + '.csv'
    file_name_test_new = '../data/new' + str(i) + '/test_new_999' + '.csv'

    train_new.to_csv(file_name_train_new , index = False)
    test_new.to_csv(file_name_test_new , index = False)

'''阶段一：结束'''


'''阶段二：将8个补充完全的特征的文件，连接在一起，然后输出到磁盘里面--1.csv 2.csv，之后将这两个文件重命名为train_new.csv test_new.csv'''
# 将几个训练集、预测集 合并在一起，然后一起来做模型
train_new_1 = pd.read_csv('../data/new1/train.csv')
train_new_2 = pd.read_csv('../data/new2/train_new_999.csv')
train_new_3_1= pd.read_csv('../data/new3_1/train_new_999.csv')
train_new_3_2 = pd.read_csv('../data/new3_2/train_new_999.csv')
train_new_3_3= pd.read_csv('../data/new3_3/train_new_999.csv')
train_new_3_4 = pd.read_csv('../data/new3_4/train_new_999.csv')
train_new_3_5= pd.read_csv('../data/new3_5/train_new_999.csv')
train_new_3_6 = pd.read_csv('../data/new3_6/train_new_999.csv')
train_new_1_col = train_new_1.columns # 一个正确的列名顺序

linshi = pd.concat([train_new_1, train_new_2])
linshi = linshi[train_new_1_col] #调整好列名顺序
linshi = pd.concat([linshi, train_new_3_1])
linshi = linshi[train_new_1_col]
linshi = pd.concat([linshi, train_new_3_2])
linshi = linshi[train_new_1_col]
linshi = pd.concat([linshi, train_new_3_3])
linshi = linshi[train_new_1_col]
linshi = pd.concat([linshi, train_new_3_4])
linshi = linshi[train_new_1_col]
linshi = pd.concat([linshi, train_new_3_5])
linshi = linshi[train_new_1_col]
linshi = pd.concat([linshi, train_new_3_6])
linshi = linshi[train_new_1_col]

linshi.to_csv('1.csv' , index = False)

# 将几个训练集、预测集 合并在一起，然后一起来做模型
test_new_1 = pd.read_csv('../data/new1/test.csv')
test_new_2 = pd.read_csv('../data/new2/test_new_999.csv')
test_new_3_1= pd.read_csv('../data/new3_1/test_new_999.csv')
test_new_3_2 = pd.read_csv('../data/new3_2/test_new_999.csv')
test_new_3_3= pd.read_csv('../data/new3_3/test_new_999.csv')
test_new_3_4 = pd.read_csv('../data/new3_4/test_new_999.csv')
test_new_3_5= pd.read_csv('../data/new3_5/test_new_999.csv')
test_new_3_6 = pd.read_csv('../data/new3_6/test_new_999.csv')
test_new_1_col = test_new_1.columns # 一个正确的列名顺序

linshi = pd.concat([test_new_1, test_new_2])
linshi = linshi[test_new_1_col] #调整好列名顺序
linshi = pd.concat([linshi, test_new_3_1])
linshi = linshi[test_new_1_col]
linshi = pd.concat([linshi, test_new_3_2])
linshi = linshi[test_new_1_col]
linshi = pd.concat([linshi, test_new_3_3])
linshi = linshi[test_new_1_col]
linshi = pd.concat([linshi, test_new_3_4])
linshi = linshi[test_new_1_col]
linshi = pd.concat([linshi, test_new_3_5])
linshi = linshi[test_new_1_col]
linshi = pd.concat([linshi, test_new_3_6])
linshi = linshi[test_new_1_col]

linshi.to_csv('2.csv' , index = False)


