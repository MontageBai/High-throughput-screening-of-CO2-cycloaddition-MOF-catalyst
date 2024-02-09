# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_Cycloaddition_Screening
@File        : data_preprocessing.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/8 14:40
"""

"""
This is a data preprocessor, all the data after preprocessing into the
classifier grid search parameters.
"""

# 导入所需包
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import ModelTraining.classifier_training as CT

# Read Data From CSV file
# 读取数据
print(os.getcwd())
Data_From_Literature = pd.read_csv("./DataResource/Data_Training/Data_From_Literature.csv")
Data_From_Literature.drop('Unnamed: 0', axis=1, inplace=True)

# Algorithms that don't need to be normalized
# 无需归一化数据
def no_scaler(classification_criteria=None):
    pd_X = Data_From_Literature[Data_From_Literature["yield"]>80].iloc[:, :-2]
    pd_Y = Data_From_Literature[Data_From_Literature["yield"]>80].iloc[:, -1]
    pd_X.drop(["Catalyst Loading","Time","Substrate"], axis=1, inplace=True)

    # 默认分类方式为75%分位数
    if classification_criteria == None:
        quantile_list = np.percentile(pd_Y, [25, 50, 75])
        pd_Y = pd_Y > quantile_list[2]
    else:
        pd_Y = pd_Y > classification_criteria
    X_train, X_test, Y_train, Y_test = train_test_split(pd_X, pd_Y, test_size=0.2, random_state=1)
    for model in [
              CT.dt_fit(X_train, X_test, Y_train, Y_test),
              CT.rf_fit(X_train, X_test, Y_train, Y_test),
              CT.bnb_fit(X_train, X_test, Y_train, Y_test),
              CT.mnb_fit(X_train, X_test, Y_train, Y_test),
              CT.gbdt_fit(X_train, X_test, Y_train, Y_test),
              CT.et_fit(X_train, X_test, Y_train, Y_test),
              CT.lgbm_fit(X_train, X_test, Y_train, Y_test),
              CT.xgb_fit(X_train, X_test, Y_train, Y_test),
              ]:
        try:
            model
        except:
            pass

# Algorithms that need to be normalized
# 无需归一化数据
def scaler(classification_criteria=None):
    pd_X = Data_From_Literature[Data_From_Literature["yield"]>80].iloc[:, :-2]
    pd_Y = Data_From_Literature[Data_From_Literature["yield"]>80].iloc[:, -1]
    pd_X.drop(["Catalyst Loading","Time","Substrate"], axis=1, inplace=True)

    min_max_scale = MinMaxScaler()
    min_max_scale.fit(pd_X)
    pd_X = min_max_scale.transform(pd_X)
    # 默认分类方式75%分位数
    if classification_criteria == None:
        quantile_list = np.percentile(pd_Y, [25, 50, 75])
        pd_Y = pd_Y > quantile_list[2]
        print(quantile_list[0])
        print(quantile_list[1])
    else:
        pd_Y = pd_Y > classification_criteria
    X_train, X_test, Y_train, Y_test = train_test_split(pd_X, pd_Y, test_size=0.2, random_state=1)
    for model in [
              CT.knn_fit(X_train, X_test, Y_train, Y_test),
              CT.lr_fit(X_train, X_test, Y_train, Y_test),
              CT.nn_fit(X_train, X_test, Y_train, Y_test),
              CT.svm_fit(X_train, X_test, Y_train, Y_test),
              CT.qda_fit(X_train, X_test, Y_train, Y_test),
              CT.sgd_fit(X_train, X_test, Y_train, Y_test),
              CT.ada_fit(X_train, X_test, Y_train, Y_test),
              ]:
        try:
            model
        except:
            pass

        pd_Y[pd_Y == True] = 1
        pd_Y[pd_Y == False] = 0
        X_train, X_test, Y_train, Y_test = train_test_split(pd_X, pd_Y, test_size=0.2, random_state=1)
        CT.cat_fit(X_train, X_test, Y_train, Y_test)

# Execute training
def training(classification_criteria=None):
    classification_criteria = classification_criteria
    scaler(classification_criteria=classification_criteria)
    no_scaler(classification_criteria=classification_criteria)
