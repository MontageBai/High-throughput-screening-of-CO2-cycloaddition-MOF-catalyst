# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_Cycloaddition_Screening
@File        : model_generation.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/10 8:41
"""

"""
The modification program is used to generate machine learning models with different values.
The models will be saved in the DataResource/model folder
"""

# 导入所需包
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# 生成模型
def model_gen():
    # 选择随机森林算法
    clf = RandomForestClassifier(max_depth=18, max_features=12, min_samples_split=2, n_estimators=225)
    Data_From_Literature = pd.read_csv("./DataResource/Data_Training/Data_From_Literature.csv")
    Data_From_Literature.drop('Unnamed: 0', axis=1, inplace=True)
    pd_X = Data_From_Literature[Data_From_Literature["yield"]>80].iloc[:, :-2]
    pd_Y = Data_From_Literature[Data_From_Literature["yield"]>80].iloc[:, -1]
    pd_X.drop(["Catalyst Loading","Time","Substrate"], axis=1, inplace=True)
    quantile_list = np.percentile(pd_Y, [25, 50, 75])
    pd_Y = pd_Y > quantile_list[2]
    X_train, X_test, Y_train, Y_test = train_test_split(pd_X, pd_Y, test_size=0.2, random_state=1)
    clf.fit(X_train,Y_train)

    # pickle保存模型
    with open(f'./DataResource/Model/RF_{str(quantile_list[2])}.pickle', 'wb') as f:
        pickle.dump(clf, f)
    return f'/DataResource/Model/RF_{str(quantile_list[2])}.pickle'
# def model_selection(choose="f1"):
#     # f1 is selected as the evaluation index
#     if choose == "f1":
#         for txt_file in
# model_gen()