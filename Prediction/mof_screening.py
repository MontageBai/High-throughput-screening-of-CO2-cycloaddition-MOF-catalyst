# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_ Cycloaddition_Screening
@File        : mof_screening.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/10 19:29
"""

import pickle

import pandas as pd


def predict():
    pd_to_predict = pd.read_csv("./DataResource/Generation_Processing/predict.csv")
    X_predict = pd_to_predict.iloc[:, 2:]
    X_predict.drop(["Catalyst Loading", "Time", "Substrate"], axis=1, inplace=True)
    result_pd2 = pd.DataFrame(index=[x for x in range(0, pd_to_predict.shape[0])], columns=["ID"], data=0.0)
    with open("./DataResource/Model/RF_51.595982145.pickle", "rb") as model:
        clf = pickle.load(model)
        t = float(input("Please enter the temperature:"))
        p = float(input("Please enter the pressure:"))
        Substrate = input("Please enter the substrate(PO&SO&BO&ECH&EBP):")
        co_catalyst = input("Please enter the co-catalyst(TBAB&TBAC&TBAI&NONE):")
        if co_catalyst.upper() == "NONE":
            for i in range(X_predict.shape[0]):
                X_predict.loc[i, "Temperature"] = t
                X_predict.loc[i, "Pressure"] = p
                X_predict.loc[i, "PO"] = 0
                X_predict.loc[i, "SO"] = 0
                X_predict.loc[i, "BO"] = 0
                X_predict.loc[i, "ECH"] = 0
                X_predict.loc[i, "EBP"] = 0
                X_predict.loc[i, Substrate.upper()] = 1
                X_predict.loc[i, "TBAB"] = 0
                X_predict.loc[i, "TBAC"] = 0
                X_predict.loc[i, "TBAI"] = 0
        else:
            co_loading = float(input("Please enter the co-catalyst concentration(mol%):"))
            for i in range(X_predict.shape[0]):
                X_predict.loc[i, "Temperature"] = t
                X_predict.loc[i, "Pressure"] = p
                X_predict.loc[i, "PO"] = 0
                X_predict.loc[i, "SO"] = 0
                X_predict.loc[i, "BO"] = 0
                X_predict.loc[i, "ECH"] = 0
                X_predict.loc[i, "EBP"] = 0
                X_predict.loc[i, Substrate.upper()] = 1
                X_predict.loc[i, co_catalyst.upper()]= co_loading
        Y_predict = clf.predict(X_predict)
        Y_predict_proba = clf.predict_proba(X_predict)
        result_pd2[f"T:{t},P:{p},S:{Substrate}"] = Y_predict
        result_pd2[f"T:{t},P:{p},S:{Substrate} proba"] = Y_predict_proba[:, 1]

    result_pd2["ID"] = pd_to_predict["ID"]
    result_pd2.to_csv("./DataResource/Result.csv")
