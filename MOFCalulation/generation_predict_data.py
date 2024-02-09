# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_ Cycloaddition_Screening
@File        : generation_predict_data.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/10 19:13
"""

import pandas as pd
import csv
import math
import os

# 电荷计算程序
def metal_charge():
    metal_index = ["ID", "Cu", "Zr", "Hf", "In", "Eu", "Zn", "Mg", "Co", "Ni", "Cd", "Cr", "Tb", "Sm", "Y", "Nd", "Pb",
                   "Sr", "Ce", "Tm", "Yb", "Ho", "Ca", "Gd", "Er", "K", "Fe", "first_metal", "second_metal"]
    pd2 = pd.DataFrame(index=[x for x in range(0, pd1.shape[0])], columns=metal_index, data=0.0)
    for i in range(pd2.shape[0]):
        list2 = str(pd2.loc[i, "ID"]).split("_")
        for a in list2:
            if a in list(pd2.columns):
                pd2.loc[i, a] = 1
    pd2.to_csv("./DataResource/Generation_Processing/metal.csv")
    # 金属元素
    metal_element = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Ga', 'Ge', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                     'Sb',
                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                     'Lu',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac']

    f = open("./DataResource/Generation_Processing/statistical_result_file.csv", mode="w", encoding="utf-8",
             newline="")
    statistical_result_file = csv.writer(f)
    statistical_result_file.writerow(['metal', 'mean', 'fangcha'])
    for file_name in os.listdir("./DataResource/RASPA_run"):
        with open(f"./DataResource/RASPA_run/{file_name}/{file_name}.cif", mode="r") as cif:
            first_metal = ""
            first_metal_num = []
            second_metal = ""
            second_metal_num = []
            for line in cif:
                test = line.split(" ")
                test = [i for i in test if i != '']
                try:
                    if test[1] in metal_element:
                        if first_metal == "":
                            first_metal = test[1]
                            first_metal_num.append(float(test[-1]))
                        elif test[1] == first_metal:
                            first_metal_num.append(float(test[-1]))
                        elif test[1] != first_metal and first_metal != "" and second_metal == "":
                            second_metal = line.split("  ")[-2]
                            second_metal_num.append(float(test[-1]))
                        elif test[1] == second_metal:
                            second_metal_num.append(float(test[-1]))
                except:
                    print(f"{file_name}")
            average_first = sum(first_metal_num) / len(first_metal_num)
            rms_first = math.sqrt(sum([x ** 2 for x in first_metal_num]) / len(first_metal_num))
            statistical_result_file.writerow([file_name, first_metal, len(first_metal_num), average_first, rms_first])
            if second_metal != "":
                average_second = sum(second_metal_num) / len(second_metal_num)
                rms_second = math.sqrt(sum([x ** 2 for x in second_metal_num]) / len(second_metal_num))
                statistical_result_file.writerow(
                    [file_name, second_metal, len(second_metal_num), average_second, rms_second])
            else:
                average_second = ""
                rms_second = ""
            # print(f"{file_name},{first_metal},{len(first_metal_num)},{second_metal},{len(second_metal_num)}")
    f.close()

    pd1 = pd.read_csv("./DataResource/Generation_Processing/metal.csv", index_col=0)
    pd2 = pd.read_csv("./DataResource/Generation_Processing/charge.csv")
    pd3 = pd.merge(pd1, pd2, on="ID", how="inner")
    pd3.to_csv("./DataResource/Generation_Processing/metal_and_charge.csv")


def ligand():
    pd1 = pd.read_csv("./DataResource/Generation_Processing/metal_and_charge.csv", index_col=0)
    ligand_index = ["ID", "Coordination carboxyl", "Triazole", "Benzene ring", "Naphthalene", "Triazine", "Pyrazole",
                    "NN double bond", "Pyridine", "Piperazine", "Amino", "CO double bond", "Phenol hydroxyl",
                    "CC triple bond", "Tetrazole", "Imidazole", "Nitro group", "CN double bond", "Fluorine ion",
                    "Sulfone", "CC double bond", "Porphyrin", "Anthracene", "Carboxyl", "Iodine ion", "Bromine ion"]
    pd2 = pd.DataFrame(index=[x for x in range(0, pd1.shape[0])], columns=ligand_index, data=0.0)
    pd2["ID"] = pd1["ID"]

    for i in range(pd2.shape[0]):
        a = pd2.loc[i, "ID"].split("_")
        if "B" in a or "2-3B" in a or "2-2B" in a or "2-1B" in a or "1-3B" in a or "1-2B" in a or "1-1B" in a or "3-1B" in a or "4B" in a:
            pd2.loc[i, "Benzene ring"] = 1
        if "4Br" in a or "2Br" in a:
            pd2.loc[i, "Bromine ion"] = 1
        if "4F" in a or "2F" in a:
            pd2.loc[i, "Fluorine ion"] = 1
        if "2NO2" in a:
            pd2.loc[i, "Nitro group"] = 1
        if "2NH2" in a:
            pd2.loc[i, "Amino"] = 1
        if "triazine" in a:
            pd2.loc[i, "Triazine"] = 1
        if "pyrene" in a or "fused" in a:
            pd2.loc[i, "Benzene ring"] = 1
        if "porphyrin" in a:
            pd2.loc[i, "Porphyrin"] = 1
        if "imadazoline" in a:
            pd2.loc[i, "Imidazole"] = 1
        pd2.loc[i, "Coordination carboxyl"] = 1
    pd3 = pd.merge(pd1, pd2, on="ID", how="inner")
    pd4 = pd.DataFrame(index=[x for x in range(0, pd1.shape[0])],
                       columns=["ID", "SO", "PO", "BO", "ECH", "EBP", "Catalyst Loading", "Substrate", "Time",
                                "Temperature", "Pressure"], data=0.0)
    pd4["ID"] = pd3["ID"]
    pd6 = pd.DataFrame(index=[x for x in range(0, pd1.shape[0])], columns=["ID", "TBAB", "TBAC", "TBAI"], data=0.0)
    pd6["ID"] = pd4["ID"]
    pd7 = pd.merge(pd3, pd6, on="ID", how="inner")
    pd8 = pd.read_csv("./DataResource/Generation_Processing/result.csv", encoding="gbk")
    pd8["pore"] = pd8["氦孔隙率"] * pd8["比容"]
    pd9 = pd.merge(pd7, pd8, on="ID", how="left")
    pd9.drop(["氦孔隙率", "比容"], axis=1, inplace=True)
    pd10 = pd.merge(pd9, pd4, on="ID", how="inner")
    pd10.to_csv("./DataResource/Generation_Processing/predict.csv")
