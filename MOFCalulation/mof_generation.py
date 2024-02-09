# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_Cycloaddition_Screening
@File        : mof_generation.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/10 9:24
"""

import os
def generation():
    try:
        f = open("./DataResource/tobacco_path.txt", mode="r")
        path = f.readline()
    except:
        path = input("Please enter the tobacco program path:")
        with open("./DataResource/tobacco_path.txt", mode="w") as f:
            f.write(path)
    os.system(f"python {path}/tobacco.py")
