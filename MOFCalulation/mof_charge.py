# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_Cycloaddition_Screening
@File        : mof_charge.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/10 9:45
"""

import pacmof
import os

def pac_cal():
    list_cif = os.listdir("./DataResource/CIF_File")

    for i in list_cif:
        try:
            pacmof.get_charges_single_serial(f"./DataResource/CIF_File/{i}", create_cif=True)
        except:
            print(f"{i} is failed")