# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_ Cycloaddition_Screening
@File        : main.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/11 9:51
"""
from ModelTraining import data_preprocessing
from ModelGeneration import model_generation
from MOFCalulation import mof_charge,mof_generation,generation_predict_data,surfacearea_porevolume
from Prediction import mof_screening
import os



def Generative_model():
    get_TOF = input("""
                    Please enter the TOF classification criteria.\n
                    The recommended range is between 5 and 80.\n
                    Enter L to select 25% quantile.\n
                    Enter M to select 50% quantile.\n
                    Enter H to select 75% quantile.\n
                    The default is 75% quantile.\n
                    """)
    if get_TOF == "L" or get_TOF == "l":
        get_TOF = 8.1683333335

    elif get_TOF == "M" or get_TOF == "m":
        get_TOF = 24.80555556

    elif get_TOF == "H" or get_TOF == "h":
        get_TOF = 51.595982145

    else:
        get_TOF = int(get_TOF)

    print("The grid search program is starting, please wait some time")
    data_preprocessing.training(classification_criteria=get_TOF)

    print("Grid search completed, model screening in progress")
    model_path = model_generation.model_gen()

    print("The model is generated.\n")
    print(model_path)
    print("Please select Generate MOF(Y) or return to main menu(N)")
    choose = input()
    if choose == "Y" or "y":
        Generative_MOF()
    else:
        get_command()


def Generative_MOF():
    command = input("Please select a data source:\n1.Self-contained data\n2.Program Generation\n")
    if command == "1":
        print("Please save the data in csv format and place it in the 'DataResource\To_Predict' directory")
    elif command == "2":
        mof_generation.generation()
        mof_charge.pac_cal()
        surfacearea_porevolume.get_input_file()
        surfacearea_porevolume.run_raspa()
        surfacearea_porevolume.search_data()
        surfacearea_porevolume.find_data()

def Predict_MOF():
    mof_screening.predict()


def get_command():
    print("Select the program you want to execute")
    command = input("1.Generative model\n2.Generative MOF\n3.Predict MOF\n4.Exit(Press any Key)\n")
    if command == "1":
        Generative_model()
    elif command == "2":
        Generative_MOF()
    elif command == "3":
        Predict_MOF()

if __name__ == '__main__':
    get_command()
