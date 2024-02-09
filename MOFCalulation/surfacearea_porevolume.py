# -*- coding: utf-8 -*-
"""
PyCharm
@project_name: MOF_CO2_Cycloaddition_Screening
@File        : surfacearea_porevolume.py
@Author      : Xuefeng-Bai@BJUT
@Date        : 2023/2/10 10:08
"""

import os
import shutil
import csv
import re


def get_input_file():
    for cif_file in os.listdir("./DataResource/CIF_File"):
        os.makedirs(f"./DataResource/RASPA_run/{cif_file[:-4]}")
        shutil.copy(f"./DataResource/CIF_File/{cif_file}", f"./DataResource/RASPA_run/{cif_file[:-4]}/{cif_file}")
        with open(f"./DataResource/RASPA_run/{cif_file[:-4]}/simulation.input", mode="w") as inputfile:
            inputfile.write(f"""
        SimulationType        MonteCarlo
        NumberOfCycles        5
        PrintEvery            10
        PrintPropertiesEvery  10

        Forcefield            ExampleMOFsForceField
        CutOff 12.8

        Framework 0
        FrameworkName {cif_file[:-4]}
        UnitCells 1 1 1
        ExternalTemperature 298.0
        SurfaceAreaProbeDistance Sigma

        Component 0 MoleculeName             helium
                    MoleculeDefinition       ExampleDefinitions
                    WidomProbability         1.0
                    SurfaceAreaProbability   1.0
                    CreateNumberOfMolecules  0
        """)
        with open(f"./DataResource/RASPA_run/{cif_file[:-4]}/simulation.run", mode="w") as runfile:
            runfile.write("""
            #! /bin/sh -f
            export RASPA_DIR=${HOME}/RASPA/
            export DYLD_LIBRARY_PATH=${RASPA_DIR}/lib
            export LD_LIBRARY_PATH=${RASPA_DIR}/lib
            $RASPA_DIR/bin/simulate $1
            """)

def run_raspa():
    for i in os.listdir("./DataResource/RASPA_run"):
        print(str(i))
        os.chdir(f"./DataResource/RASPA_run/{i}")
        os.system(f"./run")
        # print(os.getcwd())
        os.chdir("./")
        # print(os.getcwd())

def find_data(f):
    obj1 = re.compile(r"""Average surface area.*?\\n',.*? \+""", re.DOTALL)
    ret1 = obj1.findall(f)
    # print(ret)
    obj2 = re.compile(r"Average Widom Rosenbluth-weight:.*?\[")
    ret2 = obj2.findall(f)

    obj3 = re.compile(r"Framework Density:.*?\].*?\[")
    ret3 = obj3.search(f)

    return ret1, ret2, ret3

def search_data():
    f1 = open("./DataResource/Generation_Processing/Surface are_Pore volume.csv", mode="w", newline="")
    f2 = csv.writer(f1)
    for i in os.listdir("./mofgen"):
        # print(f"./mofgen/{i}/Output/System_0/{i}_1.1.1_298.000000_0.data")
        try:
            with open(f"./DataResource/RASPA_run/{i}/Output/System_0/output_{i}_1.1.1_298.000000_0.data", mode="r") as f:
                try:
                    ret1, ret2, ret3 = find_data(str(f.readlines()))
                    f2.writerow([i, ret1, ret2, ret3.group()])
                except:
                    print("reading have wrong")
        except:
            print(f"{i} is not exist")

    f1.close()