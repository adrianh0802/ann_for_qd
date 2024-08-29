from System10 import System
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
import preprocess
n_systems = 2*10**5
systems = []
num_workers = 1


def calc_system(system):
    psi = system.get_psi0()
    En = np.diag(system.get_H())
    psi_input = preprocess.getInput(psi, E=En, V=system.vnm, t=0, complex=True)
    with open("Train_Data/test_input.dat", "a") as f:
        f.write(";".join([f"{psi_input[i]}" for i in range(0, len(psi_input))]) + "\n")
    psi = system.propagate(psi)
    psi = preprocess.preprocess_psi_t(psi)
    with open("Train_Data/test_output.dat", "a") as f:
        f.write(";".join([f"{psi[i]}" for i in range(0, len(psi))]) + "\n")
    #psi_t = np.concatenate((psi_t, np.array([psi])), axis=0)


    


# n zufällige Systeme erstellen
for i in tqdm(range(n_systems)):
    # parameter: vnm_r, e0_r, width, N, sys_n
    system = System([0, 5], [-10, 10], width=[1,3], N=10, sys_n=i)
    calc_system(system)
    del system