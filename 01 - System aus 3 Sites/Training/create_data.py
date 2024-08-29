from System3 import System
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
import preprocess
n_systems = 10**5
systems = []
num_workers = 1


def calc_system(system):
    psi = system.get_psi0()
    psi_input = preprocess.getInput(psi, E=system.E0, V=system.vnm, t=0, complex=True)
    with open("Train_Data/train_input.dat", "a") as f:
        f.write(";".join([f"{psi_input[i]}" for i in range(0, len(psi_input))]) + "\n")
    psi = system.propagate(psi, 0)
    psi = preprocess.preprocess_psi_t(psi)
    with open("Train_Data/train_output.dat", "a") as f:
        f.write(";".join([f"{psi[i]}" for i in range(0, len(psi))]) + "\n")
    #psi_t = np.concatenate((psi_t, np.array([psi])), axis=0)


    


# n zufällige Systeme erstellen
for i in tqdm(range(n_systems)):
    # parameter: v_nm, e0, T, t0, sys_nr
    system = System([0, 5], [-10, 10], i)
    calc_system(system)
    del system
