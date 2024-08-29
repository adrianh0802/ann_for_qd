from System import QSDSystem
from time import time
import numpy as np
from tqdm import tqdm
N_x_arr = np.array([10, 25, 35, 50, 60, 75, 100, 150, 200, 450, 600, 750, 900, 1000])
t_steps = 500
N = 10
times = np.zeros(N)
time_sites = np.zeros(len(N_x_arr))
for i in tqdm(range(len(N_x_arr))):
    N_x = N_x_arr[i]
    for j in range(N):
        system = QSDSystem(N_x, 2, 0, 1, 1)
        start = time()
        
        for step in range(t_steps):
            if step == 0:
                system.propagate_psi(no_init=False)
            else:
                system.propagate_psi(no_init=True)
        end = time()
        times[j] = (end - start)
    time_sites[i] = np.mean(times)

time_save = np.concatenate((N_x_arr.reshape(-1,1), time_sites.reshape(-1,1)), axis=1)

np.savetxt("timing_numeric.dat", time_save, delimiter=";")