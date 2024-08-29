from System3 import System
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
n_systems = 100
systems = []
num_workers = 1
t_steps = 200
def calc_system(systems):
    for system in tqdm(systems): 
        psi = system.get_psi0()
        psi_t = np.array([psi])
        for i in range(t_steps):
            psi = system.propagate(psi, i*system.dt)
            psi_t = np.concatenate((psi_t, np.array([psi])), axis=0)
        np.savetxt(f"data/psi_sys_{system.num}.dat", psi_t, delimiter=";")

    


# n zufällige Systeme erstellen
for i in range(n_systems):
    # parameter: v_nm, e0, T, t0, sys_nr
    systems.append(System([0, 5], [-10, 10], [5, 20], [0, np.pi/2], i))


if __name__ == "__main__":
    processes = []
    N = len(systems)
    # Voraussetzung hier: N muss durch num_workers teilbar sein. Sonst geht es nicht 
    # (durch erhöhung/erniedrigung von N möglich)
    for i in range(num_workers):
        process = mp.Process(target=calc_system, args=(systems[int(N*i/num_workers): int(N*(i+1)/num_workers)], ))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    for system in systems:
        with open("Train_Data/sys_params.dat", "a") as file:
                    file.write(system.params_output())
