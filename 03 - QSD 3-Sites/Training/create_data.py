from System import SQDSystem
import numpy as np
from tqdm import tqdm
import numpy.random as rnd
# import multiprocessing as mp

# import preprocess
n_systems = 2*10 ** 4
systems = []
num_workers = 1


def calc_system(system, prop_z):
    if prop_z:
        system.prop_z()
    temp_inp = system.get_neuron_input()
    with open("Train_Data/data_input.dat", "a") as f:
        f.write(";".join([f"{temp_inp[elem]}" for elem in range(0, len(temp_inp))]) + "\n")
    system.propagate_psi(no_init=False)
    temp_out = system.get_psi_net()
    with open("Train_Data/data_output.dat", "a") as f:
        f.write(";".join([f"{temp_out[elem]}" for elem in range(0, len(temp_out))]) + "\n")
    # psi_t = np.concatenate((psi_t, np.array([psi])), axis=0)


# n zufällige Systeme erstellen
for i in tqdm(range(n_systems)):
    # parameter: N, vmn, en, gamma, Gamma
    system = SQDSystem(3, 2, 0, 1, 1)
    #times_p_z = rnd.randint(0,100)
    calc_system(system, True)
    del system


'''

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
'''