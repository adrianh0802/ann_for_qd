from System import System
import numpy as np
import pandas as pd


# psi in real und imaginärteil aufgeteilt, für die KI
# nur für einen Zeitpunkt psi_t : (1, 100) --> (1,200)
def preprocess_psi_t(psi_t):
    real = np.real(psi_t)
    imag = np.imag(psi_t)
    psi_new = []
    for i in range(len(psi_t)):
        psi_new.append(real[i])
        psi_new.append(imag[i])
    psi_preproc = np.array(psi_new)
    return psi_preproc 


def get_H_neuron(system: System, t):
    #df = pd.read_csv("Data/sys_params.dat", sep=";")
    #params = df[df["sys_nr"] == sys_nr].values[0]
    # vnm-Wert für die Wechselwirkung
    vnm = system.vnm # Wechselwirkung
    e0 = system.E0 # E_max
    T = system.T # periodendauer
    t0 = system.t0 # offset
    B = system.B # Breite Potential
    N = system.N# Anzahl Sites
    mid = int(N/2)-1
    n = np.zeros(N)
    x = np.arange(mid - B,mid + B+1)
    n[mid - B: mid + B+1] = e0*np.sin(np.pi/(2*B) * (x-(mid - B)))**2 * np.sin(2*np.pi/T * t + t0)
    n = np.round(n, decimals=10)

    # Zusammenfügen beider
    H_tot = np.concatenate((n, [vnm]), axis=0)
    return H_tot


def getInput(psi, system, t, complex=True):
    if complex:
        psi_t = preprocess_psi_t(psi)
    else:
        psi_t = psi
    h = get_H_neuron(system, t)
    input_psi = np.concatenate((psi_t, h), axis=0)

    return input_psi