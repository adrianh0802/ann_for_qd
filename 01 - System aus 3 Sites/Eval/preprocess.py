from System3 import System
import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# psi in real und imaginärteil aufgeteilt, für die KI
# nur für einen Zeitpunkt psi_t : (1, 100) --> (1,200)
def preprocess_psi_t(psi_t):
    real = torch.real(psi_t)
    imag = torch.imag(psi_t)
    psi_new = []
    for i in range(psi_t.shape[1]):
        psi_new.append(real[0,i])
        psi_new.append(imag[0,i])
    psi_preproc = torch.Tensor(psi_new)
    psi_preproc = torch.reshape(psi_preproc, (1,-1)).to(device)
    return psi_preproc 

def get_H_neuron(sys_nr, t):
    df = pd.read_csv("data/sys_params.dat", sep=";")
    params = df[df["sys_nr"] == sys_nr].values[0]
    # vnm-Wert für die Wechselwirkung
    vnm = params[4]

    # Bestimmen der Hauptdiagonalen
    # Parameter aus System für Potentialdefinition
    E0 = params[1]
    T = params[2]
    t0 = params[3]

    en = np.zeros(3)
    en[1] = E0 * np.sin(2*np.pi/(T) * (t + t0))
    #en[1] = E0
    # Zusammenfügen beider
    H_tot = np.concatenate((en, [vnm]), axis=0)
    H_tot = torch.from_numpy(H_tot).to(device)
    H_tot = torch.reshape(H_tot, (1,-1))
    return H_tot

def getInput(psi, sys_nr, t, complex=True):
    if complex:
        psi_t = preprocess_psi_t(psi)
    else:
        psi_t = psi
    h = get_H_neuron(sys_nr, t)
    input_psi = torch.cat((psi_t, h), axis=1)
    

    return input_psi



'''
df = pd.read_csv("Train_Data/psi_sys_0.dat", sep=";", header=None)
psi_0 = df.values[0,:].astype(np.complex64)
input_psi = np.concatenate((preprocess_psi_t(psi_0), get_H_neuron(0,0)), axis=0)
print(input_psi.shape)
print(input_psi)
'''
