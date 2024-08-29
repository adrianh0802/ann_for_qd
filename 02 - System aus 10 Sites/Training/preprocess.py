from System10 import System
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


def get_H_neuron(E, V, t):
    #df = pd.read_csv("Train_Data/sys_params.dat", sep=";")
    #params = df[df["sys_nr"] == sys_nr].values[0]
    # vnm-Wert für die Wechselwirkung
    #vnm = V

    # Bestimmen der Hauptdiagonalen
    # Parameter aus System für Potentialdefinition
    #E0 = E
    #T = params[2]
    #t0 = params[3]

    en = E

    # Zusammenfügen beider
    H_tot = np.concatenate((en, [V]), axis=0)
    return H_tot


def getInput(psi, E, V, t, complex=True):
    if complex:
        psi_t = preprocess_psi_t(psi)
    else:
        psi_t = psi
    h = get_H_neuron(E,V,t)
    input_psi = np.concatenate((psi_t, h), axis=0)

    return input_psi