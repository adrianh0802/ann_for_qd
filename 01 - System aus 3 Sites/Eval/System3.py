import numpy as np
from numpy.random import uniform
class System:
    # Parameter des Konstruktors
    # vnm_r : Wechselwirkung von nächsten Nachbarn
    # e0_r : Amplitude des Potentials
    # T_r : Periodendauer des Potentials
    # t0_r : Offset im Potential: e0*(sin²(b*(x-c))*sin(2*pi/T * (t+t0)))
    # sys_n : Nummer des Systems (Nur für die spätere Zuordnung, da bei Berechnung ggf. mehrere Threads laufen)
    # dt: Schrittweite (default: 10⁻²)
    # In allen Variablen wird ein bestimmter Bereich angegeben, da dieser dann zufällig generiert werden soll
    # Format: [lower_bound, upper_bound]
    def __init__(self, vnm_r, e0_r, T_r, t0_r, sys_n, dt=10**-2):
        self.vnm = np.round(uniform(low=vnm_r[0], high=vnm_r[1]), 5)
        self.E0 = np.round(uniform(low=e0_r[0], high=e0_r[1]), 5)
        self.T = np.round(uniform(low=T_r[0], high=T_r[1]), 5)
        self.t0 = np.round(uniform(low=t0_r[0], high=t0_r[1]), 5)
        #self.dx = dx
        self.dt = dt
        #self.L = L
        x = np.arange(0, 3)
        # Anfangszustand - Gaußsches-Wellenpaket
        #psi0 = np.exp(-(x-self.mu)**2/(2*self.sx**2))*np.exp(-1j*self.k*x) 
        #norm = np.sum(np.abs(psi0)**2)
        self.num = sys_n
        # Grundzustand
        self.psi0 = np.zeros(3)
        self.psi0[0] = 1


    def get_H(self, t):
        # Hauptdiagonale
        n = np.zeros(3)
        n[1] = self.E0 * np.sin(2*np.pi/(self.T) * (t + self.t0))

        # Nebendiagonale : V_nm - nächster Nachbar WW - nur auf Nebendiagonalen
        v = self.vnm*np.ones(2)
        V = np.diag(v, k=1) + np.diag(v, k=-1)

        h = np.diag(n, k=0) + V
        return h
    
    def propagate(self, psi, t):
        # Hamiltonian
        H = self.get_H(t)
        # Zeitentwicklung
        dpsidt = lambda H, psi: -1j*H@psi

        # Runge-Kutta
        k1 = dpsidt(H, psi)
        k2 = dpsidt(H, psi + self.dt/2 * k1)
        k3 = dpsidt(H, psi + self.dt/2 * k2)
        k4 = dpsidt(H, psi + self.dt*k3)
        psi_temp = psi + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Neuen Zustand normieren
        #x = np.arange(0, self.L, self.dx)
        #norm = np.trapz(np.abs(psi_temp)**2, x)
        norm = np.sum(np.abs(psi_temp)**2)
        psi_new = psi_temp/np.sqrt(norm)

        return psi_new
        
    def params_output(self):
        return f"{self.num}; {self.E0}; {self.T}; {self.t0}; {self.vnm}; {self.dt} \n"

    def get_psi0(self):
        return self.psi0


