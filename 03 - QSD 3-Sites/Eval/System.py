import numpy as np
# system mit konstanten en - e_n = e_m (m != n) & e_n != e_n(t)
class SQDSystem:
    def __init__(self, N, vmn, en, gamma, Gamma, seed = None):
        self.N = N
        self.vmn = vmn
        self.en = en
        self.gamma = gamma
        self.Gamma = Gamma
        self.dt = 10**-3
        if seed is not None and type(seed) == int:
            np.random.seed(seed)
        self.e_mat = np.diag(self.en*np.ones(self.N))
        # Wechselwirkung mit direktem Nachbar V = Vnm*(delt(n, n+1) + delt(n, n-1))
        self.v_mat = np.diag(self.vmn*np.ones(self.N-1), k=1) + np.diag(self.vmn*np.ones(self.N-1), k=-1)
        self.std = np.sqrt(gamma / 2)  # standard-deviation for initial stochastic variable z
        self.z = np.random.normal(0, self.std, N).astype(np.complex64)
        self.psi = np.random.random(N) + 1j*np.random.random(N)
        self.psi = self.psi.astype(np.complex64)
        norm = np.sum(np.abs(self.psi)**2)
        self.psi /= np.sqrt(norm)

    def propagate_psi(self, no_init=False) -> None:
        psi_temp = self.psi.copy()  # psi(t)
        if no_init:
            self.prop_z()  # calc z(t+dt) for runge-kutta
        
        k1 = self._dpsi_n_dt_new(psi_temp)
        k2 = self._dpsi_n_dt_new(psi_temp + self.dt/2 * k1)
        k3 = self._dpsi_n_dt_new(psi_temp + self.dt/2 * k2)
        k4 = self._dpsi_n_dt_new(psi_temp + self.dt * k3)
        self.psi += self.dt/6* (k1 + 2*k2 + 2*k3 + k4)
        '''
        for n in range(self.N):
            k1 = self._dpsi_n_dt(n, psi_temp)
            k2 = self._dpsi_n_dt(n, psi_temp + self.dt / 2 * k1)
            k3 = self._dpsi_n_dt(n, psi_temp + self.dt / 2 * k2)
            k4 = self._dpsi_n_dt(n, psi_temp + self.dt * k3)
            self.psi[n] = self.psi[n] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # n-th component of psi(t + dt)
        '''
    def _dpsi_n_dt_new(self, psi):
        d_psi = -1j*((self.e_mat + self.v_mat)@psi) # H*psi
        d_psi += (np.conjugate(self.z) + np.abs(psi)**2)*psi
        temp = np.sum(np.abs(psi)**2 * (np.conjugate(self.z) + np.abs(psi)**2))
        d_psi -= temp*psi
        return d_psi
    
    def _dpsi_n_dt(self, n, psi):
        d_cn = 0
        # für wechselwirkungen mit direktem nachbar Vnm = Vnm*(delt(n, n+1) + delt(n, n-1))
        if n == 0:
            d_cn += -1j * (self.en * psi[n] + self.vmn * psi[n + 1])
        elif n == self.N - 1:
            d_cn += -1j * (self.en * psi[n] + self.vmn * psi[n - 1])
        else:
            d_cn += -1j * (self.en * psi[n] + self.vmn * (psi[n - 1] + psi[n + 1]))

        d_cn += (np.conjugate(self.z)[n] + np.abs(psi[n]) ** 2) * self.psi[n]
        temp_sum = np.sum(np.abs(psi) ** 2 * (np.conjugate(self.z) + np.abs(psi) ** 2))
        d_cn -= temp_sum * psi[n]
        return d_cn

    # calculate the next z(t + dt)
    def prop_z(self) -> None:
        # Box-Mueller-Wiener Algorithmus - u = y1 + y2*i (Komplexe Gaußverteilung)
        # Numerical Recipes in C++ - The Art of Scientific Computing - Second Edition
        # (Press, Teukolsky, Vetterling, Flannery)
        # S. 293
        x1 = np.random.random(self.N)
        x2 = np.random.random(self.N)
        y1 = np.sqrt(-2 * np.log(x1)) * np.cos(2 * np.pi * x2)
        y2 = np.sqrt(-2 * np.log(x1)) * np.sin(2 * np.pi * x2)
        u = y1 + 1j * y2
        self.z = (self.z * np.exp(-self.gamma * self.dt) +
                  np.sqrt(self.Gamma * self.gamma / 2 * (1 - np.exp(-2 * self.gamma * self.dt))) * u)

    def get_psi(self) ->np.ndarray:
        return self.psi.copy()
    
    def get_psi_net(self):
        temp_psi = []
        real_psi = np.real(self.psi)
        imag_psi = np.imag(self.psi)
        for i in range(len(real_psi)):
            temp_psi.append(real_psi[i])
            temp_psi.append(imag_psi[i])
        output_psi = np.array(temp_psi).astype(np.float32)
        return output_psi
    
    def get_z_net(self):
        temp_psi = []
        real_z = np.real(self.z)
        imag_z = np.imag(self.z)
        for i in range(len(real_z)):
            temp_psi.append(real_z[i])
            temp_psi.append(imag_z[i])
        output_psi = np.array(temp_psi).astype(np.float32)
        return output_psi

    def get_neuron_input(self):
        temp_psi = []
        temp_z = []
        real_psi = np.real(self.psi)
        imag_psi = np.imag(self.psi)
        real_z = np.real(self.z)
        imag_z = np.imag(self.z)
        for i in range(len(real_psi)):
            temp_psi.append(real_psi[i])
            temp_psi.append(imag_psi[i])

        for i in range(len(real_z)):
            temp_z.append(real_z[i])
            temp_z.append(imag_z[i])
        
        #temp_z.append(self.gamma)
        #temp_z.append(self.Gamma)
        neuron_input = temp_psi + temp_z
        neuron_input = np.array(neuron_input).astype(np.float32)
        return neuron_input
    
    def get_z(self):
        return self.z.copy()