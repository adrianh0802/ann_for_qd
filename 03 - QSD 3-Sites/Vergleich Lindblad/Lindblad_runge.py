import numpy as np
# Idee für V_(n,m) von Paper - Dipolnäherung
from time import time
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science"])

N = 3
dt = 10**-3
gamma = 1 # Decay
v = 2 # Potential-Konstante
num_iter = 100


# H_ext - Hamilton-Operator
H = np.eye(N)*0
V1 = np.diag(np.ones(N-1)*v, k=1)
V2 = np.diag(np.ones(N-1)*v, k=-1)
V = V1+V2
'''
V = np.zeros((N, N)) # Dipolnäherung: V ~ 1/r³
for n in range(N):
    for m in range(N):
        if n!=m:
            V[n][m] = v/(np.abs(n-m)**3)
        else:
            continue
'''
H += V
H = H.astype(np.complex128)

def runge_rho_step(dt, rho, drhodt):
    k1 = drhodt(rho)
    k2 = drhodt(rho + dt/2*k1)
    k3 = drhodt(rho + dt/2*k2)
    k4 = drhodt(rho + dt*k3)
    rho = rho + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return rho


def lindblad(rho):
    res = -1j*(H@rho - rho@H)
    for n in range(N):
        Ln = np.zeros((N, N)); Ln[n, n] = 1
        Lna = np.conjugate(np.transpose(Ln))
        Ln = Ln.astype(np.complex128); Lna = Lna.astype(np.complex128)
        res += gamma/2 * (2*Ln@rho@Lna - Lna@Ln@rho - rho@Lna@Ln)
    return res


# Dichteoperator bei t=0

rho = np.zeros((N, N)) # rho_0
psi0 = np.array([1,0,0])
norm = np.sum(np.abs(psi0)**2)
psi0 = psi0 / np.sqrt(norm)
rho = np.outer(np.conjugate(psi0), psi0)
#rho2 = np.diag([1 for _ in range(N-1)], k=1)
#rho3 = np.diag([1 for _ in range(N-1)], k=-1)
#rho = rho + rho2 + rho3
#rho[0, 0] = 1
rho = rho.astype(np.complex64)

traces = [1]
vec1 = np.zeros(N).reshape((-1,1)); vec1[0] = 1
vec2 = np.zeros(N).reshape((-1,1)); vec2[1] = 1
vec3 = np.zeros(N).reshape((-1,1)); vec3[2] = 1
#vec4 = np.zeros(N).reshape((-1,1)); vec4[3] = 1
evolve1 = []
evolve2 = []
evolve3 = []
#evolve4= []

start = time()
for i in range(num_iter):
    rho = runge_rho_step(dt, rho, lindblad)
    traces.append(np.trace(rho))
    
    evolve1.append(np.real((rho@vec1)[0]))
    evolve2.append(np.real((rho @ vec2)[1]))
    evolve3.append(np.real((rho @ vec3)[2]))
    #evolve4.append(np.abs((rho @ vec4)[3]) ** 2)
end = time()
#print(f"{(end - start):.5f}")
print(f"(Trace-Mean  - 1): {np.mean(traces) - 1}")

#plt.figure(figsize=(9, 5))
#plt.plot(np.arange(0, num_iter*dt, dt), evolve1, label="Re($\\rho_{11}$)")
#plt.plot(np.arange(0, num_iter*dt, dt), evolve2, label="Re($\\rho_{22}$)")
#plt.plot(np.arange(0, num_iter*dt, dt), evolve3, label="Re($\\rho_{33}$)")
#plt.legend()
#plt.xlabel("t")
#plt.plot(np.arange(0, num_iter*dt, dt), evolve4)
#plt.plot(np.arange(0, num_iter*dt, dt), np.ones(num_iter)*0.01, "--", color="black")
#plt.show()
#np.savetxt("rho2.csv", rho, delimiter=";")

fig, ax = plt.subplots(1,2, figsize=(12,5))
axes = ax[0]
im = axes.matshow(np.real(rho), cmap="Wistia")
axes.set_title("Re($\\rho$)", fontsize=15)
axes.tick_params(labelsize=12)
for i in range(N):
    for j in range(N):
        c = np.round(np.real(rho)[j,i], decimals=4)
        axes.text(i,j,str(c), va="center", ha="center", fontsize=11)
fig.colorbar(im, ax=axes)
#plt.imshow(np.real(rho), origin='lower')

axes = ax[1]
im = axes.matshow(np.imag(rho), cmap="Wistia")
axes.set_title("Im($\\rho$)", fontsize=15)
axes.tick_params(labelsize=12)
#axes.set_xticks([0, 1, 2])
#axes.set_yticks([0, 1, 2])
for i in range(N):
    for j in range(N):
        c = np.round(np.imag(rho)[j,i], decimals=4)
        axes.text(i,j,str(c), va="center", ha="center", fontsize=11)
fig.colorbar(im, ax=axes)

#fig.savefig(f"lindblad-{num_iter}.png", dpi=200, format="png")
np.savetxt("rho_lind100.dat", rho, delimiter=";")
plt.show()

