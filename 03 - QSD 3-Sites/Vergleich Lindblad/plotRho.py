import matplotlib.pyplot as plt
import scienceplots
import numpy as np
plt.style.use(["science"])
N = 3
lindblad_rho = np.loadtxt("rho_lind200.dat", delimiter=";", dtype=np.complex64)
qsd_rho = np.loadtxt("rho_ann200.dat", dtype=np.complex64)




fig, ax = plt.subplots(2,2, figsize=(12,9))

# Lindblad
axes = ax[1,0]
im = axes.matshow(np.real(lindblad_rho), cmap="Wistia")
#axes.set_title("Re($\\rho$)", fontsize=15)
axes.tick_params(labelsize=12)
for i in range(N):
    for j in range(N):
        c = np.round(np.real(lindblad_rho)[j,i], decimals=4)
        axes.text(i,j,str(c), va="center", ha="center", fontsize=11)
fig.colorbar(im, ax=axes)
axes.set_ylabel("$\\rho_{\\text{Lindblad}}$", fontsize=15)
#plt.imshow(np.real(rho), origin='lower')

axes = ax[1, 1]
im = axes.matshow(np.imag(lindblad_rho), cmap="Wistia")
#axes.set_title("Im($\\rho$)", fontsize=15)
axes.tick_params(labelsize=12)
#axes.set_xticks([0, 1, 2])
#axes.set_yticks([0, 1, 2])
for i in range(N):
    for j in range(N):
        c = np.round(np.imag(lindblad_rho)[j,i], decimals=4)
        axes.text(i,j,str(c), va="center", ha="center", fontsize=11)
fig.colorbar(im, ax=axes)


## Referenz
axes = ax[0, 0]
im = axes.matshow(np.real(qsd_rho), cmap="Wistia")
axes.set_title("Re($\\rho$)", fontsize=15)
axes.tick_params(labelsize=12)
for i in range(N):
    for j in range(N):
        c = np.round(np.real(qsd_rho)[j,i], decimals=4)
        axes.text(i,j,str(c), va="center", ha="center", fontsize=11)
fig.colorbar(im, ax=axes)
axes.set_ylabel("$\\rho_{\\text{ANN}}$", fontsize=15)
#plt.imshow(np.real(rho), origin='lower')

axes = ax[0, 1]
im = axes.matshow(np.imag(qsd_rho), cmap="Wistia")
axes.set_title("Im($\\rho$)", fontsize=15)
axes.tick_params(labelsize=12)
#axes.set_xticks([0, 1, 2])
#axes.set_yticks([0, 1, 2])
for i in range(N):
    for j in range(N):
        c = np.round(np.imag(qsd_rho)[j,i], decimals=4)
        axes.text(i,j,str(c), va="center", ha="center", fontsize=11)
fig.colorbar(im, ax=axes)

fig.savefig("lindblad_compare_100.png", dpi=400, format="png")
plt.show()