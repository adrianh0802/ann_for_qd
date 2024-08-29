import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "grid"])

s_abs = np.loadtxt("s_abs_10.dat")
s_phase =  np.loadtxt("s_phase_10.dat")

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.plot(np.arange(len(s_abs)), s_abs, color="blue")
ax2.plot(np.arange(len(s_phase)), np.abs(s_phase), color="blue")
plt.show()