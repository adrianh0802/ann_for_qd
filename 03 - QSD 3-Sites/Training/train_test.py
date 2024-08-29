import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])

df_train = pd.read_csv("train_losses.csv", sep=";")
df_test = pd.read_csv("test_losses.csv", sep=";")
n_epochs = df_train.values[:,0]
train_loss = df_train.values[:,1]
test_loss = df_test.values[:,1]
ratio = test_loss/train_loss
plt.figure(figsize=(9,5))
plt.semilogy(n_epochs, train_loss, color="blue", label="Trainingsdaten")
plt.semilogy(n_epochs, test_loss, "--", color="red", label="Testdaten")
plt.xlabel("Epoche", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
#plt.grid()
plt.savefig("train_test.png", dpi=200, format="png")
plt.show()

#
plt.figure(figsize=(9,5))
plt.plot(n_epochs, ratio, color="blue")
plt.xlabel("Epoche", fontsize=12)
plt.ylabel("Verhätnisse der Fehler", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("train_test_ratio.png", dpi=200, format="png")
plt.show()