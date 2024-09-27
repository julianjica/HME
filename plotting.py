import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("results.txt", delimiter = " ")
a = a[a[:, 0].argsort()]

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

plt.plot(a[:,0], a[:,1], label = r"$a_1$")
plt.plot(a[:,0], a[:,2], label = r"$a_2$")
plt.legend()
plt.xlabel(r"$\sigma_d$")
plt.show()
plt.plot(a[:,0], a[:,2], label = r"$t_c$")
plt.plot(a[:,0], a[:,3], label = r"$t_d$")
plt.plot(a[:,0], a[:,4], label = r"$t_k$")
plt.legend()
plt.xlabel(r"$\sigma_d$")
plt.show()
