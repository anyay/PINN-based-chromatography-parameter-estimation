import numpy as np
from matplotlib import pyplot

data = np.genfromtxt("test.dat", delimiter=' ')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

fig, ax = pyplot.subplots(nrows=1)
cntr = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax)
pyplot.show()
