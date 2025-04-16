#!/usr/local/bin/python3
import os
import pickle as pkl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator

with open('RES.pkl', 'rb') as fp:
    RES = pkl.load(fp)

x_ = np.linspace(-0.5,0.5,51)
y_ = np.linspace(-0.5,0.5,51)

xx_, yy_ = np.meshgrid(x_, y_)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(
    xx_, yy_, RES[:,0].reshape((51,51)),
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False
)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

