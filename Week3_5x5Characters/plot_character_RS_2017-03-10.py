import matplotlib.pyplot as plt
import numpy as np

c = [1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 0, 0, 0, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 0, 1, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 1, 0, 0,
     1, 0, 0, 0, 0, 0, 0, 1, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 1]

c = np.reshape(c, (9, 9))

fig, ax = plt.subplots()
im = ax.matshow(c, cmap='Wistia')
locs = np.arange(len(c))
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_ticks(locs + 0.5, minor=True)
    axis.set(ticks=locs, ticklabels=np.arange(0, 9))
ax.grid(True, which='minor', color='black', linewidth=1)
plt.show()
