import numpy as np
import os
import utils.data_processing as ct
import matplotlib.pyplot as plt

path = "../dataset/data/conditions/    "
files = [path + i for i in os.listdir(path)]
print(files[0])

data = np.load(files[0])[0]
data = ct.polar_to_cartesian(data, threshold=0.1)

print(data[0])

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
p = data[:, 3]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, alpha=p)
ax.view_init(elev=20, azim=-180)
plt.show()