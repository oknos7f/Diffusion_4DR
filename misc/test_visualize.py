import numpy as np
import os
import utils.data_processing as ct
import matplotlib.pyplot as plt

path = "../dataset/data/conditions/"
files = [path + i for i in os.listdir(path)]

data_path = files[700]
print(data_path)

data = np.load(data_path, mmap_mode='r')
data = ct.polar_to_cartesian(data, threshold=99)

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