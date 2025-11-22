import numpy as np
import os
import utils.data_processing as ct
import matplotlib.pyplot as plt

# 10136 ->
# 1421 /0500599.npy)
# 9887 /0500228.npy)

path = "../dataset/data/conditions/"
files = [path + i for i in os.listdir(path)]

num = 599
data_path = files[num - 34]
print(data_path)

data = np.load(data_path, mmap_mode='r')
data = ct.polar_to_cartesian(data, threshold=99, normalize_coord=True)
data = ct.voxelize(data, agg='max')

print(data.shape)

coordinates = data[:, :3]
values = data[:, 3]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=-180)

scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        s=20,  # 점 크기
        c='k',
        alpha=values,  # **투명도: value에 따라 다르게**
        marker='o'
    )

ax.set_title(f"{num}", fontsize=16)
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)

plt.show()