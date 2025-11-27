import matplotlib.pyplot as plt
import numpy as np

# 0, 1, 2, 3, 4, 5, 6,  7,   8,     9,     10
# r, a, e, x, y, z, pw, dop, idx_r, idx_a, idx_e

path = r'../dataset/data/radar_tensor/rpc_00033.npy'
data = np.load(path, mmap_mode='r')

print(data.shape)

coordinates = data[:, 3:6]
values = data[:, 6]
v_min, v_max = values.min(), values.max()
values = (values - v_min) / (v_max - v_min + 1e-8)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=30)

scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        s=20,  # 점 크기
        c='k',
        alpha=values,  # **투명도: value에 따라 다르게**
        marker='o'
    )

ax.set_title("sparse tensor", fontsize=16)
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)

plt.show()