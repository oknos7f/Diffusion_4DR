import os
import shutil

# path = "D:/"

# for i in os.listdir(path):
#     if "cam-" in i:
#         print(f"{path+i+'/'+os.listdir(path + i)[0]}")
#         for j in ["cam-left", "cam-right", "cam-rear"]:
#             try:
#                 shutil.rmtree(f"{path+i}/{j}")
#             except: pass
#         os.rename(f"{path+i+'/'+os.listdir(path + i)[0]}", f"{path+i[:-4]}")
#         os.rmdir(f"{path+i}")

# polar3d_00033.npy cam-front_00001.png

# for i in range(1, 7):
#     folder = path + str(i)
#     polars = [j.replace("polar3d_", "cam-front_").replace(".npy", ".png") for j in os.listdir(folder)]
#     pictures = os.listdir(f"{path}{i}_cam/")
#
#     removes = list(set(pictures) - set(polars))
#
#     for j in removes:
#         os.remove(f"{path}{i}_cam/{j}")

path = "../dataset/data/images/"

for i in os.listdir(path):
    file = path + i
    if i[-11:-9] != "05":
        os.remove(file)