import os


# D:/1_cam/cam-front_00033.png
# D:/1/polar3d_00033.npy

def transform(src):
    if "cam" in src:
        return src.replace("_cam/cam-front_", "/polar3d_").replace(".png", ".npy")
    else:
        return src.replace("/polar3d_", "_cam/cam-front_").replace(".npy", ".png")
    
    
path = "D:/"
data_cp = []
data_cam = []

for _i in map(str, range(1,7)):
    folder = path + _i + "/"
    data_cp.append([folder + _j for _j in os.listdir(folder)])
    
for _i in map(str, range(1,7)):
    folder = path + _i + "_cam/"
    data_cam.append([folder + _j for _j in os.listdir(folder)])
