import numpy as np
import cv2, os

root = '/Volumes/xavier/dataset/sintel1/'
cate = 'alley_1'
frame = 'frame_0001.png'

clean_path = os.path.join(root, 'clean', cate, frame)
albedo_path = os.path.join(root, 'albedo', cate, frame)
shading_path = os.path.join(root, 'shading', cate, 'out_0001.png')

clean = cv2.imread(clean_path) / 255.
albedo = cv2.imread(albedo_path) / 255.
shading = cv2.imread(shading_path) / 255.

albedo = np.clip(albedo, 0.001, 1)
myshading = clean / albedo
mi, ma = myshading.min(), myshading.max()
myshading = (myshading - mi) / (ma - mi) 

mse = np.mean((shading - myshading) ** 2)

print mse
