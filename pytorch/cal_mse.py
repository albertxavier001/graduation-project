import numpy as np
import cv2
import os

def si_mse(im,gt):
	alpha = np.sum(im * gt) / np.sum(im**2)
	return np.mean((im*alpha-gt)**2)

scene = 'market_6'
frame = 10
sintel_root = '/home/lwp/workspace/sintel2/'

gt = cv2.imread(os.path.join(sintel_root, 'albedo', scene, 'frame_%04d.png'%(frame)))[0:416,:,:] / 255.
im = cv2.imread('out_merge.png') / 255.

print(si_mse(im,gt))
