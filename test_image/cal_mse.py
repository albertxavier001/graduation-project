import numpy as np
import cv2

def mse(im,gt):
    # m = np.mean((a-b)**2)
    alpha = np.sum(im * gt) / np.sum(im**2)
    si_mse = np.average((im*alpha - gt)**2)
    return si_mse

ori = cv2.imread('out_merge.png')/255.
di = cv2.imread('direct_intrinsics.png')/255.
gt = cv2.imread('gt.png')/255.
res = cv2.imread('res.png')/255.

gt = gt[0:416,:,:]
di = di[0:416,:,:]

mse_ori = mse(ori, gt)
mse_res = mse(res, gt)
mse_di = mse(di, gt)

print('direct intrinsics', mse_di)
print('original', mse_ori)
print('result', mse_res)
