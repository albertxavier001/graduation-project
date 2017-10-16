import argparse
import cv2
import numpy as np


parser = argparse.ArgumentParser(description='LMSE Parameters')
parser.add_argument('-i', '--image', dest='im', type=str)
parser.add_argument('-g', '--truth', dest='gt', type=str)

args = parser.parse_args()

gt = cv2.imread(args.gt) / 255.
im = cv2.imread(args.im) / 255.

h,w,c = gt.shape
k = max(h,w) // 10
step = k // 2
lmse = 0.
cnt = 0

for hb in range(0, h-k, step):
	for wb in range(0, w-k, step):
		wd_im = im[hb:hb+k, wb:wb+k, :]
		wd_gt = gt[hb:hb+k, wb:wb+k, :]
		si = np.sum(wd_im * wd_gt) / np.sum(wd_im**2)
		lmse += np.average((wd_im*si - wd_gt)**2)
		cnt += 1
lmse  = lmse / cnt
print lmse