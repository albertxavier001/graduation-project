import argparse
import cv2
import numpy as np


parser = argparse.ArgumentParser(description='LMSE Parameters')
parser.add_argument('-i', '--image', dest='im', type=str)
parser.add_argument('-g', '--truth', dest='gt', type=str)

args = parser.parse_args()

gt = cv2.imread(args.gt) / 255.
im = cv2.imread(args.im) / 255.
alpha = np.sum(im * gt) / np.sum(im**2)
si_mse = np.average((im*alpha - gt)**2)
print "si_mse = ", si_mse
si_mse




