import argparse
import cv2
import numpy as np


??? argparse ???

gt = cv2.imread(args.gt).astype(np.float32) / 255.
im = cv2.imread(args.im).astype(np.float32) / 255.
print "mse = ", np.average((im - gt)**2)




