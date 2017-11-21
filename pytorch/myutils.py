import torch 
import cv2
import numpy as np


"""
    input: gt: tensor
    output: gt: variable
"""
def processGt(gt, scale_factor, dx=False, dy=False):
    s = scale_factor
    n,c,h,w = gt.shape
    gt = gt[0,:,:,:]
    gt = gt.transpose((1,2,0))
    # resize
    if s > 1:
        gt = cv2.resize(gt, (h//s, w//s))
    
    # make gradient
    if dx == True:
        pass

    if dy == True:
        pass

    gt = gt.transpose((2,0,1))
    gt = gt[np.newaxis, :]
    gt = Variable(torch.from_numpy(gt))


# test
if __name__ == '__main__':
    pass