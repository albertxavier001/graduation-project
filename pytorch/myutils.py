import torch 
import cv2
import numpy as np
import torch
from torch.autograd import Variable



class MyUtils(object):
    """docstring for MyUtils"""
    def __init__(self):
        super(MyUtils, self).__init__()

    """
        input: gt: tensor
        output: gt: variable
    """
    def processGt(self, gt, scale_factor=1, gd=False, return_image=True):
        s = scale_factor
        gt = gt[0,:,:,:]
        gt = gt.transpose((1,2,0))
        # resize
        while s > 1:
            h,w,c = gt.shape
            gt = cv2.resize(gt, (h//2, w//2))
            s //= 2
        
        # make gradient
        if gd == True:
            gt = self.makeGradient(gt)

        if return_image == True: 
            display = np.copy(gt)
            mi, ma = display.min(), display.max()
            if gd == True: 
                display = (display - display.max()) / (display.max()-display.min())
        
        gt = gt.transpose((2,0,1))
        gt = gt[np.newaxis, :]
        gt = Variable(torch.from_numpy(gt))

        if return_image == True: return gt, display
        return gt

    def makeGradient(self, image):
        diff_x = np.diff(image, axis=1);
        diff_y = np.diff(image, axis=0);
        diff = np.zeros((image.shape[0],image.shape[1],image.shape[2]*2))
        diff[0:image.shape[0]-1,:,0:3] = diff_y
        diff[:,0:image.shape[1]-1,3:6] = diff_x
        return diff.astype(np.float32);

# test
if __name__ == '__main__':
    pass