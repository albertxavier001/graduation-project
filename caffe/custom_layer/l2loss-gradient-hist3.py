# import caffe
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

import os

import sys
# sys.path.append('/home/flex/workspace/caffe/python')
from scipy import fftpack
import caffe
import glob
import numpy as np
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import cv2
import random
import json

class L2LossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        """
        example: params = dict(nyud_dir="/path/to/data_dir", split="val",
                               tops=['color', 'hha', 'label'])
        """
        print('>>>begin setup image layer')
        print "param_str = ", self.param_str
        if self.param_str == "": self.param_str = "{\"display\":false}"
        params = json.loads(self.param_str)
        print params
        self.display_ = params.get('display', False)
        self.suffix = params.get('suffix', "")
        self.bp_mult = params.get('bp_mult', 1)
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.inv = 0

        self.image_log = './image_log'
        if not os.path.exists(self.image_log):
            os.makedirs(self.image_log)

    def reshape(self, bottom, top):
        # check input dimensions match
        # if bottom[0].count != bottom[1].count:
        #     print 'bottom 0 shape', bottom[0].data.shape
        #     print 'bottom 1 shape', bottom[1].data.shape
        #     raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data[:,:,:,:], dtype=np.float32)
        self.hist_coeff = np.zeros_like(bottom[0].data[:,:,:,:], dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        self.C = 1.0

    def forward(self, bottom, top):

        predict = bottom[0].data
        ground_truth = bottom[1].data
        imageSize = predict.size

        (n, bins) = np.histogram(ground_truth, bins=20, normed=True)
        n *= (bins[1] - bins[0])

        pp = np.copy(predict[:,:,:,:])
        pp_shape = pp.shape
        pp = np.reshape(pp, (pp.size))
        # print pp.shape
        inds = np.searchsorted(bins, pp)
        inds = np.clip(inds,0,19)
        # print inds.max()
        hist_coeff = n[inds]
        hist_coeff = np.clip(hist_coeff, 0.001, 1)
        self.hist_coeff = np.reshape(hist_coeff, pp_shape)

        print 'l2 loss = ', np.sum((predict - ground_truth)**2) / imageSize
        self.diff[...] = (predict - ground_truth) * (self.hist_coeff)
        # top[0].data[...] =  np.sum(self.diff_r0**2 + self.diff_r1**2 + self.diff_r01**2) / (imageSize*3)
        top[0].data[...] =  \
            np.sum(self.diff**2) / imageSize

        if self.display_ == False:
            return

        if self.inv % 8 == 0:
            im_r0 = np.copy(predict[0,...])
            im_r1 = np.copy(ground_truth[0,...])
            im_r0 = im_r0 / 4 + 0.5
            im_r1 = im_r1 / 4 + 0.5
            im_r0 = im_r0.transpose(1,2,0)
            im_r1 = im_r1.transpose(1,2,0)

            cv2.imwrite(os.path.join(self.image_log, "albedo_predict_gy{}.png".format("_"+self.suffix)), im_r0[:,:,0:3]*255)
            cv2.imwrite(os.path.join(self.image_log, "albedo_ground_truth_gy{}.png".format("_"+self.suffix)), im_r1[:,:,0:3]*255)


            if self.inv != 0:
                self.inv = 0
            else:
                self.inv += 1
        else :
            self.inv += 1


    def backward(self, top, propagate_down, bottom):

        predict = bottom[0].data
        ground_truth = bottom[1].data
        imageSize = predict.size

        print 'predict    = ', predict.min(), predict.max()
        print 'ground_truth    = ', ground_truth.min(), ground_truth.max()
        print 'self.hist_coeff min max = ', self.hist_coeff.min(), self.hist_coeff.max()
        print ''

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                bottom[i].diff[:,:,:,:] = 2.0 * (self.diff[:,:,:,:]) / imageSize * self.bp_mult
