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

class ImageLayer3(caffe.Layer):

    def setup(self, bottom, top):
        """
        example: params = dict(nyud_dir="/path/to/data_dir", split="val",
                               tops=['color', 'hha', 'label'])
        """
        # config
        print('>>>begin setup image layer')
        print "param_str = ", self.param_str
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.split = params['split']
        self.tops = params['tops']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.list_file = params['list_file']
        self.mean_bgr = np.array(params['mean_bgr'])
        self.scale = params.get('scale', 1)
        # store top data for reshape + forward
        self.data = {}
        self.indices = open(self.list_file, 'r').read().splitlines()
        #print 'self.indices =', self.indices

        self.wcrop, self.hcrop = np.array(params['crop_size'])

        self.blur_ksize = 41
        self.blur_step = 200
        self.count = 1
        # tops: check configuration
        if len(top) != len(self.tops):
           raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # self.interval = random.randint(5, 20)
        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

        self.cnt = 0


    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        self.data[self.tops[0]], self.data[self.tops[1]], self.data[self.tops[2]], self.data[self.tops[3]] = \
            self.load(self.indices[self.idx])
        for i,t in enumerate(self.tops):
            top[i].reshape(1, *self.data[t].shape)

    def forward(self, bottom, top):
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, files):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """

        # Load image
        input_, albedo_gt, shading_gt, mask_ = files.split(' ')
        print "img = ", input_
        im = Image.open(self.data_dir + input_)
        im_albedo_gt = Image.open(self.data_dir + albedo_gt)
        im_shading_gt = Image.open(self.data_dir + shading_gt)

        # convert images to numpt format
        in_ = np.array(im, dtype=np.float32)
        albedo_gt = np.array(im_albedo_gt, dtype=np.float32)
        shading_gt = np.array(im_shading_gt, dtype=np.float32)
        # in_ = synthesis(i=in_, a=albedo_gt, s=shading_gt)

        # convert rgb to bgr format
        in_ = in_[:,:,::-1]
        albedo_gt = albedo_gt[:,:,::-1]

        # data augmentation: scale
        scaled_height = random.randint(int(self.wcrop) + 10, int(in_.shape[0]*2))
        scaled_width = int(1.0*scaled_height/in_.shape[0]*in_.shape[1])

        in_ = cv2.resize(in_, (scaled_width, scaled_height))
        albedo_gt = cv2.resize(albedo_gt, (scaled_width, scaled_height))
        # albedo_gt_r1 = cv2.resize(albedo_gt_r1, (scaled_width, scaled_height))

        # data augmentation: random crop
        h = in_.shape[0]
        w = in_.shape[1]


        woff = random.randint(0, w - self.wcrop - 1)
        hoff = random.randint(0, h - self.hcrop - 1)

        esp = 0.

        # clip
        in_ = in_[hoff:hoff + self.hcrop, woff:woff + self.wcrop, :]
        albedo_gt = albedo_gt[hoff:hoff + self.hcrop, woff:woff + self.wcrop, :]

        # copy
        albedo_gt_2 = np.copy(albedo_gt)
        in_2 = np.copy(in_);

        # log
        albedo_gt = mySigmoid(generatePairwiseGroundTruth(albedo_gt, 1))
        albedo_gt_2 = np.log(albedo_gt_2/255. + 0.5)


        # data augmentation: flip input image
        flip_mode = random.randint(0, 4)
        in_ = flipImage(in_, flip_mode)
        in_2 = flipImage(in_2, flip_mode)
        albedo_gt = flipImage(albedo_gt, flip_mode)
        albedo_gt_2 = flipImage(albedo_gt_2, flip_mode)

        gray = (in_2[:,:,0]+in_2[:,:,1]+in_2[:,:,2])/3;
        gray = np.clip(gray, 10, 300);
        in_2[:,:,0] /= gray; in_2[:,:,1] /= gray;
        in_2[:,:,2] /= gray;
        in_2 = (in_2 - in_2.min()) / (in_2.max() - in_2.min())*255 - 128
        gray = np.expand_dims(gray, axis=2)

        ## blur
        # if self.count % self.blur_step == 0: self.blur_ksize -= 4
        # if self.blur_ksize > 1:
        #     print "blur!!!", self.blur_ksize
        #     in_ = cv2.GaussianBlur(in_, (self.blur_ksize,self.blur_ksize), 0)
        #     albedo_gt_2 = cv2.GaussianBlur(albedo_gt_2, (self.blur_ksize,self.blur_ksize), 0)
        # self.count += 1

        cv2.imwrite('in.png', (in_[:,:,0:3]-in_[:,:,0:3].min())/(in_[:,:,0:3].max()-in_[:,:,0:3].min())*255)
        cv2.imwrite('in2.png', (in_2[:,:,0:3]-in_2[:,:,0:3].min())/(in_2[:,:,0:3].max()-in_2[:,:,0:3].min())*255)
        cv2.imwrite('albedo_gt.png', (albedo_gt_2[:,:,0:3]-albedo_gt_2[:,:,0:3].min())/(albedo_gt_2[:,:,0:3].max()-albedo_gt_2[:,:,0:3].min())*255)
        cv2.imwrite('albedo_diff_gt.png', (albedo_gt[:,:,0:3]-albedo_gt[:,:,0:3].min())/(albedo_gt[:,:,0:3].max()-albedo_gt[:,:,0:3].min())*255)

        lum = 1.
        in_ *= lum
        mean_bgr = self.mean_bgr
        in_ -= mean_bgr
        in_ *= self.scale

        in_2 = np.concatenate((in_2, in_), axis=2)




        return \
        in_.transpose((2,0,1)), \
        in_2.transpose((2,0,1)), \
        albedo_gt.transpose((2,0,1)),\
        albedo_gt_2.transpose((2,0,1))



def dspImage(array, title, mean):
    array = array.transpose((1,2,0))
    array = array[:,:,::-1]
    print array.shape
    plt.figure()
    plt.title(title)
    plt.imshow(np.array(array*255.0, dtype=np.uint8))
    plt.show()

def rotateImage(image, angle):
    image_center = tuple((image.shape[1]/2,image.shape[0]/2))
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR)
    return result

def flipImage(image, mode):
    if mode == 1:
        return cv2.flip(image, 1)
    if mode == 2:
        return cv2.flip(image, 0)
    if mode == 3:
        return cv2.flip(image, -1)
    else:
        return image

def generatePairwiseGroundTruth(image, radius):
    # kernel_size = 2 * radius + 1
    # kernel = -np.ones((kernel_size, kernel_size))
    # kernel[radius, radius] = 0
    # kernel /= kernel_size * kernel_size - 1
    # kernel[radius, radius] = 1
    # return cv2.filter2D(image, -1, kernel)

    diff_x = np.diff(image, axis=1);
    diff_y = np.diff(image, axis=0);
    # print "diff_y.shape = ", diff_y.shape
    diff = np.zeros((image.shape[0],image.shape[1],image.shape[2]*2))
    # print "image.shape = ", image.shape
    # print "diff.shape = ", diff.shape
    diff[0:image.shape[0]-1,:,0:3] = diff_y
    diff[:,0:image.shape[1]-1,3:6] = diff_x

    # cv2.imwrite("albedo_gy.png", (diff_y - diff_y.min()) / (diff_y.max()-diff_y.min())*255)
    # cv2.imwrite("albedo_gx.png", (diff_x - diff_x.min()) / (diff_x.max()-diff_x.min())*255)

    return diff;

def mySigmoid(x):
    return x
    # scale = 4.
    # return 2. * scale /(1.+np.exp(-x*4./50.))-scale


def synthesis(i, a, s):
    if random.randint(0,100) % 3 == 0:
        i2 = a * s
        i2 = (i2 - i2.min()) / (i2.max() - i2.min()) * 255
        return i2
    return i
