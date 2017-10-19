import caffe
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

class L2LossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        # if len(bottom) != 3:
        #     raise Exception("Need two inputs to compute distance.")
        self.inv = 0
        print('>>>begin setup l2loss layer')
        print "param_str = ", self.param_str
        if self.param_str == "": self.param_str = "{\"display\":false}"
        params = json.loads(self.param_str)
        print params
        self.display_ = params.get('display', False)
        self.suffix = params.get('suffix', "")
        self.bp_mult = params.get('bp_mult', 1)

    def reshape(self, bottom, top):
        # check input dimensions match
        # if bottom[0].count != bottom[1].count:
        #     print 'bottom 0 shape', bottom[0].data.shape
        #     print 'bottom 1 shape', bottom[1].data.shape
        #     raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data[:,0:3,:,:], dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        self.C = 1.0

    def forward(self, bottom, top):
        #print 'begin forward'
        #self.diff[...] = bottom[0].data - np.log(bottom[1].data+0.5)
        predict = bottom[0].data
        self.diff[...] = predict - bottom[1].data
        if len(bottom) == 3: top[0].data[...] =  np.sum(self.diff**2 * bottom[2].data) / predict.size
        if len(bottom) == 2: top[0].data[...] =  np.sum(self.diff**2) / predict.size
        print "albedo loss = ", top[0].data[...]
        #print "true albedo loss = ", np.sum((bottom[0].data[:,0:3,:,:] - bottom[1].data[:,0:3,:,:] )**2 * bottom[2].data[:,0:3,:,:] ) / bottom[0].data.size * 2
        #top[0].data[...] = 1


        
        # im = np.copy(bottom[0].data[0,:,:,:])
        # im = im.transpose(1,2,0)
        # im = im
        # print "min =", im.min(), "  max =", im.max()
        # cv2.imwrite("/home/albertxavier/a/train_albedo.png", im*255.)

        # im = np.copy(bottom[1].data[0,:,:,:])
        # im = im.transpose(1,2,0)
        # im = im
        # cv2.imwrite("/home/albertxavier/a/gt_albedo.png", im*255.)


        # Y = bottom[0].data[0, 0, :, :]
        # U = bottom[0].data[0, 1, :, :]
        # V = bottom[0].data[0, 2, :, :]

        # im = np.copy(bottom[0].data[0,:,:,:])
        # im = np.exp(im)-0.5
        # im = im.transpose(1,2,0)
        # cv2.imwrite("albedo.png", im*255.)

        # Y = bottom[1].data[0, 0, :, :]
        # U = bottom[1].data[0, 1, :, :]
        # V = bottom[1].data[0, 2, :, :]

        # im = np.copy(bottom[1].data[0,:,:,:])
        # im = im.transpose(1,2,0)
        # im[:,:,0] = (3.-U-V)*Y
        # im[:,:,1] = V*Y
        # im[:,:,2] = U*Y
        # cv2.imwrite("/home/albertxavier/a/gt_albedo.png", im*255.)

        # print bottom[0].data.shape
       
        # print 'loss =', top[0].data
        # im = np.copy(bottom[0].data[0,:,:,:])
        # im = im.transpose(1,2,0)
        # im = im[:,:,::-1]
        # print "min =", im.min(), "  max =", im.max()
        # plt.figure()
        # plt.title('out')
        # # plt.imshow(im+0.5)
        # # plt.imshow(np.clip(im,0,1))
        # im = im+0.5
        # plt.imshow(np.clip(im,0,10000))
        # # plt.imshow((im-im.min())/(im.max()-im.min() ))

        # # plt.show()
        
        if self.display_ == False:
            return

        if self.inv % 4*8 == 0:
            im = np.copy(bottom[0].data[0,:,:,:])
            im = im.transpose(1,2,0)
            im = (np.exp(im )-0.5)*255.
            cv2.imwrite("albedo{}.png".format("_"+self.suffix), im)
            if self.inv != 0:
                self.inv = 0
            else:
                self.inv += 1
        else :
            self.inv += 1



        # im = np.copy(bottom[1].data[0,0:3,:,:])
        # im = im.transpose(1,2,0)
        # im = im[:,:,::-1]
        # im = np.exp(im)-0.5
        # print "min =", im.min(), "  max =", im.max()
        # plt.figure()
        # plt.title('gt-albedo')
        # plt.imshow(im/im.max())
        # plt.show()


    def backward(self, top, propagate_down, bottom):
        #print 'begin backward'
        #print propagate_down
        predict = bottom[0].data
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1.0
            else:
                sign = -1.0
            # s = np.mean(bottom[0].data)
            # s = np.exp(s) - 0.5
            # s = -np.sin((1-s)*np.pi)*50.0+51.0
            # mul = np.zeros_like(bottom[0].data)
            # mul[:,0,:,:] = s
            # mul[:,1,:,:] = s
            # mul[:,2,:,:] = s
            
            if len(bottom) == 3: diff1 = 2.0 * sign * self.diff  * bottom[2].data / predict.size
            if len(bottom) == 2: diff1 = 2.0 * sign * self.diff / predict.size
            bottom[i].diff[...] = diff1 * self.bp_mult
            # bottom[i].diff[:,0:3,:,:] = diff1
            # bottom[i].diff[:,3:6,:,:] = diff1
            # bottom[i].diff[:,6:9,:,:] = diff1
