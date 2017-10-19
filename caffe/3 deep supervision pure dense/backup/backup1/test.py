import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import caffe
import scipy.io as sio

# Mean BGR values of input
mean_bgr = np.array([104, 117, 123], dtype=np.float)

def get_pad_multiple_32(shape):
    # Get pad values for y and x axeses such that an image
    # has a shape of multiple of 32 for each side
    def _f(s):
        n = s // 32
        r = s % 32
        if r == 0:
            return 0
        return 32 - r
    return _f(shape[0]), _f(shape[1])

def bgr2rgb(img):
    # Convert an BGR image to RGB
    return img[..., ::-1]

def minmax_01(img):
    # Put values in a range of [0, 1]
    ma, mi = img.max(), img.min()
    print 'ma=',ma, 'mi=',mi
    return (img - mi) / (ma - mi)

def unpad_img(img, pad):
    # Crop an image according to pad, used in post_process
    y = img.shape[0] - pad[0]
    x = img.shape[1] - pad[1]
    return img[:y, :x, :]

def post_process(img, pad, i=0):
    # Post processes of Direct intrinsics net.
    # The output of direct intrinsics net is in log domain with bias +0.5, and in BGR order.
    #return unpad_img(minmax_01(bgr2rgb((np.exp(img[i]) - 0.5).transpose(1, 2, 0))), pad)
    # return unpad_img((bgr2rgb((np.exp(img[i]) - 0.5).transpose(1, 2, 0))), pad)
    return unpad_img(((((img[i]) - 0).transpose(1, 2, 0))), pad)

def generatePairwiseGroundTruth(image):
    diff_x = np.diff(image, axis=1);
    diff_y = np.diff(image, axis=0);
    
    diff = np.zeros((image.shape[0],image.shape[1],image.shape[2]*2))
    
    diff[0:image.shape[0]-1,:,0:3] = diff_y
    diff[:,0:image.shape[1]-1,3:6] = diff_x
    
    return diff

# def mySigmoid(x):
#     return 4./(1.+np.exp(-x*4./50.))-2.
def mySigmoid(x):
    return x
    # scale = 4.
    # return 2. * scale /(1.+np.exp(-x*4./50.))-scale

def invMySigmoid(y):
    return y
    # return -50.0/4.0*np.log((2.0-y)/(2.0+y))

def predict(net, img):
    # Predicting function

    # Pad an image so that it has a shape of a multiple of 32
    pad = get_pad_multiple_32(img.shape)
    pad_img = cv2.copyMakeBorder(img, 0, pad[0], 0, pad[1], cv2.BORDER_CONSTANT)

    # in_2 = generatePairwiseGroundTruth(pad_img)
    # in_2 = np.concatenate((in_2, pad_img), axis=2)
    in_2 = np.copy(pad_img)
    gray = (in_2[:,:,0]+in_2[:,:,1]+in_2[:,:,2])/3;
    gray = np.clip(gray, 10, 300);
    in_2[:,:,0] /= gray; in_2[:,:,1] /= gray;
    in_2[:,:,2] /= gray;
    in_2 = (in_2 - in_2.min()) / (in_2.max() - in_2.min())*255 - 128
    print 'in_.shape = ', pad_img.shape
    print 'in_2.shape = ', in_2.shape
    in_2 = np.concatenate((in_2, pad_img), axis=2)

    # Reshape and fill input with the image
    net.blobs['Python1'].reshape(1, 3, *pad_img.shape[:2])
    net.blobs['Python1'].data[0, ...] = np.rollaxis(pad_img - mean_bgr, 2)

    net.blobs['Python2'].reshape(1, 6, *in_2.shape[:2])
    net.blobs['Python2'].data[0, ...] = np.rollaxis(in_2, 2)
    
    # Predict and get the outputs albedo and shading
    out = net.forward()
    a = out['Deconvolution4'].copy()
    # a_super = out['Deconvolution5'].copy()

    # Appy post processes before returning
    # return post_process(a, pad), post_process(a_super, pad)
    return post_process(a, pad)

def save(img, name):
    cv2.imwrite(name, (img-img.min())/(img.max()-img.min())*255)

if __name__ == '__main__':
    # load network and solver

    modelfile = './snapshot/albedonet_deep_supervision_iter_94754.caffemodel'
    caffe.set_mode_gpu()
    net = caffe.Net('test_albedonet.prototxt', modelfile, caffe.TEST)

    # load images
    dataset = '/home/albertxavier/dataset/sintel/images'
    frame_index = '0001'
    scene = "bandage_1"
    scene = "alley_1"

    img = np.array(cv2.imread(dataset + '/clean/{}/frame_{}.png'.format(scene, frame_index)), dtype=np.float32)
    mask = np.array(cv2.imread(dataset + '/albedo_defect_mask/{}/frame_{}.png'.format(scene, frame_index)), dtype=np.float32)
    gt_albedo = np.array(cv2.imread(dataset + '/albedo/{}/frame_{}.png'.format(scene, frame_index)), dtype=np.float32)

    # predict
    E = 0.1
    # albedo, albedo_super = predict(net, img)
    albedo = predict(net, img)
    albedo = np.clip(albedo, -2+E, 2-E)
    # albedo_super = np.exp(albedo_super) - 0.5
    albedo = invMySigmoid(albedo)
    
    # ground truth gradient
    diff_gt = generatePairwiseGroundTruth(gt_albedo)
    
    # output result
    # save(albedo_super[:,:,0:3], "test_albedo_super.png")
    save(albedo[:,:,0:3], "test_albedo_gy.png")
    save(albedo[:,:,3:6], "test_albedo_gx.png")
    save(diff_gt[:,:,0:3], "test_albedo_gt_gy.png")
    save(diff_gt[:,:,3:6], "test_albedo_gt_gx.png")
    
    # compute L2 error
    print np.sum(((diff_gt - albedo)/255.)**2) / diff_gt.size