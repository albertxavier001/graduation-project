
# coding: utf-8

# In[1]:


import os, glob, platform, datetime, random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch import functional as F
# import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import cv2
from PIL import Image
from tensorboardX import SummaryWriter

import numpy as np
from numpy.linalg import inv as denseinv
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv as spinv
import scipy.misc

from myimagefolder import MyImageFolder
from mymodel import GradientNet
from myargs import Args


# In[2]:


def loadimg(path):
    im = Image.open(path).convert('RGB')
    print(im.size)
    im = transforms.ToTensor()(im)
    x = torch.zeros(1,3,416,1024)
    x[0,:,:,:] = im[:,0:416,0:1024]
    x = Variable(x)
    return x


# In[3]:


gpu_num = 2
gradient = False
type2 = 'rgb' if gradient == False else 'gd'

res_root = './results/images/'
scenes = glob.glob('/home/lwp/workspace/sintel2/clean/*')
cnt_albedo = 0
cnt_shading = 0
for scene in scenes:
    scene = scene.split('/')[-1]
    res_dir = os.path.join(res_root, scene)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for type_ in ['albedo', 'shading']:
        root = '/media/lwp/xavier/graduation_results/showcase_model/{}/{}/{}/'.format(scene, type_, type2)
        print (root+'snapshot-239.pth.tar')
        if not os.path.exists(root+'snapshot-239.pth.tar'): continue
        snapshot = torch.load(root+'snapshot-239.pth.tar')
        state_dict = snapshot['state_dict']
        args = snapshot['args']
        densenet = models.__dict__[args.arch](pretrained=True).cuda(gpu_num)
        
#         net.load_state_dict(state_dict)
#         net.train()
        net = None
        num = 40 if scene=='market_6' else 50
        for ind in range(1, num+1):
            if net is not None: del net
#             torch.cuda.empty_cache()
            net = GradientNet(densenet=densenet, growth_rate=32, 
                          transition_scale=2, pretrained_scale=4,
                         gradient=gradient).cuda(gpu_num)
            net.load_state_dict(state_dict)
            net.train()
            frame = 'frame_%04d.png'%(ind)
            print('/home/lwp/workspace/sintel2/clean/{}/{}'.format(scene, frame))
            im = loadimg('/home/lwp/workspace/sintel2/clean/{}/{}'.format(scene, frame)).cuda(gpu_num)
            _, mergeRGB = net(im.cuda(gpu_num), go_through_merge=True)
            merged = mergeRGB[5]
            merged = merged[0]
            merged = merged.cpu().data.numpy()
            print (merged.shape)
            merged = merged.transpose(1,2,0)
            print (merged.shape)
            dx = merged[:,:,0:3]
            res_frame = 'albedo_%04d.png'%(ind) if type_ == 'albedo' else 'shading_%04d.png'%(ind)
            cv2.imwrite(os.path.join(res_dir,res_frame), dx[:,:,::-1]*255)            


# In[ ]:


# if gradient == False:
#     merged = mergeRGB[5]
#     merged = merged[0]
#     merged = merged.cpu().data.numpy()
#     print (merged.shape)
#     merged = merged.transpose(1,2,0)
#     print (merged.shape)
#     dx = merged[:,:,0:3]
#     cv2.imwrite('out_merge.png', dx[:,:,::-1]*255)


# In[ ]:


# if gradient == True:
#     merged = mergeRGB[5]
#     merged = merged[0]
#     merged = merged.cpu().data.numpy()
#     print (merged.shape)
#     merged = merged.transpose(1,2,0)
#     print (merged.shape)
#     dy = merged[:,:,0:3]+0.5
#     dx = merged[:,:,3:6]+0.5
#     cv2.imwrite('out_merge_dx.png', dx[:,:,::-1]*255)
#     cv2.imwrite('out_merge_dy.png', dy[:,:,::-1]*255)

