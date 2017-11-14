# https://github.com/pytorch/examples/blob/master/imagenet/main.py

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
from mymodel import PreTrainedModel, GradientNet

snapshot = torch.load('snapshot/snapshot-179.pth.tar')

print(snapshot)