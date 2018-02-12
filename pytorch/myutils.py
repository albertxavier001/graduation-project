import torch 
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from graphviz import Digraph


class MyUtils(object):
    """docstring for MyUtils"""
    def __init__(self):
        super(MyUtils, self).__init__()

    """
        input: gt: tensor
        output: gt: variable
    """
    def tensor2Numpy(self, x, use_cuda=True):
        if use_cuda == True: 
            x = x.cpu().data.numpy()
        else:
            x = x.data.numpy()
        x = x[0,:,:,:]
        x = x.transpose((1,2,0))
        return x

    def numpy2Tensor(self, x):
        x = x.transpose((2,0,1))
        x = x[np.newaxis, :]
        x = Variable(torch.from_numpy(x))
        return x

    # def processGtEccv(self, gt, lambda_=None, return_image=True):
    #     gt = self.tensor2Numpy(gt)
    #     if lambda_ is not None: lambda_ = self.tensor2Numpy(lambda_)
        
    #     gradient = self.makeGradient(gt)

    #     h,w,c = gt.shape

    #     res = np.concatenate((gt, gradient, lambda_), axis=2)


    def save_snapshot(self, epoch, args, net, optimizer):
        torch.save({
            'epoch': epoch,
            'args' : args,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'snapshot{}/snapshot-{}.pth.tar'.format(args.gpu_num, epoch))


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
            if gd == True: 
                display += 0.5
        
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

    def makeGradientTorch(self, image, direction='x'):
        if direction == 'x':
            [n,c,h,w] = image.size()
            a = image[:,:,:,0:w-1]
            b = image[:,:,:,1:w]
            return a - b
        elif direction == 'y':
            [n,c,h,w] = image.size()
            a = image[:,:,0:h-1,:]
            b = image[:,:,1:h,:]
            return a - b

    def adjust_learning_rate(self, optimizer, args, epoch, beg, end, reset_lr=None):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        base_lr = args.base_lr
        for param_group in optimizer.param_groups:
            if reset_lr != None:
                param_group['lr'] = reset_lr
                continue
            param_group['lr'] = base_lr * (float(end-epoch)/(end-beg)) ** (args.power)
            if param_group['lr'] < 1.0e-8: param_group['lr'] = 1.0e-8

    def make_dot(self, var, params=None):
        """ Produces Graphviz representation of PyTorch autograd graph
        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function
        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        if params is not None:
            assert isinstance(params.values()[0], Variable)
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="10240,10240"), format='svg')
        seen = set()

        def size_to_str(size):
            return '('+(', ').join(['%d' % v for v in size])+')'

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    name = param_map[id(u)] if params is not None else ''
                    node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')
                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)
        add_nodes(var.grad_fn)
        return dot
# test
if __name__ == '__main__':
    pass