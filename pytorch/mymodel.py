""" model v2 """

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import math

class PreTrainedModel(nn.Module):
    def __init__(self, pretrained):
        super(PreTrainedModel, self).__init__()
        common_features_net = nn.Sequential(*list(pretrained.children())[0:1])
        self.net_16M = nn.Sequential(OrderedDict([
            ('conv0', common_features_net[0].conv0),
            ('norm0', common_features_net[0].norm0),
            ('relu0', common_features_net[0].relu0)
        ]))
        self.net_8M = nn.Sequential(OrderedDict([
            ('pool0', common_features_net[0].pool0)
        ]))
        self.net_4M = nn.Sequential(OrderedDict([
            ('denseblock1', common_features_net[0].denseblock1),
            ('transition1', common_features_net[0].transition1)
        ]))
        self.net_2M = nn.Sequential(OrderedDict([
            ('denseblock2', common_features_net[0].denseblock2),
            ('transition2', common_features_net[0].transition2)
        ]))
        self.net_1M = nn.Sequential(OrderedDict([
            ('denseblock3', common_features_net[0].denseblock3),
            ('transition3', common_features_net[0].transition3),
            ('denseblock4', common_features_net[0].denseblock4)
        ]))
    def forward(self, ft_32M):
        
        pretrained_features = [0]*5
        pretrained_features[0] = self.net_16M(ft_32M)
        pretrained_features[1]  = self.net_8M(pretrained_features[0])
        pretrained_features[2]  = self.net_4M(pretrained_features[1])
        pretrained_features[3]  = self.net_2M(pretrained_features[2])
        pretrained_features[4]  = self.net_1M(pretrained_features[3])
        return pretrained_features

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, ks=3, deconv=False):
        super(_DenseLayer, self).__init__()
        
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                    growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        
        if deconv == False:
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=ks, stride=1, padding=(ks-1)//2, bias=False)),
        else:
            self.add_module('deconv2', nn.ConvTranspose2d(bn_size * growth_rate, growth_rate,
                        kernel_size=ks, stride=1, padding=(ks-1)//2, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DeconvLayer(nn.Sequential):
    def __init__(self, ch_in, ch_out, bn_size=4, ks=4, stride=2, padding=1):
        super(_DeconvLayer, self).__init__()
        ch_mid = ch_out // 2* bn_size
        self.add_module('norm1', nn.BatchNorm2d(ch_in)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(ch_in, ch_mid, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(ch_mid)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('deconv', nn.ConvTranspose2d(ch_mid, ch_out,
                    kernel_size=ks, stride=stride, padding=(ks-1)//2, bias=False)),
        """init weight"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                std = math.sqrt(2. / n / 100.)
                # std = 1e-10
                m.weight.data.normal_(0, std)
                print ('_ ConvTranspose2d weight', std)
    def forward(self, x):
        new_features = super(_DeconvLayer, self).forward(x)
        return new_features


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, ks=3, deconv=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, ks=ks, deconv=deconv)
            self.add_module('denselayer%d' % (i + 1), layer)

class _MyTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, pool_ks=3):
        super(_MyTransition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        # if pool_ks>1: self.add_module('pool', nn.AvgPool2d(kernel_size=pool_ks, stride=1, padding=(pool_ks-1)//2))


    
class GradientNet(nn.Module):
    

    def build_blocks(self, num_block, num_init_features, pool_ks=3, ks=3, bn_size=4, growth_rate=32, transition_scale=2, deconv=False):
        drop_rate = 0
        num_features = num_init_features
        features = nn.Sequential()
        for i, num_layers in enumerate(num_block):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, ks=ks, deconv=deconv)
            features.add_module('mydenseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            trans = _MyTransition(num_input_features=num_features, num_output_features=num_features // transition_scale, pool_ks=pool_ks)
            features.add_module('mytransition%d' % (i + 1), trans)
            num_features = num_features // transition_scale
#         return features.cuda()
        return features
    
    def calOutputChannel(self, input_channel, blocks, bn_size=4, growth_rate=32, transition_scale=2):
        output_channel = input_channel
        for b in blocks:
            output_channel += b * growth_rate
            output_channel //= transition_scale
        return output_channel

    def __init__(self, densenet, use_gpu=True, bn_size=4, growth_rate=32, 
                 transition_scale=2, pretrained_scale=1, debug=False, gradient=False):
        super(GradientNet, self).__init__()
        self.debug = debug
        final_channel = 3 
        if gradient == True: final_channel = 6
        self.block_config = [(5,5),(5,5),(5,5),(5,5),(5,5)]
        self.num_pretrained_features = [64,64,128,256,1024]
        self.pretrained_scale = pretrained_scale
        self.num_input_features = [64,64,128,128,256]


        """ features channels after denseblocks """
        self.ch_after_DB = [0] * len(self.block_config)
        for i, blocks in enumerate(self.block_config):
            self.ch_after_DB[i] = self.calOutputChannel(self.num_input_features[i], blocks, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale) 

        self.upsample_config = [2*2,4*2,8*2,16*2,32*2]

        self.pretrained_model = PreTrainedModel(densenet)

        grow_16M = 16
        
        """ compress pretrained features """
        i=1; self.compress_pretrained_08M = nn.Conv2d(self.num_pretrained_features[i], grow_16M, 1)
        i=2; self.compress_pretrained_04M = nn.Conv2d(self.num_pretrained_features[i], grow_16M, 1)
        i=3; self.compress_pretrained_02M = nn.Conv2d(self.num_pretrained_features[i], grow_16M, 1)
        i=4; self.compress_pretrained_01M = nn.Conv2d(self.num_pretrained_features[i], grow_16M, 1)
        
        # upsample pretrained features
        self.upsample_8M_for_16M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[1],grow_16M,1)),
            ('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))
        self.upsample_4M_for_16M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[2],grow_16M,1)),
            ('upsample1', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample2', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))
        self.upsample_2M_for_16M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[3],grow_16M,1)),
            ('upsample1', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample2', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample3', nn.Upsample(scale_factor=2, mode='bilinear')),
        ]))
        self.upsample_1M_for_16M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[4],grow_16M,1)),
            ('upsample1', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample2', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample3', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample4', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))

        
        i=0; self.denseblock16 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        
        final_channel = 3 + 6 + 1
        i=0; self.merge_toRGB_32M = nn.ConvTranspose2d(self.ch_after_mg[i], final_channel, 4, stride=2, padding=1)

    def forward(self, ft_input):
        ft_pretrained = self.pretrained_model(ft_input)

        ft_predict = [0]*len(ft_pretrained)
        
        """ compress pretrained features """
        if self.pretrained_scale > 1:
            i=1; ft_pretrained[i] = self.compress_pretrained_08M(ft_pretrained[i])
            i=2; ft_pretrained[i] = self.compress_pretrained_04M(ft_pretrained[i])
            i=3; ft_pretrained[i] = self.compress_pretrained_02M(ft_pretrained[i])
            i=4; ft_pretrained[i] = self.compress_pretrained_01M(ft_pretrained[i])

        if self.debug==True: 
            for i in range(len(ft_pretrained)): print('compress pretrained', i, ft_pretrained[i].size())
        
        """ combine different scale features """
        upsampled_8M_for_16M = self.upsample_8M_for_16M(ft_pretrained[1])
        upsampled_4M_for_16M = self.upsample_4M_for_16M(ft_pretrained[2])
        upsampled_2M_for_16M = self.upsample_2M_for_16M(ft_pretrained[3])
        upsampled_1M_for_16M = self.upsample_1M_for_16M(ft_pretrained[4])
        
        if self.debug==True: 
            print('16M upsampled size', upsampled_8M_for_16M.size())
            print('16M upsampled size', upsampled_4M_for_16M.size())
            print('16M upsampled size', upsampled_2M_for_16M.size())
            print('16M upsampled size', upsampled_1M_for_16M.size())

        _16M = torch.cat([
            ft_pretrained[0],
            upsampled_8M_for_16M,
            upsampled_4M_for_16M,
            upsampled_2M_for_16M,
            upsampled_1M_for_16M
        ], 1)
        
        
        """ denseblocks for each scale """
        i=0; ft_predict[i] = self.denseblock16(_16M)
        
        if self.debug==True: 
            for i in range(len(ft_predict)): print('after denseblocks', i, ft_predict[i].size())
        
        
        i=0; res = self.merge_toRGB_32M(ft_predict[i])
        return res
