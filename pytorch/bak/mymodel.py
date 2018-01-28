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
                                          kernel_size=1, stride=1, padding=0, bias=False))
        if pool_ks>1: self.add_module('pool', nn.AvgPool2d(kernel_size=pool_ks, stride=1, padding=(pool_ks-1)//2))


    
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
                 transition_scale=2, pretrained_scale=1, debug=False, gradient=False, deconv_ks=4):
        super(GradientNet, self).__init__()
        self.debug = debug
        final_channel = 3 
        if gradient == True: final_channel = 6
        # self.block_config = [(3,3,3),(3,3,3),(3,3,3),(6,6,6),(12,12,12)]
        self.block_config = [(4,4),(4,4),(4,4),(4,4),(4,4)]
        self.num_pretrained_features = [64,64,128,256,1024]
        self.pretrained_scale = pretrained_scale
        self.num_input_features = [64,64,128,128,256]
        # for i in range(len(self.num_input_features)): self.num_input_features[i] //= pretrained_scale 


        """ features channels after denseblocks """
        self.ch_after_DB = [0] * len(self.block_config)
        for i, blocks in enumerate(self.block_config):
            self.ch_after_DB[i] = self.calOutputChannel(self.num_input_features[i], blocks, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale) 

        self.upsample_config = [2*2,4*2,8*2,16*2,32*2]
        self.merge_config = (3,3,3)
        self.merge_v3_config = [(4,4),(4,4),(4,4),(4,4),(4,4),(4,4)]

        self.pretrained_model = PreTrainedModel(densenet)
        
        """ compress pretrained features """
        i=0; self.compress_pretrained_16M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        i=1; self.compress_pretrained_08M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        i=2; self.compress_pretrained_04M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        i=3; self.compress_pretrained_02M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        i=4; self.compress_pretrained_01M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        
        
        
        # upsample pretrained features
        grow_16M = 8
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

        self.compress16M = nn.Conv2d(self.num_input_features[0]+4*grow_16M, self.num_input_features[0], 1)

        # upsample pretrained features
        grow_08M = 8
        self.upsample_4M_for_8M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[2],grow_08M,1)),
            ('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))
        self.upsample_2M_for_8M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[3],grow_08M,1)),
            ('upsample1', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample2', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))
        self.upsample_1M_for_8M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[4],grow_08M,1)),
            ('upsample1', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample2', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample3', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))

        self.compress8M = nn.Conv2d(self.num_input_features[1]+3*grow_08M, self.num_input_features[1], 1)

        # upsample pretrained features
        grow_04M = 8
        self.upsample_2M_for_4M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[3],grow_04M,1)),
            ('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))
        self.upsample_1M_for_4M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[4],grow_04M,1)),
            ('upsample1', nn.Upsample(scale_factor=2, mode='bilinear')),
            ('upsample2', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))

        self.compress4M = nn.Conv2d(self.num_input_features[2]+2*grow_04M, self.num_input_features[2], 1)

        # upsample pretrained features
        grow_02M = 8
        self.upsample_1M_for_2M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[4],grow_02M,1)),
            ('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))

        self.compress2M = nn.Conv2d(self.num_input_features[3]+grow_02M, self.num_input_features[3], 1)

        i=0; self.denseblock16 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=1; self.denseblock08 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=2; self.denseblock04 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=3; self.denseblock02 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=4; self.denseblock01 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        
        
        """ to RGB """
        i=0; self.compress16_3ch = nn.Conv2d(self.ch_after_DB[i], final_channel, 1)
        i=1; self.compress08_3ch = nn.Conv2d(self.ch_after_DB[i], final_channel, 1)
        i=2; self.compress04_3ch = nn.Conv2d(self.ch_after_DB[i], final_channel, 1)
        i=3; self.compress02_3ch = nn.Conv2d(self.ch_after_DB[i], final_channel, 1)
        i=4; self.compress01_3ch = nn.Conv2d(self.ch_after_DB[i], final_channel, 1)
        

        """ merge v3 """
        """ use 3-channel """
        self.ch_after_mg = [0]*6
        i=4; self.ch_after_mg[i] = self.calOutputChannel(self.ch_after_DB[i], self.merge_v3_config[i], bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=3; self.ch_after_mg[i] = self.calOutputChannel(self.ch_after_DB[i], self.merge_v3_config[i], bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=2; self.ch_after_mg[i] = self.calOutputChannel(self.ch_after_DB[i], self.merge_v3_config[i], bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=1; self.ch_after_mg[i] = self.calOutputChannel(self.ch_after_DB[i], self.merge_v3_config[i], bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=0; self.ch_after_mg[i] = self.calOutputChannel(self.ch_after_DB[i], self.merge_v3_config[i], bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=5; self.ch_after_mg[i] = 64
        
        # i=4; self.deconv_01M_to_02M = nn.Sequential(OrderedDict([
        #     ('deconv', nn.ConvTranspose2d(self.ch_after_DB[i], self.ch_after_DB[i-1], 4, stride=2, padding=1))
        #     # ('norm', nn.BatchNorm2d())
        # ]))
        # i=3; self.deconv_02M_to_04M = nn.ConvTranspose2d(self.ch_after_mg[i], self.ch_after_DB[i-1], 4, stride=2, padding=1)
        # i=2; self.deconv_04M_to_08M = nn.ConvTranspose2d(self.ch_after_mg[i], self.ch_after_DB[i-1], 4, stride=2, padding=1)
        # i=1; self.deconv_08M_to_16M = nn.ConvTranspose2d(self.ch_after_mg[i], self.ch_after_DB[i-1], 4, stride=2, padding=1)
        # i=0; self.deconv_16M_to_32M = nn.ConvTranspose2d(self.ch_after_mg[i], 64, 4, stride=2, padding=1)

        i=4; self.deconv_01M_to_02M = _DeconvLayer(self.ch_after_DB[i], self.ch_after_DB[i-1])
        i=3; self.deconv_02M_to_04M = _DeconvLayer(self.ch_after_mg[i], self.ch_after_DB[i-1])
        i=2; self.deconv_04M_to_08M = _DeconvLayer(self.ch_after_mg[i], self.ch_after_DB[i-1])
        i=1; self.deconv_08M_to_16M = _DeconvLayer(self.ch_after_mg[i], self.ch_after_DB[i-1])
        i=0; self.deconv_16M_to_32M = _DeconvLayer(self.ch_after_mg[i], 64, 4)

        i=4; self.transition02M = _MyTransition(self.ch_after_DB[i-1]*2, self.ch_after_DB[i-1])
        i=3; self.transition04M = _MyTransition(self.ch_after_DB[i-1]*2, self.ch_after_DB[i-1])
        i=2; self.transition08M = _MyTransition(self.ch_after_DB[i-1]*2, self.ch_after_DB[i-1])
        i=1; self.transition16M = _MyTransition(self.ch_after_DB[i-1]*2, self.ch_after_DB[i-1])

        deconv=False
        self.merge_block_02M = self.build_blocks(self.merge_v3_config[3], self.ch_after_DB[3], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale, deconv=deconv)
        self.merge_block_04M = self.build_blocks(self.merge_v3_config[2], self.ch_after_DB[2], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale, deconv=deconv)
        self.merge_block_08M = self.build_blocks(self.merge_v3_config[1], self.ch_after_DB[1], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale, deconv=deconv)
        self.merge_block_16M = self.build_blocks(self.merge_v3_config[0], self.ch_after_DB[0], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale, deconv=deconv)
        # self.merge_block_32M = self.build_blocks(self.merge_v3_config[5], 64, ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        
        deconv_ks=deconv_ks
        i = 3; self.merge_toRGB_02M = nn.ConvTranspose2d(self.ch_after_mg[i], final_channel, deconv_ks, stride=2, padding=(deconv_ks-2)//2)
        i = 2; self.merge_toRGB_04M = nn.ConvTranspose2d(self.ch_after_mg[i], final_channel, deconv_ks, stride=2, padding=(deconv_ks-2)//2)
        i = 1; self.merge_toRGB_08M = nn.ConvTranspose2d(self.ch_after_mg[i], final_channel, deconv_ks, stride=2, padding=(deconv_ks-2)//2)
        i = 0; self.merge_toRGB_16M = nn.ConvTranspose2d(self.ch_after_mg[i], final_channel, deconv_ks, stride=2, padding=(deconv_ks-2)//2)
         
        # i = 5; self.merge_toRGB_32M = nn.Sequential(OrderedDict([
        #     ('toRGB', nn.ConvTranspose2d(self.ch_after_mg[0], final_channel, 4, stride=2, padding=1))
        # ]))

        """init weight"""
        # for m in self.modules():
        #     if isinstance(m, nn.ConvTranspose2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         # std = math.sqrt(2. / n)
        #         std = 1e-10
        #         m.weight.data.normal_(0, std)
        #         print ('ConvTranspose2d weight', std)
        
    def forward(self, ft_input, go_through_merge=False):
        ft_pretrained = self.pretrained_model(ft_input)

        ft_predict = [0]*len(ft_pretrained)
        ft_upsampled = [0]*len(ft_pretrained)
        
        """ compress pretrained features """
        if self.pretrained_scale > 1:
            i=0; ft_pretrained[i] = self.compress_pretrained_16M(ft_pretrained[i])
            i=1; ft_pretrained[i] = self.compress_pretrained_08M(ft_pretrained[i])
            i=2; ft_pretrained[i] = self.compress_pretrained_04M(ft_pretrained[i])
            i=3; ft_pretrained[i] = self.compress_pretrained_02M(ft_pretrained[i])
            i=4; ft_pretrained[i] = self.compress_pretrained_01M(ft_pretrained[i])

        if self.debug==True: 
            for i in range(len(ft_pretrained)): print(i, ft_pretrained[i].size())
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
        
        if self.debug==True: print('_16M', _16M.size())
        _16M = self.compress16M(_16M)

        upsampled_4M_for_8M = self.upsample_4M_for_8M(ft_pretrained[2])
        upsampled_2M_for_8M = self.upsample_2M_for_8M(ft_pretrained[3])
        upsampled_1M_for_8M = self.upsample_1M_for_8M(ft_pretrained[4])
        
        _8M = torch.cat([
            ft_pretrained[1],
            upsampled_4M_for_8M,
            upsampled_2M_for_8M,
            upsampled_1M_for_8M
        ], 1)
        
        if self.debug==True: print('_8M', _8M.size())
        _8M = self.compress8M(_8M)

        
        upsampled_2M_for_4M = self.upsample_2M_for_4M(ft_pretrained[3])
        upsampled_1M_for_4M = self.upsample_1M_for_4M(ft_pretrained[4])
        
        _4M = torch.cat([
            ft_pretrained[2],
            upsampled_2M_for_4M,
            upsampled_1M_for_4M
        ], 1)
        
        if self.debug==True: print('_4M', _4M.size())
        _4M = self.compress4M(_4M)

        upsampled_1M_for_2M = self.upsample_1M_for_2M(ft_pretrained[4])
        
        _2M = torch.cat([
            ft_pretrained[3],
            upsampled_1M_for_2M
        ], 1)
        
        if self.debug==True: print('_2M', _2M.size())
        _2M = self.compress2M(_2M)

        """ denseblocks for each scale """
        i = 0; ft_predict[i] = self.denseblock16(_16M)
        i = 1; ft_predict[i] = self.denseblock08(_8M)
        i = 2; ft_predict[i] = self.denseblock04(_4M)
        i = 3; ft_predict[i] = self.denseblock02(_2M)
        i = 4; ft_predict[i] = self.denseblock01(ft_pretrained[i])
        
        if self.debug==True: 
            for i in range(len(ft_predict)): print('after denseblocks', i, ft_predict[i].size())
        
        """ to RGB """
        RGB = [0] * 6
        i = 0; RGB[i] = self.compress16_3ch(ft_predict[i])
        i = 1; RGB[i] = self.compress08_3ch(ft_predict[i])
        i = 2; RGB[i] = self.compress04_3ch(ft_predict[i])
        i = 3; RGB[i] = self.compress02_3ch(ft_predict[i])
        i = 4; RGB[i] = self.compress01_3ch(ft_predict[i])
        
        
        if go_through_merge == False: return RGB, []
        
        if self.debug == True: print('go thr', go_through_merge)
        

        """merge v3"""
        merged_RGB = [0]*6
        # 02M
        i=3;
        ft_predict[i] = self.transition02M(torch.cat([self.deconv_01M_to_02M(ft_predict[i+1]) , ft_predict[i], merged_RGB[i+1]],1))
        ft_predict[i] = self.merge_block_02M(ft_predict[i])
        merged_RGB[i] = self.merge_toRGB_02M(ft_predict[i])
        if go_through_merge == '02M': return RGB, merged_RGB
        
        # 04M
        i=2;
        ft_predict[i] = self.transition04M(torch.cat([self.deconv_02M_to_04M(ft_predict[i+1]) , ft_predict[i], merged_RGB[i+1]],1))
        ft_predict[i] = self.merge_block_04M(ft_predict[i])
        merged_RGB[i] = self.merge_toRGB_04M(ft_predict[i])
        if go_through_merge == '04M': return RGB, merged_RGB
        
        # 08M
        i=1;
        ft_predict[i] = self.transition08M(torch.cat([self.deconv_04M_to_08M(ft_predict[i+1]) , ft_predict[i], merged_RGB[i+1]],1))
        ft_predict[i] = self.merge_block_08M(ft_predict[i])
        merged_RGB[i] = self.merge_toRGB_08M(ft_predict[i])
        if go_through_merge == '08M': return RGB, merged_RGB
        
        # 16M
        i=0;
        ft_predict[i] = self.transition16M(torch.cat([self.deconv_08M_to_16M(ft_predict[i+1]) , ft_predict[i], , merged_RGB[i+1]],1))
        ft_predict[i] = self.merge_block_16M(ft_predict[i])
        merged_RGB[i] = self.merge_toRGB_16M(ft_predict[i])
        if go_through_merge == '16M': return RGB, merged_RGB
        
        # 32M
        i=-1;
        merged_RGB[i] = merged_RGB[0]
        # ft_predict[-1] = self.deconv_16M_to_32M(ft_predict[i+1])
        # merged_RGB[i] = self.merge_toRGB_32M(ft_predict[i])
        return RGB, merged_RGB
