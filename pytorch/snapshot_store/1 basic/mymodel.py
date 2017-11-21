import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict

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
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _MyTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, pool_ks=3):
        super(_MyTransition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=pool_ks, stride=1, padding=(pool_ks-1)//2))


    
class GradientNet(nn.Module):
    def build_blocks(self, num_block, num_init_features, pool_ks=3):
        bn_size = 4
        growth_rate = 32
        drop_rate = 0
        num_features = num_init_features
        features = nn.Sequential()
        for i, num_layers in enumerate(num_block):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            features.add_module('mydenseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            trans = _MyTransition(num_input_features=num_features, num_output_features=num_features // 2, pool_ks=pool_ks)
            features.add_module('mytransition%d' % (i + 1), trans)
            num_features = num_features // 2
#         return features.cuda()
        return features
    
    def __init__(self, pretrained_model, use_gpu=True):
        super(GradientNet, self).__init__()
        self.block_config = [(3,3,3),(6,6,6),(12,12,12),(16,16,16),(24,24,24)]
        self.num_input_features = [64,64,128,256,1024]
        self.upsample_config = [2*2,4*2,8*2,16*2,32*2]
        
        self.pretrained_model = pretrained_model
        # if use_gpu: self.pretrained_model.cuda()
        
        i=0; self.denseblock16 = self.build_blocks(self.block_config[i], self.num_input_features[i])
        i=1; self.denseblock08 = self.build_blocks(self.block_config[i], self.num_input_features[i])
        i=2; self.denseblock04 = self.build_blocks(self.block_config[i], self.num_input_features[i])
        i=3; self.denseblock02 = self.build_blocks(self.block_config[i], self.num_input_features[i])
        i=4; self.denseblock01 = self.build_blocks(self.block_config[i], self.num_input_features[i])
        
        # if use_gpu: self.denseblock16.cuda()
        # if use_gpu: self.denseblock08.cuda()
        # if use_gpu: self.denseblock04.cuda()
        # if use_gpu: self.denseblock02.cuda()
        # if use_gpu: self.denseblock01.cuda()
            
        self.num_upsample_input_features = [92,176,352,480,800]
#         for i in range(0, len(self.block_config)):

        i=0; self.upsample16 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1)
        i=1; self.upsample08 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=4, padding=2, output_padding=0, groups=1, bias=True, dilation=1)
        i=2; self.upsample04 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=8, padding=4, output_padding=0, groups=1, bias=True, dilation=1)
        i=3; self.upsample02 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=16, padding=8, output_padding=0, groups=1, bias=True, dilation=1)
        i=4; self.upsample01 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=32, padding=16, output_padding=0, groups=1, bias=True, dilation=1)

        # if use_gpu: self.upsample16.cuda()
        # if use_gpu: self.upsample08.cuda()
        # if use_gpu: self.upsample04.cuda()
        # if use_gpu: self.upsample02.cuda()
        # if use_gpu: self.upsample01.cuda()
        

        """merge v1"""
        self.merge = nn.Sequential()
        self.merge_in_channels =  (3*len(self.block_config), 64, 32, 16)
        self.merge_out_channels = (                      64, 32, 16,  3)
        for i in range(0, len(self.merge_out_channels)): 
            self.merge.add_module('merge.norm.%d'%i, nn.BatchNorm2d(self.merge_in_channels[i])),
            self.merge.add_module('merge.relu.%d'%i, nn.ReLU(inplace=True)),
            self.merge.add_module('merge.conv.%d'%i, nn.Conv2d(in_channels=self.merge_in_channels[i], 
                                out_channels=self.merge_out_channels[i], kernel_size=1))
        self.merge.add_module('merge.final', nn.Sigmoid())
        

        """merge v2"""
        # self.merge = nn.Sequential()
        # for i, block_num in enumerate([3,3,3]):
        #     self.merge.add_module('merge.final', self.build_blocks(block_num, 3*len(self.block_config)))
        # self.merge.add_module('merge.final', nn.Sigmoid())
        

    def forward(self, ft_input):
        ft_pretrained = self.pretrained_model(ft_input)

        ft_predict   = [0]*len(ft_pretrained)
        ft_upsampled = [0]*len(ft_pretrained)
        
        i = 0; ft_predict[i] = self.denseblock16(ft_pretrained[i])
        i = 1; ft_predict[i] = self.denseblock08(ft_pretrained[i])
        i = 2; ft_predict[i] = self.denseblock04(ft_pretrained[i])
        i = 3; ft_predict[i] = self.denseblock02(ft_pretrained[i])
        i = 4; ft_predict[i] = self.denseblock01(ft_pretrained[i])
        
        i = 0; ft_upsampled[i] = self.upsample16(ft_predict[i])
        i = 1; ft_upsampled[i] = self.upsample08(ft_predict[i])
        i = 2; ft_upsampled[i] = self.upsample04(ft_predict[i])
        i = 3; ft_upsampled[i] = self.upsample02(ft_predict[i])
        i = 4; ft_upsampled[i] = self.upsample01(ft_predict[i])
        
        ft_concated = torch.cat(ft_upsampled, 1)
        ft_merged = self.merge(ft_concated)
        ft_output = ft_upsampled + [ft_merged]
        return ft_output
