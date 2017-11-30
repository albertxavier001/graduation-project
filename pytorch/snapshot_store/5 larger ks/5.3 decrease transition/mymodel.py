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
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, ks=3):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=ks, stride=1, padding=(ks-1)//2, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, ks=3):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, ks=ks)
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
    

    def build_blocks(self, num_block, num_init_features, pool_ks=3, ks=3, bn_size=4, growth_rate=32, transition_scale=2):
        drop_rate = 0
        num_features = num_init_features
        features = nn.Sequential()
        for i, num_layers in enumerate(num_block):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, ks=ks)
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

    def __init__(self, densenet, use_gpu=True, bn_size=4, growth_rate=32, transition_scale=2, pretrained_scale=1):
        super(GradientNet, self).__init__()
        # self.block_config = [(3,3,3),(6,6,6),(12,12,12),(16,16,16),(24,24,24)]
        self.block_config = [(6,6,6),(6,6,6),(12,12,12),(16,16,16),(24,24,24)]
        self.num_pretrained_features = [64,64,128,256,1024]
        self.pretrained_scale = pretrained_scale
        # for i in range(len(self.num_input_features)): self.num_input_features[i] //= pretrained_scale 
        self.num_input_features = [16,16,64,64,128]


        """ features channels after denseblocks """
        self.num_upsample_input_features = [0] * len(self.block_config)
        for i, blocks in enumerate(self.block_config):
            self.num_upsample_input_features[i] = self.calOutputChannel(self.num_input_features[i], blocks, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale) 

        self.upsample_config = [2*2,4*2,8*2,16*2,32*2]
        self.merge_config = (3,3,3)
        self.merge_v3_config = (3,3,3)

        self.pretrained_model = PreTrainedModel(densenet)
        
        """ compress pretrained features """
        i=0; self.compress_pretrained_16M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        i=1; self.compress_pretrained_08M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        i=2; self.compress_pretrained_04M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        i=3; self.compress_pretrained_02M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        i=4; self.compress_pretrained_01M = nn.Conv2d(self.num_pretrained_features[i], self.num_input_features[i], 1)
        
        
        
        # upsample pretrained features
        self.upsample_8M_for_16M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[1],16,1)),
            ('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))
        self.upsample_4M_for_16M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[2],16,1)),
            ('upsample', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.upsample_2M_for_16M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[3],16,1)),
            ('upsample', nn.Upsample(scale_factor=8, mode='bilinear'))
        ]))
        self.upsample_1M_for_16M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[4],16,1)),
            ('upsample', nn.Upsample(scale_factor=16, mode='bilinear'))
        ]))

        self.compress16M = nn.Conv2d(self.num_input_features[0]+4*16, self.num_input_features[0], 1)

        # upsample pretrained features
        self.upsample_4M_for_8M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[2],16,1)),
            ('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))
        self.upsample_2M_for_8M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[3],16,1)),
            ('upsample', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.upsample_1M_for_8M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[4],16,1)),
            ('upsample', nn.Upsample(scale_factor=8, mode='bilinear'))
        ]))

        self.compress8M = nn.Conv2d(self.num_input_features[1]+3*16, self.num_input_features[1], 1)

        # upsample pretrained features
        self.upsample_2M_for_4M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[3],64,1)),
            ('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))
        self.upsample_1M_for_4M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[4],64,1)),
            ('upsample', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        self.compress4M = nn.Conv2d(self.num_input_features[2]+2*64, self.num_input_features[2], 1)

        # upsample pretrained features
        self.upsample_1M_for_2M = nn.Sequential(OrderedDict([
            ('compress', nn.Conv2d(self.num_input_features[4],256,1)),
            ('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        ]))

        self.compress2M = nn.Conv2d(self.num_input_features[3]+256, self.num_input_features[3], 1)

        i=0; self.denseblock16 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=7, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=1; self.denseblock08 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=5, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=2; self.denseblock04 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=3; self.denseblock02 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        i=4; self.denseblock01 = self.build_blocks(self.block_config[i], self.num_input_features[i], ks=3, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale)
        
        
        # upsample final

        # i=0; self.upsample16 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1)
        # i=1; self.upsample08 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=4, padding=2, output_padding=0, groups=1, bias=True, dilation=1)
        # i=2; self.upsample04 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=8, padding=4, output_padding=0, groups=1, bias=True, dilation=1)
        # i=3; self.upsample02 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=16, padding=8, output_padding=0, groups=1, bias=True, dilation=1)
        # i=4; self.upsample01 = nn.ConvTranspose2d(in_channels=self.num_upsample_input_features[i], out_channels=3, kernel_size=self.upsample_config[i], stride=32, padding=16, output_padding=0, groups=1, bias=True, dilation=1)

        
        i=0; self.compress16 = nn.Conv2d(self.num_upsample_input_features[i], 3, 1)
        i=1; self.compress08 = nn.Conv2d(self.num_upsample_input_features[i], 3, 1)
        i=2; self.compress04 = nn.Conv2d(self.num_upsample_input_features[i], 3, 1)
        i=3; self.compress02 = nn.Conv2d(self.num_upsample_input_features[i], 3, 1)
        i=4; self.compress01 = nn.Conv2d(self.num_upsample_input_features[i], 3, 1)
        
        i=0; self.upsample16 = nn.Sequential(OrderedDict([
                ('bilinear', nn.Upsample(scale_factor=2, mode='bilinear'))
            ]))
        i=1; self.upsample08 = nn.Sequential(OrderedDict([
                ('bilinear', nn.Upsample(scale_factor=4, mode='bilinear'))
            ]))
        i=2; self.upsample04 = nn.Sequential(OrderedDict([
                ('bilinear', nn.Upsample(scale_factor=8, mode='bilinear'))
            ]))
        i=3; self.upsample02 = nn.Sequential(OrderedDict([
                ('bilinear', nn.Upsample(scale_factor=16, mode='bilinear'))
            ]))
        i=4; self.upsample01 = nn.Sequential(OrderedDict([
                ('bilinear', nn.Upsample(scale_factor=32, mode='bilinear'))
            ]))

        """merge v1"""
        # self.merge = nn.Sequential()
        # self.merge_in_channels =  (3*len(self.block_config), 64, 32, 16)
        # self.merge_out_channels = (                      64, 32, 16,  3)
        # for i in range(0, len(self.merge_out_channels)): 
        #     self.merge.add_module('merge.norm.%d'%i, nn.BatchNorm2d(self.merge_in_channels[i])),
        #     self.merge.add_module('merge.relu.%d'%i, nn.ReLU(inplace=True)),
        #     self.merge.add_module('merge.conv.%d'%i, nn.Conv2d(in_channels=self.merge_in_channels[i], 
        #                         out_channels=self.merge_out_channels[i], kernel_size=1))
        # self.merge.add_module('merge.final', nn.Sigmoid())
        

        """merge v2"""
        self.merge = nn.Sequential()
        self.merge.add_module('merge_denseblock', self.build_blocks(self.merge_config, 3*len(self.block_config), pool_ks=1, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale))
        self.merge.add_module('merge_final_conv', nn.Conv2d(in_channels=self.calOutputChannel(3*len(self.block_config), self.merge_config, bn_size=bn_size, growth_rate=growth_rate, transition_scale=transition_scale), out_channels=3, kernel_size=1)) # (3,3,3)->85
        self.merge.add_module('merge_final_sigmoid', nn.Sigmoid())
        
        """ IMPORTTANT!!!"""
        return 

        """ merge v3 """
        """ use 3-channel """
        # self.deconv_1M_to_2M = nn.Sequential(OrderedDict([
        #     ('compress', nn.Conv2d(in_channels=???, out_channels=???, kernel_size=1)),
        #     ('deconv', nn.ConvTranspose2d(in_channels=???, out_channels=???, kernel_size=4, stride=2, padding=1))
        # ]))
        # self.deconv_block_2M = self.build_blocks(???)

        # self.deconv_2M_to_4M = nn.Sequential(OrderedDict([
        #     ('compress', nn.Conv2d(in_channels=???, out_channels=???, kernel_size=1)),
        #     ('deconv', nn.ConvTranspose2d(in_channels=???, out_channels=???, kernel_size=4, stride=2, padding=1))
        # ]))
        # self.deconv_block_4M = self.build_blocks(???)

        # self.deconv_4M_to_8M = nn.Sequential(OrderedDict([
        #     ('compress', nn.Conv2d(in_channels=???, out_channels=???, kernel_size=1)),
        #     ('deconv', nn.ConvTranspose2d(in_channels=???, out_channels=???, kernel_size=4, stride=2, padding=1))
        # ]))
        # self.deconv_block_8M = self.build_blocks(???)

        # self.deconv_8M_to_16M = nn.Sequential(OrderedDict([
        #     ('compress', nn.Conv2d(in_channels=???, out_channels=???, kernel_size=1)),
        #     ('deconv', nn.ConvTranspose2d(in_channels=???, out_channels=???, kernel_size=4, stride=2, padding=1))
        # ]))
        # self.deconv_block_16M = self.build_blocks(???)

        # self.deconv_16M_to_32M = nn.Sequential(OrderedDict([
        #     ('compress', nn.Conv2d(in_channels=???, out_channels=???, kernel_size=1)),
        #     ('deconv', nn.ConvTranspose2d(in_channels=???, out_channels=???, kernel_size=4, stride=2, padding=1))
        # ]))
        # self.deconv_block_32M = self.build_blocks(???)

        # self.merge.add_module('merge_final_conv', nn.Conv2d(in_channels=???, out_channels=3, kernel_size=1))
        # self.merge.add_module('merge_final_sigmoid', nn.Sigmoid())



    def forward(self, ft_input):
        ft_pretrained = self.pretrained_model(ft_input)

        ft_predict   = [0]*len(ft_pretrained)
        ft_upsampled = [0]*len(ft_pretrained)
        
        if self.pretrained_scale > 1:
            i=0; ft_pretrained[i] = self.compress_pretrained_16M(ft_pretrained[i])
            i=1; ft_pretrained[i] = self.compress_pretrained_08M(ft_pretrained[i])
            i=2; ft_pretrained[i] = self.compress_pretrained_04M(ft_pretrained[i])
            i=3; ft_pretrained[i] = self.compress_pretrained_02M(ft_pretrained[i])
            i=4; ft_pretrained[i] = self.compress_pretrained_01M(ft_pretrained[i])

        for i in range(len(ft_pretrained)): print(i, ft_pretrained[i].size())
        upsampled_8M_for_16M = self.upsample_8M_for_16M(ft_pretrained[1])
        upsampled_4M_for_16M = self.upsample_4M_for_16M(ft_pretrained[2])
        upsampled_2M_for_16M = self.upsample_2M_for_16M(ft_pretrained[3])
        upsampled_1M_for_16M = self.upsample_1M_for_16M(ft_pretrained[4])
        

        _16M = torch.cat([
            ft_pretrained[0],
            upsampled_8M_for_16M,
            upsampled_4M_for_16M,
            upsampled_2M_for_16M,
            upsampled_1M_for_16M
        ], 1)

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
        
        _8M = self.compress8M(_8M)

        
        upsampled_2M_for_4M = self.upsample_2M_for_4M(ft_pretrained[3])
        upsampled_1M_for_4M = self.upsample_1M_for_4M(ft_pretrained[4])
        
        _4M = torch.cat([
            ft_pretrained[2],
            upsampled_2M_for_4M,
            upsampled_1M_for_4M
        ], 1)
        
        _4M = self.compress4M(_4M)

        upsampled_1M_for_2M = self.upsample_1M_for_2M(ft_pretrained[4])
        
        _2M = torch.cat([
            ft_pretrained[3],
            upsampled_1M_for_2M
        ], 1)
        
        _2M = self.compress2M(_2M)


        i = 0; ft_predict[i] = self.denseblock16(_16M)
        i = 1; ft_predict[i] = self.denseblock08(_8M)
        i = 2; ft_predict[i] = self.denseblock04(_4M)
        i = 3; ft_predict[i] = self.denseblock02(_2M)
        i = 4; ft_predict[i] = self.denseblock01(ft_pretrained[i])
        
        i = 0; ft_predict[i] = self.compress16(ft_predict[i])
        i = 1; ft_predict[i] = self.compress08(ft_predict[i])
        i = 2; ft_predict[i] = self.compress04(ft_predict[i])
        i = 3; ft_predict[i] = self.compress02(ft_predict[i])
        i = 4; ft_predict[i] = self.compress01(ft_predict[i])
        
        i = 0; ft_upsampled[i] = self.upsample16(ft_predict[i])
        i = 1; ft_upsampled[i] = self.upsample08(ft_predict[i])
        i = 2; ft_upsampled[i] = self.upsample04(ft_predict[i])
        i = 3; ft_upsampled[i] = self.upsample02(ft_predict[i])
        i = 4; ft_upsampled[i] = self.upsample01(ft_predict[i])
        
        ft_concated = torch.cat(ft_upsampled, 1)
        ft_merged = self.merge(ft_concated)
        ft_output = ft_predict[0:5] + [ft_merged]
        return ft_output
