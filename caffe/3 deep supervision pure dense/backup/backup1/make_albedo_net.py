from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe


def res_group(bottom, ks, nout, stride, pad, dropout, weight_filler=dict(type='msra'), project=False):
    # branch 1
    branch1 = bottom
    if project == True: 
        branch1 = L.Convolution(bottom, kernel_size=1, stride=1, num_output=nout, pad=0, bias_term=False, weight_filler=weight_filler)
        branch1 = L.BatchNorm(branch1, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        branch1 = L.Scale(branch1, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
        branch1 = L.ReLU(branch1, in_place=True)
    branch1 = L.Dropout(branch1, dropout_ratio=dropout)
    
    # branch 2
    branch2 = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, bias_term=True, weight_filler=weight_filler, bias_filler=dict(type='constant', value=0))
    branch2 = L.BatchNorm(branch2, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    branch2 = L.Scale(branch2, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    branch2 = L.ReLU(branch2, in_place=True)
    branch2 = L.Dropout(branch2, dropout_ratio=dropout)
    branch2 = L.Convolution(branch2, kernel_size=ks, stride=stride, num_output=nout, pad=pad, bias_term=True, weight_filler=weight_filler, bias_filler=dict(type='constant', value=0))
    branch2 = L.BatchNorm(branch2, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    branch2 = L.Scale(branch2, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    branch2 = L.ReLU(branch2, in_place=True)
    branch2 = L.Dropout(branch2, dropout_ratio=dropout)
    
    # add
    fuse = L.Eltwise(branch1, branch2)

    return fuse 

def dense_group(bottom, ks, nout, stride, pad, dropout, weight_filler=dict(type='msra')):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, bias_term=True, weight_filler=weight_filler, bias_filler=dict(type='constant', value=0))
    batch_norm = L.BatchNorm(conv, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    prelu = L.ReLU(scale, in_place=True)
    drop = L.Dropout(prelu, dropout_ratio=dropout)
    branch1 = bottom
    branch2 = drop

    return branch1, branch2

def add_layer(bottom, num_filter, dropout, res=False, project=False):
    if res == False:
        branch1, branch2 = dense_group(bottom, ks=3, nout=num_filter, stride=1, pad=1, dropout=dropout)
        concate = L.Concat(branch1, branch2, axis=1)
        return concate
    else:
        return res_group(bottom, ks=3, nout=num_filter, stride=1, pad=1, dropout=dropout, project=project)


def transition(bottom, num_filter, dropout, weight_filler=dict(type='msra')):
    conv = L.Convolution(bottom, kernel_size=1, stride=1, 
                    num_output=num_filter, pad=0, bias_term=False, weight_filler=weight_filler, bias_filler=dict(type='constant', value=0))
    batch_norm = L.BatchNorm(conv, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    prelu = L.ReLU(scale, in_place=True)
    if dropout > 1e-6: 
        drop = L.Dropout(prelu, dropout_ratio=dropout)
        return drop
    return prelu

def upsampleVGG(bottom, dropout, nout, upsample):
    # weight_filler = dict(type='gaussian', std=0.01)
    weight_filler = dict(type='msra')
    if upsample <= 1:
        if upsample == 1: 
            s, k, p = 1, 3, 1
        else:
            s, k, p = 2, 4, 1
        conv = L.Convolution(bottom, kernel_size=k, stride=s, num_output=nout, pad=p, bias_term=True, weight_filler=weight_filler)
        batch_norm = L.BatchNorm(conv, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
        prelu = L.ReLU(scale, in_place=True)
        drop = L.Dropout(prelu, dropout_ratio=dropout)
        return drop

    k = upsample * 2
    s = upsample
    conv = L.Convolution(bottom, kernel_size=1, stride=1, num_output=nout, pad=0, bias_term=False, weight_filler=weight_filler)
    batch_norm = L.BatchNorm(conv, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    prelu = L.ReLU(scale, in_place=True)
    deconv = L.Deconvolution(prelu, convolution_param=dict(kernel_size=k, stride=s, num_output=nout, group=nout, pad=(k-s)/2, bias_term=False, weight_filler=dict(type='bilinear')), param=[dict(lr_mult=0, decay_mult=0)])
    drop = L.Dropout(deconv, dropout_ratio=dropout)
    return drop

def make_VGG(bottom):
    # stage 1
    nout=64
    conv1_1 = L.Convolution(bottom, name='conv1_1', kernel_size=3,num_output=nout, pad=1)
    relu1_1 = L.ReLU(conv1_1, name='relu1_1')
    conv1_2 = L.Convolution(relu1_1, name='conv1_2', kernel_size=3,num_output=nout, pad=1)
    relu1_2 = L.ReLU(conv1_2, name='relu1_2')
    pool1 = L.Pooling(relu1_2, name='pool1', pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))

    # stage 2
    nout=128
    conv2_1 = L.Convolution(pool1, name='conv2_1', kernel_size=3,num_output=nout, pad=1)
    relu2_1 = L.ReLU(conv2_1, name='relu2_1')
    conv2_2 = L.Convolution(relu2_1, name='conv2_2', kernel_size=3,num_output=nout, pad=1)
    relu2_2 = L.ReLU(conv2_2, name='relu2_2')
    pool2 = L.Pooling(relu2_2, name='pool2', pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))

    # stage 3
    nout=256
    conv3_1 = L.Convolution(pool2, name='conv3_1', kernel_size=3,num_output=nout, pad=1)
    relu3_1 = L.ReLU(conv3_1, name='relu3_1')
    conv3_2 = L.Convolution(relu3_1, name='conv3_2', kernel_size=3,num_output=nout, pad=1)
    relu3_2 = L.ReLU(conv3_2, name='relu3_2')
    conv3_3 = L.Convolution(relu3_2, name='conv3_3', kernel_size=3,num_output=nout, pad=1)
    relu3_3 = L.ReLU(conv3_2, name='relu3_3')
    pool3 = L.Pooling(relu3_2, name='pool3', pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))

    # stage 4
    nout=512
    conv4_1 = L.Convolution(pool3, name='conv4_1', kernel_size=3,num_output=nout, pad=1)
    relu4_1 = L.ReLU(conv4_1, name='relu4_1')
    conv4_2 = L.Convolution(relu4_1, name='conv4_2', kernel_size=3,num_output=nout, pad=1)
    relu4_2 = L.ReLU(conv4_2, name='relu4_2')
    conv4_3 = L.Convolution(relu4_2, name='conv4_3', kernel_size=3,num_output=nout, pad=1)
    relu4_3 = L.ReLU(conv4_2, name='relu4_3')
    pool4 = L.Pooling(relu4_2, name='pool4', pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))

    # stage 5
    nout=512
    conv5_1 = L.Convolution(pool4, name='conv5_1', kernel_size=3,num_output=nout, pad=1)
    relu5_1 = L.ReLU(conv5_1, name='relu5_1')
    conv5_2 = L.Convolution(relu5_1, name='conv5_2', kernel_size=3,num_output=nout, pad=1)
    relu5_2 = L.ReLU(conv5_2, name='relu5_2')
    conv5_3 = L.Convolution(relu5_2, name='conv5_3', kernel_size=3,num_output=nout, pad=1)
    relu5_3 = L.ReLU(conv5_2, name='relu5_3')
    pool5 = L.Pooling(relu5_2, name='pool5', pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))

    return pool1, pool2, pool3, pool4, pool5

def AbdNet():
    growth_rate = 16
    dropout = 0.2
    vgg_nout = 64
    N = 5
    nchannels = 16
    imsize = 256
    msra = dict(type='msra')
    gs_1e_2 = dict(type='gaussian', std=0.01)
    # n = caffe.NetSpec()
    data, data2, albedo_diff_gt, albedo_gt = L.Python(ntop=4, \
        python_param=dict(\
            module='image_layer3_gradient',\
            layer='ImageLayer3',\
            param_str="{{'data_dir': '/home/albertxavier/dataset/sintel/images/', 'tops': ['data', 'data2', 'albedo_diff_gt', 'albedo_gt'],'seed': 1337,'split': 'train', 'list_file':'train_two_folds_split_scene.txt', 'mean_bgr': (104.00699, 116.66877, 122.67892), 'crop_size':({imsize},{imsize})}}".format(imsize=imsize)\
        )\
    )

    pool1, pool2, pool3, pool4, pool5 = make_VGG(data)

    # scale 2
    model = L.Convolution(data2, kernel_size=4, stride=2, 
                    num_output=96, pad=1, bias_term=True, weight_filler=msra, bias_filler=dict(type='constant', value=0))
    model = L.BatchNorm(model, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    model = L.Scale(model, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    model = L.ReLU(model, in_place=True)
    model = L.Pooling(model, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))
    model = L.Dropout(model, dropout_ratio=dropout)



    # concat VGG
    vgg1 = upsampleVGG(pool1, upsample = 2/4, dropout=dropout, nout=vgg_nout)
    vgg2 = upsampleVGG(pool2, upsample = 4/4, dropout=dropout, nout=vgg_nout)
    vgg3 = upsampleVGG(pool3, upsample = 8/4, dropout=dropout, nout=vgg_nout)
    vgg4 = upsampleVGG(pool4, upsample = 16/4, dropout=dropout, nout=vgg_nout)
    vgg5 = upsampleVGG(pool5, upsample = 32/4, dropout=dropout, nout=vgg_nout)

    model = L.Concat(model, vgg1, vgg2, vgg3, vgg4, vgg5, axis=1)


    # block 1: dense
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout, weight_filler=msra)

    # block 2: dense
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout, weight_filler=msra)

    # block 3: res
    # nchannels = int(nchannels * 0.6)
    # for i in range(N):
    #     if i == 0: project = True
    # else: project = False
    #     model = add_layer(bottom, nchannels, dropout, project=project)

    block 3: dense
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout, weight_filler=msra)

    # deep supervision
    model_deep = L.Convolution(model, kernel_size=1, stride=1, num_output=96, pad=0, bias_term=False, weight_filler=gs_1e_2, param=[dict(lr_mult=1, decay_mult=1)])
    model_deep = L.Deconvolution(model_deep,  convolution_param=dict(kernel_size=8, stride=4, num_output=3, pad=2, bias_term=True, weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0)), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    loss_deep = L.Python(\
        model_deep, albedo_gt,\
        loss_weight=1.0, ntop=1,\
        python_param=dict(\
            module='l2loss',\
            layer='L2LossLayer',\
        )\
    )
    # model = L.Concat(model, model_deep, propagate_down=[True, False])

    # block 4
    for i in range(N):
        model = add_layer(model, growth_rate, dropout)
        nchannels += growth_rate
    model = transition(model, nchannels, dropout=0., weight_filler=msra)

    # fuse feature
    model = L.Convolution(model, kernel_size=1, stride=1, num_output=96, pad=0, bias_term=False, weight_filler=gs_1e_2, bias_filler=dict(type='constant'))
    # upsample
    model = L.Deconvolution(model,  convolution_param=dict(kernel_size=8, stride=4, num_output=6, pad=2, bias_term=True, weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0)), param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])

    # loss
    loss = L.Python(\
        model, albedo_diff_gt,\
        loss_weight=1.0, ntop=1,\
        python_param=dict(\
            module='l2loss-gradient-hist',\
            layer='L2LossLayer',\
            param_str="{'display': True}"\
        )\
    )

    return to_proto(loss, loss_deep)

def make_net(suffix=""):
    # with open('vgg.txt', 'r') as f:
    #     vgg = f.read()
    with open('train_albedonet_{}.prototxt'.format(suffix), 'w') as f:
        f.write(str(AbdNet()))

def make_solver(suffix=""):
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    s.train_net = 'train_albedonet_{}.prototxt'.format(suffix)
    # s.test_net.append('test_densenet.prototxt')
    # s.test_interval = 800
    # s.test_iter.append(200)

    s.max_iter = 100000
    s.type = 'Nesterov'
    s.display = 1

    s.base_lr = 0.02
    s.momentum = 0.9
    s.weight_decay = 1e-4
    s.iter_size = 2

    s.lr_policy='multistep'
    s.gamma = 0.1
    s.stepvalue.append(int(0.15 * s.max_iter))
    s.stepvalue.append(int(0.30 * s.max_iter))
    s.stepvalue.append(int(0.80 * s.max_iter))

    s.solver_mode = caffe_pb2.SolverParameter.GPU

    s.snapshot_prefix = './snapshot/albedonet_{}'.format(suffix)
    s.snapshot = 5000
    
    solver_path = 'solver_{}.prototxt'.format(suffix)
    with open(solver_path, 'w') as f:
        f.write(str(s))

def make_train_bash(suffix=''):
    s = \
"""#!/usr/bin/env sh
$CAFFE_ROOT/build/tools/caffe train \\
    -solver  solver_{}.prototxt \\
    -weights /home/albertxavier/caffe_model/vgg16.caffemodel \\
    -gpu 0
""".format(suffix)
    path = 'train_{}.sh'.format(suffix)
    with open(path, 'w') as f:
        f.write(str(s))

if __name__ == '__main__':
    suffix = "deep_supervision"
    make_net(suffix)
    make_solver(suffix)
    make_train_bash(suffix)