#!/usr/bin/env sh
/home/lwp/workspace/direct-intrinsics/modified_caffe/caffe/build/tools/caffe train \
    -solver  solver.prototxt \
    -weights /home/lwp/workspace/caffe_model/vgg16.caffemodel \
    -gpu 3
