#!/usr/bin/env sh
$CAFFE_ROOT/build/tools/caffe train \
    -solver  solver_deep_supervision.prototxt \
    -weights /home/albertxavier/caffe_model/vgg16.caffemodel \
    -gpu 0
