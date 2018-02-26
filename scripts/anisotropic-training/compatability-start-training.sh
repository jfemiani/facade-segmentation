#!/usr/bin/env bash

INITIAL_WEIGHTS=${PWD}/test_weights_from_peihao.caffemodel

caffe.bin train -solver=compatability-solver.prototxt -weights ${INITIAL_WEIGHTS} -gpu=3