#!/usr/bin/env bash

ITER=$(./get_iter)
cp deploy/test_weights.caffemodel ../../models/facades_12_independent_labels.${ITER}.caffemodel
#ln  ../../models/facades_12_independent_labels.${ITER}.caffemodel  ../../models/facades_12_independent_labels.caffemodel
