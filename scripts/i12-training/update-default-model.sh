#!/usr/bin/env bash

ITER=$(python -m pyfacades.get_iter /home/shared/Projects/Facades/mybook/facade_iter_*.caffemodel)
cp deploy/test_weights.caffemodel ../../models/facades_12_independent_labels.${ITER}.caffemodel
#ln  ../../models/facades_12_independent_labels.${ITER}.caffemodel  ../../models/facades_12_independent_labels.caffemodel
