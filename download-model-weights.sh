#!/usr/bin/env bash

echo "Downloading Caffe Models"

curl -C - "http://handouts1.csi.miamioh.edu:82/vision/facades/{i12-weights-caffe,crf-deploy,crf-initial-weights-caffe,low-rank-weights-caffe,segnet5-weights-caffe,segnet_driving}.zip" -o "data/#1.zip"

echo "Extracting Caffe Models"

unzip data/i12-weights-caffe.zip
unzip data/crf-deploy.zip
unzip data/crf-initial-weights-caffe.zip
unzip data/low-rank-weights-caffe.zip
unzip data/segnet5-weights-caffe.zip
unzip data/segnet_driving.zip
