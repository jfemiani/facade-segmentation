#!/usr/bin/env bash

echo "Downloading Caffe Models"

# I put all of the files into one 'curl' command so that we ger 
# a unified progress message. 
# The `-C -` option prevents us from downloading files that
# we already have locally. 
curl -C - "http://handouts1.csi.miamioh.edu:82/vision/facades/{i12-weights-caffe,crf-deploy,crf-initial-weights-caffe,low-rank-weights-caffe,segnet5-weights-caffe,segnet_driving}.zip" -o "data/#1.zip"

echo "Extracting Caffe Models"

# The -n flag is used to prevent us from modifing files that 
# already exist UNLESS the one in the zip file is newer. 
unzip -n data/i12-weights-caffe.zip
unzip -n data/crf-deploy.zip
unzip -n data/crf-initial-weights-caffe.zip
unzip -n data/low-rank-weights-caffe.zip
unzip -n data/segnet5-weights-caffe.zip
unzip -n data/segnet_driving.zip
