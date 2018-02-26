#!/usr/bin/env bash

echo "Downloading 12-label dataset"
echo " (This includes preprocessed CMP and GSV images)"

curl curl -C - "http://handouts1.csi.miamioh.edu:82/vision/facades/{i12}.zip" -o "data/#1.zip" "http://handouts1.csi.miamioh.edu:82/vision/{dataset}.zip" -o "#data/1.zip"

echo "I am not extracting these right now; on my system I keep them in a particular location"