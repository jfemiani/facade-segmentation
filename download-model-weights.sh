
echo "Downloading Caffe Models"

curl http://handouts1.csi.miamioh.edu:82/vision/facades/i12-weights-caffe.zip 
curl http://handouts1.csi.miamioh.edu:82/vision/facades/crf-initial-weights-caffe.zip
curl http://handouts1.csi.miamioh.edu:82/vision/facades/i12-weights-caffe.zip 
curl http://handouts1.csi.miamioh.edu:82/vision/facades/low-rank-weights-caffe.zip 
curl http://handouts1.csi.miamioh.edu:82/vision/facades/segnet5-weights-caffe.zip 
curl http://handouts1.csi.miamioh.edu:82/vision/facades/segnet_driving.zip

echo "Extracting Caffe Models"

unzip i12-weights-caffe.zip
unzip crf-initial-weights-caffe.zip
unzip i12-weights-caffe.zip 
unzip low-rank-weights-caffe.zip
unzip segnet5-weights-caffe.zip
unzip segnet_driving.zip
