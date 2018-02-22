
echo "Downloading Caffe Models"

curl -C - "http://handouts1.csi.miamioh.edu:82/vision/facades/{i12-weights-caffe,crf-initial-weights-caffe,low-rank-weights-caffe,segnet5-weights-caffe,segnet_driving}.zip" -o "#1.zip"
#curl http://handouts1.csi.miamioh.edu:82/vision/facades/crf-initial-weights-caffe.zip -o crf-initial-weights-caffe.zip
#curl http://handouts1.csi.miamioh.edu:82/vision/facades/low-rank-weights-caffe.zip -o low-rank-weights-caffe.zip
#curl http://handouts1.csi.miamioh.edu:82/vision/facades/segnet5-weights-caffe.zip -o segnet5-weights-caffe.zip
#curl http://handouts1.csi.miamioh.edu:82/vision/facades/segnet_driving.zip -o segnet_driving.zip

echo "Extracting Caffe Models"

unzip i12-weights-caffe.zip
unzip crf-initial-weights-caffe.zip
unzip low-rank-weights-caffe.zip
unzip segnet5-weights-caffe.zip
unzip segnet_driving.zip
