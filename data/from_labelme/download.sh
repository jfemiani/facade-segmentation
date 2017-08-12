echo
echo "Hey. in case you get an error about libcrypto or something, DO NOT use this with an anaconda environment."
echo "Also make sure the .pem file for vision.csi.miamioh.edu has been added using ssh-add"
echo

scp -r ubuntu@vision.csi.miamioh.edu:/var/www/html/LabelMe/Annotations .
scp -r ubuntu@vision.csi.miamioh.edu:/var/www/html/LabelMe/Images .

echo
echo "Alright, now you should import the images for training. See data/training/i12 for some scripts to do that."
echo
