# Installation

# Basics
* I work on ubuntu wih anaconda2
* Install [anaconda2](https://www.continuum.io/downloads)
* Install cuda-8.0 from NVvidia
* Install cudnn-3 (I put it in a local folder so it does not conflict with other projects that use cuda-5)
* Install caffe-segnet (look on github). It is an older fork of caffe tweaked for segnet
* Create a 'env.sh' script like the following:
```bash
#optional: source activate mypythonenvironment
export CAFFE_HOME=/path/to/segnet-caffe/distribute
export CUDA_HOME=/path/to/cuda-8
export CUDNN_HOME=/path/to/cudnn
export LD_LIBRARY_PATH=${CAFFE_HOME}/lib:${CUDA_HOME}/lib64:${CUDNN_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CAFFE_HOME}\bin:${PATH}
```
* use `source env.sh` in a new terminal before running this code. That way you will have access to the right versions of its dependancies. 

# Troublshooting
* If a module cannot be found, ```conda install -y modulename```. If that fails, ```pip install modulename```. 

# Examples

## Convert images so that they can be used by labelme.
```bash

python -m pyfacades.to_labelme --pat=*.png --rectify --debug ~/Downloads/10_April_4_Datasets

# Make a folder on the labelme server and copy the data
ssh ubuntu@vision.csi.miamioh.edu 'mkdir -p /var/www/html/LabelMe/Images/batch2-rectified'
scp ./data/for-labelme/*.jpg ubuntu@vision.csi.miamioh.edu:/var/www/html/LabelMe/Images/batch2-rectified

# Rebuild the list of images in the labelme collection
ssh ubuntu@vision.csi.miamioh.edu 'rm  /var/www/html/LabelMe/annotationCache/DirLists/labelme.txt'
ssh ubuntu@vision.csi.miamioh.edu 'cd  /var/www/html/LabelMe/annotationTools/sh && ./populate_dirlist.sh'

# Make the starter web pages for the annotators
ssh ubuntu@vision.csi.miamioh.edu 'cd  /var/www/html/LabelMe && . /make-tables.sh'
```