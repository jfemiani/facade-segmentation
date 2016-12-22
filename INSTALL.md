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
