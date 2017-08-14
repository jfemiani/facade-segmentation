# from https://github.com/BVLC/caffe/blob/master/docker/standalone/gpu/Dockerfile
FROM nvidia/cuda:7.5-cudnn3-devel-ubuntu14.04
MAINTAINER femianjc@miamioh.edu


# Handy instructions:
# Install nvidia-docker
#    https://github.com/NVIDIA/nvidia-docker/wiki/Installation
#
# I will ASSUME you are in the top level directory of this repo when you type this...
#
# To build the image:
#
#    nvidia-docker build -t facades docker/gpu
#
# To start the container:
#
#    sudo nvidia-docker run -it -v ${PWD}:/workspace facades
#


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        vim && \
    rm -rf /var/lib/apt/lists/*

# Used by the facade processing shell scripts in order to package up our results
RUN apt-get install zip

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

RUN python -m pip install -U pip
RUN pip install scikit-learn -U
RUN pip install scikit-image -U

# NOTE:  Do not install any python packages after building segnet!
#        The risk of an incompatability is just too great.
#        Segnet ends up compiling against the particular version of numpy/opencv that
#        are present when it is build.


# NOTE: I _wanted_ to use the latest (or a leter) version of cudnn because I thought it could
#       improve performance, but TomoSaemann's repo does not unclude the modifications needed
#       to the Dropout layer that are needed for bayesian inference.
#
#       # ENV SEGNET_REPO=https://github.com/TimoSaemann/caffe-segnet-cudnn5
ENV SEGNET_REPO=https://github.com/alexgkendall/caffe-segnet
RUN git clone --depth 1 ${SEGNET_REPO} .
RUN for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build


# NOTE: I _could_ make a Makefile and usin the Dockerfile 'ADD' command instead
#       of appendining to the example like this, but this seems to work.
RUN cp Makefile.config.example Makefile.config
RUN echo USE_CUDNN := 1 >> Makefile.config
RUN echo WITH_PYTHON_LAYER := 1 >> Makefile.config
RUN make -j
RUN make -j python
RUN make -j pycaffe

#NOTE: I am considering shipping a Jupyter interface to help guide people through
#      the software, but for processing lot's of data I prefer bash
RUN pip install jupyter scipy

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

#NOTE: I am not sure that this line is necessary given the 'ldconfig' stuff above; I
#      had an unrelated issue and added this just-in-case
ENV LD_LIBRARY_PATH=${CAFFE_ROOT}/build/lib:$LD_LIBRARY_PATH


# This is used by the finish-training.sh script to set the I12 caffe model weights in an
# environment variable.
RUN apt-get install coreutils



#NOTE: I am not sure what the purpose of a 'VOLUME' in a Dockerfile is, as it seems that
#      you still have to
# This is a volume used to hold the output
# Use -v /path/to/output:/output in order to preserve the output
VOLUME /output

# This is a voume used to hold the data
# Use -v /path/to/data:/data in order to provide your data
VOLUME /data


# I generate plots (often to files), and this does not work by default on systems without
# an X-server setup properly (such as docker images...) so I am setting it up to use Agg for plotting.
ENV MATPLOTLIBRC ${HOME}/.config/matplotlib
RUN mkdir -p ${MATPLOTLIBRC}
RUN echo backend: Agg > ${MATPLOTLIBRC}/matplotlibrc


# Expose some ports for http or ipynb
# EXPOSE 80
# EXPOSE 8888

RUN \
  apt-get update && \
  apt-get install -y sudo curl git && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
  sudo apt-get install git-lfs

# Add files to the container
WORKDIR /opt
RUN git clone   https://github.com/jfemiani/facade-segmentation /opt/facades

ENV PYTHONPATH=/opt/facades:${PYTHONPATH}
RUN ln -s /opt/facades/scripts/i12-inference/generate /usr/bin/i12-inference
RUN ln -s /opt/facades/scripts/i12-inference/generate /usr/bin/inference

VOLUME /workspace
WORKDIR /workspace
CMD /bin/bash
