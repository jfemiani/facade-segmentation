# from https://github.com/BVLC/caffe/blob/master/docker/standalone/gpu/Dockerfile
FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
MAINTAINER caffe-maint@googlegroups.com

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

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

RUN python -m pip install -U pip
ENV SEGNET_REPO=https://github.com/TimoSaemann/caffe-segnet-cudnn5
RUN git clone --depth 1 ${SEGNET_REPO} . 
RUN for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build

RUN cp Makefile.config.example Makefile.config
RUN echo USE_CUDNN := 1 >> Makefile.config
RUN echo WITH_PYTHON_LAYER := 1 >> Makefile.config 
RUN make -j

RUN pip install jupyter scipy

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

VOLUME /workspace
WORKDIR /workspace

CMD /bin/bash
