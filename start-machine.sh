#!/usr/bin/env bash

sudo nvidia-docker build -t facades docker/gpu

# This is an example of how I use the machine to process data.
#
# FIRST:
#    I put the images (they are sent to me as a zip file) into an input data folder.
# SECOND:
#    I create a text file

sudo nvidia-docker run -it \
    -v "/media/femianjc/My Book/facade-output/":/output \
    -v "/home/shared/Projects/Facades/src/data/":/data \
    facades


#    -v ${PWD}:/workspace \
