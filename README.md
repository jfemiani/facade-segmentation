# facade-segmentation

# Installation
See the Dockerfile in docker/gpu. 
You may install use nvidia-docker to run this code (run start-machine.sh) or you may 
consider it as instructions on how to configure your own machine. 

# Development

I am using PyCharm to develop this code on an Ubuntu Machine with a
Tesla K40 GPU. 

# Example 
First, edit `start-machine.sh` so that you mount an appropriate folder 
for input as `/data` and one for output as `/output`. They can be the
same folders if you want to process data in-place.

```bash
source start-machine.sh
```
Then you should find yourself inside a bash shell of a docker image as root

To process a list of files use:

```bash
source inference ~/code/scripts/jobs/one-file.txt
```


