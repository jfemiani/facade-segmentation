# SEGNET-FACADE

# Installation
See the Dockerfile in docker/gpu. 
You may may use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation) to run this code 
(run start-machine.sh) or you may consider the Dockerfile as instructions on how to configure your own machine. 

In this README I will assume you have followed the installation instructions for nvidia-docker which include putting your current user in 'dockers' group.

To start the docker image you may also use:
```
nvidia-docker run -it \
    -v "/media/femianjc/My Book/facade-output/":/output \
    -v "/home/shared/Projects/Facades/src/data/":/data \
    jfemiani/segnet-facade
```
which will put you in a bash shell, with the code from this git repo under `/opt/facades`

**NOTE:** You should, of course, replace those paths with ones that make sense on your system.

Once you are in the docker image's shell, type:
```bash
cp /opt/facades/data/cock-congo /data
cp /opt/facades/scripts/jobs/one-file.txt /data

inference /data/one-file.txt
```

If everything works, this will copy an example input file into 
the `/data` folder and produce output (including figures and a YAML file with facade elements) in the `/output` folder. 


Some explanations:
1.  The `-v` tag allows you to map folders on your host computer with 
    volumes in the docker image. 
2.  Most of the default-arguments to my scripts will assume that input 
    imagery is under the `/data` folder. Usually you can change this 
    by passing other paths or changing config files, but if you use a 
    docker container you can also just map whatever folder has your files 
    to the `/data ` volume in the docker image. 
3.  My scripts, by default, write output to the `/output` folder.

The `inference` script just runs a python module that does the real work, you can use that directly to get more control of the output. Try:

```bash
python -m pyfacades --help
```

To get the most up-to-date help on how to use the script. 
# Development

I am using PyCharm to develop this code on an Ubuntu Machine with a
Tesla K40 GPU. 



