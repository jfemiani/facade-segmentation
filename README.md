# SEGNET-FACADE

# Installation
See the Dockerfile in docker/gpu. 
You may may use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation) to run this code 
(run start-machine.sh) or you may consider the Dockerfile as instructions on how to configure your own machine. 
In this README I will assume you have followed the installation instructions for nvidia-docker which include putting your current user in 'dockers' group.

# Jupyter Demo
I have put jupyter on the docker image so that you can follow along with a simple tutorial in `scripts/process.ipynb`. 
I will try to describe how to use it here:
1. Figure out which port you want to use for a jupyter server; the default is port 8888, but you may already be serving a jupyter notebook on that port (if you are not currently using jupyter, just remember that the port is 8888).  Once you have settled on a port, let's save it to an environment variable:
```bash
export MYPORT=8888
```
2. You will probably want to produce output to a folder on your computer, and you may also want to provide your own inputs via a folder on your computer. For now, I will asume that that the input and output will be placed under your `/tmp` directory. Let's set some environment variables for our input and output locations. 
```bash
mkdir -p /tmp/segnet-facade/output
mkdir -p /tmp/segent-facade/input
export MYINPUT=/tmp/segent-facade/input
export MYOUTPUT=/tmp/segnet-facade/output
```
3. Now you can run a jupyter server using my docker image:
```bash
nvidia-docker run -v "${MYOUTPUT}":/output -v "${MYINPUT}":/data -p ${MYPORT}:${MYPORT} \
    jfemiani/segnet-facade jupyter notebook --allow-root --ip=* --port ${MYPORT} \
    /opt/facades/scripts/process.ipynb
```
4. You should see a link on your terminal to `http://localhost:8888/<a bunch of randomy text>`. Copy and paste the link into your browser. 


# Basic Demo

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
cp -r /opt/facades/data/cock-congo /data
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



