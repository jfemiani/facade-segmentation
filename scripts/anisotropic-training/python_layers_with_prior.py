"""
NOTE: Make sure PYTHONPATH is set so that we can find the pyfacades module
NOTE: You can export FOLD to control which fold we use for training and testing
"""
import os
from os.path import join
import numpy as np; import random

import pyfacades.models.independant_12_layers.caffe_layers
from pyfacades import PROJECT_ROOT
from prepare_crf_input import prepare, FEATURES

# Use the environment to pass variables in.....
FOLD = os.environ.get('FOLD', 1)
DATA_ROOT = '/media/femianjc/My Book/independant_12_layers'
FOLD_PATH = join(DATA_ROOT,'data/training/independant_12_layers/fold_{:02}'.format(FOLD))
TRAIN_PATH = join(FOLD_PATH, 'train.txt')
EVAL_PATH = join(FOLD_PATH, 'eval.txt')

def noop(*args): pass

class TrainInputLayerWithPrior(pyfacades.models.independant_12_layers.caffe_layers.InputLayer):
    def __init__(self, p_object, *args, **kwargs):
        super(TrainInputLayerWithPrior, self).__init__(p_object, *args, **kwargs)
        self.source = TRAIN_PATH
        self.files = [join(DATA_ROOT, l.strip()) for l in open(self.source).readlines()]
        
        self.num_channels = 3 + len(FEATURES)*3
        
        # We will not be doing random transforms, the book-keeping for 
        # remembering the prior results is too messy
        self._transform = noop
        
        self.priors = None
        
        print "Initialized 'TrainInputLayerWithPrior"
        print "   For fold number", FOLD
        print "   Getting files from", TRAIN_PATH
        print "   Found", len(self.files), "files."
        
    def forward(self, bottom, top):
        for i in range(self.batch_size):
            if self.verbose:
                print self.files[self.counter]
            
            if self.priors is None:
                filename = self.files[self.counter]
                self.counter += 1
                current = np.load(filename)

                data = prepare(current[:3])
                labels = current[3:]
                top[0].data[i, ...] = data
                for j in range(1, len(labels)):
                    if np.all(labels[j] == 1):
                        labels[j][0, :] = 0
                    top[j].data[i, ...] = labels[j]
            else:
                current = top[0].data[i]
                data = np.concatenate([current[:3], self.priors])
                labels = current[3:]
                self.proirs = None
                
                top[0].data[i, ...] = data
            


        # Reshuffle the images at the end of each epoch
        if self.counter >= len(self.files):
            random.shuffle(self.files)
            self.counter = 0
            self.epochs += 1