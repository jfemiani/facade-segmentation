"""
The version of caffe used by segnet does not allow parameters to be set for python layers
via their PythonLayer class.   We can configure it here.

NOTE: Make sure PYTHONPATH is set so that we can find the pyfacades module
NOTE: You can export FOLD to control which fold we use for training and testing
"""
import os
from os.path import join

import pyfacades.models.independant_12_layers.caffe_layers
from pyfacades import PROJECT_ROOT

# Use the environment to pass variables in.....
FOLD = os.environ.get('FOLD', 1)
DATA_ROOT = PROJECT_ROOT
FOLD_PATH = join(DATA_ROOT,'data/training/independant_12_layers/fold_{:02}'.format(FOLD))
TRAIN_PATH = join(FOLD_PATH, 'train.txt')
EVAL_PATH = join(FOLD_PATH, 'eval.txt')

def noop(*args): pass

class TrainInputLayer(pyfacades.models.independant_12_layers.caffe_layers.InputLayer):
    def __init__(self, p_object, *args, **kwargs):
        super(TrainInputLayer, self).__init__(p_object, *args, **kwargs)
        self.source = TRAIN_PATH
        self.files = [join(DATA_ROOT, l.strip()) for l in open(self.source).readlines()]
        
        self._transform = noop
        print "Initialized 'TrainInputLayer"
        print "   For fold number", FOLD
        print "   Getting files from", TRAIN_PATH
        print "   Found", len(self.files), "files."


class EvalInputLayer(pyfacades.models.independant_12_layers.caffe_layers.InputLayer):
    def __init__(self, p_object, *args, **kwargs):
        super(EvalInputLayer, self).__init__(p_object, *args, **kwargs)
        self.source = EVAL_PATH
        self.files = [join(DATA_ROOT, l.strip()) for l in open(self.source).readlines()]

        print "Initialized 'EvalInputLayer"
        print "   For fold number", FOLD
        print "   Getting files from", EVAL_PATH
        print "   Found", len(self.files), "files."

