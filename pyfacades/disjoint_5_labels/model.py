import caffe
import os

LABELS = ['ignore',
          'window',
          'door',
          'facade-edge',
          'other-facade']

WINDOW = LABELS.index('window')
DOOR = LABELS.index('door')
FACADE_EDGE = LABELS.index('facade-edge')

MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
LAYERS = os.path.abspath(os.path.join(MODELS, 'facades_512_deploy.prototxt'))
print LAYERS
WEIGHTS = os.path.join(MODELS, 'facades_512.caffemodel')

net = caffe.Net(LAYERS, WEIGHTS, caffe.TEST)


