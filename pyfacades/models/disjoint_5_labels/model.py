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

__FOLDER = os.path.abspath(os.path.dirname(__file__))
LAYERS = os.path.join(__FOLDER, 'facades_512_deploy.prototxt')
WEIGHTS = os.path.join(__FOLDER, 'facades_512.caffemodel')

net = caffe.Net(LAYERS, WEIGHTS, caffe.TEST)


