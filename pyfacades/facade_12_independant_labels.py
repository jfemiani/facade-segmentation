import caffe
import os

LABELS = ['background',
          'facade',
          'window',
          'door',
          'cornice',
          'sill',
          'balcony',
          'blind',
          'deco',
          'molding',
          'pillar',
          'shop']


MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
LAYERS = os.path.join(MODELS, 'facades_12_independent_labels.prototxt')
WEIGHTS = os.path.join(MODELS, 'facades_12_independent_labels.caffemodel')

net = caffe.Net(LAYERS, WEIGHTS, caffe.TEST)
