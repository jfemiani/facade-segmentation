import os
from math import sqrt, ceil
from os.path import isfile

import caffe
import numpy as np

from pyfacades.util import colorize

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

BACKGROUND = LABELS.index('background')
FACADE = LABELS.index('facade')
WINDOW = LABELS.index('window')
DOOR = LABELS.index('door')
CORNICE = LABELS.index('cornice')
SILL = LABELS.index('sill')
BALCONY = LABELS.index('balcony')
BLIND = LABELS.index('blind')
DECO = LABELS.index('deco')
MOLDING = LABELS.index('molding')
PILLAR = LABELS.index('pillar')
SHOP = LABELS.index('shop')

MODELS = os.path.dirname(__file__)
LAYERS = os.path.join(MODELS, 'facades_12_independent_labels.prototxt')
WEIGHTS = os.path.join(MODELS, 'facades_12_independent_labels.caffemodel')

# During training and testing, it is convenient to use different weights other than the
# weights in the git repo. For this, I use an environment variable.
#  (see $PROJECT/training/i12/finish-training.sh)
if 'I12_WEIGHTS' in os.environ and isfile(os.environ['I12_WEIGHTS']):
    WEIGHTS = os.environ['I12_WEIGHTS']


_net = None

JET_12 = np.array([[0,   0,    0],
                   [0,   0,   232],
                   [0,   56,  255],
                   [0,   148,  255],
                   [12,  244,  234],
                   [86,  255,  160],
                   [160, 255,  86],
                   [234, 255,  12],
                   [255, 170,  0],
                   [255, 85,   0],
                   [232, 0,    0],
                   [255, 255,  255]], dtype=np.uint8)


def net(weights=WEIGHTS):
    """
    Get the caffe net that has been trained to segment facade features.

    This initializes or re-initializes the global network with weights. There are certainly side-effects!

    The weights default to a caffe model that is part of the same sourcecode repository as this file.
    They can be changed by setting the I12_WEIGHTS environment variable, by passing a command line argument
    to some programs, or programatically (of course).

    :param weights: The weights to use for the net.
    :return:
    """
    global WEIGHTS
    global _net
    if _net is None or weights != WEIGHTS:
        if weights is not None:
            WEIGHTS = weights
        _net = caffe.Net(LAYERS, WEIGHTS, caffe.TEST)
    return _net


class FeatureMap(object):
    def __init__(self, features, confidence=None, image=None):
        super(FeatureMap, self).__init__()
        self.features = features
        self.image = image
        self.labels = LABELS
        self.colors = JET_12
        self.confidence = confidence

    def layers(self):
        return self.features.shape[0]

    def rows(self):
        return self.features.shape[1]

    def cols(self):
        return self.features.shape[2]

    def __getitem__(self, item):
        return self.features[item]

    def background(self):
        return self.features[BACKGROUND, 1]

    def facade(self):
        return self.features[FACADE, 1]

    def window(self):
        return self.features[WINDOW, 1]

    def door(self):
        return self.features[DOOR, 1]

    def cornice(self):
        return self.features[CORNICE, 1]

    def sill(self):
        return self.features[SILL, 1]

    def balcony(self):
        return self.features[BALCONY, 1]

    def blind(self):
        return self.features[BLIND, 1]

    def deco(self):
        return self.features[DECO, 1]

    def molding(self):
        return self.features[MOLDING, 1]

    def pillar(self):
        return self.features[PILLAR, 1]

    def shop(self):
        return self.features[SHOP, 1]

    def facade_edge(self):
        return self.features[FACADE, 2]

    def window_edge(self):
        return self.features[WINDOW, 2]

    def door_edge(self):
        return self.features[DOOR, 2]

    def cornice_edge(self):
        return self.features[CORNICE, 2]

    def sill_edge(self):
        return self.features[SILL, 2]

    def balcony_edge(self):
        return self.features[BALCONY, 2]

    def blind_edge(self):
        return self.features[BLIND, 2]

    def deco_edge(self):
        return self.features[DECO, 2]

    def molding_edge(self):
        return self.features[MOLDING, 2]

    def pillar_edge(self):
        return self.features[PILLAR, 2]

    def shop_edge(self):
        return self.features[SHOP, 2]

    def plot(self, overlay_alpha=0.5):
        import pylab as pl
        rows = int(sqrt(self.layers()))
        cols = int(ceil(self.layers()/rows))

        for i in range(rows*cols):
            pl.subplot(rows, cols, i+1)
            pl.axis('off')
            if i >= self.layers():
                continue
            pl.title('{}({})'.format(self.labels[i], i))
            pl.imshow(self.image)
            pl.imshow(colorize(self.features[i].argmax(0),
                               colors=np.array([[0,     0, 255],
                                                [0,   255, 255],
                                                [255, 255, 0],
                                                [255, 0,   0]])),
                      alpha=overlay_alpha)



def facade(facade_feature_map):
    return facade_feature_map[FACADE, 1]


def window(facade_feature_map):
    return facade_feature_map[WINDOW, 1]


def door(facade_feature_map):
    return facade_feature_map[DOOR, 1]


def cornice(facade_feature_map):
    return facade_feature_map[CORNICE, 1]


def sill(facade_feature_map):
    return facade_feature_map[SILL, 1]


def balcony(facade_feature_map):
    return facade_feature_map[BALCONY, 1]


def blind(facade_feature_map):
    return facade_feature_map[BLIND, 1]


def deco(facade_feature_map):
    return facade_feature_map[DECO, 1]


def molding(facade_feature_map):
    return facade_feature_map[MOLDING, 1]


def pillar(facade_feature_map):
    return facade_feature_map[PILLAR, 1]


def shop(facade_feature_map):
    return facade_feature_map[SHOP, 1]


def facade_edge(facade_feature_map):
    return facade_feature_map[FACADE, 2]


def window_edge(facade_feature_map):
    return facade_feature_map[WINDOW, 2]


def door_edge(facade_feature_map):
    return facade_feature_map[DOOR, 2]


def cornice_edge(facade_feature_map):
    return facade_feature_map[CORNICE, 2]


def sill_edge(facade_feature_map):
    return facade_feature_map[SILL, 2]


def balcony_edge(facade_feature_map):
    return facade_feature_map[BALCONY, 2]


def blind_edge(facade_feature_map):
    return facade_feature_map[BLIND, 2]


def deco_edge(facade_feature_map):
    return facade_feature_map[DECO, 2]


def molding_edge(facade_feature_map):
    return facade_feature_map[MOLDING, 2]


def pillar_edge(facade_feature_map):
    return facade_feature_map[PILLAR, 2]


def shop_edge(facade_feature_map):
    return facade_feature_map[SHOP, 2]
