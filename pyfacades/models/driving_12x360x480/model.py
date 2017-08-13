import caffe
import os
from collections import OrderedDict
import numpy as np
from pyfacades.util import colorize


# This model came from the segnet web demo, which used these colors to visualize results
COLORS = OrderedDict()
COLORS['Sky'] = [128, 128, 128]
COLORS['Building'] = [128, 0, 0]
COLORS['Pole'] = [192, 192, 128]
COLORS['Road Marking'] = [255, 69, 0]
COLORS['Road'] = [128, 64, 128]
COLORS['Pavement'] = [60, 40, 222]
COLORS['Tree'] = [128, 128, 0]
COLORS['Sign Symbol'] = [192, 128, 128]
COLORS['Fence'] = [64, 64, 128]
COLORS['Car'] = [64, 0, 128]
COLORS['Pedestrian'] = [64, 64, 0]
COLORS['Bicyclist'] = [0, 128, 192]
COLORS['Unlabelled'] = [0, 0, 0]

LABELS = COLORS.keys()

SKY = LABELS.index('Sky')
BUILDING = LABELS.index('Building')
POLE = LABELS.index('Pole')
ROAD_MARKING = LABELS.index('Road Marking')
ROAD = LABELS.index('Road')
PAVEMENT = LABELS.index('Pavement')
TREE = LABELS.index('Tree')
SIGN_SYMBOL = LABELS.index('Sign Symbol')
FENCE = LABELS.index('Fence')
CAR = LABELS.index('Car')
PEDESTRIAN = LABELS.index('Pedestrian')
BICYCLIST = LABELS.index('Bicyclist')
UNLABELLED = LABELS.index('Unlabelled')

__FOLDER = os.path.abspath(os.path.dirname(__file__))
LAYERS = os.path.join(__FOLDER, 'segnet_model_driving_webdemo.prototxt')
WEIGHTS = os.path.join(__FOLDER, 'segnet_weights_driving_webdemo.caffemodel')

__net = None


def net():
    """Delay loading the net until the last possible moment.

    Loading the net is SLOW and produces a ton of terminal garbage.
    Also we want to wait to load it until we have called some other
    caffe initializations code (caffe.set_mode_gpu(), caffe.set_device(0), etc)

    """
    global __net
    if __net is None:
        __net = caffe.Net(LAYERS, WEIGHTS, caffe.TEST)
    return __net


JET_12 = np.array([[0, 0, 127],
                   [0, 0, 232],
                   [0, 56, 255],
                   [0, 148, 255],
                   [12, 244, 234],
                   [86, 255, 160],
                   [160, 255, 86],
                   [234, 255, 12],
                   [255, 170, 0],
                   [255, 85, 0],
                   [232, 0, 0],
                   [127, 0, 0]], dtype=np.uint8)


class FeatureMap(object):
    def __init__(self, features, image=None, colors=None):
        super(FeatureMap, self).__init__()
        self.features = features
        self.image = image
        if colors is None:
            colors = JET_12
        self.colors = colors

    def layers(self):
        return self.features.shape[0]

    def rows(self):
        return self.features.shape[1]

    def cols(self):
        return self.features.shape[2]

    def __getitem__(self, item):
        return self.features[item]

    def sky(self):
        return self.features[SKY]

    def building(self):
        return self.features[BUILDING]

    def pole(self):
        return self.features[POLE]

    def road_marking(self):
        return self.features[ROAD_MARKING]

    def road(self):
        return self.features[ROAD]

    def pavement(self):
        return self.features[PAVEMENT]

    def tree(self):
        return self.features[TREE]

    def sign_symbol(self):
        return self.features[SIGN_SYMBOL]

    def fence(self):
        return self.features[FENCE]

    def car(self):
        return self.features[CAR]

    def pedestrian(self):
        return self.features[PEDESTRIAN]

    def bicyclist(self):
        return self.features[BICYCLIST]

    def unlabelled(self):
        return self.features[UNLABELLED]

    def plot(self, overlay_alpha=0.5):
        import pylab as pl
        #pl.imshow(self.image)
        tinted = ((1-overlay_alpha)*self.image
                  + overlay_alpha*colorize(np.argmax(self.features, 0), self.colors))
        from skimage.segmentation import mark_boundaries
        tinted = mark_boundaries(tinted.clip(0, 255).astype(np.uint8), np.argmax(self.features, 0))
        pl.imshow(tinted)


def sky(feature_map):
    return feature_map[SKY]


def building(feature_map):
    return feature_map[BUILDING]


def pole(feature_map):
    return feature_map[POLE]


def road_marking(feature_map):
    return feature_map[ROAD_MARKING]


def road(feature_map):
    return feature_map[ROAD]


def pavement(feature_map):
    return feature_map[PAVEMENT]


def tree(feature_map):
    return feature_map[TREE]


def sign_symbol(feature_map):
    return feature_map[SIGN_SYMBOL]


def fence(feature_map):
    return feature_map[FENCE]


def car(feature_map):
    return feature_map[CAR]


def pedestrian(feature_map):
    return feature_map[PEDESTRIAN]


def bicyclist(feature_map):
    return feature_map[BICYCLIST]


def unlabelled(feature_map):
    return feature_map[UNLABELLED]
