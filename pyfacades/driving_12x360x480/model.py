import caffe
import os
from collections import OrderedDict

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

MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
# noinspection SpellCheckingInspection
LAYERS = os.path.join(MODELS, 'segnet_model_driving_webdemo.prototxt')
# noinspection SpellCheckingInspection
WEIGHTS = os.path.join(MODELS, 'segnet_weights_driving_webdemo.caffemodel')

net = caffe.Net(LAYERS, WEIGHTS, caffe.TEST)


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

