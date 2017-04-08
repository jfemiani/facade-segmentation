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

MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
LAYERS = os.path.join(MODELS, 'facades_12_independent_labels.99000.prototxt')
WEIGHTS = os.path.join(MODELS, 'facades_12_independent_labels.99000.caffemodel')

net = caffe.Net(LAYERS, WEIGHTS, caffe.TEST)


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
