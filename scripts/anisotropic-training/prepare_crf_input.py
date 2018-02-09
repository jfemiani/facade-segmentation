
FEATURES = ['facade', 'window', 'door', 'cornice', 'sill', 'balcony', 'blind', 'deco', 'molding', 'pillar', 'shop']
WEIGHT = 'deploy/test_weights.caffemodel'
LAYOUT = 'modified-inference-net.prototxt'

NEG = NEGATIVE = 0
POS = POSITIVE = 2
EDG = EDGE = 3

import numpy as np
import caffe
net = caffe.Net(LAYOUT, WEIGHT, caffe.TEST)

# Set the batch size to one
net.blobs['data'].reshape(1, 3, 512, 512)
net.reshape()

def priors(im):
    """
    :param im: An input image, shape 3x512x512
    :type im: np.ndarray
    """
    result = net.forward(data=np.array([im[:3]]))
    probs = [net.blobs['prob-{}'.format(feature)].data[0,(NEG,POS,EDG)] for feature in FEATURES]
    probs = np.concatenate(probs, axis=0)
    return probs

def prepare(im, prior=None):
    if prior is None:
        prior = priors(im)
    assert prior.shape == (len(FEATURES)*3, 512, 512)
    concat = np.concatenate([im[:3], prior])
    return np.array([concat])