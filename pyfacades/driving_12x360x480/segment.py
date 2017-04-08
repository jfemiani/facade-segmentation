from pyfacades.driving_12x360x480.model import LABELS, net
from pyfacades.process_strip import split_tiles, combine_tiles
from pyfacades.util import softmax
import numpy as np

from pyfacades.driving_12x360x480 import model

def segment(image):
    """ Segment the image -- return the scores.

    :param image:  An image with shape 3 x 360 x 480 (sized to match the net)
    :return: The output scores (len(LABELS) x 512 x 512)
    """
    assert image.shape == (3, 360, 480)

    net.blobs['data'].data[...] = np.array([image])
    results = net.forward(blobs=['conv1_1_D'])
    # noinspection SpellCheckingInspection
    batch = results['conv1_1_D'][:, :12, :, :]
    for sample in batch:
        sample[...] = softmax(sample.copy())  # last label is ignored
    predicted = batch.mean(0)

    return predicted


def process_strip(img):
    """Process an image in vertical strips.

    The input image is rescaled vertically and split horizontally into square tiles that
    are processed separately and then combined to form a complete segmented image.

    Each class of object is represented by 3 scores; 0=NEGATIVE, 1=POSITIVE, 2=EDGE

    :param img:  An input image (e.g. from google street view)
    :return: Scores, shape (len(LABELS) x HEIGHT x WIDTH) where img is a 3 x HEIGHT x WIDTH image.
    :return: Deviation, or uncertainty, shape (HEIGHT x WIDTH)
    """
    assert img.shape[0] == 3, "Image must be channels-first"
    assert img.max() > 1, "Must use 0..255 images, not 0..1"

    th, tw = 360, 480

    images = list(split_tiles(img, (th, tw)))
    n = len(images)
    oh, ow = img.shape[-2:]
    oc = len(LABELS)-1 # last label is ignored (not output)

    outputs = np.zeros((n, oc, th, tw))
    for i, image in enumerate(images):
        outputs[i] = segment(image)

    output = combine_tiles(outputs.reshape(n, oc, th, tw), (oc, oh, ow))
    return output


def occlusion(feature_map):
    result = model.bicyclist(feature_map) > 0.5
    result |= model.pedestrian(feature_map) > 0.5
    result |= model.car(feature_map) > 0.5
    result |= model.pavement(feature_map) > 0.5
    result |= model.tree(feature_map) > 0.5
    return result