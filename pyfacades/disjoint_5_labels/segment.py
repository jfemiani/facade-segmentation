from pyfacades.disjoint_5_labels.model import LABELS, net, WINDOW, FACADE_EDGE, DOOR
from pyfacades.process_strip import split_tiles, combine_tiles
from pyfacades.util import softmax
import numpy as np


def _probabilistic_segment(image):
    """ Segment the image multiple times -- return the mean and deviation of scores.

    :param image:  An image with shape 3 x 512 x 512 (sized to match the net)
    :return: The output scores (len(LABELS) x 512 x 512)
    :return: The output deviation (512 x 512)
    """
    assert image.shape == (3, 512, 512)

    batch_size = net.blobs['data'].num
    net.blobs['data'].data[...] = np.array([image] * batch_size)
    results = net.forward()
    batch = results['conv1_1_D']
    for sample in batch:
        sample[...] = softmax(sample.copy())
    predicted = batch.mean(0)
    conf = batch.std(0).sum(0)

    return predicted, conf


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

    images = list(split_tiles(img, (512, 512)))
    n = len(images)
    oh, ow = img.shape[-2:]
    oc = len(LABELS)

    outputs = np.zeros((n, oc, 512, 512))
    outputs_conf = np.zeros((n, 1, 512, 512))
    for i, image in enumerate(images):
        predicted, conf = _probabilistic_segment(image)
        outputs[i] = predicted
        outputs_conf[i, 0] += conf

    output = combine_tiles(outputs.reshape(n, oc, 512, 512), (oc, oh, ow))
    output_conf = combine_tiles(outputs_conf, (1, oh, ow))
    return output, output_conf


def windows(output):
    if isinstance(output, tuple):
        # The return value of process_strip is actually a tuple (output, conf)
        output = output[0]
    return output[WINDOW]


def doors(output):
    if isinstance(output, tuple):
        # The return value of process_strip is actually a tuple (output, conf)
        output = output[0]
    return output[DOOR]


def facade_edge(output):
    if isinstance(output, tuple):
        # The return value of process_strip is actually a tuple (output, conf)
        output = output[0]
    return output[FACADE_EDGE]
