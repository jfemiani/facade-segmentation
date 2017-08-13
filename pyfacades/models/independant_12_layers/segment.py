from math import ceil

import numpy as np

from pyfacades.models.independant_12_layers.model import LABELS, net, FeatureMap
from pyfacades.util import softmax, channels_last
from pyfacades.util.process_strip import split_tiles, combine_tiles


def probabilistic_segment(image, n=5):
    """ Segment the image multiple times -- return the mean and deviation of scores.

    :param image:  An image with shape 3 x 512 x 512 (sized to match the net)
    :return: The output scores (len(LABELS) x 3 x 512 x 513)
    """
    assert image.shape == (3, 512, 512)

    batch_size = net().blobs['data'].num
    passes = int(ceil(float(n) / batch_size))*batch_size

    big_batch = np.zeros( (passes, len(LABELS), 3, 512, 512), dtype=np.float32)

    for p in range(0, passes, batch_size):
        net().blobs['data'].data[...] = np.array([image] * batch_size)
        results = net().forward()
        for label in range(1, 12):
            name = LABELS[label]
            # noinspection SpellCheckingInspection
            layer = 'conv-{}'.format(name)
            blob = results[layer]
            for k in range(batch_size):
                big_batch[p + k, label] = softmax(blob[k][[0, 2, 3]].copy())

    predicted = np.zeros((len(LABELS), 3, 512, 512), dtype=np.float32)
    conf = np.zeros((len(LABELS), 512, 512), dtype=np.float32)

    for label in range(1, 12):
        for k in range(passes):
            batch = big_batch[:, label, :, :, :].squeeze()
            predicted[label] = batch[:, :, :, :].mean(0)
            conf[label] = batch.std(0).sum(0)

    return predicted, conf


def process_strip(img):
    """Process an image in vertical strips.

    The input image is rescaled vertically and split horizontally into square tiles that
    are processed separately and then combined to form a complete segmented image.

    Each class of object is represented by 3 scored; 0=NEGATIVE, 1=POSITIVE, 2=EDGE

    :param img:  An input image (e.g. from google street view)
    :return: Scores, shape (len(LABELS) x 3 x HEIGHT x WIDTH) where img is a 3 x HEIGHT x WIDTH image.
    """
    assert img.shape[0] == 3, "Image must be channels-first"
    assert img.max() > 1, "Must use 0..255 images, not 0..1"

    images = list(split_tiles(img, (512, 512)))
    n = len(images)
    m = len(LABELS)
    oh, ow = img.shape[-2:]
    on, oc = m, 3

    outputs = np.zeros((n, m, 3, 512, 512))
    outputs_conf = np.zeros((n, 12, 512, 512))
    for i, image in enumerate(images):
        predicted, conf = probabilistic_segment(image)
        for j in range(1, 12):
            outputs[i, j] = predicted[j]
            outputs_conf[i, j] += conf[j]

    output = combine_tiles(outputs.reshape(n, on * oc, 512, 512), (on * oc, oh, ow))
    output = output.reshape(on, oc, oh, ow)

    output_conf = combine_tiles(outputs_conf, (12, oh, ow))
    return FeatureMap(output, confidence=output_conf, image=channels_last(img))
