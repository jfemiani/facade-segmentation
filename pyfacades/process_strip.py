import numpy as np
import skimage

from pyfacades.facade_12_independant_labels import net, LABELS
from skimage.transform import rescale


def softmax(a, axis=0):
    a = np.exp(a - a.max(axis=axis))
    a /= a.max(axis=axis)
    return a


def channels_first(image):
    return image.transpose(2, 0, 1)


def channels_last(image):
    return image.transpose(1, 2, 0)


def probabilistic_segment(image):
    """ Segment the image multiple times -- return the mean and deviation of scores.

    :param image:  An image with shape 3 x 512 x 512 (sized to match the net)
    :return: The output scores (len(LABELS) x 3 x 512 x 513)
    """
    assert image.shape == (3, 512, 512)

    batch_size = net.blobs['data'].num
    net.blobs['data'].data[...] = np.array([image] * batch_size)
    results = net.forward()
    predicted = np.zeros((len(LABELS), 3, 512, 512), dtype=np.float32)
    conf = np.zeros((len(LABELS), 512, 512), dtype=np.float32)

    for label in range(1, 12):
        name = LABELS[label]
        layer = 'conv-{}'.format(name)
        blob = results[layer]
        batch = np.empty((batch_size, 3, 512, 512), dtype=np.float32)
        for k in range(batch_size):
            #  Get rid of label '1', it means IGNORED / UNLABELED.
            batch[k] = softmax(blob[k][[0, 2, 3]].copy())
        predicted[label] = batch.mean(0)
        conf[label] = batch.std(0).sum(0)

    return predicted, conf


def split_images(image, shape, overlap=16):
    """ Rescale and split the input images to get several overlapping images of a given shape.

    The input image is rescaled so that height matches the output height.
    It is split into possibly overlapping tiles, each sized to match the output shape
    """
    # image_channels = image.shape[0]
    image_height = image.shape[-2]
    # image_width = image.shape[-1]
    output_height = shape[-2]
    output_width = shape[-1]

    # Rescale to match vertical size
    scale = output_height / float(image_height)
    scaled_image = rescale(image.transpose(1, 2, 0), (scale, scale), order=0, preserve_range=True).transpose(2, 0, 1)

    scaled_width = scaled_image.shape[-1]

    if scaled_width < output_width:
        padding = output_width - scaled_width

        if len(scaled_image.shape) == 3:
            scaled_image = np.pad(scaled_image, ((0, 0), (0, 0), (padding / 2, padding - padding / 2)), mode='constant')
        else:
            scaled_image = np.pad(scaled_image, ((0, 0), (padding / 2, padding - padding / 2)), mode='constant')

    # Since the input is not a multiple of the output width, we will evenly divide the image
    # to produce overlapping tiles. Work it out.
    #   -- The last tile always fits, and does not overlap with the _next_ tile (there is none)
    #   -- The remaining tiles each overlap with the following tile. The width of uncovered portion
    #      evenly divides the rest of the strip
    #   -- I need an integer number of tiles to cover the remaining strip (so I use a ceil)
    num_tiles = 1 + int(np.ceil((scaled_width - output_width) / float(output_width - overlap)))
    for x in np.linspace(0, scaled_width - output_width, num_tiles):
        yield scaled_image[:, :, int(x):int(x) + output_width]


def combine_images(images, output_shape):
    """ Combine a set of overlapping images to match the output shape

    This routine is the inverse os split_images.
    The 'images' parameter is a sequence of identically-shaped images.
    They are evenly spaced in the output_shape.

    Overlapping regions are averages (arithmetic mean)

    """

    # You may need to reshape in order to curry extra dimensions into the 'channels'
    tn, tc, th, tw = images.shape
    oc, oh, ow = output_shape

    # The number of channels should match. In my case I have 5D inputs so I slice...
    assert tc == oc

    if oh != th:
        s = float(th) / oh
        result = combine_images(images, (tc, th, int(ow * s)))
        result = result.transpose(1, 2, 0)  # Change to channels-last for skimage
        result = skimage.transform.resize(result, (oh, ow, oc), preserve_range=True)
        result = result.transpose(2, 0, 1)  # Back to channels-first
        return result

    assert oh == th

    if ow < tw:
        x = (tw - ow) / 2
        result = combine_images(images, (oc, oh, tw))
        result = result[:, :, x:x + ow]
        return result

    assert ow >= tw

    tx = np.linspace(0, ow - tw, tn).astype(int)
    output = np.zeros(output_shape)
    counts = np.zeros((oh, ow))

    for i, x in enumerate(tx):
        x = int(x)

        weight = np.ones((th, tw))
        if i > 0:
            left_padding = tx[i - 1] + tw - x
            weight[:, :left_padding] *= np.linspace(0, 1, left_padding)
        if i < len(tx)-1:
            right_padding = x + tw - tx[i + 1]
            weight[:, -right_padding:] *= np.linspace(1, 0, right_padding)

        output[:, :, x:x + tw] += images[i]*weight
        counts[:, x:x + tw] += weight
    output /= counts
    output = output.astype(images.dtype)

    return output


def process_strip(img):
    """Process an image in vertical strips.

    The input image is rescaled vertically and split horizontally into square tiles that
    are processed separately and then combined to form a complete segmented image.

    Each class of object is represented by 3 scored; 0=NEGATIVE, 1=POSITIVE, 2=EDGE

    :param img:  An input image (e.g. from google streetview)
    :return: Scores, shape (len(LABELS) x 3 x HEIGHT x WIDTH) where img is a 3 x HEIGHT x WIDTH image.
    """
    assert img.shape[0] == 3, "Image must be channels-first"
    assert img.max() > 1, "Must use 0..255 images, not 0..1"

    images = list(split_images(img, (512, 512)))
    n = len(images)
    m = len(LABELS)
    oh, ow = img.shape[-2:]
    on, oc = m, 3

    outputs = np.zeros((n, m, 3, 512, 512))
    outputs_conf = np.zeros((n, 1, 512, 512))
    for i, image in enumerate(images):
        predicted, conf = probabilistic_segment(image)
        for j in range(1, 12):
            outputs[i, j] = predicted[j]
            outputs_conf[i, 0] += conf[j]

    output = combine_images(outputs.reshape(n, on * oc, 512, 512), (on * oc, oh, ow))
    output = output.reshape(on, oc, oh, ow)

    output_conf = combine_images(outputs_conf, (1, oh, ow))
    return output, output_conf


