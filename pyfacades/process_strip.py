import numpy as np
import skimage
from skimage.transform import rescale


def split_tiles(image, shape, overlap=16):
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


def combine_tiles(images, output_shape):
    """ Combine a set of overlapping images to match the output shape

    This routine is the inverse os split_tiles.
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
        result = combine_tiles(images, (tc, th, int(ow * s)))
        # noinspection PyTypeChecker
        result = result.transpose(1, 2, 0)  # Change to channels-last for skimage
        result = skimage.transform.resize(result, (oh, ow, oc), preserve_range=True)
        result = result.transpose(2, 0, 1)  # Back to channels-first
        return result

    assert oh == th

    if ow < tw:
        x = (tw - ow) / 2
        result = combine_tiles(images, (oc, oh, tw))
        result = result[:, :, x:x + ow]
        return result

    assert ow >= tw

    # noinspection PyUnresolvedReferences
    tx = (np.linspace(0, ow - tw, tn)).astype(int)
    output = np.zeros(output_shape)
    counts = np.zeros((oh, ow))

    for i, x in enumerate(tx):
        x = int(x)

        weight = np.ones((th, tw))
        if i > 0:
            left_padding = tx[i - 1] + tw - x
            weight[:, :left_padding] *= np.linspace(0, 1, left_padding)
        if i < len(tx) - 1:
            right_padding = x + tw - tx[i + 1]
            weight[:, -right_padding:] *= np.linspace(1, 0, right_padding)

        output[:, :, x:x + tw] += images[i] * weight
        counts[:, x:x + tw] += weight
    output /= counts
    output = output.astype(images.dtype)

    return output
