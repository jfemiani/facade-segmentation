import os

import numpy as np


def softmax(a, axis=0):
    a = np.exp(a - a.max(axis=axis))
    a /= a.sum(axis=axis)
    return a


def channels_first(image):
    return image.transpose(2, 0, 1)


def channels_last(image):
    return image.transpose(1, 2, 0)


def colorize(labels, colors):
    result = np.zeros(labels.shape + (3,), dtype=np.uint8)
    if not isinstance(colors, dict):
        colors = {i: colors[i] for i in range(len(colors))}
    rgb = colors.values()
    indices = colors.keys()
    for i in range(len(indices)):
        mask = labels == indices[i]
        color = rgb[i]
        result[mask, 0] = color[0]
        result[mask, 1] = color[1]
        result[mask, 2] = color[2]
    return result


def replace_ext(f, e):
    return os.path.splitext(f)[0] + e


def find_files(dir, pattern):
    import fnmatch
    import os

    matches = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches