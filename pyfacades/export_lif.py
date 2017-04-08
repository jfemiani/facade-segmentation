import json
from base64 import b64encode
from collections import OrderedDict, defaultdict

import StringIO

import skimage
from PIL import Image
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage.measure import approximate_polygon
import skimage.measure
import numpy as np

DEFAULT_LINECOLOR = (0, 255, 0, 255)
DEFAULT_FILLCOLOR = (0, 255, 255, 127)


def format_lif(image_path, image_data, one_hots, labels,
               threshold=0.5, tolerance=5,
               line_color=None, fill_color=None,
               rectangles=None, include=None):

    if rectangles is None:
        rectangles = {}

    if include is None:
        include = labels

    data = OrderedDict()
    data['imagePath'] = image_path
    data['imageData'] = encode_image(image_data)
    data['lineColor'] = line_color or DEFAULT_LINECOLOR
    data['fillColor'] = fill_color or DEFAULT_FILLCOLOR

    shapes = []
    for label, onehot in zip(labels, one_hots):
        if label not in include:
            print label, 'not in', include
            continue
        print 'Processing', label

        mask = (onehot >= threshold).astype(np.uint8)

        # Make sure regions do not extend past the edge of the image
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0
        mask = skimage.measure.label(mask)

        regions = regionprops(mask)
        for region in regions:
            t, l, b, r = region.bbox

            # Make sure there are no holes (we cannot represent those)
            filled = region.filled_image

            # Make sure we get a complete polygon
            # #(instead of falling of the edge of the image)
            filled = np.pad(filled, ((1, 1), (1, 1)), mode='constant')

            # I expect only one... but just in case...
            contours = find_contours(filled, threshold, fully_connected='high')

            for contour in contours:
                if rectangles.get(label, False):
                    simplified = [(b, l), (b, r), (t, r), (t, l), (b, l)]
                    print t, l, b, r
                else:
                    simplified = approximate_polygon(contour, tolerance) + np.array((t, l))
                if len(simplified) < 4:
                    # Since polygons are duplicate that first and last point,
                    # 4 'points' is actually a triangle.
                    continue  # Skip degenrate shapes

                shape = OrderedDict()
                shape['label'] = unicode(label)
                shape['points'] = [(x, y) for y, x in simplified]
                shape['line_color'] = line_color
                shape['fill_color'] = fill_color

                shapes.append(shape)

    data['shapes'] = shapes
    return data


def encode_image(image_data):
    output = StringIO.StringIO()
    img = Image.fromarray(image_data)
    img.save(output, format='jpeg')
    result = b64encode(output.getvalue())
    img.close()
    return result


# noinspection PyPep8Naming,SpellCheckingInspection
def export_lif(path, imagepath, imagedata, onehots, labels,
               threshold=0.5, tolerance=5,
               lineColor=None, fillColor=None,
               rectangles=None, include=None):
    """ Export segmentation as a .lif file that can be edited with pylabelme

    :param path: Output file (.lif)
    :param imagepath:
    :param imagedata:
    :param onehots:
    :param labels:
    :param threshold:
    :param tolerance:
    :param lineColor:
    :param fillColor:
    :param rectangles:
    :return:
    """
    with open(path, 'wb') as f:
        data = format_lif(imagepath, imagedata, onehots, labels,
                          threshold, tolerance,
                          lineColor, fillColor, rectangles, include)
        json.dump(data, f, ensure_ascii=True, indent=2)
