from glob import glob

from PIL import Image
from PIL import ImageDraw
import numpy as np
from collections import defaultdict

from os.path import join, getmtime, splitext, basename, isfile

from skimage.util.dtype import img_as_ubyte

import pyfacades
from pyfacades.labelme.annotation import Annotation, Collection
import pyfacades.independant_12_layers.model
import os

# These are the values for positive, negative, and unknown labels
LABEL_POSITIVE = 1
LABEL_EDGE = 2
LABEL_UNKNOWN = 3
LABEL_NEGATIVE = 0

EDGE_WIDTH = 6


DEFAULT_LABELME_INPUT = join(pyfacades.DATA_ROOT, 'from_labelme')
DEFAULT_TRAINING_OUTPUT = join(pyfacades.DATA_ROOT, 'training', 'independant_12_layers')


class LabelMask(object):
    def __init__(self, annotation):
        super(LabelMask, self).__init__()
        self.annotation = annotation

        assert annotation.imagesize.nrows is not None, "Must know the image size first (see update_image_size)"

        self.image = Image.new('L', (annotation.imagesize.ncols, annotation.imagesize.nrows), LABEL_NEGATIVE)
        self.artist = ImageDraw.Draw(self.image)

    def fill_polygon(self, poly, color=(1,)):
        """
        
        :type poly: pyfacades.labelme.annotation.Polygon
        :param poly: 
        :return: 
        """
        self.artist.polygon([tuple(p) for p in poly.points], fill=color)

    def fill_polygons(self, polys, color=LABEL_POSITIVE):
        for p in polys:
            self.fill_polygon(p, color)

    def outline_polygons(self, width=EDGE_WIDTH, color=LABEL_NEGATIVE):
        from skimage.morphology import binary_dilation, disk
        im = np.asarray(self.image).copy()
        outset = binary_dilation(im != 0, disk(width / 2))
        inset = binary_dilation(im == 0, disk(width - width / 2))
        boundary = outset & inset
        im[boundary] = color
        self.image = Image.fromarray(im)

    def plot(self, show=True):
        import pylab
        pylab.imshow(self.image, interpolation='nearest')
        if show:
            pylab.show()

    def mark_positives(self, polygons):
        self.fill_polygons(polygons, LABEL_POSITIVE)

    def mark_edges(self, polygons, width=EDGE_WIDTH):
        self.outline_polygons(width=width)

    def mark_unknown(self, polygons):
        self.fill_polygons(polygons, color=LABEL_UNKNOWN)


class LabelDataGenerator(object):
    def __init__(self, annotations=None, edge_width=6):
        """

        :param annotations:
        :type annotations: Annotation
        :param label_names:
        :type label_names: list[str]
        """
        super(LabelDataGenerator, self).__init__()

        self.edge_width = defaultdict(default_factory=lambda : edge_width)
        self.annotations = annotations
        self.label_names = pyfacades.independant_12_layers.model.LABELS
        self.nlayers = len(self.label_names)
        self.nrows = self.annotations.source.nrows
        self.ncols = self.annotations.source.ncols
        self.data = np.zeros((self.nrows, self.ncols, self.nlayers+3), dtype=np.uint8)
        self.label_data = self.data[:, :, 3:]
        self.color_data = self.data[:, :, :3]

    def mark_all(self):
        unknowns = [o.polygon for o in self.annotations if o.name not in self.label_names]
        for i, label in enumerate(self.label_names):
            lm = LabelMask(self.annotations)
            assert isinstance(lm, LabelMask)
            positives = [o.polygon for o in self.annotations if o.name == label]
            lm.mark_positives(positives)
            lm.mark_edges(positives, self.edge_width[label])
            lm.mark_unknown(unknowns)
            self.label_data[:, :, i] = np.asarray(lm.image)

    def mark_facade_edges(self, thickness=12):
        label = 'facade'
        i = self.label_names.index(label)
        lm = LabelMask(self.annotations)
        h, w, layers = self.label_data.shape
        assert isinstance(lm, LabelMask)
        positives = [o.polygon for o in self.annotations if o.name == label]
        for p in positives:
            left = min(p.points[:, 0])
            right = max(p.points[:, 1])
            self.label_data[:, max(0,left-thickness):min(w, left+thickness), i] = LABEL_EDGE

    def save_data(self, path):
        np.save(path, self.data)

    def load_data(self, path):
        self.data[...] = np.load(path)

    def set_colors(self, image):
        self.color_data[...] = img_as_ubyte(image)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--from-labelme', default=DEFAULT_LABELME_INPUT, help="Local XML files produced by LabelMe")
    p.add_argument('--training-out', default=DEFAULT_TRAINING_OUTPUT, help="Local location to put data for training")
    p.add_argument('--files', help="Text file with the names of images that are ready to process")

    args = p.parse_args()

    with open(args.files) as f:
        files = f.readlines()
    xml_files = [join(args.from_labelme, 'Annotations', splitext(f)[0] + '.xml') for f in files]
    jpg_files = [join(args.from_labelme, 'Images', splitext(f)[0] + '.jpg') for f in files]
    training_outputs = [join(args.training_out, splitext(basename(xml))[0] + '.npy') for xml in xml_files]




if __name__ == '__main__':
    main()

