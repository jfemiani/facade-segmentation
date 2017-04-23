import os
from os.path import dirname
from unittest import TestCase

import numpy as np

import pyfacades
from pyfacades.independant_12_layers.ingest_labelme import LabelMask
from pyfacades.labelme.annotation import Annotation

PROJECT_ROOT = dirname(dirname(pyfacades.__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'test')
ANNOTATION_PATH = os.path.join(DATA_ROOT, 'orthographic04.xml')


class TestLabelMask(TestCase):
    def setUp(self):
        self.annotations = Annotation(path=ANNOTATION_PATH)
        assert isinstance(self.annotations, Annotation)
        self.annotations.collection.image_root = DATA_ROOT
        self.annotations.remove_deleted()
        self.annotations.update_image_size()

    def test_fill_polygon(self):
        lm = LabelMask(self.annotations)
        assert isinstance(lm, LabelMask)

        p = self.annotations.objects[0].polygon
        lm.fill_polygon(self.annotations.objects[0].polygon, 255)

        # lm.plot()
        print p.points

        im = np.asarray(lm.image)

        # Poke at the corners to make sure we set the right pixels
        self.assertEqual(im[554, 153], 0)
        self.assertEqual(im[554, 154], 255)
        self.assertEqual(im[554, 195], 255)
        self.assertEqual(im[554, 196], 0)

        self.assertEqual(im[614, 153], 0)
        self.assertEqual(im[614, 154], 255)
        self.assertEqual(im[614, 195], 255)
        self.assertEqual(im[614, 196], 0)

    def test_fill_polygons(self):
        lm = LabelMask(self.annotations)
        assert isinstance(lm, LabelMask)

        lm.fill_polygons([o.polygon for o in self.annotations.objects if o.name == 'window'])

        im = np.asarray(lm.image)
        # lm.plot()

        # I verified this by showing the image and comparing against the LabelMe visualization
        self.assertEqual(np.count_nonzero(im), 193665)

    def test_stroke_polygon(self):
        lm = LabelMask(self.annotations)
        assert isinstance(lm, LabelMask)

        windows = [o.polygon for o in self.annotations.objects if o.name == 'window']

        lm.fill_polygons(windows)
        lm.outline_polygons(width=6)

        im = np.asarray(lm.image)
        lm.plot()

        # I verified this by showing the image and comparing against the LabelMe visualization
        self.assertEqual(np.count_nonzero(im), 224660)
