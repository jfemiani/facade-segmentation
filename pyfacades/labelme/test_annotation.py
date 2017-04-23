import os
from unittest import TestCase
import numpy as np
from os.path import dirname

import pyfacades
from pyfacades.labelme.annotation import Annotation

PROJECT_ROOT = dirname(dirname(pyfacades.__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'test')
ANNOTATION_PATH = os.path.join(DATA_ROOT, 'orthographic04.xml')


class TestAnnotation(TestCase):
    def setUp(self):
        super(TestAnnotation, self).setUp()

    def test_parse_xml(self):
        a = Annotation()

        xml = open('../data/test/orthographic04.xml')
        a.parse_xml(xml)

        self.assertEqual(a.filename, 'orthographic04.jpg')
        self.assertEqual(a.folder, 'cock')

        self.assertIsNone(a.imagesize.nrows)
        self.assertIsNone(a.imagesize.ncols)

        self.assertEqual(len(a.objects), 136)

        self.assertEqual(len(a.objects[0].polygon.points), 1)
        self.assertEqual(len(a.objects[1].polygon.points), 1)
        self.assertEqual(len(a.objects[2].polygon.points), 1)
        self.assertEqual(len(a.objects[3].polygon.points), 4)

        self.assertEqual(a.objects[0].deleted, 1)
        self.assertEqual(a.objects[1].deleted, 1)
        self.assertEqual(a.objects[2].deleted, 1)
        self.assertEqual(a.objects[3].deleted, 1)

        self.assertEqual(a.objects[30].deleted, 0)
        self.assertEqual(a.objects[31].deleted, 0)

        self.assertEqual(a.objects[30].polygon.username, 'michelle')


        #self.fail()

    def test_remove_deleted(self):
        a = Annotation()

        xml = open(ANNOTATION_PATH)
        a.parse_xml(xml)
        a.remove_deleted()
        self.assertEqual(len(a.objects), 83)
        self.assertEqual(a.objects[-1].name, 'balcony')
        self.assertEqual(a.objects[-2].name, 'obstruction')
        self.assertEqual(a.objects[0].name, 'window')

        # Make sure we have the expected set of names after removing the deleted items
        d = {}
        for o in a.objects:
            if o.name not in d:
                d[o.name] = []
            d[o.name].append(o.polygon.points)

        self.assertEqual(set(d.keys()),
                         {u'shop', u'door', u'molding', u'sill', u'sky', u'obstruction', u'window', u'facade',
                          u'cornice', u'balcony'})

        # Make sure we have the expected number of windows after removing deleted items
        self.assertEqual(len(d['window']), 34)


class TestPolygon(TestCase):
    def setUp(self):
        super(TestPolygon, self).setUp()
        self.annotation = Annotation(path=ANNOTATION_PATH, images_root=DATA_ROOT)
        assert isinstance(self.annotation, Annotation)
        self.annotation.remove_deleted()

    def test_points(self):
        polygon = self.annotation.objects[0].polygon
        print polygon.points
        expected = [[154, 554],
                    [195, 554],
                    [195, 614],
                    [154, 614]]
        self.assertTrue(np.all(polygon.points == np.array(expected)))

    def test_centroid(self):
        polygon = self.annotation.objects[0].polygon

        cx, cy = polygon.centroid()

        self.assertEqual(cx, 174.5)
        self.assertEqual(cy, 584)

    def test_bounds(self):
        polygon = self.annotation.objects[0].polygon

        t, l, b, r = polygon.bounds()
        print polygon.points

        self.assertEqual(t, 554)
        self.assertEqual(l, 154)
        self.assertEqual(b, 614)
        self.assertEqual(r, 195)

