from unittest import TestCase
from urllib2 import urlopen

from collections import OrderedDict

from annotation import Annotation


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

        xml = open('../data/test/orthographic04.xml')
        a.parse_xml(xml)
        a.remove_deleted()
        self.assertEqual(len(a.objects), 83)
        self.assertEqual(a.objects[-1].name, 'balcony')
        self.assertEqual(a.objects[-2].name, 'obstruction')
        self.assertEqual(a.objects[0].name, 'window')


        d = {}
        for o in a.objects:
            if o.name not in d:
                d[o.name] = []
            d[o.name].append(o.polygon.points)

        self.assertEqual(set(d.keys()),
                         {u'shop', u'door', u'molding', u'sill', u'sky', u'obstruction', u'window', u'facade',
                          u'cornice', u'balcony'})
        self.assertEqual(len(d['window']), 34)
