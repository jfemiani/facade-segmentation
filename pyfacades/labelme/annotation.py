from datetime import date, datetime

from PIL import Image, ImageDraw
import xmltodict
import numpy as np


class Source(object):
    def __init__(self):
        super(Source, self).__init__()
        self.sourceImage = ""
        self.sourceAnnotation = ""

    def from_dict(self, d):
        self.sourceAnnotation = d.pop('sourceAnnotation', "")
        self.sourceImage = d.pop('sourceImage', "")


class Polygon(object):
    def __init__(self):
        super(Polygon, self).__init__()
        self.username = ""
        """Username of the annotator"""

        self.points = []

    def set_from_dict(self, d):
        self.username = d.pop('username', '')

        points = d.pop('pt', [])
        if not isinstance(points, list):
            points = [points]
        self.points = np.array([(int(p.get('x', 0)), int(p.get('y', 0))) for p in points])


TYPE_POLYGON = 'polygon'
TYPE_BOUNDING_BOX = 'bounding_box'


class Object(object):
    def __init__(self):
        super(Object, self).__init__()

        self.name = ""
        """The label of this object"""

        self.deleted = 0
        """Whether this object has been deleted (1) or not (0)"""

        self.verified = 0
        """Whether this object has been verified (1) or not (0)"""

        self.occluded = 0
        """Whether this object is partially covered by somethign else"""

        self.attributes = ""
        """A comma separated list of attributes (?)"""

        # self.parts?

        self.date = datetime.now()
        """The date that an annotation was created"""

        self.id = -1
        """An ID assigned to this object (for cross referencing ?)"""

        self.polygon = Polygon()
        self.type = TYPE_POLYGON

    def set_from_dict(self, d):
        """

        :param d: A dictionary like object
        :type d: dict
        :return:
        """
        self.name = d.pop('name', '')
        self.deleted = int(d.pop('deleted', 0))
        self.verified = int(d.pop('verified', 0))
        self.occluded = d.pop('occluded', None)
        self.attributes = d.pop('attributes', None)

        # TODO: Do not know how to handle this
        parts = d.pop('parts', {})

        date = d.pop('date')
        self.date = datetime.strptime(date, "%d-%b-%Y %H:%M:%S")

        self.id = int(d.pop('id'))

        self.type = d.pop('type', TYPE_POLYGON)

        self.polygon.set_from_dict(d.get('polygon', {}))


class ImageSize(object):
    def __init__(self):
        super(ImageSize, self).__init__()
        self.nrows = None
        self.ncols = None

    def from_dict(self, d):
        self.nrows = d.get('nrows', None)
        self.ncols = d.get('ncols', None)


class Annotation(object):
    def __init__(self):
        super(Annotation, self).__init__()

        self.filename = ""
        """The filename of the image that  was annotated"""

        self.folder = ""
        """The folder on the LabelMe server that contains the image"""

        self.source = Source()
        """Source of this annotation"""

        self.objects = []
        """List of objects that have been annotated"""

        self.imagesize = ImageSize()
        """The size fo the image"""

    def parse_xml(self, f):
        """

        :param f: Either a string of XML or a file-like-object
        :return:
        """
        data = xmltodict.parse(f)

        annotation = data['annotation']
        self.filename = annotation['filename']
        self.folder = annotation['folder']

        if 'source' in annotation:
            self.source.from_dict(annotation['source'])

        objects = annotation.get('object', [])

        self.imagesize.from_dict(annotation.get('imagesize', {}))

        for o in objects:
            new_object = Object()
            new_object.set_from_dict(o)
            self.objects.append(new_object)

    def remove_deleted(self):
        self.objects = [o for o in self.objects if not o.deleted]