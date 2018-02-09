from datetime import date, datetime

from PIL import Image, ImageDraw
import xmltodict
import numpy as np
import os



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
        self.points = np.array([(float(p.get('x', 0)), float(p.get('y', 0))) for p in points], dtype=float)
        assert isinstance(self.points, np.ndarray)

    def __iter__(self):
        for point in self.points:
            yield point

    def centroid(self):
        """Get the centroid (mean coordinate) of the polygon.
        """
        return self.points.mean(0)

    def bounds(self):
        """Get the min-max box (top, left, bottom, right) of this polygon.
        """
        return tuple(np.roll(self.points.min(0), 1)) + tuple(np.roll(self.points.max(0),1))

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
        """The polygon that bounds this object"""
        # type self.polygon: Polygon

        self.type = TYPE_POLYGON

    def __iter__(self):
        for point in self.polygon:
            yield point

    def bounds(self):
        """
        :return: top, left, bottom, right
        """
        return self.polygon.bounds()

    def get_mask(self, fill=1, outline=0):
        """
        Get a mask that is nonzero within the object.
        :param fill:  A color to use on the object's interior
        :param outline: A color to use on the object's edge (single pixel)
        :return: A 2D numpy array of uint8
        """
        if self.type in (TYPE_POLYGON, TYPE_BOUNDING_BOX):
            (top, left, bottom, right) = self.polygon.bounds()
            w = int(right-left)
            h = int(bottom-top)
            mask = Image.new("I", (w, h))
            d = ImageDraw.Draw(mask)
            d.polygon([(px-left, py-top) for (px, py) in self.polygon.points],
                        fill=fill, outline=outline)
            del d
            return np.asarray(mask)
        else:
            assert False, "Unhandled Type"

    def draw(self, image, fill=1, outline=0):
        if not isinstance(image, Image.Image):
            image2 = Image.fromarray(image)
        else:
            image2 = image

        d = ImageDraw.Draw(image2)
        d.polygon([(px, py) for (px, py) in self.polygon.points],
                  fill=fill, outline=outline)
        del d

        if image2 is not  image:
            image[...] = np.asarray(image2)
        return image

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

        try:
            date = d.pop('date')
            self.date = datetime.strptime(date, "%d-%b-%Y %H:%M:%S")
        except:
            print("Object", self.name, "is missing a date")

        self.id = int(d.pop('id'))

        self.type = d.pop('type', TYPE_POLYGON)

        self.polygon.set_from_dict(d.get('polygon', {}))
    
    def __repr__(self):
        return "Object({})".format(self.name)

class ImageSize(object):
    def __init__(self):
        super(ImageSize, self).__init__()
        self.nrows = None
        self.ncols = None

    def set_from_dict(self, d):
        """
        :param d: A dictionary with serialized data for this object.
        :type d: dict
        """
        self.nrows = d.get('nrows', None)
        self.ncols = d.get('ncols', None)
        
    def __repr__(self):
        return "ImageSize(rows:{},cols:{})".format(self.nrows, self.ncols)


class Annotation(object):
    def __init__(self, path=None, 
                       images_root=None, 
                       xml_root=None, 
                       collection=None,
                       update_image_size=True,
                       remove_deleted=True):
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

        if collection is None:
            collection = Collection()
            if path is not None:
                collection.guess_from_path(path)

        self.collection = collection
        """The paths to local annotation and image files"""
        assert isinstance(self.collection, Collection)

        if images_root is not None:
            self.collection.image_root = images_root

        if xml_root is not None:
            self.collection.xml_root = xml_root

        self.path = path
        """The last annotation file we opened (iff we got it from a file)"""

        if path is not None:
            with open(path) as f:
                self.parse_xml(f)
        
        if update_image_size:
            self.update_image_size()
        
        if remove_deleted:
            self.remove_deleted()

    def parse_xml(self, f, path=None):
        """

        :param f: Either a string of XML or a file-like-object
        :return:
        """
        data = xmltodict.parse(f)

        self.path = path

        annotation = data['annotation']
        self.filename = annotation['filename']
        self.folder = annotation['folder']

        if 'source' in annotation:
            self.source.from_dict(annotation['source'])

        objects = annotation.get('object', [])
        if isinstance(objects, dict):
            objects = [objects]

        self.imagesize.set_from_dict(annotation.get('imagesize', {}))

        for o in objects:
            new_object = Object()
            if not isinstance(o, dict):
                print o
                print type(o)
                import ipdb; ipdb.set_trace()
            new_object.set_from_dict(o)
            self.objects.append(new_object)

    def remove_deleted(self):
        self.objects = [o for o in self.objects if not o.deleted]

    def get_image_path(self):
        """ Get the path to the image that corresponds to this annotation
        """
        root = self.collection.image_root
        folder = self.folder
        filename = self.filename
        return os.path.join(root, folder, filename)

    def get_image(self):
        """ Get the PIL image for this annotation
        :rtype: Image.Image
        :except IOError: If the image file cannot be found or the image cannot be opened.
        """
        return Image.open(self.get_image_path())

    def update_image_size(self):
        """
        :except IOError: If the file cannot be found or the image cannot be opened.
        """
        image = self.get_image()
        self.imagesize.nrows = image.height
        self.imagesize.ncols = image.width

    def __iter__(self):
        for o in self.objects:
            if not o.deleted:
                yield o

    def __repr__(self):
        return "Annoation({}, {} objects)".format(self.filename, len(self.objects))
    

class Collection(object):
    def __init__(self, root='.', xml_root="Annotations", image_root="Images"):
        super(Collection, self).__init__()
        self.root = os.path.abspath(root)
        self.xml_root = os.path.abspath(os.path.join(self.root, xml_root))
        self.image_root = os.path.abspath(os.path.join(self.root, image_root))

    def guess_from_path(self, annotation_or_image):
        path = os.path.abspath(annotation_or_image)
        collection = os.path.dirname(path)
        annotations_or_images = os.path.dirname(collection)
        root = os.path.dirname(annotations_or_images)
        self.root = root
        self.image_root = os.path.join(root, "Images")
        self.xml_root = os.path.join(root, "Annotations")
