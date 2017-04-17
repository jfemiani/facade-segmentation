from PIL import Image
from PIL import ImageDraw

from pyfacades.labelme.annotation import Annotation, Collection
import os


class LabelMask(object):
    def __init__(self, annotation):
        super(LabelMask, self).__init__()
        self.annotation = annotation

        assert annotation.imagesize.nrows is not None, "Must know the image size first (see update_image_size)"

        self.image = Image.new('L', (annotation.imagesize.ncols, annotation.imagesize.nrows), 0)
        self.artist = ImageDraw.Draw(self.image)

    def fill_polygon(self, poly, color=1):
        """
        
        :type poly: pyfacades.labelme.annotation.Polygon
        :param poly: 
        :return: 
        """
        self.artist.polygon(poly.points, fill=color)

    def fill_polygons(self, polys, color=1):
        for p in polys:
            self.fill_polygon(p, color)

    def stroke_polygon(self, poly, color=1):
        self.artist.line()
