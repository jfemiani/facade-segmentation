import matplotlib
matplotlib.use('Agg', force=True)  # Avoid issues with X11 when running over ssh/tmux

from glob import glob

from PIL import Image
from PIL import ImageDraw
import numpy as np

from os.path import join, splitext, basename, isfile

from skimage.util.dtype import img_as_ubyte

import pyfacades
from pyfacades.labelme.annotation import Annotation, Polygon
import pyfacades.models.independant_12_layers.model
import os

# These are the values for positive, negative, and unknown labels
from pyfacades.util.process_strip import split_tiles
from pyfacades.util import channels_first, colorize

LABEL_POSITIVE = 2
LABEL_EDGE = 3
LABEL_UNKNOWN = 1
LABEL_NEGATIVE = 0
EDGE_WIDTH = 3

DEFAULT_LABELME_INPUT = join(pyfacades.DATA_ROOT, 'from_labelme')
DEFAULT_TRAINING_OUTPUT = join(pyfacades.DATA_ROOT, 'training', 'independant_12_layers')



class SingleLayerLabels(object):
    def __init__(self, annotation, nrows=512, ncols=None):
        super(SingleLayerLabels, self).__init__()
        self.annotation = annotation

        assert annotation.imagesize.nrows is not None, "Must know the image size first (see update_image_size)"

        self.aspect = annotation.imagesize.ncols/float(annotation.imagesize.nrows)
        self.y_scale = nrows / float(annotation.imagesize.nrows)
        self.nrows = nrows
        if ncols is None:
            self.x_scale = self.y_scale
            self.ncols = int(self.x_scale*annotation.imagesize.ncols)
        else:
            self.x_scale = ncols / float(annotation.imagesize.ncols)
            self.ncols = ncols

        self.image = Image.new('L', (self.ncols, self.nrows), LABEL_NEGATIVE)
        self.artist = ImageDraw.Draw(self.image)

    def fill_polygon(self, poly, color=LABEL_POSITIVE, outline=None):
        """
        
        :type poly: pyfacades.labelme.annotation.Polygon
        :param poly: 
        :return: 
        """
        self.artist.polygon([(px*self.x_scale, py*self.y_scale) for (px, py) in poly.points],
                            fill=color,
                            outline=outline)

    def fill_polygons(self, polys, color=LABEL_POSITIVE, outline=LABEL_EDGE):
        for p in polys:
            if len(p.points) > 3:
                self.fill_polygon(p, color, outline=outline)

    def outline_polygons(self, width=EDGE_WIDTH, color=LABEL_EDGE):
        from skimage.morphology import binary_dilation, disk
        im = np.asarray(self.image).copy()
        outset = binary_dilation(im == LABEL_POSITIVE, disk(width / 2))
        inset = binary_dilation(im != LABEL_POSITIVE, disk(width - width / 2))
        boundary = outset & inset
        im[boundary] = color
        self.image = Image.fromarray(im)
        self.artist = ImageDraw.Draw(self.image)

    def mark_rectangle(self, top, left, bottom, right, label=LABEL_POSITIVE):
        left *= self.x_scale
        right *= self.x_scale
        top *= self.y_scale
        bottom *= self.y_scale
        self.artist.rectangle([(left, top), (right, bottom)], fill=label)

    def outline_rectangle(self, top, left, bottom, right, label=LABEL_POSITIVE):
        left *= self.x_scale
        right *= self.x_scale
        top *= self.y_scale
        bottom *= self.y_scale
        self.artist.rectangle([(left, top), (right, bottom)], outline=label)

    def plot(self):
        """Visualize the label images (e.g. for debugging)

        :return: The current figure
        """
        import pylab
        pylab.imshow(self.image, interpolation='nearest')
        return pylab.gcf()

    def mark_positives(self, polygons, outline=None):
        """
        Shorthand to mark all polygons as positive.
        :param polygons: A list of polygons to mark.
        :type polygons: list[Polygon]
        :return:
        """
        self.fill_polygons(polygons, LABEL_POSITIVE, outline=outline)

    def mark_edges(self, width=EDGE_WIDTH):
        """Shorthand to mark the edges of all polygons as object boundaries
        :return:
        """
        self.outline_polygons(width=width, color=LABEL_EDGE)

    def mark_unknown(self, polygons):
        """Shorthand to mark all polygons as unknown
        :param polygons: A list of polygons to mark.
        :type polygons: list[Polygon]
        :return:
        """
        self.fill_polygons(polygons, color=LABEL_UNKNOWN, outline=LABEL_UNKNOWN)

    def mark_boxes(self, polygons, label=LABEL_POSITIVE):
        """For some features we may want to label the box, not the precise polygon.

        :param polygons: A list of polygons to label
        :type polygons: list[Polygon]
        :param label: The label to fill the polygon with
        :type label: int
        """
        for p in polygons:
            assert isinstance(p, Polygon)
            self.mark_rectangle(*p.bounds(), label=label)

    def mark_vertical_box_edges(self, polygons, width=16, label=LABEL_POSITIVE):
        """For facades, we just want the left and right edges.

        :param polygons: A list of polygons to label
        :type polygons: list[Polygon]
        :param width: The width (thickness) of the vertical edges
        :param label: The label to fill the polygon with
        :type label: int
        """
        for p in polygons:
            assert isinstance(p, Polygon)
            top, left, bottom, right = p.bounds()
            self.mark_rectangle(top, left-width/2, bottom, left+width-width/2,  label=label)
            self.mark_rectangle(top, right-width/2, bottom, right+width-width/2, label=label)


class MultiLayerLabels(object):
    def __init__(self, annotations=None, nrows=512, ncols=None, edge_width=EDGE_WIDTH):
        """

        :param annotations:  The LabelMe annotations
        :type annotations: Annotation
        """
        super(MultiLayerLabels, self).__init__()

        assert isinstance(annotations, Annotation)

        self.edge_width = dict()
        self.edge_width['facade'] = 15
        self.default_edge_width = edge_width
        self.annotations = annotations
        self.label_names = pyfacades.models.independant_12_layers.model.LABELS
        self.unknown_names = ['unknown', 'unlabeled']
        self.unknowns = [o.polygon for o in self.annotations if o.name in self.unknown_names]
        self.nlayers = len(self.label_names)
        self.nrows = nrows
        y_scale = nrows / float(annotations.imagesize.nrows)
        self.ncols = ncols if ncols is not None else int(y_scale*annotations.imagesize.ncols)

        self.data = np.zeros((self.nrows, self.ncols, self.nlayers+3), dtype=np.uint8)
        self.label_data = self.data[:, :, 3:]
        self.color_data = self.data[:, :, :3]
        self.set_colors(self.annotations.get_image().resize((self.ncols, self.nrows), resample=Image.BICUBIC))

    def mark_all(self):
        for i, label in enumerate(self.label_names):
            if label == 'facade':
                self.label_data[:, :, i] = self._mark_vertical_edges(label, self.unknowns)
            else:
                self.label_data[:, :, i] = self._mark(label, self.unknowns)

    def _mark(self, label, unknowns, edge_width=None):
        if edge_width is None:
            edge_width = self.edge_width.get(label, self.default_edge_width)

        layer = SingleLayerLabels(self.annotations, ncols=self.ncols, nrows=self.nrows)
        assert isinstance(layer, SingleLayerLabels)
        positives = [o.polygon for o in self.annotations if o.name == label]
        layer.mark_unknown(unknowns)
        layer.mark_positives(positives, outline=LABEL_EDGE)
        layer.mark_edges(edge_width)
        return np.asarray(layer.image)

    def _mark_vertical_edges(self, label, unknowns, edge_width=None):
        if edge_width is None:
            edge_width = self.edge_width.get(label, self.default_edge_width)

        layer = SingleLayerLabels(self.annotations)
        assert isinstance(layer, SingleLayerLabels)
        positives = [o.polygon for o in self.annotations if o.name == label]
        layer.mark_unknown(unknowns)
        layer.mark_boxes(positives, label=LABEL_POSITIVE)
        layer.mark_vertical_box_edges(positives, width=edge_width, label=LABEL_EDGE)
        return np.asarray(layer.image)

    def save_data(self, path):
        np.save(path, self.data)

    def save_tiles(self, path, rows=None, cols=None, overlap=16):
        if rows is None:
            rows = self.nrows
        if cols is None:
            cols = rows

        tiles = split_tiles(channels_first(self.data), (rows, cols), overlap)

        for i, tile in enumerate(tiles):
            fn = '{}_tile{:04}.npy'.format(path, i+1)
            np.save(fn, tile)

    def load_data(self, path):
        self.data[...] = np.load(path)

    def set_colors(self, image):
        self.color_data[...] = img_as_ubyte(image)

    def plot(self):
        """ Plot the layer data (for debugging)
        :return: The current figure
        """
        import pylab as pl
        aspect = self.nrows / float(self.ncols)
        figure_width = 6 #inches

        rows = max(1, int(np.sqrt(self.nlayers)))
        cols = int(np.ceil(self.nlayers/rows))
        # noinspection PyUnresolvedReferences
        pallette = {i:rgb for (i, rgb) in enumerate(pl.cm.jet(np.linspace(0, 1, 4), bytes=True))}
        f, a = pl.subplots(rows, cols)
        f.set_size_inches(6 * cols, 6 * rows)
        a = a.flatten()
        for i, label in enumerate(self.label_names):
            pl.sca(a[i])
            pl.title(label)
            pl.imshow(self.color_data)
            pl.imshow(colorize(self.label_data[:, :, i], pallette), alpha=0.5)
            # axis('off')
        return f

def main():
    import pylab as pl
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--from-labelme',
                   default=DEFAULT_LABELME_INPUT,
                   help="Local XML files produced by LabelMe")
    p.add_argument('--training-out', '-o',
                   default=DEFAULT_TRAINING_OUTPUT,
                   help="Local location to put data for training")
    p.add_argument('--files',
                   help="Text file with the names of images that are ready to process. "
                         "Names should be relative to the 'Images' folder on the LabelMe site.")
    p.add_argument('--plot', action='store_true', help="Save plots of each image")
    p.add_argument('--tilesize', type=int, nargs=2, help="The size of each tile (width, height)", default=(512, 512))
    p.add_argument('--ignore', help="A comma-separated list of labels to mark as ignored/missing")
    p.add_argument('--resume', action='store_true', help="Skip over files with existing output")
    p.add_argument('--clean', action='store_true', help="Remove old outputs before processing. "
                                                        "This only removes outputs that match the "
                                                        "names of our input")
    p.add_argument('--summary', action='store_true', help="Produce a summary of the labels in each file only.")

    args = p.parse_args()

    print "Collecting input file names..."
    with open(args.files) as f:
        files = f.readlines()
    xml_root = join(args.from_labelme, 'Annotations')
    img_root = join(args.from_labelme, 'Images')
    xml_files = [join(xml_root, splitext(f)[0] + '.xml') for f in files]
    img_files = [join(img_root, splitext(f)[0] + '.jpg') for f in files]
    training_outputs = [join(args.training_out, 'npy', splitext(basename(xml))[0]) for xml in xml_files]
    plot_outputs = [join(args.training_out, 'plot', splitext(basename(xml))[0] + '.png') for xml in xml_files]

    if args.clean and args.resume:
        print "You asked me to clean and resume, I cannot do both."
        print "Not cleaning"

    if args.summary:
	print "# Generating a summary of labels only. No other processing will occure."
	labels = pyfacades.models.independant_12_layers.LABELS
	labels += ['tree', 'sky', 'occlusion', 'unknown']
        fields = ['index','name']+labels
        print ',\t'.join(fields)
	for i in range(len(xml_files)):
            values = ['']*len(fields)
            values[0] = '{} of {}'.format(i+1, len(xml_files))
	    values[1] = basename(xml_files[i])
	    annotations = Annotation(xml_files[i],
                                     images_root=img_root,
                                     xml_root=xml_root)
            assert isinstance(annotations, Annotation)
            annotations.remove_deleted()
            annotations.update_image_size()

	    for label in labels:
	        objects = [o for o in annotations.objects if o.name.lower() == label]
		values[fields.index(label)] = str(len(objects))
	    print ',\t'.join(values)
        return 
            

    # Clean any old outputs
    if args.clean and not args.resume:
        for fn in training_outputs:
            if isfile(fn):
                os.remove(fn)
            path = splitext(fn)[0]
            for i in range(10):
                if isfile('{}_tile{:04}.npy'.format(path, i)):
                    os.remove('{}_tile{:04}.npy'.format(path, i))
            if isfile(path + '-plot.png'):
                os.remove(path + '-plot.png')

    num_files = len(xml_files)
    print "Found", num_files, "files to ingest"

    for i in range(num_files):
        print i+1, 'of', num_files, basename(xml_files[i]),

        if args.resume:
            output_files = glob(training_outputs[i]+'_tile*.npy')
            if len(output_files) > 0:
                print "already processed, skipping (resume is true)"
                continue

        annotations = Annotation(xml_files[i],
                                 images_root=img_root,
                                 xml_root=xml_root)
        assert isinstance(annotations, Annotation)
        annotations.remove_deleted()

        # The LabelMe tool does not set the image size
        annotations.update_image_size()

        data = MultiLayerLabels(annotations, nrows=args.tilesize[1])
        assert isinstance(data, MultiLayerLabels)
        data.mark_all()

        # Ignore some labels completely
        if args.ignore != '':
            for label in args.ignore.split(','):
                assert label in data.label_names
                index = data.label_names.index(label)
                data.label_data[:, :, index] = LABEL_UNKNOWN

        # But no layer can be completely ignored [caffe issue]
        ensure_unignored_labels(annotations, data)

        if args.plot:
            pl.close('all')
            pl.suptitle(basename(training_outputs[i]))
            data.plot().savefig(plot_outputs[i], dpi=100)
            print 'figure and',

        data.save_tiles(training_outputs[i], args.tilesize[1], args.tilesize[0])
        print 'labels saved.',
        labels_found = sorted(np.unique([o.name for o in annotations.objects]))
        print '\t(' + ','.join(labels_found) + ')'


def ensure_unignored_labels(annotations, data):
    # BEGIN HACK -- remove sky from all layers (no layer can be completely ignored)
    mask = SingleLayerLabels(data.annotations, data.nrows, data.ncols)
    assert isinstance(mask, SingleLayerLabels)
    mask.mark_positives([o.polygon for o in annotations.objects if o.name == 'sky'])
    sky = np.array(mask.image) == LABEL_POSITIVE
    for layer in range(data.nlayers):
        layer_labels = data.label_data[:, :, layer]

        #print
        #print layer_labels.shape
        #print
        layer_labels[sky] = LABEL_NEGATIVE

        # Further hack -- in case there is not enough sky, the top row should safely be negative
        if np.all(layer_labels == LABEL_UNKNOWN):
            layer_labels[0, :] = LABEL_NEGATIVE
    # END HACKISH THING


if __name__ == '__main__':
    main()

