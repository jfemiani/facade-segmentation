from collections import OrderedDict
from os.path import join, isfile

import networkx as nx
import numpy as np
import skimage
import yaml
from skimage.measure import regionprops
from skimage.morphology import disk, binary_opening

def complete(path, paths):
    for d in paths:
        cur = join(d, path.strip())
        if isfile(cur):
            return cur
    return path


class BBox(object):
    def __init__(self, row_min, col_min, row_max, col_max, **args):
        super(BBox, self).__init__()
        self.row_min = row_min
        self.row_max = row_max
        self.col_min = col_min
        self.col_max = col_max

    def intersection(self, other):
        assert isinstance(other, BBox)
        col_min = max(self.col_min, other.col_min)
        col_max = min(self.col_max, other.col_max)
        row_min = max(self.row_min, other.row_min)
        row_max = min(self.row_max, other.row_max)
        return BBox(row_min, col_min, row_max, col_max)

    def union(self, other):
        assert  isinstance(other, BBox)
        col_min = min(self.col_min, other.col_min)
        col_max = max(self.col_max, other.col_max)
        row_min = min(self.row_min, other.row_min)
        row_max = max(self.row_max, other.row_max)
        return BBox(row_min, col_min, row_max, col_max)

    @property
    def _slice(self):
        return slice(self.row_min, self.row_max), slice(self.col_min, self.col_max)

    @property
    def area(self):
        return self.width * self.height

    @property
    def width(self):
        return self.col_max - self.col_min

    @property
    def height(self):
        return self.row_max - self.row_min

    def __repr__(self):
        return "BBox({},{},{},{})".format(self.row_min, self.col_min, self.row_max, self.col_max)


def match_objects(candidates, targets):
    hits = []
    for t in targets:
        intersection_over_union = 0
        match = None
        tbb = BBox(*t.bbox)
        for c in candidates:
            cbb = BBox(*c.bbox)
            intersection = tbb.intersection(cbb).area
            union = tbb.area + cbb.area - intersection

            try:
                current_iou = intersection / float(union)
            except ZeroDivisionError:
                current_iou = float('inf')
                pass

            if current_iou > intersection_over_union:
                intersection_over_union = current_iou
                match = c
        if intersection_over_union > 0.5:
            hits.append((t, match))
    return hits


def match_objects_uniquely(candidates, targets, threshold=0.5):
    g = nx.Graph()
    for t in targets:
        g.add_node(t)
    for c in candidates:
        g.add_node(c)

    for t in targets:
        tbb = BBox(*t.bbox)
        for c in candidates:
            cbb = BBox(*c.bbox)
            intersection = tbb.intersection(cbb).area
            union = tbb.area + cbb.area - intersection

            try:
                current_iou = intersection / float(union)
            except ZeroDivisionError:
                current_iou = float('inf')
                pass

            if current_iou >= threshold:
                g.add_edge(t, c, weight=current_iou)

    target_set = set(targets)
    # Between versions 1.9 and 2.2, networkx did something evil here. 
    # They changed the _semantics_ AND the _api_ without changing the name OR deprecating first.
    # In the 2.2 API, it returns a set of edges in arbitrary order -- e.g. (t, c) or (c,t) are the same
    # and one does not know which will be returned. 
    # By contrast, in the 1.9 API it would return each edge twice; that is (t,c) and (c,t) would both 
    # be returned (in a sencse, via kv pairs in a dict).
    matching = nx.max_weight_matching(g, maxcardinality=True)  # <- a dict with  v->c and c->v both
    #hits = [(t, c) for (t, c) in matching if t in target_set]
    return matching # hits


class Metrics(object):

    def __init__(self, expected=None, predicted=None, min_iou=0.5, threshold=0.5,
                 feature='', source='',
                 TP=0, FP=0, FN=0, TN=0, hits=0, targets=0, candidates=0, images=0,
                 label_positive=1, label_negative=0, **kwargs):
        """
        Compute evaluation metrics from two labeled images (can also be aggregated over multiple images)

        This class compares two labels and calculates the per-pixel and also per-object statistics such as precision,
        recall, and f-score.

        You provide the `expected` and the `predicted` labels, and then the Metric's exposes various stats as properties

        :param expected: The expected labels
        :param predicted: The predicted labels (output of out method)
        :param min_iou: The minimum intersection-over-union for a pair of objects to be considered a match
        :param threshold: TBD
        :param feature: For documenting the saved files, what it the name of feature (object type) are the labels
                        represent.
        :param source: For documenting saved files, what image was the source used to calculate the labele
        :param TP: For accumulating multiple metrics, what are the per-pixel scores from past images
        :param FP: For accumulating multiple metrics, what are the per-pixel scores from past images
        :param FN: For accumulating multiple metrics, what are the per-pixel scores from past images
        :param TN: For accumulating multiple metrics, what are the per-pixel scores from past images
        :param hits: For accumulating multiple metrics, what are the number of object matches from past images
        :param targets: For accumulating multiple metrics, what are the number of objects expected in past images
        :param candidates: For accumulating multiple metrics, what are the number of objects detected in past images
        :param images: For accumulating multiple metrics, how many images have been aggregated?

        :param label_negative: That label should be interpreted as a negative? Labels that are not + or - are ignored
        :param label_positive: What label should be interpreted as a positive? Labels that are not + or - are ignored

        """
        super(Metrics, self).__init__()

        self.min_intersection_over_union = min_iou
        self.threshold = threshold
        self.feature = feature
        self.source = source
        self.label_positive = label_positive
        self.label_negative = label_negative

        if predicted is None:
            self.TP = TP
            self.FP = FP
            self.FN = FN
            self.TN = TN
            self.hits = hits
            self.targets = targets
            self.candidates = candidates
            self.images = images
        else:
            self.TP = float(np.count_nonzero((expected == self.label_positive) & (predicted == self.label_positive)))
            self.FP = float(np.count_nonzero((expected == self.label_negative) & (predicted == self.label_positive)))
            self.TN = float(np.count_nonzero((expected == self.label_negative) & (predicted == self.label_negative)))
            self.FN = float(np.count_nonzero((expected == self.label_positive) & (predicted == self.label_negative)))

            targets = regionprops(skimage.measure.label(expected == self.label_positive))

            mask = predicted == self.label_positive
            se = disk(3)
            mask = binary_opening(mask, selem=se)

            candidates = regionprops(skimage.measure.label(mask))
            hits = match_objects_uniquely(candidates, targets)

            self.targets = len(targets)
            self.candidates = len(candidates)
            self.hits = len(hits)
            self.images = 1

    def __iadd__(self, other):
        assert self.min_intersection_over_union == other.min_intersection_over_union
        assert self.threshold == other.threshold

        self.TP += other.TP
        self.FP += other.FP
        self.TN += other.TN
        self.FN += other.FN
        self.targets += other.targets
        self.candidates += other.candidates
        self.hits += other.hits
        self.images += other.images

        return self

    @property
    def pixel_total(self):
        return self.TP + self.TN + self.FP + self.FN

    @property
    def object_f1(self):
        if (self.object_precision + self.object_recall) > 0:
            return (2.0 * self.object_precision * self.object_recall) / (self.object_precision + self.object_recall)
        else:
            return float('NaN')

    @property
    def object_precision(self):
        if self.candidates == 0:
            return float('NaN')
        else:
            return self.hits / float(self.candidates)

    @property
    def object_recall(self):
        if self.targets == 0:
            return float('NaN')
        else:
            return self.hits / float(self.targets)

    @property
    def pixel_f1(self):
        if (self.pixel_precision + self.pixel_recall) > 0:
            return (2.0 * self.pixel_precision * self.pixel_recall) / (self.pixel_precision + self.pixel_recall)
        else:
            return float('NaN')

    @property
    def pixel_recall(self):
        if (self.TP + self.FN) == 0:
            return float('NaN')
        else:
            return self.TP / float(self.TP + self.FN)

    @property
    def pixel_precision(self):
        if self.TP + self.FP == 0:
            return float('NaN')
        else:
            return self.TP / float(self.TP + self.FP)

    @property
    def pixel_accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def as_dict(self):
        result = OrderedDict()
        result.update(feature=self.feature)
        result.update(source=self.source)
        result.update(TP=self.TP)
        result.update(TN=self.TN)
        result.update(FP=self.FP)
        result.update(FN=self.FN)
        result.update(targets=self.targets)
        result.update(candidates=self.candidates)
        result.update(hits=self.hits)
        result.update(pixel_recall=self.pixel_recall)
        result.update(pixel_precision=self.pixel_precision)
        result.update(pixel_f1=self.pixel_f1)
        result.update(object_recall=self.object_recall)
        result.update(object_precision=self.object_precision)
        result.update(object_f1=self.object_f1)
        result.update(min_intersection_over_union=self.min_intersection_over_union)
        result.update(threshold=self.threshold)
        result.update(images=self.images)
        return result

    def save_yaml(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.as_dict(), f)

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as f:
            result = Metrics(**yaml.load(f))
            return result

    def __repr__(self):
        return "Metric(P={:04.2f}, R={:04.2f}, F={:04.2f},  OP={:04.2f}, OR={:04.2f}, OF={:04.2f})".format(
            self.pixel_precision,
            self.pixel_recall,
            self.pixel_f1,
            self.object_precision,
            self.object_recall,
            self.object_f1)

def viz_pixel_labels(expected, predicted, rgb, label_positive, label_negative):
    viz = np.zeros(expected.shape + (3,), dtype=np.uint8)
    viz[(predicted == label_positive) & (expected == label_positive)] = (0, 255, 0)
    viz[(predicted == label_positive) & (expected == label_negative)] = (255, 0, 0)
    viz[(predicted == label_negative) & (expected == label_positive)] = (0, 0, 255)
    viz = (0.5 * viz + 0.4 * rgb).astype(np.uint8)
    return viz
