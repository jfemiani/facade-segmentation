from __future__ import print_function

import os
from collections import OrderedDict
from os.path import dirname, join, isfile

import numpy as np
import pyfacades.models.segnet_5_labels.segment as d5
import skimage
import skimage.color
import skimage.io
import yaml
import yamlordereddictloader
from configargparse import ArgumentParser
from skimage.measure import regionprops
from skimage.morphology import binary_opening
from skimage.morphology import disk

import pyfacades
import pyfacades.models.independant_12_layers.segment as i12
from pyfacades.util import channels_first, channels_last
from pyfacades.util.metrics import BBox


LABEL_POSITIVE = 2
LABEL_EDGE = 3
LABEL_UNKNOWN = 1
LABEL_NEGATIVE = 0


def complete(path, paths):
    for d in paths:
        cur = join(d, path.strip())
        if isfile(cur):
            return cur
    return path



import networkx as nx

def match_objects(candidates, targets):
    hits = []
    for t in targets:
        intersection_over_union = 0
        match = None
        tbb = BBox(*t.bbox)
        for c in candidates:
            cbb = BBox(*c.bbox)
            current_iou = tbb.intersection(cbb).area / float(tbb.union(cbb).area)
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
    matching = nx.max_weight_matching(g, maxcardinality=True) # <- a dict with  v->c and c->v both
    hits = [(t, c) for (t, c) in matching.items() if t in target_set]
    return hits


class Metrics(object):
    def __init__(self, expected=None, predicted=None, min_iou=0.5, threshold=0.5,
                 feature='', source='',
                 TP=0, FP=0, FN=0, TN=0, hits=0, targets=0, candidates=0, images=0, **kwargs):
        super(Metrics, self).__init__()

        self.min_intersection_over_union = min_iou
        self.threshold = threshold
        self.feature = feature
        self.source = source

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
            self.TP = float(np.count_nonzero((expected==LABEL_POSITIVE) & (predicted==LABEL_POSITIVE)))
            self.FP = float(np.count_nonzero((expected==LABEL_NEGATIVE) & (predicted==LABEL_POSITIVE)))
            self.TN = float(np.count_nonzero((expected==LABEL_NEGATIVE) & (predicted==LABEL_NEGATIVE)))
            self.FN = float(np.count_nonzero((expected==LABEL_POSITIVE) & (predicted==LABEL_NEGATIVE)))

            targets = regionprops(skimage.measure.label(expected==LABEL_POSITIVE))

            mask = predicted==LABEL_POSITIVE
            se = disk(3)
            mask = binary_opening(mask, selem=se)

            candidates = regionprops(skimage.measure.label(mask))
            # hits = match_objects(candidates, targets)
            hits = match_objects_uniquely(candidates, targets)

            self.targets = len(targets)
            self.candidates = len(candidates)
            self.hits = len(hits)
            self.images = 1

    def __add__(self, other):
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
        if  self.candidates == 0:
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
            yaml.dump(self.as_dict(), path)

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as f:
            result = Metrics(**yaml.load(f))
            return result


def eval():
    p = ArgumentParser(default_config_files=[join(dirname(__file__), '.eval_d5.cfg')])
    p.add('--config', '-c', is_config_file=True, help="config file path")
#    p.add('--weights', type=str, help="path to the classifier weights")
    p.add('--data', type=str, help="test data, a file with each line as an .npy filename")
    p.add('--output', '-o', type=str, help="folder to hold outputs", default='.')
    p.add('--path', type=str, action='append', help='search paths for files')
    args = p.parse_args()

    try:
        os.makedirs(args.output)
    except OSError:
        pass


    files = [complete(p, args.path) for p in open(args.data).readlines()]

    summary = OrderedDict()
    LABELS=('window', 'door')

    for label in LABELS:
        summary[label] = OrderedDict()
        summary[label].update(TP=0)
        summary[label].update(TN=0)
        summary[label].update(FP=0)
        summary[label].update(FN=0)
        summary[label].update(targets=0)
        summary[label].update(candidates=0)
        summary[label].update(hits=0)


    for file_index, file in enumerate(files):
        print(file_index+1, "of", len(files), ":", file)
        cached_file = os.path.join(args.output, "metrics-{:04}.yml".format(file_index))

        data = np.load(file)
        rgb = channels_last(data[:3].astype(np.uint8))
        if not os.path.isfile(os.path.join(args.output, 'file-{}.jpg'.format(file_index))):
            skimage.io.imsave(os.path.join(args.output, 'file-{}.jpg'.format(file_index)), rgb)
        all_expected = data[3:]

        if not os.path.isfile(os.path.join(args.output, 'file-{}-windows.jpg'.format(file_index))):
            skimage.io.imsave(os.path.join(args.output, 'file-{}-windows.jpg'.format(file_index)),
                              skimage.color.gray2rgb((all_expected[
                                                          pyfacades.models.independant_12_layers.model.WINDOW] == 2).astype(float)))

        if not os.path.isfile(cached_file):
            print("Calculating metrics for file", file)
            #data = np.load(file)
            #rgb = channels_last(data[:3].astype(np.uint8))
            #all_expected = data[3:]
            all_predicted, conf = d5.process_strip(channels_first(rgb))
            label_out = all_predicted.argmax(0)

            results = OrderedDict()
            for label in LABELS:

                # Skip the top row and the bottom row -- I gave those dummy labels to work around a caffe error
                expected = all_expected[i12.LABELS.index(label)][1:-1]

                # The labels are at different indexes...
                if label == 'window':
                    predicted = (label_out == d5.WINDOW)*2  # 0->LABEL_NEGATIVE, 2->LABEL_POSITIVE
                elif label == 'door':
                    predicted = (label_out == d5.DOOR)*2
                predicted = predicted[1:-1]

                metrics = Metrics(expected, predicted, source=file, feature=label, threshold=0.5)
                results[label] = metrics.as_dict()

                viz = np.zeros(expected.shape + (3,), dtype=np.uint8)
                viz[(predicted == LABEL_POSITIVE) & (expected == LABEL_POSITIVE)] = (0, 255, 0)
                viz[(predicted == LABEL_POSITIVE) & (expected == LABEL_NEGATIVE)] = (255, 0, 0)
                viz[(predicted == LABEL_NEGATIVE) & (expected == LABEL_POSITIVE)] = (0, 0, 255)
                viz = (0.5*viz + 0.4*rgb[1:-1]).astype(np.uint8)
                skimage.io.imsave(os.path.join(args.output, 'file-{}-viz-{}.jpg'.format(file_index, label)), viz)

            with open(cached_file, 'w') as f:
                yaml.dump(results, f, Dumper=yamlordereddictloader.Dumper)

        cached = yaml.load(open(cached_file))
        print("Cumulative:")
        for label in LABELS:
            metrics = Metrics(**cached[label])
            assert metrics.source == file

            summary[label]['TP'] += metrics.TP
            summary[label]['FP'] += metrics.FP
            summary[label]['TN'] += metrics.TN
            summary[label]['FN'] += metrics.FN
            summary[label]['targets'] += metrics.targets
            summary[label]['candidates'] += metrics.candidates
            summary[label]['hits'] += metrics.hits

            cum = Metrics(**summary[label])
            print("{:10}: pix(P:{:2.5f}, R:{:2.5f}, F:{:2.5f}), obj:(P:{:2.5f}, R:{:2.5f}, F:{:2.5f})".format(
                label,
                cum.pixel_precision,
                cum.pixel_recall,
                cum.pixel_f1,
                cum.object_precision,
                cum.object_recall,
                cum.object_f1))



if __name__ == '__main__':
    eval()
