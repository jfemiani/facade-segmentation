from __future__ import print_function

import os
from collections import OrderedDict
from os.path import dirname, join

import numpy as np
import pyfacades.models.independant_12_layers.segment as i12
import skimage
import skimage.color
import skimage.io
import yaml
import yamlordereddictloader
from configargparse import ArgumentParser

from pyfacades.util import channels_first, channels_last
from pyfacades.util.metrics import complete, Metrics, viz_pixel_labels

LABEL_POSITIVE = 2
LABEL_EDGE = 3
LABEL_UNKNOWN = 1
LABEL_NEGATIVE = 0


def eval():
    p = ArgumentParser(default_config_files=[join(dirname(__file__), '.eval.cfg')])
    p.add('--config', '-c', is_config_file=True, help="config file path")
    p.add('--weights', type=str, help="path to the classifier weights")
    p.add('--data', type=str, help="test data, a file with each line as an .npy filename")
    p.add('--output', '-o', type=str, help="folder to hold outputs", default='.')
    p.add('--path', type=str, action='append', help='search paths for files')
    args = p.parse_args()

    try:
        os.makedirs(args.output)
    except OSError:
        pass


    # If weights is None, this will default to the weights in the models folder,
    # or the ones indicated by the environment variable
    net = i12.net(args.weights)

    files = [complete(p, args.path) for p in open(args.data).readlines()]

    summary = OrderedDict()

    for label in i12.LABELS:
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
            all_predicted = i12.process_strip(channels_first(rgb))

            results = OrderedDict()
            for label_index, label in enumerate(i12.LABELS):
                if label == 'background': continue
                # Skip the top row and the bottom row -- I gave those dummy labels to work around a caffe error
                expected = all_expected[label_index][1:-1]

                # Uggh!  Since the 'UNKNOWN' label is removed, argmax gives the wrong values; I need 0->0, 1->2, 2->3
                predicted = all_predicted.features[label_index].argmax(0)[1:-1]+1
                predicted[predicted==LABEL_UNKNOWN] = LABEL_NEGATIVE

                metrics = Metrics(expected, predicted, source=file, feature=label, threshold=0.5)
                results[label] = metrics.as_dict()

                viz = viz_pixel_labels(expected, predicted, rgb[1:-1],
                                       label_negative=LABEL_NEGATIVE,
                                       label_positive=LABEL_POSITIVE)

                skimage.io.imsave(os.path.join(args.output, 'file-{}-viz-{}.jpg'.format(file_index, label)), viz)

            with open(cached_file, 'w') as f:
                yaml.dump(results, f, Dumper=yamlordereddictloader.Dumper)

        cached = yaml.load(open(cached_file))
        print("Cumulative:")
        for label in i12.LABELS:
            if label == 'background':
                continue

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
            print("{:10}: pix(P:{:2.5f}, R:{:2.5f},F:{:2.5f}), obj:(P:{:2.5f}, R:{:2.5f},F:{:2.5f})".format(
                label,
                cum.pixel_precision,
                cum.pixel_recall,
                cum.pixel_f1,
                cum.object_precision,
                cum.object_recall,
                cum.object_f1))


if __name__ == '__main__':
    eval()
