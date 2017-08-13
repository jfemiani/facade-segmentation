import os
import re

import numpy as np
import skimage.io
from PIL import Image

from pyfacades.util.metrics import Metrics, viz_pixel_labels

# Lama's results are provided as text files with the bounding boxes and confidences on each line
LAMA_RESULTS_PATH = '/home/shared/Projects/Facades/data/lama_results/boxes'

# Our results are shared binary images [0/1]
OUR_RESULTS_PATH = '/home/shared/Projects/Facades/data/lama_results/ours'

# Get the raw list of files, sorted by the numbers I included in the names
TEST_FILES = [f for f in os.listdir(OUR_RESULTS_PATH) if re.match(r"file-([0-9]+)\.jpg", f)]
TEST_FILES = sorted(TEST_FILES, key=lambda f: int(re.match(r"file-([0-9]+)\.jpg", f).group(1)))


def load_lama_result(fn, shape):
    with open(fn) as f:
        lines = [line.strip().split(',') for line in f.readlines()]

    predicted = np.zeros(shape, dtype=np.uint8)
    for line in lines:
        confidence = float(line[-1])
        col_min, row_min, col_max, row_max = [int(v) for v in line[:-1]]

        predicted[row_min:row_max, col_min:col_max] = 1

    return predicted


total = Metrics(feature='windows')
for f in TEST_FILES:
    print f,
    stem = os.path.splitext(f)[0]
    lama_input_file = os.path.join(LAMA_RESULTS_PATH, stem + '.png')
    lama_results_file = os.path.join(LAMA_RESULTS_PATH, stem + '.txt')
    our_expected_file = os.path.join(OUR_RESULTS_PATH, stem + '-windows.jpg')
    source = Image.open(lama_input_file)

    predicted = load_lama_result(lama_results_file, (source.height, source.width))
    raw_expected = np.asarray(Image.open(our_expected_file), dtype=np.uint8)
    expected = np.asarray(Image.open(our_expected_file), dtype=np.uint8)[:, :, 0] > 128

    current = Metrics(expected, predicted, feature='windows')
    total += current

    viz = viz_pixel_labels(expected, predicted, np.asarray(source), label_negative=0, label_positive=1)
    lama_results_viz_file = os.path.join(LAMA_RESULTS_PATH, stem + '-viz-windows.jpg')
    skimage.io.imsave(lama_results_viz_file, viz)

    current.save_yaml(os.path.join(LAMA_RESULTS_PATH, stem + '-windows.yml'))

    print current, total

total.save_yaml(os.path.join(LAMA_RESULTS_PATH, 'windows.yml'))
