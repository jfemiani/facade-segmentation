"""
Visually compare the test results using side by side images
"""

import re
import os
import webbrowser


# from urllib import pathname2url
def pathname2url(p):
    return 'file://' + p

import numpy as np

LAMA_RESULTS_PATH = '/home/shared/Projects/Facades/data/lama_results/boxes'
OUR_I12_RESULTS_PATH = '/home/shared/Projects/Facades/data/lama_results/ours'
OUR_D5_RESULTS_PATH = '/media/femianjc/My Book/independant_12_layers/eval-d5-unique-open3'

# Get the raw list of files, sorted by the numbers I included in the names
TEST_FILES = [f for f in os.listdir(OUR_I12_RESULTS_PATH) if re.match(r"file-([0-9]+)\.jpg", f)]
TEST_FILES = sorted(TEST_FILES, key=lambda f: int(re.match(r"file-([0-9]+)\.jpg", f).group(1)))

with open('compare_methods.html', 'w') as f:

    print >> f, "<HTML>"
    print >> f, "<BODY>"

    print >> f, "<TABLE>"

    for file in TEST_FILES:
        stem = os.path.splitext(file)[0]
        print >> f, "<TR>"

        print >> f, "<TD>"
        print >> f, "<IMG src=\"{}\" alt=\"source\">".format(pathname2url(os.path.join(OUR_I12_RESULTS_PATH, file)))
        print >> f, "</TD>"

        print >> f, "<TD>"
        print >> f, "<IMG src=\"{}\" alt=\"lama\">".format(
            pathname2url(os.path.join(LAMA_RESULTS_PATH, stem + '-viz-windows.jpg')))

        print >> f, "</TD>"

        print >> f, "<TD>"
        print >> f, "<IMG src=\"{}\" alt=\"ours\">".format(
            pathname2url(os.path.join(OUR_I12_RESULTS_PATH, stem + '-viz-window.jpg')))

        print >> f, "</TD>"

        print >> f, "</TR>"

    print >> f, "</TABLE>"

    print >> f, "</BODY>"
    print >> f, "</HTML>"


webbrowser.open('compare_methods.html')