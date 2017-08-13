import os
from glob import glob

def get_iter(path):
    s = os.path.splitext(path)[0]
    s = s.rsplit('_')[-1]
    iter = int(s)
    return iter


def get_last_iter(names):
    iters = [get_iter(name) for name in names]
    max_iter = max(iters)
    return max_iter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('names', help="A list of possible files", nargs='*', default=glob("Models/Training/*.solverstate"))
args = parser.parse_args()

print get_last_iter(args.names)