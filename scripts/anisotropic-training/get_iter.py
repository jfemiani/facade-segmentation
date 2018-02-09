#!/home/femianjc/anaconda2/envs/femiani/bin/python
import os
from glob import glob
import re


def get_iter(path):
    s = os.path.splitext(path)[0]
    s = s.rsplit('_')[-1]
    iter = int(s)
    return iter


def get_last_iter(names):
    iters = [get_iter(name) for name in names]
    max_iter = max(iters) if len(iters) > 0 else ''
    return max_iter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('names', 
                    help="A list of possible files", nargs='*',
                    default=[])
parser.add_argument('--snapshot', '-s', action='store_true', 
                    help="output the name of the snapshot, instead of just the iteration number")
parser.add_argument('--model', '-m', action='store_true', 
                    help="output the name of the model, instead of just the iteration number")
parser.add_argument('--solver', '-p', 
                    help="The solver.prototxt file", 
                    default='solver.prototxt')
args = parser.parse_args()

solver_proto = open(args.solver).read()
snapshot_prefix = re.search('^snapshot_prefix *: &*\"(.*)\"', solver_proto, re.MULTILINE).group(1).strip(' "')

if len(args.names) == 0:
    args.names = glob(snapshot_prefix + "*.solverstate")

if args.model and args.snapshot:
    print "Invalid argument, expect _eithe_ `--model` _or_ `--snapshot` but not both."
    parser.print_help()
elif args.model:
    print snapshot_prefix + '_iter_{}.caffemodel'.format(get_last_iter(args.names))
elif args.snapshot:
    print snapshot_prefix + '_iter_{}.solverstate'.format(get_last_iter(args.names))
else:
    print get_last_iter(args.names)