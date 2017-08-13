""" Split data into folds

"""
from glob import glob
import numpy as np
import os
from os.path import dirname
import argparse

p = argparse.ArgumentParser()
p.add_argument('--folds', type=int, help="The number of folds to generate for cross validation", default=5)
p.add_argument('files', nargs='+', help="A list of files")
p.add_argument('--output', help="A prefix to put in front of each output file.")
p.add_argument('--debug', action='store_true', help="debug mode [dev only]")

args = p.parse_args()

if args.debug:
    import ipdb
    ipdb.set_trace()

files = [f for pattern in args.files for f in glob(pattern)]
# files = [f.split('_tile')[0] for f in files]
files = np.unique(files)
np.random.shuffle(files)

num_test = len(files) / args.folds
folds = [np.roll(files, num_test*i) for i in range(args.folds)]
folds = [(fold[:num_test], fold[num_test:]) for fold in folds]


# noinspection SpellCheckingInspection
def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


for i, (eval_, train) in enumerate(folds):
    train_path = args.output + 'fold_{:02}/train.txt'.format(i + 1)
    eval_path = args.output + 'fold_{:02}/eval.txt'.format(i + 1)
    mkdirp(dirname(train_path))
    mkdirp(dirname(eval_path))

    with open(eval_path, 'w') as f:
        f.writelines([s + '\n' for s in eval_])

    with open(train_path, 'w') as f:
        f.writelines([s + '\n' for s in train])

    print "Finished generating input file lists for fold", i + 1
