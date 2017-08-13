import argparse
from os.path import splitext

from pylab import *

from pyfacades.models.independant_12_layers.model import  LABELS
from pyfacades.util import channels_last

p  = argparse.ArgumentParser()
p.add_argument('labels', nargs='*', help="A .npy file of label output for this classifier")
p.add_argument('--save', '-s', action='store_true', help="Whether to save the figure")
p.add_argument('--output', '-o', help="The output file, if saving", default=[])


args = p.parse_args()

if args.save and len(args.output) == 0:
    p.output = [splitext(fn)[0] + '-plot.png' for fn in args.labels]



def colorize(labels, colors=None):
    if colors is None:
        colors = array(cm.jet(linspace(0,1,4), alpha=0.7, bytes=True), dtype=np.uint8)
    rgb = np.zeros(labels.shape[:2]+(4,), dtype=np.uint8)
    rgb.flat = colors[labels.astype(np.uint8).flatten(), :]
    return rgb

for k in range(len(args.labels)):
    data = np.load(args.labels[k])

    color = channels_last(data[:3])
    print "Color data:"
    print "  Max:", color.reshape(-1, 3).max(0)
    print "  Min:", color.reshape(-1, 3).min(0)
    print "  Mean:", color.reshape(-1, 3).mean(0)
    labels = data[3:]

    f, a = subplots(3, 4)
    f.set_size_inches(6*4, 6*3)
    a = a.flatten()
    for i, label in enumerate(LABELS):
        sca(a[i])
        title(label)
        imshow(color)
        imshow(colorize(labels[i]))
        #axis('off')
    show()