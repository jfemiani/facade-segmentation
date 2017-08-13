from __future__ import division
from __future__ import print_function

import argparse
import json
import os
from glob import glob
from os.path import join, relpath, splitext, dirname, basename

import numpy as np
from pylab import *
from skimage.color.colorconv import rgb2gray, gray2rgb
from skimage.transform._geometric import AffineTransform, warp
from skimage.util.dtype import img_as_float

from pyfacades.labelme.annotation import Annotation
from pyfacades.models.independant_12_layers.import_labelme import SingleLayerLabels, LABEL_POSITIVE
from pyfacades.rectify import Homography

LABELME_ROOT = "../from_labelme"
if 'LABELME_ROOT' in os.environ:
    LABELME_ROOT = os.environ['LABELME_ROOT']

today = datetime.datetime.now()
ts = today.strftime("%Y-%m-%d")

# Command Line Arguments
p = argparse.ArgumentParser()
p.add_argument("--labelme",
               help="the root location of LabelMe data",
               default=LABELME_ROOT)

#      the blacklist may be the 'annotationCache/Dirlist/xxxx.txt' files from past runs.
p.add_argument("--since",
               help="the earliest date of an annotation to process. Format is YYYY-MM-DD",
               default='2001-01-01')

p.add_argument("--output", "-o",
               help="output folder",
               default="./output-{}".format(ts))

p.add_argument("--collection", "-c",
               help="The name of the collection (dirlist used by LabelMe)",
               default="facades-{}.txt".format(ts))

p.add_argument("--ofolder",
               help="the output filder (all facades are copied into a single output folder)",
               default="facades-{}".format(ts))

p.add_argument("--pad", type=int,
               help="The amount to padd each side of the facade by when rectifying. "
                    "Padding will include some features of the input that may extend past "
                    "the main facade",
               default=100)
p.add_argument("--use-quads",
               help="Use the quads to rectify, instead of using Lama's automatic approach",
               action='store_true')
p.add_argument("--start-at", type=int, default=0)

p.add_argument("--list", 
               help="A list of XML files to process, if they exist under the LabelMe"
                    " annotations folder")

args = p.parse_args()

images_root = join(args.labelme, "Images")

try:
    os.makedirs(args.output)
except OSError:
    pass

if args.list is None:
    xml_files = glob(join(args.labelme, 'Annotations', '*/*.xml'))
else:
    print("Reading files from a list")
    with open(args.list) as f:
        xml_files = [join(args.labelme, 'Annotations', filename.strip()) for filename in f]

results = []

since = datetime.datetime.strptime(args.since, "%Y-%m-%d")

total_new_facades = 0

collection = []

for i, xml in enumerate(xml_files):
    if i < args.start_at:
        print (i + 1, ":", xml, "[SKIPPED]")
        continue
      
    if not os.path.isfile(xml):
       print(i+1, ":", "Missing file", xml)
       continue

    stem = splitext(relpath(xml, join(args.labelme, 'Annotations')))[0]
    folder = os.path.dirname(stem)
    stem = basename(stem)
    img = join(args.labelme, 'Images', folder, stem + '.jpg')
    xml = join(args.labelme, 'Annotations', folder, stem + '.xml')

    print(i + 1, ":", stem)

    a = Annotation(xml, images_root=images_root)
    assert isinstance(a, Annotation)
    a.remove_deleted()

    all_facades = [o for o in a.objects if o.name.lower() == 'facade']
    facades = [o for o in all_facades if o.date > since]

    print("   ", len(facades), "of", len(all_facades), "since ", since)
    total_new_facades += len(facades)

    if len(facades) == 0:
        # Nothing else to do!
        continue

    a.update_image_size()
    data = a.get_image()
    data_array = np.asarray(data)

    use_quad = args.use_quads

    if not use_quad:
        masker = SingleLayerLabels(a, nrows=a.imagesize.nrows)
        assert isinstance(masker, SingleLayerLabels)
        masker.mark_positives([o.polygon for o in a if o.name == 'occlusion'])
        masker.mark_positives([o.polygon for o in a if o.name == 'tree'])
        masker.mark_positives([o.polygon for o in a if o.name == 'sky'])
        mask = np.asarray(masker.image) == LABEL_POSITIVE

    for j, f in enumerate(all_facades):
        if f.date <= since:
            continue
        in_quad = f.polygon.points

        x0 = int(in_quad[:, 0].min())
        y0 = int(in_quad[:, 1].min())
        x1 = int(in_quad[:, 0].max())
        y1 = int(in_quad[:, 1].max())

        if x0 < 20 or (data.width-x1) < 20:
            #Skip facades at the ends
            continue

        i0 = np.argmin([np.hypot(x - x0, y - y0) for (x, y) in in_quad])
        i1 = np.argmin([np.hypot(x - x1, y - y0) for (x, y) in in_quad])
        i2 = np.argmin([np.hypot(x - x1, y - y1) for (x, y) in in_quad])
        i3 = np.argmin([np.hypot(x - x0, y - y1) for (x, y) in in_quad])

        in_quad = in_quad[(i0, i1, i2, i3), :]

        pad = args.pad
        width = in_quad[:, 0].max() - in_quad[:, 0].min()
        height = in_quad[:, 1].max() - in_quad[:, 1].min()
        out_quad = array([(0, 0), (width, 0), (width, height), (0, height)]) + pad

        # import ipdb; ipdb.set_trace()

        metadata = dict(folder=folder, stem=stem)
        metadata['polygon'] = f.polygon.points.tolist()
        highlight = np.zeros((data.height, data.width), dtype=np.uint8)
        f.draw(highlight, fill=255, outline=128)

        if use_quad:
            P = AffineTransform()
            P.estimate(out_quad, in_quad)
            output = warp(data, P, output_shape=(height + 2 * pad, width + 2 * pad))
            sub_highlight = warp(highlight, P, output_shape=(height + 2 * pad, width + 2 * pad))
            projection_matrix = P.params
            metadata['use_quad'] = True
            metadata['projection'] = projection_matrix.tolist()
            metadata['subimage'] = None
        else:
            # import ipdb; ipdb.set_trace()
            data_array = img_as_float(data_array)

            ptop = max(0, y0 - pad)
            pbottom = min(data.height, y1 + pad)
            pright = min(data.width, x1 + pad)
            pleft = max(0, x0 - pad)

            sub_image = data_array[ptop:pbottom, pleft:pright, :].copy()
            sub_mask = mask[ptop:pbottom, pleft:pright]
            sub_highlight = highlight[ptop:pbottom, pleft:pright]

            H = Homography(sub_image, mask=sub_mask)
            output = H.rectified
            sub_highlight = warp(sub_highlight,
                                 AffineTransform(H.H),
                                 preserve_range=True)

            gs = gray2rgb(rgb2gray(output))

            highlighted = output.copy()
            highlighted[sub_highlight==0] = 0.5*gs[sub_highlight==0]
            highlighted[sub_highlight==128] = (255, 0, 0)

            projection_matrix = H.H
            metadata['use_quad'] = False
            metadata['projection'] = projection_matrix.tolist()
            metadata['subimage'] = dict(left=ptop, right=pbottom, bottom=pright, top=pleft)

        out_folder = args.ofolder
        out_basename = stem + '-facade-{:02}'.format(j + 1)
        fname = join(args.output, 'Images', out_folder, out_basename + '.jpg')
        try:
            os.makedirs(dirname(fname))
        except OSError:
            pass
        imsave(splitext(fname)[0]+'-original.jpg', output)
        imsave(splitext(fname)[0]+'-highlighted.jpg', highlighted)
        imsave(splitext(fname)[0]+'-mask.jpg', sub_highlight)


        with open(splitext(fname)[0] + '.json', 'w') as mdf:
            json.dump(metadata, mdf)

        collection.append(','.join([out_folder, out_basename + '-highlighted' + '.jpg']) + '\n')

    try:
        os.makedirs(join(args.output, 'annotationCache', 'DirLists'))
    except OSError:
        pass

    with open(join(args.output, 'annotationCache', 'DirLists', args.collection), 'w') as f:
        f.writelines(collection)
