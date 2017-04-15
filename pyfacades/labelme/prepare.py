import os
import urlparse

import sys

import skimage
from scipy.misc import imread, imsave
import zipfile
import numpy as np

import pyfacades
from pyfacades.rectify import rectify, Homography
from pyfacades.util import find_files, channels_first
import pyfacades.driving_12x360x480.segment
import pyfacades.driving_12x360x480.model


def rename_file(filename, basedir):
    newname = os.path.relpath(filename, basedir)
    newname = newname.replace('/', '-')
    newname = os.path.splitext(newname)[0] + '.jpg'
    return newname


def rectify_building(image, meta=None):
    labels = pyfacades.driving_12x360x480.segment.process_strip(channels_first(image))
    non_rectangular = building = np.argmax(labels, 0) == pyfacades.driving_12x360x480.model.BUILDING
    h = pyfacades.rectify.Homography(image, mask=non_rectangular)

    if meta is not None:
        meta['labels'] = labels
        meta['building'] = building
        meta['homography'] = h
    return h.rectified


def process_files(files, basedir='./data', debug=False, rectify=False,
                  outdir='./data/for-labelme', **kwargs):
    attempts = 0
    n = len(files)

    print "Rectify is set to", rectify

    try:
        os.makedirs(outdir)
    except OSError as e:
        pass

    if debug:
        try:
            os.makedirs(os.path.join(outdir, 'debug'))
        except OSError as e:
            # Directory already exists
            pass

    for i, f in enumerate(files):
        try:
            newbasename = rename_file(f, basedir)
            newname = os.path.join(outdir, newbasename)
            print i + 1, 'of', n, newname

            image = imread(f)

            if rectify:
                try:
                    meta = {}
                    rectified = rectify_building(image, meta)
                    if debug:
                        import pylab as pl
                        h = meta['homography']
                        pl.suptitle('u:{} d:{} l:{} r:{}'.format(h.du, h.dd, h.dl, h.dr))
                        pl.subplot(221)
                        pl.imshow(image)
                        pl.axis('off')
                        pl.subplot(222)
                        pl.imshow(meta['building'])
                        pl.axis('off')
                        pl.subplot(223)
                        h.plot_original()
                        pl.subplot(224)
                        h.plot_rectified()
                        pl.savefig(os.path.join(outdir, 'debug', newbasename))
                    imsave(newname, rectified)
                except Exception as e:
                    print e
                    pass
            else:
                imsave(newname, image)
        except Exception as e:
            print e


def is_url(url):
    return urlparse.urlparse(url).scheme != ""


def is_zip(filename):
    return zipfile.is_zipfile(filename)


def process_dir(dir, pattern='orthographic.png', debug=False, **kwargs):
    files = find_files(dir, pattern)
    print "Found", len(files), "files."
    process_files(files, basedir=dir, debug=debug, **kwargs)


def process_zip(path_to_zip_file, out_dir=None, pattern='orthographic.png', debug=False, **kwargs):
    if out_dir is None:
        out_dir = os.path.splitext(path_to_zip_file)[0]
        try:
            os.makedirs(out_dir)
        except Exception as e:
            print e
            print type(e)

    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    print "Extracting contents of the zip file.."
    zip_ref.extractall(out_dir)
    zip_ref.close()

    files = find_files(out_dir, pattern=pattern)
    process_files(files, debug=debug, **kwargs)


def process_url(url, filename=None, out_dir=None, pattern='orthographic.png', debug=False, **kwargs):
    import urllib
    filename, headers = urllib.urlretrieve(url, filename=filename)
    process_zip(filename, out_dir, pattern=pattern, debug=debug, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', default='-')
    parser.add_argument('--pat', type=str, default='orthographic.png')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--rectify', dest='rectify', action='store_true')
    parser.add_argument('--no-rectify', dest='rectify', action='store_false')
    parser.add_argument('--gpu', type=int, help="Which GPU to use (we are not using the CPU)", default=0)
    parser.add_argument('--cpu', action='store_true', help="Use the CPU (and not the GPU)")

    args = parser.parse_args()

    # init caffe
    import caffe
    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()

    print os.getcwd()

    if args.dir == '-':
        args.dir = raw_input("Enter a source url, zip, or folder:")
        print args.dir

    options = dict(
        rectify=args.rectify,
        pattern=args.pat,
        debug=args.debug
    )

    if is_url(args.dir):
        print "Processing from a URL..."
        process_url(args.dir, **options)
    elif is_zip(args.dir):
        print "Processing from a ZIP"
        process_zip(args.dir, **options)
    elif os.path.isdir(args.dir):
        print "Processing a Directory"
        process_dir(args.dir, **options)
    else:
        print "Did not recognize argument as a URL, ZIP, or Folder..."

    print "Now you need to copy the images to the labelme server and rebuild the db"


if __name__ == '__main__':
    main()
