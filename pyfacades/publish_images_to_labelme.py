import os
import urlparse

import sys
from scipy.misc import imread, imsave

import pyfacades
from pyfacades.rectify import rectify, Homography
from pyfacades.util import find_files
import zipfile


def rename_file(filename, basedir):
    newname = os.path.relpath(filename, basedir)
    newname = newname.replace('/', '-')
    newname = os.path.splitext(newname)[0] + '.jpg'
    return newname

def process_files(files, basedir='./data', debug=False, **kwargs):
    attempts = 0
    n = len(files)

    try:
        os.makedirs('./data/for-labelme')
    except OSError as e:
        pass

    for i, f in enumerate(files):
        newname = rename_file(f, basedir)
        newname = os.path.join('./data/for-labelme', newname)
        image = imread(f)
        # h = Homography(image)
        # rectified = h.rectified
        # if debug:
        #     import pylab
        #     pylab.suptitle('u:{} d:{} l:{} r:{}'.format(h.du, h.dd, h.dl, h.dr))
        #     pylab.subplot(121)
        #     h.plot_original()
        #     pylab.subplot(122)
        #     h.plot_rectified()
        #     pylab.savefig(os.path.splitext(newname)[0] + '-rectification.jpg')

        print i+1, 'of', n, newname

        # TODO: Figure out a better way to rectify the images...
        imsave(newname, image)


def is_url(url):
    return urlparse.urlparse(url).scheme != ""

def is_zip(filename):
    return zipfile.is_zipfile(filename)

def process_dir(dir, pattern='orthographic.png', debug=False, **kwargs):
    files = find_files(dir, pattern)
    process_files(files, basedir=dir, debug=debug,  **kwargs)


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
    parser.add_argument('--debug',
                        type=bool,
                        default=False)

    args = parser.parse_args()
    print os.getcwd()

    if args.dir == '-':
        args.dir = raw_input("Enter a source url, zip, or folder:")
        print args.dir

    if is_url(args.dir):
        print "Processing from a URL..."
        process_url(args.dir, debug=args.debug)
    elif is_zip(args.dir):
        print "Processing from a ZIP"
        process_zip(args.dir, debug=args.debug)
    elif os.path.isdir(args.dir):
        print "Processing a Directory"
        process_dir(args.dir, debug=args.debug)
    else:
        print "Did not recognize argument as a URL, ZIP, or Folder..."


    "Now you need to copy the images to the labelme server and rebuild the db"

if __name__ == '__main__':
    main()
