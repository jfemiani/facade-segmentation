import os

import pyfacades
import pyfacades.export_parametric_facade
from pyfacades.util import find_files


def process_files(files, **kwargs):
    attempts = 0
    while len(files) > 0 and attempts < 5:
        attempts += 1
        failures = []
        for i, f in enumerate(files):
            mf = pyfacades.export_parametric_facade.MegaFacade()
            try:
                mf.load_image(f, **kwargs)
                mf.save_params(**kwargs)
            except Exception as e:
                print e
                import traceback
                traceback.print_exc()
                failures.append(f)
            print '\r', i + 1, 'of', len(files), f,

        files = failures

def process_dir(dir, pattern='orthographic.png', **kwargs):
    files = find_files(dir, pattern)
    process_files(files, **kwargs)


def process_zip(path_to_zip_file, out_dir=None, pattern='orthographic.png', **kwargs):
    import zipfile

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
    process_files(files, **kwargs)


def process_url(url, filename=None, out_dir=None, pattern='orthographic.png', **kwargs):
    import urllib
    filename, headers = urllib.urlretrieve(url, filename=filename)
    process_zip(filename, out_dir, pattern=pattern, **kwargs)