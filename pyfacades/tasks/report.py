from __future__ import print_function
import argparse
import os
import webbrowser
from shutil import copyfile
from urllib import pathname2url

import pdfkit

from os.path import join, dirname

from rope.base.pyobjectsdef import _AssignVisitor

p = argparse.ArgumentParser()

p.add_argument('--root', help="The root folder of a dataset, which is "
                              "a folder. Under this folder I expect to find "
                              "a name (e.g. 'congo') and then a mega-facade "
                              "index and an image-index. For example 'congo/1/2'")
p.add_argument('--outdir', help="Folder for results, broken up into small pages")

args = p.parse_args()

counter = 1

def save_html(outpath, root, datasets):
    global counter
    f = open(outpath, 'w')
    print("<html><body>", file=f)

    print("<!--", "args.root =", root, "-->", file=f)

    for dataset in datasets:
        if not os.path.isdir(join(root, dataset)):
            continue

        print('<div  style="float:top;">', file=f)
        print('<h1>  Dataset ', dataset, '</h1>', file=f)
        for megafacade in os.listdir(join(root, dataset)):
            print('<div style="float:top;white-space: nowrap;">', file=f)
            print('<h2> Megafacade ', megafacade, '</h2>', file=f)

            for image in os.listdir(join(root, dataset, megafacade)):
                image_folder = join(root, dataset, megafacade, image)
                regions_jpg = join(image_folder, 'regions.jpg')
                if not os.path.isdir(image_folder):
                    continue
                print('<div style="display:inline-block;">', file=f)
                localname = "img_{:06}.jpg".format(counter)
                counter += 1
                copyfile(regions_jpg, join(dirname(outpath), localname))
                print('<img height="400", src="{}"></img>'.format(localname), file=f)
                print('</div>', file=f)
            print('<div style="clear: both"></div>', file=f)
            print('</div>', file=f)

        print('</div>', file=f)

    print("</body></html>", file=f)


outdir = args.outdir
try:
    os.makedirs(outdir)
except OSError as e:
    pass

datasets = [d for d in os.listdir(args.root) if os.path.isdir(join(args.root, d))]
n = 5
pages = [datasets[i:min(len(datasets), i + n)] for i in range(0, len(datasets), n)]

idx = open(join(outdir, 'index.html'), 'w')
print("<html><body><ol>", file=idx)
for i, page in enumerate(pages):
    print(i+1)
    outpath = join(outdir, 'report-page-{:04}.html'.format(i + 1))

    print("<li><a href={url}>{url}</a>".format(url=pathname2url(os.path.relpath(outpath, outdir))), file=idx)
    save_html(outpath, args.root, page)

print("</ol></body></html>", file=idx)


webbrowser.open(join(outdir, 'index.html'))
