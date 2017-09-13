# coding=utf-8
from __future__ import print_function
from pylab import *
import argparse
import os
from os.path import join, basename, dirname, splitext
from glob import glob

# Environment arguments (defaults for command line)
from pyfacades.labelme.annotation import Annotation, Object

LABELME_ROOT = "/labelme"

# Command Line Arguments
p = argparse.ArgumentParser()

p.add_argument("--labelme",
               help="the root location of LabelMe data",
               default=LABELME_ROOT)

args = p.parse_args()

xml_files = glob(join(args.labelme, 'Annotations', '*/*.xml'))

results = []

days = {}

names = {}

NAMES = {
    'shop',
#    'occluded',
    'sign',
    'obstruction',
    'roof',
    'facade',
    'cornice',
    'balcony',
    'molding',
    'sky',
    'negative',
    'window',
    'unlabeled',
    'sill',
    'tree',
    'bay',
    'unknown',
    'awning',
    'pillar',
    'door',
    'ledge',
    'deco',
    'flag',
    'air-conditioner',
    'fire-escape-ladder',
    'fire-escape-balcony'
}

ALIASES = {
    u'tre': 'tree',
    #    u'asdf': None,
    u'obstructionç': 'obstruction',
    u'egative': 'negative',
    u'unlabed': 'unlabeled',
    u'unlabel': 'unlabeled',
    u'unlabeled_': 'unlabeled',
    u'occluision': 'obstruction',
    u'unlabled': 'unlabeled',
    u'unlabale': 'unlabeled',
    #    u'v': None,
    u'modeling': 'molding',
    u'c': None,
    u'occluded': 'obstruction',
    u'occlusion': 'obstruction',
    u'ob': 'obstruction',
    #    u'o': None,
    #    u'sh': None,
    u'unknown': 'unlabeled',
    u'sil': 'sill',
    u'corniceç': 'cornice',
    u'unknow': 'unlabeled',
    u'negattive':'negative',
    u'neattive':'negative',
    u'license plate':None,
    u'siil':'sill',
    u'occluded': 'obstruction',
    u'occlusion': 'obstruction',
    u'corice':'cornice',
    u'ngeative':'negative',
    u'obstuction':'obstruction',
    u'0':None,
    u'unknonw':'unknown',
    u'o bstruction':'obstruction',
    u'negaitve':'negative',
    u'obatruction':'obstruction',
    u'obstrucion':'obstruction',
    u'blacony':'balcony',
    u'sing':'sign',
    u'negarive':'negative',
    u'unknown':'unknown',
    u'molging': 'molding',
    u'neative': 'negative',
    u'ledg':'ledge',
    u'facode':'facade',
    u'legde':'ledge',
    u'unknwon':'unknown',
}

xml_warnings = {}
xml_errors = {}

for i, xml in enumerate(xml_files):
    print(i + 1, xml)
    a = Annotation(xml)
    assert isinstance(a, Annotation)
    a.remove_deleted()

    for o in a:
        assert isinstance(o, Object)
        date = o.date
        assert isinstance(date, datetime.datetime)
        day = date.date()

        if day not in days:
            days[day] = {}

        if o.polygon.username not in days[day]:
            days[day][o.polygon.username] = {}

        name = o.name
        if name not in NAMES:
            if unicode(name) in ALIASES:
                xml_warnings[xml] = xml_warnings.get(xml, [])
                xml_warnings[xml].append(u'{}: {}->{}'.format(o.polygon.username, name, ALIASES[name]))
                name = ALIASES[name]
            else:
                xml_errors[xml] = xml_errors.get(xml, [])
                xml_errors[xml].append(u'{}:{}'.format(o.polygon.username, name))

        if o.name not in days[day][o.polygon.username]:
            days[day][o.polygon.username][o.name] = []

        days[day][o.polygon.username][o.name].append(date)
        if name:
            name = name.encode('ascii', 'ignore')
            names[name] = names.get(name, 0) + 1


for day in days:
    print(day)
    total = 0
    for username in days[day]:
        combined = [t for name in days[day][username] for t in days[day][username][name]]
        combined.sort()
        print("  {:20}".format(username), ":", len(combined), "objects in", combined[-1] - combined[0])
        total += len(combined)
        for name in days[day][username]:
            name = name.encode('ascii', 'ignore')
            print("    {:20}".format(username), ":", end='')
            times = days[day][username][name]
            times.sort()
            duration = times[-1] - times[0]
            print(u"{:6}".format(len(times)), name)
    print("  Labeled", total, "objects.")



def make_url(xml):
    folder = basename(dirname(xml))
    stem = splitext(basename(xml))[0]

    pattern = "http://vision.csi.miamioh.edu/?folder={folder}&image={stem}.jpg"
    return pattern.format(folder=folder, stem=stem)


print("WARNINGS:")
for xml in xml_warnings:
    print(join(basename(dirname(xml)), basename(xml)))
    print(make_url(xml))
    for warning in xml_warnings[xml]:
        warning = warning.encode('ascii', 'ignore')
        print("    ", warning)

print("ERRORS:")
for xml in xml_errors:
    print(join(basename(dirname(xml)), basename(xml)))
    print(make_url(xml))
    for err in xml_errors[xml]:
        err = err.encode('ascii', 'ignore')
        print("    ", err)

print("TOTALS:")
for name in names:
    print("    {:30}:{: 6}".format(name, names[name]))