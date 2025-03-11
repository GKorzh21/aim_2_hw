#!/opt/conda/envs/dsenv/bin/python

import sys
import os
from glob import glob
import logging

sys.path.append('.')
from model import fields

filter_cond_files = glob('filter_cond*.py')

if len(filter_cond_files) != 1:
    logging.critical("Must supply exactly one filter")
    sys.exit(1)

exec(open(filter_cond_files[0]).read())

outfields = [f for f in fields if f != 'label']

for line in sys.stdin:
    if line.startswith(fields[0]):
        continue

    values = line.rstrip().split('\t')
    row = dict(zip(outfields, values))

    if filter_cond(row):
        output = "\t".join([row[x] for x in outfields if x in row])
        print(output)
