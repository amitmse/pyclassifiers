#!/usr/bin/env python
#
# Copyright (c) 2013 Kyle Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# csv_help.py: some helper methods for dealing with CSV files

from csv import DictReader
from collections import defaultdict


def table(X, Y):
    d = defaultdict(list)
    for (X, Y) in zip(X, Y):
        d[Y].append(X)
    return d.values()


def read_csv(fid, drop_NA=[]):
    """
    Read a CSV file into a dictionary in which columns are represented as
    key-lists in a dictionary; for any column name in drop_NA, if
    a row has a NA value for that row, the row is discarded
    """
    source = DictReader(open(fid, 'r'))
    data = {field: [] for field in source.fieldnames}
    for row in source:
        if any(k in drop_NA and v == 'NA' for (k, v) in row.iteritems()):
            continue
        for (key, val) in row.iteritems():
            try:
                val = float(val)
            except ValueError:
                pass
            data[key].append(val)
    return data


def rotate(X):
    return [list(i) for i in zip(*X)]
