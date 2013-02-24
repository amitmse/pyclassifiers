#!/usr/bin/env python
#
# Copyright (c) 2013 Kyle Gorman <gormanky@ohsu.edu>
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
# Given a binomial variable Y (represented as a string) and a continuous
# variable X (represented as number), find a split-point which minimizes
# classification error, and report classification statistics

from binaryclassifier import BinaryClassifier, Threshold, NINF


class Stump(BinaryClassifier):
    """
    Compute a classifier which makes a single "cut" in a continuous
    predictor vector X that splits the outcomes into "hit" and "miss"
    so as to maximize the number of correct classifications

    >>> from csv import DictReader
    >>> from binaryclassifier import AUC
    >>> X = []
    >>> Y = []
    >>> for row in DictReader(open('iris.csv', 'r')):
    ...     X.append([float(row['Sepal.Length']),
    ...               float(row['Sepal.Width']),
    ...               float(row['Petal.Length']),
    ...               float(row['Petal.Width'])])
    ...     Y.append(row['Species'])
    >>> s = Stump(X, Y, 'versicolor')
    >>> cm = s.leave_one_out(X, Y)
    >>> round(cm.accuracy, 2)
    0.88
    >>> round(AUC(Stump, X, Y), 2)
    0.98
    """

    def __repr__(self):
        return 'Stump(col={}, {})'.format(self.best_col, self.thresh)

    def train(self, X, Y):
        """
        Find the optimal split point
        """
        # init w/ leftmost column
        self.best_col = None  # index of most informative column in X
        best_accuracy = NINF
        # go through other columns
        for (i, col) in enumerate(zip(*X)):
            thresh = Threshold(col, Y, self.hit)
            accuracy = thresh.accuracy
            if accuracy > best_accuracy:
                self.thresh = thresh
                self.best_col = i
                best_accuracy = accuracy
                if best_accuracy == 1.:  # separable
                    return

    def score(self, x):
        return x[self.best_col]

    def classify(self, x):
        return self.hit if self.thresh.is_hit(self.score(x)) else self.miss


if __name__ == '__main__':
    import doctest
    doctest.testmod()
