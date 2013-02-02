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
# lda.py: classification using a traditional linear discriminant

from numpy.linalg import inv
from numpy import cov, mean, dot

from binaryclassifier import BinaryClassifier, Threshold

class LDA(BinaryClassifier):
    """
    Compute a linear discriminant classifier

    >>> from random import seed
    >>> seed(62485)

    # read in Iris data and score
    >>> from csv import DictReader
    >>> X = []
    >>> Y = []
    >>> for row in DictReader(open('iris.csv', 'r')):
    ...     X.append([float(row['Sepal.Length']),
    ...               float(row['Sepal.Width']),
    ...               float(row['Petal.Length']), 
    ...               float(row['Petal.Width'])])
    ...     Y.append(row['Species'])
    >>> versicolor = LDA(X, Y, 'versicolor')
    >>> virginica = LDA(X, Y, 'virginica')
    >>> versicolor.leave_one_out(X, Y)
    >>> virginica.leave_one_out(X, Y)
    >>> versicolor.accuracy() == virginica.accuracy()
    True
    >>> round(versicolor.accuracy(), 2)
    0.98
    >>> round(versicolor.AUC(X, Y), 2)
    1.0
    """

    def __repr__(self):
        return 'LDA({})'.format(', '.join(self.W))

    def train(self, X, Y):
        # construct table of values
        table = {self.hit: [], self.miss: []}
        for (x, y) in zip(X, Y):
            table[y].append(x)
        # transpose columns in table
        for col in (self.hit, self.miss):
            table[col] = zip(*table[col])
        # compute difference in means
        mu_hit  = mean(table[self.hit],  axis=1)
        mu_miss = mean(table[self.miss], axis=1)
        md = mu_hit - mu_miss
        # compute weights
        self.W = dot(inv(cov(table[self.hit]) + cov(table[self.miss])), md)
        self.thresh = Threshold([self.score(x) for x in X],
                                [y == self.hit for y in Y])
        # population confusion matrix
        self.evaluate(X, Y)
        
    def score(self, x):
        return sum(w * f for w, f in zip(self.W, x))

    def classify(self, x):
        return self.hit if self.thresh.is_hit(self.score(x)) else self.miss


if __name__ == '__main__':
    import doctest
    doctest.testmod()
