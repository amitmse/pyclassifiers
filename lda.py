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

# user classes

class LDA(BinaryClassifier):
    """
    Compute a linear discriminant classifier

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
    >>> L = LDA(X, Y, 'versicolor')
    >>> cm = L.leave_one_out(X, Y)
    >>> round(cm.accuracy, 2)
    0.97
    >>> round(AUC(LDA, X, Y), 2)
    1.0
    """

    def __repr__(self):
        W = ', '.join('{: 02.3f}'.format(w) for w in self.W)
        return 'LDA({})'.format(W)

    def train(self, X, Y):
        # construct table of values
        table = {self.hit: [], self.miss: []}
        for (x, y) in zip(X, Y):
            table[y].append(x)
        # transpose columns in table
        for col in (self.hit, self.miss):
            table[col] = zip(*table[col])
        # compute weights
        self.W = dot(inv(cov(table[self.hit]) + cov(table[self.miss])),
                             mean(table[self.hit],  axis=1) -
                             mean(table[self.miss], axis=1))
        self.thresh = Threshold((self.score(x) for x in X), Y, self.hit)
        
    def score(self, x):
        return sum(w * f for w, f in zip(self.W, x))

    def classify(self, x):
        return self.hit if self.thresh.is_hit(self.score(x)) else self.miss


if __name__ == '__main__':
    import doctest
    doctest.testmod()
