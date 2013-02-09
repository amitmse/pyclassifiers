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
# logistic_regression.py: classification using logistic regression

from numpy.linalg import inv
from numpy import allclose, array, dot, exp, log, zeros

from binaryclassifier import BinaryClassifier

_TOL = 1.e-7

class LogisticRegression(BinaryClassifier):
    """
    Compute a logistic regression classifier, stopping estimation when
    log-likelihood is stable, or when at least one training observation 
    hits ceiling or floor probability. This is loosely based on code by 
    Jeffrey Whitaker.

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

    >>> L = LogisticRegression(X, Y, 'versicolor')
    >>> L.leave_one_out(X, Y)
    >>> round(L.accuracy(), 2)
    0.94
    >>> round(L.AUC(X, Y), 3)
    0.995
    """

    def __repr__(self):
        W = ', '.join('{: 02.3f}'.format(w) for w in self.W)
        return 'LogisticRegression({})'.format(W)

    @staticmethod
    def logis(alpha):
        ex = exp(alpha)
        return ex / (1. + ex)

    @staticmethod
    def Newton_Raphson(X, Y, n_iter=100):
        Xt = X.T
        W = zeros(X.shape[0])    # weights
        old_L = float('-inf')    # log-likelihood of previous iteration
        for i in xrange(n_iter):
            # probabilities that each observation is a hit
            p = LogisticRegression.logis(dot(W, X))
            # (quasi)separation
            if any(p == 0.) or any(p == 1.):
                return old_W
            # readjust your expectations of "what's old", as you age
            old_W = W
            # first derivative
            dXdY = dot(X, Y - p) 
            # magic update
            W = old_W + dot(inv(dot(X * (p * (1. - p)), Xt)), dXdY)
            # check for convergence using log-likelihood change
            L = sum(Y * log(p) + (1. - Y) * log(1. - p))
            if old_L + _TOL > L:
                return W
            old_L = L
        # no convergence reached!
        raise ValueError('Convergence failure')

    def train(self, X, Y, n_iter=100):
        self.W = LogisticRegression.Newton_Raphson(
                       array([[1.] + x for x in X]).T,
                       array([int(y == self.hit) for y in Y]),
                       n_iter=n_iter)

    def score(self, x):
        return dot(self.W, array([1.] + x))

    def classify(self, x):
        return self.hit if self.score(x) > 0. else self.miss


if __name__ == '__main__':
    import doctest
    doctest.testmod()
