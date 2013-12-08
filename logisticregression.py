#!/usr/bin/env python -O
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

from numpy import asarray, asmatrix, dot, exp, log, ravel, zeros

from binaryclassifier import AUC, BinaryClassifier

# constants

TOL = 1.e-7
INF = float('inf')

# user classes


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
    >>> cm = L.leave_one_out(X, Y)
    >>> round(cm.accuracy, 2)
    0.97
    >>> round(AUC(LogisticRegression, X, Y), 2)
    1.0
    """

    def __repr__(self):
        return '{}(weights=[{}])'.format(self.__class__.__name__,
                        ', '.join('{: 02.3f}'.format(w) for w in self.W))

    @staticmethod
    def logis(alpha):
        ex = exp(alpha)
        return ex / (1. + ex)

    @staticmethod
    def Newton_Raphson(X, Y, n_iter=100):
        Xt = X.T
        W = zeros(X.shape[0])       # weights
        old_L = -INF                # log-likelihood of previous iteration
        for i in xrange(n_iter):
            # probability of a "hit" for each observation
            P = LogisticRegression.logis(dot(W, X))
            # check for (quasi)separation
            for p in P:
                if p == 0. or p == 1.:
                    return old_W
            # adjust your expectations of what it means to be "old"
            old_W = W
            # first derivative
            dXdY = dot(X, Y - P)
            # magic update
            W = old_W + ravel(dot(asmatrix(
                                  dot(X * (P * (1. - P)), Xt)).I, dXdY))
            # compute log-likelihood
            L = sum(Y * log(P) + (1. - Y) * log(1. - P))
            # check for convergence by looking for a stable log-likelihood
            if old_L + TOL > L:
                return W
            old_L = L
        # no convergence reached
        raise ValueError('Convergence failure')

    def train(self, X, Y, n_iter=100):
        self.W = LogisticRegression.Newton_Raphson(
                         asarray([[1.] + x for x in X]).T,
                         asarray([int(y == self.hit) for y in Y]),
                         n_iter=n_iter)

    def score(self, x):
        return dot(self.W, asarray([1.] + x))

    def classify(self, x):
        return self.hit if self.score(x) > 0. else self.miss


if __name__ == '__main__':
    import doctest
    doctest.testmod()
