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
# perceptron.py: A single-layer perceptron with some fancy features

from random import choice, random

from binaryclassifier import BinaryClassifier

# user functions

class Perceptron(BinaryClassifier):
    """
    Compute a binary classifier using the "pocket" and "rachet" variant
    of a single-layer perceptron, after:

    S.I. Gallant. 1990. Perceptron-based learning algorithms. IEEE 
    Transactions on Neural Networks 1: 179-191.

    >>> from random import seed
    >>> from csv import DictReader
    >>> from binaryclassifier import AUC
    >>> seed(62485)
    >>> X = []
    >>> Y = []
    >>> for row in DictReader(open('iris.csv', 'r')):
    ...     X.append([float(row['Sepal.Length']),
    ...               float(row['Sepal.Width']),
    ...               float(row['Petal.Length']), 
    ...               float(row['Petal.Width'])])
    ...     Y.append(row['Species'])
    >>> p = Perceptron(X, Y, 'versicolor')
    >>> cm = p.leave_one_out(X, Y)
    >>> round(cm.accuracy, 2)
    0.95
    >>> round(AUC(Perceptron, X, Y), 2)
    0.99
    """

    def __repr__(self):
        W = ', '.join('{: 02.3f}'.format(w) for w in self.W)
        return 'Perceptron({})'.format(W)

    def train(self, X, Y, n_iter=200):
        """
        Train with "pocket" and "rachet"    

        FYI: what Gallant calls $W$ (the correct weights) is here called P,
        and what he calls $\pi$ is here called self.W
        """
        assert all(len(Y) == len(feat) for feat in zip(*X))
        self.W = [0. for i in xrange(len(X[0]) + 1)]
        run_W = 0
        run_P = 0
        cor_W = 0
        cor_P = 0
        P = self.W
        for i in xrange(n_iter):
            while True:
                (x, y) = choice(zip(X, Y))
                if self.classify(x) == y:
                    run_W += 1
                    if run_W > run_P:
                        cm = self.evaluate(X, Y) # check current values
                        cor_W = cm.tp + cm.tn
                        if cor_W > cor_P:
                            P = self.W
                            run_P = run_W
                            cor_P = cor_W
                            if cm.fp + cm.fn == 0: # separability
                                return
                else:
                    s = 1 if y == self.hit else -1
                    self.W = [w + s * f for w, f in zip(self.W, [1.] + x)]
                    run_P = 0
                    break
        self.W = P

    def score(self, x):
        return sum(w * f for w, f in zip(self.W, [1.] + x))

    def classify(self, x):
        return self.hit if self.score(x) > 0. else self.miss

## perceptron with dropout

class Coin(object):
    """
    Simple class representing a weighted coin with a fixed p(heads)
    """
    
    def __init__(self, p_heads):
        assert 0. < p_heads < 1., 'Value error'
        self.p_heads = p_heads

    def __repr__(self):
        return 'coin(p(heads) = {:.7})'.format(self.p_heads)

    def flip_once(self):
        return 1 if random() < self.p_heads else 0

    def flip(self):
        while True:
            yield self.flip_once()


class PerceptronWithDropout(Perceptron):
    """
    Compute a binary classifier using the "pocket", "rachet", and "dropout"
    variant of a single-layer perceptron. "Dropout" potentially prevents
    bizarre schemes of co-adaptation and is based loosely on:

    G.E. Hinton, N.  Srivastava, A. Krizhevsky, I. Sutskever & R.R. 
    Salakhutdinov. 2012. Improving neural networks by preventing 
    co-adaptation of feature detectors. Ms., University of Toronto.

    >>> from random import seed
    >>> from csv import DictReader
    >>> from binaryclassifier import AUC
    >>> seed(62485)
    >>> X = []
    >>> Y = []
    >>> for row in DictReader(open('iris.csv', 'r')):
    ...     X.append([float(row['Sepal.Length']),
    ...               float(row['Sepal.Width']),
    ...               float(row['Petal.Length']), 
    ...               float(row['Petal.Width'])])
    ...     Y.append(row['Species'])
    >>> p = PerceptronWithDropout(X, Y, 'versicolor', .1)
    >>> cm = p.leave_one_out(X, Y)
    >>> round(cm.accuracy, 2)
    0.93
    >>> round(AUC(PerceptronWithDropout, X, Y, p_dropout=.1), 2)
    0.99
    """

    def __init__(self, X, Y, hit, p_dropout=.5):
        # make a coin for dropout
        self.coin = Coin(1. - p_dropout)
        # call superclass init
        super(PerceptronWithDropout, self).__init__(X, Y, hit)

    def __repr__(self):
        weight = ', '.join(str(round(i, 3)) for i in self.W)
        return 'PerceptronWithDropout({})'.format(weight)

    def train(self, X, Y, n_iter=200):
        """
        Training with "pocket", rachet", and "dropout"

        FYI: what Gallant calls $W$ (the correct weights) is here called P,
        and what he calls $\pi$ is here called self.W
        """
        assert all(len(Y) == len(feat) for feat in zip(*X))
        self.W = [0. for i in xrange(len(X[0]) + 1)]
        run_W = 0
        run_P = 0
        cor_W = 0
        cor_P = 0
        P = self.W
        for i in xrange(n_iter):
            while True:
                (x, y) = choice(zip(X, Y))
                if self.classify(x) == y:
                    run_W += 1
                    if run_W > run_P:
                        cm = self.evaluate(X, Y) # check current values
                        cor_W = cm.tp + cm.tn
                        if cor_W > cor_P:
                            P = self.W
                            run_P = run_W
                            cor_P = cor_W
                            if cm.fp + cm.fn == 0: # separability
                                return
                else:
                    sgn = 1 if y == self.hit else -1
                    self.W = [w + sgn * f for (w, f) in zip(self.W,
                                              [1.] + self._dropout(x))]
                    run_P = 0
                    break
        self.W = P

    def _dropout(self, x):
        return [d * f for (d, f) in zip(self.coin.flip(), x)]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
