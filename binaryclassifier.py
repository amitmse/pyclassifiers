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
# binaryclassifier.py: An abstract class for a binary classifier, a class
# representing a confusion matrix and associated binary classification
# metrics, and a class representing a single-attribute threshold finder

from __future__ import division

from math import sqrt
from operator import xor
from collections import Counter
from itertools import combinations

# constants

NAN = float('nan')
INF = float('inf')

# user functions


def AUC(model_class, X, Y, **kwargs):
    """
    Compute accuracy as measured by area under the ROC curve (AUC)
    using all-pairs analysis
    """
    assert all(len(Y) == len(feat) for feat in zip(*X))
    assert len(Y) > 1
    # init base model
    # select first outcome to be "hit", since it doesn't matter
    hit = Y[0]
    m = model_class(X, Y, hit, **kwargs)
    assert hasattr(m, 'train')
    # compute U statistic
    U = 0
    denom = 0
    for (i, j) in combinations(xrange(len(Y)), 2):
        if Y[i] == Y[j]:  # tie
            continue
        # train on held-out
        m.train(X[:i] + X[i + 1:j] + X[j + 1:],
                Y[:i] + Y[i + 1:j] + Y[j + 1:])
        # score the held-out data
        i_score = m.score(X[i])
        j_score = m.score(X[j])
        if i_score == j_score:  # tie
            continue
        U += xor(Y[i] == hit, i_score < j_score)
        denom += 1
    # compute AUC from U
    U /= denom  # now is AUC, though direction may be wrong
    return (U if U > .5 else 1. - U)

# user classes


class ConfusionMatrix(object):

    """
    Binary confusion matrix and various scoring methods

    To initialize, pass an iterable containing 2-tuples of booleans
    specifying for each response whether a hit or a miss was guessed
    and whether it was in fact a hit or a miss
    """

    def __repr__(self):
        return 'ConfusionMatrix(tp = {}, fp = {}, fn = {}, tn = {})'.format(self.tp, self.fp, self.fn, self.tn)

    def __init__(self, iterable):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        for (hit_guessed, is_hit) in iterable:
            if hit_guessed:
                if is_hit:
                    self.tp += 1
                else:
                    self.fp += 1
            elif is_hit:
                self.fn += 1
            else:
                self.tn += 1

    def __len__(self):
        return self.tp + self.fp + self.fn + self.tn

    # aggregate scores

    @property
    def accuracy(self):
        return (self.tp + self.tn) / len(self)

    def Fscore(self, ratio=1.):
        """
        F-score, by default F_1; ratio is the importance of recall vs.
        precision
        """
        assert ratio > 0.
        r_square = ratio * ratio
        P = self.precision
        R = self.recall
        return ((1. + r_square) * P * R) / (r_square * P + R)

    @property
    def F1(self):
        return self.Fscore()

    def Sscore(self, ns_ratio=1.):
        """
        Same idea as F-score, but defined in terms of specificity and
        sensitivity; ratio is the importance of specificity vs. sensitivity
        """
        assert ratio > 0.
        r_square = ratio * ratio
        Sp = self.specificity
        Se = self.sensitivity
        return ((1. + r_square) * Sp * Se) / (r_square * Sp * Se)

    @property
    def S1(self):
        return self.Sscore()

    @property
    def MCC(self):
        N = len(self)
        if N == 0:
            return NAN
        S = (self.tp + self.fn) / N
        P = (self.tp + self.fp) / N
        PS = P * S
        denom = sqrt(PS * (1. - S) * (1. - P))
        if denom == 0:
            return NAN
        return ((self.tp / N) - PS) / denom

    # specific scores

    # precision

    @property
    def precision(self):
        denom = self.tp + self.fp
        if denom == 0:
            return INF
        return self.tp / denom

    @property
    def PPV(self):
        return self.precision

    # recall

    @property
    def recall(self):
        denom = self.tp + self.fn
        if denom == 0:
            return INF
        return self.tp / denom

    @property
    def sensitivity(self):
        return self.recall

    @property
    def TPR(self):
        return self.recall

    # specificity

    @property
    def specificity(self):
        denom = self.fp + self.tn
        if denom == 0:
            return INF
        return self.tp / denom

    @property
    def TNR(self):
        return self.specificity

    # others, rarely used

    @property
    def FPR(self):
        denom = self.fp + self.tp
        if denom == 0:
            return INF
        return self.fp / denom

    @property
    def NPV(self):
        denom = self.tn + self.fn
        if denom == 0:
            return INF
        return self.tn / denom

    @property
    def FDR(self):
        denom = self.fp + self.tp
        if denom == 0:
            return INF
        return self.fp / denom


class BinaryClassifier(object):

    """
    Dummy class representing a binary classifier
    """

    # FIXME implement me!
    def __repr__(self):
        raise NotImplementedError

    def __init__(self, X, Y, hit, **kwargs):
        self.hit = hit
        self.miss = None
        for y in Y:  # figure out what the "miss" is called
            if y != hit:
                self.miss = y
                break
        if self.miss == None:
            raise ValueError('Outcomes are invariant')
        self.train(X, Y, **kwargs)

    # FIXME implement me!
    def score(self, x):
        raise NotImplementedError

    # FIXME implement me!
    def classify(self, x):
        raise NotImplementedError

    # FIXME implement me!
    def train(self, X, Y):
        raise NotImplementedError

    def evaluate(self, X, Y):
        """
        Compute scores measuring the classification Y ~ X
        """
        assert all(len(Y) == len(feat) for feat in zip(*X))
        return ConfusionMatrix((self.classify(x) == self.hit,
                                y == self.hit) for x, y in zip(X, Y))

    def _LOO_gen(self, X, Y):
        """
        Generator for leave-one-out cross-validation
        """
        for i in xrange(1, len(Y)):
            self.train(X[:i] + X[i + 1:], Y[:i] + Y[i + 1:])
            yield self.classify(X[i]) == self.hit, Y[i] == self.hit

    def leave_one_out(self, X, Y):
        """
        Score using leave-one-out cross-validation
        """
        assert all(len(Y) == len(feat) for feat in zip(*X))
        return ConfusionMatrix(self._LOO_gen(X, Y))


class Threshold(object):

    """
    Class representing a single split applied to a vector of continuous
    data, used to construct classifiers

    # invariant data
    >>> Threshold([1, 2, 3], ['T', 'T', 'T'], 'T')
    Threshold(MISS < -inf < HIT)

    # separable data
    >>> Threshold([1, 2, 3], ['T', 'T', 'F'], 'T')
    Threshold(HIT < inf < MISS)

    >>> from csv import DictReader
    >>> S = []
    >>> Y = []
    >>> for row in DictReader(open('iris.csv', 'r')):
    ...     S.append(float(row['Petal.Width']))
    ...     Y.append(row['Species'])
    >>> Threshold(S, Y, 'virginica')
    Threshold(MISS < 1.65 < HIT)
    >>> Threshold(S, Y, 'versicolor')
    Threshold(HIT < 1.65 < MISS)
    """

    def __repr__(self):
        if self.upper_is_hit:
            return 'Threshold(MISS < {:.3} < HIT)'.format(self.split)
        else:
            return 'Threshold(HIT < {:.3} < MISS)'.format(self.split)

    def __init__(self, S, Y, hit):
        (my_S, my_Y) = (list(i) for i in zip(*sorted(
                                         zip(S, (y == hit for y in Y)))))
        N = len(my_Y)
        # initializing at the infinite left edge...
        self.split = -INF
        upper_h = sum(my_Y)
        upper_m = N - upper_h
        self.upper_is_hit = (upper_h >= upper_m)
        self.accuracy = abs(upper_h - upper_m) / N  # best so far
        # check for invariance
        if self.accuracy == 1.:
            return
        lower_h = 0
        lower_m = 0
        # hacky stuff...
        prev_s = my_S[0]  # doesn't run the first iteration that way...
        my_S.append(INF)           # so the last split point is infinite
        for (i, y) in enumerate(my_Y, 1):
            if my_S[i] != prev_s:  # scores have changed
                acc_s = ((upper_h - upper_m) + (lower_m - lower_h)) / N
                acc = abs(acc_s)
                if acc > self.accuracy:
                    # set the split as the mean between the current and
                    # next point, a sort of max-margin principle
                    self.split = (my_S[i - 1] + my_S[i]) / 2.
                    # compute new unscaled best accuracy
                    self.accuracy = acc
                    # set which side is up
                    self.upper_is_hit = (acc_s > 0.)
                    # check for separability
                    if self.accuracy == N:
                        return
                prev_s = my_S[i]
            # deal with y
            if y:
                upper_h -= 1
                lower_h += 1
            else:
                upper_m -= 1
                lower_m += 1

    def is_hit(self, s):
        return xor(s < self.split, self.upper_is_hit)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
