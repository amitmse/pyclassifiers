#!/usr/bin/env python -u 
# Scoring with the perceptron classifier 
# Kyle Gorman <gormanky@ohsu.edu>

from itertools import product
from termcolor import colored

from csv_help import read_csv
from demo_help import score_me
from perceptron import Perceptron, PerceptronWithDropout

_hit = 'versicolor'

if __name__ == '__main__':

    # read in
    d = read_csv('iris.csv') # setosa has alread been culled
    X = [col for (lab, col) in d.iteritems() if lab != 'Species']
    X = [list(t) for t in zip(*X)] # rotate it
    Y = d['Species']

    print colored('No dropout', 'red')
    p = Perceptron(X, Y, _hit)
    score_me(p, X, Y)
    for i in xrange(1, 11):
        pdrop = i / 20.
        print colored('Dropout p = {:.2}'.format(pdrop), 'red')
        p = PerceptronWithDropout(X, Y, _hit, pdrop)
        score_me(p, X, Y)
