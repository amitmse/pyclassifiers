#!/usr/bin/env python
# helper functions for the demos
# Kyle Gorman <gormanky@ohsu.edu>

def score_me(model, X, Y):
    print model
    print 'AUC:         {:.3}'.format(model.AUC(X, Y))
    model.leave_one_out(X, Y)
    print 'Accuracy:    {:.3}'.format(model.accuracy())
    print 'F1:          {:.3}'.format(model.F1())
    print 'MCC:         {:.3}'.format(model.MCC())
    print 'Precision:   {:.3}'.format(model.precision())
    print 'Sensitivity: {:.3}'.format(model.sensitivity())
    print 'Specificity: {:.3}'.format(model.specificity())
