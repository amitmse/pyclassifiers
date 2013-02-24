#!/usr/bin/env python
# helper functions for the demos
# Kyle Gorman <gormanky@ohsu.edu>


def score_me(model, X, Y):
    print model
    cm = model.leave_one_out(X, Y)
    print 'Accuracy:    {:.3f}'.format(cm.accuracy)
    print 'F1:          {:.3f}'.format(cm.F1)
    print 'MCC:         {:.3f}'.format(cm.MCC)
    print 'Precision:   {:.3f}'.format(cm.precision)
    print 'Sensitivity: {:.3f}'.format(cm.sensitivity)
    print 'Specificity: {:.3f}'.format(cm.specificity)
