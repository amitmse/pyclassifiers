This package contains some statistical classifiers, largely in pure
Python. Some code currently depends on `numpy`; I'd like to move away
from this dependency ultimately. Whereas you can surely get classifiers
elsewhere, "elsewhere" rarely provides high quality means to evaluate
classification quality. Here, you will find no such lacuna; there are
methods for all the better-known metrics, as well as all-pairs AUC and
leave-one-out scoring. These can be hard to write and test, so I think
this is a big win. One can quite quickly construct a new binary classifier
by subclassing `binaryclassifier.BinaryClassifier` and defining the 
`train`, `score`, and `classify` methods!

Development on this package is solely based on my needs (mostly binary 
classification) and wants (staying away from R) and so will be sporadic.
Multiclass classification is not implausible, someday, though it is not
something I need right now.

`binaryclassifier.py`
=====================

An abstract base class for binary classifiers; also includes a complex
thresholding class used by many classifiers

`lda.py`
========

Traditional linear discriminant analysis using the covariance matrix, 
with the threshold is chosen to maximize classification accuracy.

`perceptron.py`
===============

A perceptron classifier which is trained with "ratchet" and "pocket";
this will be of less interest to people interested in the renowned
online learning characteristics of the perceptron, but of more interest
to those interested in good classification. There is also the option to
use Hinton-style "dropout" so as to prevent bizarre co-adaptation.

`stump.py`
==========

A simple, univariate "decision rule" or "stump" classifier.
