This package contains several classifiers, largely in pure Python. Some 
code currently depends on `numpy`; I may move away from this dependency 
eventually. Whereas you can surely get classifiers elsewhere, "elsewhere" 
rarely provides high quality means to evaluate classification quality. 
Here, you will find no such lacuna; there are methods for all the
better-known metrics, as well as all-pairs AUC and leave-one-out scoring.
These can be hard to write and test, so I think this is a big win. One can
quickly construct a new binary classifier just by subclassing 
`binaryclassifier.BinaryClassifier` and defining the `train`, `score`, and
`classify` methods.

Development on this package is solely based on my needs (mostly binary
classification with continuous features) and wants (staying away from R)
and so will be sporadic. Multiclass classification is not implausible,
someday, though it is not something I need right now. Contributions are 
always welcome.

Thanks to Steven Bedrick, Constantine Lignos, and Brian Roark for help and
feedback.

`binaryclassifier.py`
=====================

An abstract base class for binary classifiers; also includes a class 
representing a confusion matrix, and a complex thresholding class used by
many classifiers.

`lda.py`
========

Traditional linear discriminant analysis using the covariance matrix, with
the threshold is chosen to maximize classification accuracy.

`logisticregression.py`
=======================

Classifier based on a logistic regression model using the Newton-Raphson
algorithm. Some tricky stuff deals with the spectre of (quasi)separability.

`perceptron.py`
===============

A perceptron classifier which is trained with "ratchet" and "pocket"; this
will be of less interest to people interested in the renowned online 
learning characteristics of the perceptron, but of more interest to those 
interested in good classification. There is also the option to use 
Hinton-style "dropout" so as to prevent bizarre co-adaptation.

`stump.py`
==========

A simple, univariate "decision rule" or "stump" classifier: it selects the
best predictor column (using accuracy) and then uses this going forward.
