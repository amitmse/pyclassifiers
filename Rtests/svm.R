# Compare those results to what you'd get with a defualt SVM, just for fun
# Kyle Gorman <gormanky@ohsu.edu>

require(e1071)

## scoring metrics

precision <- function(tp, tn, fp, fn) {
    (tp / (tp + fp))
}

recall <- function(tp, tn, fp, fn) {
    (tp / (tp + fn))
}

accuracy <- function(tp, tn, fp, fn) {
    num <- tp + tn
    (num / (num + fp + fn))
}

F1 <- function(tp, tn, fp, fn) {
    p <- precision(tp, tn, fp, fn)
    r <- recall(tp, tn, fp, fn)
    ((2. * p * r) / (p + r))
}

## helpers

Fdrop <- function(data) {
    data[] <- lapply(data, function(x) x[, drop=1])
    return(data)
}

## read in
d <- Fdrop(subset(iris, Species != 'setosa'))

## score
model <- svm(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=d, kernel='linear', cross=nrow(d))
right <- ifelse(model$accuracies == 100, TRUE, FALSE)
hit <- 'versicolor'
tp <- sum(right  & d$Species == hit)
tn <- sum(right  & d$Species != hit)
fp <- sum(!right & d$Species != hit)
fn <- sum(!right & d$Species == hit)
cat('Accuracy:', round(accuracy(tp, tn, fp, fn), 3), '\n')
cat('F1:', round(F1(tp, fn, fp, fn), 3), '\n')
